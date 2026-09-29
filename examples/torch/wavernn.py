# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# Original model: pytorch/audio (torchaudio/models/wavernn.py)
#
# This adaptation adds tensor shape type annotations for pyrefly.

# ## Inventory
# - [x] ResBlock.__init__ — Dims: n_freq
# - [x] ResBlock.forward
# - [x] MelResNet.__init__ — Dims: n_freq, n_hidden, n_output, kernel_size; int: n_res_block
# - [x] MelResNet.forward
# - [x] Stretch2d.__init__ — Dims: time_scale, freq_scale
# - [x] Stretch2d.forward
# - [x] UpsampleNetwork.__init__ — Dims: n_freq, n_hidden, n_output, kernel_size; int: n_res_block, upsample_scales
# - [x] UpsampleNetwork.forward
# - [x] WaveRNN.__init__ — Dims: n_classes, n_rnn, n_fc, n_freq, n_hidden, n_output, kernel_size; int: n_res_block, hop_length
# - [x] WaveRNN.forward
# - [x] WaveRNN.infer

import math
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from shape_extensions import Elements, Int, IntTuple, IntVar
    from torch import Tensor


class ResBlock[NF: IntVar = 128](nn.Module):
    def __init__(self, n_freq: Int[NF] = 128) -> None:
        super().__init__()
        self.resblock_model = nn.Sequential(
            nn.Conv1d(
                in_channels=n_freq, out_channels=n_freq, kernel_size=1, bias=False
            ),
            nn.BatchNorm1d(n_freq),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=n_freq, out_channels=n_freq, kernel_size=1, bias=False
            ),
            nn.BatchNorm1d(n_freq),
        )

    def forward[B: IntVar, L: IntVar](
        self, specgram: Tensor[[B, NF, L]]
    ) -> Tensor[[B, NF, L]]:
        residual = self.resblock_model(specgram)
        out = residual + specgram
        return out


class MelResNet[
    NF: IntVar = 128,
    NH: IntVar = 128,
    NO: IntVar = 128,
    K: IntVar = 5,
](nn.Module):
    def __init__(
        self,
        n_res_block: int = 10,
        n_freq: Int[NF] = 128,
        n_hidden: Int[NH] = 128,
        n_output: Int[NO] = 128,
        kernel_size: Int[K] = 5,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(
            in_channels=n_freq,
            out_channels=n_hidden,
            kernel_size=kernel_size,
            bias=False,
        )
        self.bn_in = nn.BatchNorm1d(n_hidden)
        self.relu_in = nn.ReLU(inplace=True)
        self.res_blocks = nn.ModuleList(
            [ResBlock(n_hidden) for _ in range(n_res_block)]
        )
        self.conv_out = nn.Conv1d(
            in_channels=n_hidden, out_channels=n_output, kernel_size=1
        )

    def forward[B: IntVar, L: IntVar](
        self, specgram: Tensor[[B, NF, L]]
    ) -> Tensor[[B, NO, (1 + L) + (-1 * K)]]:
        x = self.conv_in(specgram)
        x = self.bn_in(x)
        x = self.relu_in(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.conv_out(x)
        return x


class Stretch2d[TS: IntVar, FS: IntVar](nn.Module):
    def __init__(self, time_scale: Int[TS], freq_scale: Int[FS]) -> None:
        super().__init__()
        self.freq_scale = freq_scale
        self.time_scale = time_scale

    def forward[B: IntVar, C: IntVar, F: IntVar, T: IntVar](
        self, specgram: Tensor[[B, C, F, T]]
    ) -> Tensor[[B, C, F * FS, T * TS]]:
        x = specgram.repeat_interleave(self.freq_scale, -2)
        x = x.repeat_interleave(self.time_scale, -1)
        return x


class UpsampleNetwork[
    NF: IntVar = 128,
    NH: IntVar = 128,
    NO: IntVar = 128,
    K: IntVar = 5,
](nn.Module):
    def __init__(
        self,
        upsample_scales: list[int],
        n_res_block: int = 10,
        n_freq: Int[NF] = 128,
        n_hidden: Int[NH] = 128,
        n_output: Int[NO] = 128,
        kernel_size: Int[K] = 5,
    ) -> None:
        super().__init__()

        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale
        self.total_scale: int = total_scale

        self.indent = (kernel_size - 1) // 2 * total_scale
        self.resnet = MelResNet(n_res_block, n_freq, n_hidden, n_output, kernel_size)
        self.resnet_stretch = Stretch2d(total_scale, 1)

        up_layers: list[nn.Module] = []
        for scale in upsample_scales:
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=(1, scale * 2 + 1),
                padding=(0, scale),
                bias=False,
            )
            torch.nn.init.constant_(conv.weight, 1.0 / (scale * 2 + 1))
            up_layers.append(stretch)
            up_layers.append(conv)
        self.upsample_layers = nn.Sequential(*up_layers)

    def forward[B: IntVar, T: IntVar](
        self, specgram: Tensor[[B, NF, T]]
    ) -> tuple[Tensor[[B, NF, Any]], Tensor[[B, NO, Any]]]:
        resnet_output = self.resnet(specgram)
        resnet_output_4d = resnet_output.unsqueeze(1)
        resnet_output_4d = self.resnet_stretch(resnet_output_4d)
        # total_scale is int from dynamic loop product → time dim becomes Any
        resnet_out = resnet_output_4d.squeeze(1)

        specgram_4d = specgram.unsqueeze(1)
        upsampling_raw = self.upsample_layers(specgram_4d)
        upsampling_sliced = upsampling_raw.squeeze(1)[:, :, self.indent : -self.indent]
        # Annotation fallback: NF freq channels preserved through upsampling
        # Receipt: Sequential(*list) with dynamic layer count; NF is bridge dim
        upsampling_output: Tensor[[B, NF, Any]] = upsampling_sliced

        return upsampling_output, resnet_out


class WaveRNN[
    NC: IntVar,
    NR: IntVar = 512,
    NFC: IntVar = 512,
    NF: IntVar = 128,
    NH: IntVar = 128,
    NO: IntVar = 128,
    K: IntVar = 5,
](nn.Module):
    def __init__(
        self,
        upsample_scales: list[int],
        n_classes: Int[NC],
        hop_length: int,
        n_res_block: int = 10,
        n_rnn: Int[NR] = 512,
        n_fc: Int[NFC] = 512,
        kernel_size: Int[K] = 5,
        n_freq: Int[NF] = 128,
        n_hidden: Int[NH] = 128,
        n_output: Int[NO] = 128,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self._pad = (kernel_size - 1 if kernel_size % 2 else kernel_size) // 2
        self.n_rnn = n_rnn
        self.n_aux = n_output // 4
        self.hop_length = hop_length
        self.n_classes = n_classes
        self.n_bits: int = int(math.log2(n_classes))

        total_scale = 1
        for upsample_scale in upsample_scales:
            total_scale *= upsample_scale
        if total_scale != hop_length:
            raise ValueError(
                f"Expected: total_scale == hop_length, but found {total_scale} != {hop_length}"
            )

        self.upsample = UpsampleNetwork(
            upsample_scales, n_res_block, n_freq, n_hidden, n_output, kernel_size
        )
        self.fc = nn.Linear(n_freq + self.n_aux + 1, n_rnn)

        self.rnn1 = nn.GRU(n_rnn, n_rnn, batch_first=True)
        self.rnn2 = nn.GRU(n_rnn + self.n_aux, n_rnn, batch_first=True)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(n_rnn + self.n_aux, n_fc)
        self.fc2 = nn.Linear(n_fc + self.n_aux, n_fc)
        self.fc3 = nn.Linear(n_fc, n_classes)

    # The result is (B, 1, T, NC) at runtime, but the aux concatenations below
    # leave the batch and time axes gradual, so a bare `Tensor` is what this port
    # can actually prove.
    def forward[B: IntVar, T: IntVar](
        self, waveform: Tensor[[B, 1, T]], specgram: Tensor[[B, 1, NF, Any]]
    ) -> Tensor:
        if waveform.size(1) != 1:
            raise ValueError("Require the input channel of waveform is 1")
        if specgram.size(1) != 1:
            raise ValueError("Require the input channel of specgram is 1")

        waveform_2d = waveform.squeeze(1)
        specgram_3d = specgram.squeeze(1)

        batch_size = waveform_2d.size(0)
        h1 = torch.zeros(
            1,
            batch_size,
            self.n_rnn,
            dtype=waveform_2d.dtype,
            device=waveform_2d.device,
        )
        h2 = torch.zeros(
            1,
            batch_size,
            self.n_rnn,
            dtype=waveform_2d.dtype,
            device=waveform_2d.device,
        )

        specgram_up, aux = self.upsample(specgram_3d)
        specgram_up_t = specgram_up.transpose(1, 2)
        aux_t = aux.transpose(1, 2)

        aux_idx = [self.n_aux * i for i in range(5)]
        a1 = aux_t[:, :, aux_idx[0] : aux_idx[1]]
        a2 = aux_t[:, :, aux_idx[1] : aux_idx[2]]
        a3 = aux_t[:, :, aux_idx[2] : aux_idx[3]]
        a4 = aux_t[:, :, aux_idx[3] : aux_idx[4]]

        # Nothing upstream relates the waveform length T to the upsampled aux
        # extent: the upsampling factor is a product accumulated over a
        # `list[int]`, so the aux time axis is already Any here. Comparing the two
        # is undecidable, so this cat recovers gradually and every value derived
        # from it inherits that.
        x = torch.cat((waveform_2d.unsqueeze(-1), specgram_up_t, a1), dim=-1)
        # Each Linear and GRU still pins its own output feature count, so the
        # trailing axis stays exact even though the leading rank never recovers.
        x = self.fc(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat((x, a2), dim=-1)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat((x, a3), dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)

        x = torch.cat((x, a4), dim=-1)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        result = x.unsqueeze(1)
        return result

    def infer[B: IntVar](
        self, specgram: Tensor[[B, NF, Any]], lengths: Tensor[[B]] | None = None
    ) -> tuple[Tensor, Tensor[[B]] | None]:
        device = specgram.device
        dtype = specgram.dtype

        specgram_padded: Tensor[[B, NF, Any]] = F.pad(  # type: ignore[pyrefly:bad-assignment]
            specgram, (self._pad, self._pad)
        )

        specgram_up, aux = self.upsample(specgram_padded)
        if lengths is not None:
            lengths = lengths * self.upsample.total_scale

        output: list[Tensor] = []
        b_size, _, seq_len = specgram_up.size()

        h1 = torch.zeros((1, b_size, self.n_rnn), device=device, dtype=dtype)
        h2 = torch.zeros((1, b_size, self.n_rnn), device=device, dtype=dtype)
        x = torch.zeros((b_size, 1), device=device, dtype=dtype)

        aux_split = [aux[:, self.n_aux * i : self.n_aux * (i + 1), :] for i in range(4)]

        for i in range(seq_len):
            m_t = specgram_up[:, :, i]

            a1_t, a2_t, a3_t, a4_t = [a[:, :, i] for a in aux_split]

            x = torch.cat((x, m_t, a1_t), dim=1)
            x = self.fc(x)
            _, h1 = self.rnn1(x.unsqueeze(1), h1)

            x = x + h1[0]
            inp = torch.cat((x, a2_t), dim=1)
            _, h2 = self.rnn2(inp.unsqueeze(1), h2)

            x = x + h2[0]
            x = torch.cat((x, a3_t), dim=1)
            x = F.relu(self.fc1(x))

            x = torch.cat((x, a4_t), dim=1)
            x = F.relu(self.fc2(x))

            logits = self.fc3(x)

            posterior = F.softmax(logits, dim=1)

            x = torch.multinomial(posterior, 1).float()
            # The scalar divisor is a `float`, so tensor arithmetic preserves the shape.
            x = 2 * x / (2 ** (self.n_bits * 1.0) - 1.0) - 1.0

            output.append(x)

        # dynamic loop accumulation → stack/permute result is bare
        result = torch.stack(output).permute(1, 2, 0)
        return result, lengths


def _smoke_test() -> None:
    model = WaveRNN(upsample_scales=[5, 5, 8], n_classes=512, hop_length=200)
    waveform = torch.randn(2, 1, 6000)
    specgram = torch.randn(2, 1, 128, 30)
    out = model(waveform, specgram)
