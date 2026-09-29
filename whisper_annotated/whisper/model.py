from __future__ import annotations

import base64
import gzip
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function

if TYPE_CHECKING:
    from shape_extensions import Int, IntTuple, IntVar

try:
    from torch.nn.functional import scaled_dot_product_attention

    SDPA_AVAILABLE = True
except (ImportError, RuntimeError, OSError):
    scaled_dot_product_attention = None
    SDPA_AVAILABLE = False


@dataclass
class ModelDimensions[
    NMels: IntVar,
    NACtx: IntVar,
    NAState: IntVar,
    NAHead: IntVar,
    NALayer: IntVar,
    NVocab: IntVar,
    NTCtx: IntVar,
    NTState: IntVar,
    NTHead: IntVar,
    NTLayer: IntVar,
]:
    n_mels: Int[NMels]
    n_audio_ctx: Int[NACtx]
    n_audio_state: Int[NAState]
    n_audio_head: Int[NAHead]
    n_audio_layer: Int[NALayer]
    n_vocab: Int[NVocab]
    n_text_ctx: Int[NTCtx]
    n_text_state: Int[NTState]
    n_text_head: Int[NTHead]
    n_text_layer: Int[NTLayer]


class LayerNorm(nn.LayerNorm):
    def forward[S: IntTuple](self, input: Tensor[S]) -> Tensor[S]:
        return super().forward(input.float()).type(input.dtype)


class Linear[IN: IntVar, OUT: IntVar](nn.Linear):
    # NOTE: the base is deliberately not subscripted: a generic base alias is
    # evaluated eagerly at runtime and real torch classes are not
    # subscriptable without shape_extensions installed. Precision is recovered
    # through the explicit __init__ / forward below.
    # An explicit forwarding __init__ is required: the inherited generic
    # __init__ binds Linear[int, int] instead of propagating the arguments.
    def __init__(
        self,
        in_features: Int[IN],
        out_features: Int[OUT],
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward[Bs: IntTuple](
        self, input: Tensor[[*Bs, IN]]
    ) -> Tensor[[*Bs, OUT]]:
        # self.weight/self.bias come from the unsubscripted base; re-establish
        # the (in_features, out_features) contract they were built with.
        weight: Tensor[[OUT, IN]] = self.weight
        bias: Tensor[[OUT]] | None = self.bias
        return F.linear(
            input,
            weight.to(input.dtype),
            None if bias is None else bias.to(input.dtype),
        )


class Conv1d[
    InC: IntVar,
    OutC: IntVar,
    K: IntVar,
    S: IntVar = 1,
    P: IntVar = 0,
    D: IntVar = 1,
](nn.Conv1d):
    # NOTE: the base is deliberately not subscripted: a generic base alias is
    # evaluated eagerly at runtime and real torch classes are not
    # subscriptable without shape_extensions installed. Precision is recovered
    # through the explicit __init__ / forward below.
    # An explicit forwarding __init__ is required: the inherited generic
    # __init__ binds gradual type args instead of propagating the arguments.
    def __init__(
        self,
        in_channels: Int[InC],
        out_channels: Int[OutC],
        kernel_size: Int[K],
        stride: Int[S] = 1,
        padding: Int[P] = 0,
        dilation: Int[D] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        # super().__init__ sees the unsubscripted base, whose S/P/D take the
        # stub's PEP 696 defaults; the values are forwarded opaquely and the
        # subclass signature above tracks them.
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,  # type: ignore[pyrefly:bad-argument-type]
            padding,  # type: ignore[pyrefly:bad-argument-type]
            dilation,  # type: ignore[pyrefly:bad-argument-type]
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward[B: IntVar, L: IntVar](
        self, input: Tensor[[B, InC, L]]
    ) -> Tensor[[B, OutC, (L + 2 * P - D * (K - 1) - 1) // S + 1]]:
        return super().forward(input)

    def _conv_forward[B: IntVar, L: IntVar](
        self, x: Tensor[[B, InC, L]], weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids[L: IntVar, C: IntVar](
    length: Int[L], channels: Int[C], max_timescale: int = 10000
) -> Tensor[[L, C]]:
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    # The concatenation has [(C // 2) + (C // 2)] columns, which equals C only
    # given the even-channels assertion above (A1-style divisibility gap).
    return cast(
        "Tensor[[L, C]]",
        torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
    )


@contextmanager
def disable_sdpa():
    prev_state = MultiHeadAttention.use_sdpa
    try:
        MultiHeadAttention.use_sdpa = False
        yield
    finally:
        MultiHeadAttention.use_sdpa = prev_state


class MultiHeadAttention[NS: IntVar, NH: IntVar](nn.Module):
    use_sdpa = True

    def __init__(self, n_state: Int[NS], n_head: Int[NH]):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward[B: IntVar, Cq: IntVar, Cxa: IntVar](
        self,
        x: Tensor[[B, Cq, NS]],
        xa: Optional[Tensor[[B, Cxa, NS]]] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Dict[nn.Module, Tensor]] = None,
    ) -> Tuple[Tensor[[B, Cq, NS]], Optional[Tensor[[B, NH, Cq, Cxa]]]]:
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            if xa is None:
                k = self.key(x)
                v = self.value(x)
            else:
                k = self.key(xa)
                v = self.value(xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        # Boundary: the branches produce [[B, Cq, NS]] (self-attention),
        # [[B, Cxa, NS]] (cross-attention) or a gradual cache hit; the cast
        # re-establishes the key/value contract for qkv_attention.
        k_typed = cast("Tensor[[B, Cxa, NS]]", k)
        v_typed = cast("Tensor[[B, Cxa, NS]]", v)
        wv, qk = self.qkv_attention(q, k_typed, v_typed, mask)
        return self.out(wv), qk

    def qkv_attention[B: IntVar, Cq: IntVar, Ck: IntVar](
        self,
        q: Tensor[[B, Cq, NS]],
        k: Tensor[[B, Ck, NS]],
        v: Tensor[[B, Ck, NS]],
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor[[B, Cq, NS]], Optional[Tensor[[B, NH, Cq, Ck]]]]:
        n_batch, n_ctx, n_state = q.shape
        scale: float = (n_state // self.n_head) ** -0.25
        qh = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        kh = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        vh = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if SDPA_AVAILABLE and MultiHeadAttention.use_sdpa:
            # SDPA_AVAILABLE guards the import above, but the checker cannot
            # correlate the flag with it.
            assert scaled_dot_product_attention is not None
            if kh.shape[0] == 1 and qh.shape[0] != 1:
                # Cross-attention K/V have batch 1 and broadcast against the
                # beam-expanded query; the fused SDPA kernels reject the batch
                # mismatch and fall back to math, so expand K/V to a stride-0
                # view (no copy) to keep them on the fast path.
                kh = kh.expand(qh.shape[0], -1, -1, -1)
                vh = vh.expand(qh.shape[0], -1, -1, -1)
            a = scaled_dot_product_attention(
                qh, kh, vh, is_causal=mask is not None and n_ctx > 1
            )
            # A1: NH * (NS // NH) reassembles to NS only when NS is divisible
            # by NH, which the architecture guarantees.
            out = cast(
                "Tensor[[B, Cq, NS]]", a.permute(0, 2, 1, 3).flatten(start_dim=2)
            )
            qk = None
        else:
            qk = (qh * scale) @ (kh * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()

            w = F.softmax(qk, dim=-1).to(q.dtype)
            # A1: NH * (NS // NH) reassembles to NS only when NS is divisible
            # by NH, which the architecture guarantees.
            out = cast(
                "Tensor[[B, Cq, NS]]", (w @ vh).permute(0, 2, 1, 3).flatten(start_dim=2)
            )
            qk = qk.detach()

        return out, qk


class ResidualAttentionBlock[NS: IntVar, NH: IntVar](nn.Module):
    def __init__(
        self, n_state: Int[NS], n_head: Int[NH], cross_attention: bool = False
    ):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward[B: IntVar, Cq: IntVar, Ck: IntVar](
        self,
        x: Tensor[[B, Cq, NS]],
        xa: Optional[Tensor[[B, Ck, NS]]] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[Dict[nn.Module, Tensor]] = None,
    ) -> Tensor[[B, Cq, NS]]:
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        cross_attn = self.cross_attn
        cross_attn_ln = self.cross_attn_ln
        if cross_attn is not None and cross_attn_ln is not None:
            x = x + cross_attn(cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder[NM: IntVar, NAC: IntVar, NAS: IntVar, NAH: IntVar](nn.Module):
    positional_embedding: Tensor[[NAC, NAS]]

    def __init__(
        self,
        n_mels: Int[NM],
        n_ctx: Int[NAC],
        n_state: Int[NAS],
        n_head: Int[NAH],
        n_layer: int,
    ):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: nn.ModuleList[ResidualAttentionBlock[NAS, NAH]] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward[B: IntVar, NF: IntVar](
        self, x: Tensor[[B, NM, NF]]
    ) -> Tensor[[B, NAC, NAS]]:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        h = F.gelu(self.conv1(x))
        h = F.gelu(self.conv2(h))
        h = h.permute(0, 2, 1)

        assert h.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        # Boundary: the strided convolutions map NF frames to NAC audio
        # contexts only for conforming inputs, as asserted above; the static
        # convolution arithmetic cannot prove (NF - 1) // 2 + 1 == NAC.
        h_conv = cast("Tensor[[B, NAC, NAS]]", h)
        h_embed: Tensor[[B, NAC, NAS]] = (h_conv + self.positional_embedding).to(
            h_conv.dtype
        )

        for block in self.blocks:
            h_embed = block(h_embed)

        h_out = self.ln_post(h_embed)
        return h_out


class TextDecoder[NV: IntVar, NTC: IntVar, NTS: IntVar, NTH: IntVar](nn.Module):
    mask: Tensor[[NTC, NTC]]

    def __init__(
        self,
        n_vocab: Int[NV],
        n_ctx: Int[NTC],
        n_state: Int[NTS],
        n_head: Int[NTH],
        n_layer: int,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: nn.ModuleList[ResidualAttentionBlock[NTS, NTH]] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True)
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward[B: IntVar, T: IntVar, AC: IntVar, NAS: IntVar](
        self,
        x: Tensor[[B, T]],
        xa: Tensor[[B, AC, NAS]],
        kv_cache: Optional[Dict[nn.Module, Tensor]] = None,
    ) -> Tensor[[B, T, NV]]:
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        # Boundary (conditional equality): cross-attention projects xa with
        # NTS-wide linears, so whisper configurations always set
        # n_audio_state == n_text_state; view the features at width NTS.
        xa_nts = cast("Tensor[[B, AC, NTS]]", xa)
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        h: Tensor[[B, T, NTS]] = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        h = h.to(xa_nts.dtype)

        for block in self.blocks:
            h = block(h, xa_nts, mask=self.mask, kv_cache=kv_cache)

        h = self.ln(h)
        logits = (
            h @ torch.transpose(self.token_embedding.weight.to(h.dtype), 0, 1)
        ).float()

        return logits


class Whisper[
    NM: IntVar,
    NAC: IntVar,
    NAS: IntVar,
    NAH: IntVar,
    NAL: IntVar,
    NV: IntVar,
    NTC: IntVar,
    NTS: IntVar,
    NTH: IntVar,
    NTL: IntVar,
](nn.Module):
    alignment_heads: Tensor[[NTL, NTH]]

    def __init__(
        self, dims: ModelDimensions[NM, NAC, NAS, NAH, NAL, NV, NTC, NTS, NTH, NTL]
    ):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio[B: IntVar, NF: IntVar](
        self, mel: Tensor[[B, NM, NF]]
    ) -> Tensor[[B, NAC, NAS]]:
        return self.encoder(mel)

    def logits[B: IntVar, T: IntVar, AC: IntVar, NAS2: IntVar](
        self, tokens: Tensor[[B, T]], audio_features: Tensor[[B, AC, NAS2]]
    ) -> Tensor[[B, T, NV]]:
        return self.decoder(tokens, audio_features)

    def forward[B: IntVar, NF: IntVar, T: IntVar](
        self, mel: Tensor[[B, NM, NF]], tokens: Tensor[[B, T]]
    ) -> Tensor[[B, T, NV]]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(
        self, cache: Optional[Dict[nn.Module, Tensor]] = None
    ) -> Tuple[Dict[nn.Module, Tensor], List[Any]]:
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
