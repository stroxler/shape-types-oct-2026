from __future__ import annotations

import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import TYPE_CHECKING, Literal, Optional, Union, cast, overload

import numpy as np
import torch
import torch.nn.functional as F

from .utils import exact_div

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
    from torch import Tensor

# hard-coded audio hyperparameters
SAMPLE_RATE: Int[16000] = 16000
N_FFT: Int[400] = 400
HOP_LENGTH: Int[160] = 160
CHUNK_LENGTH: Int[30] = 30
N_SAMPLES: Int[480000] = (
    CHUNK_LENGTH * SAMPLE_RATE
)  # 480000 samples in a 30-second chunk
N_FRAMES: Int[3000] = exact_div(
    N_SAMPLES, HOP_LENGTH
)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN: Int[320] = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND: Int[100] = exact_div(
    SAMPLE_RATE, HOP_LENGTH
)  # 10ms per audio frame
TOKENS_PER_SECOND: Int[50] = exact_div(
    SAMPLE_RATE, N_SAMPLES_PER_TOKEN
)  # 20ms per audio token


def load_audio[N: IntVar](file: str, sr: int = SAMPLE_RATE) -> np.ndarray[[N]]:
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


@overload
def pad_or_trim[N: IntVar, L: IntVar](
    array: Tensor[[N]], length: Int[L] = N_SAMPLES, *, axis: int = -1
) -> Tensor[[L]]: ...
@overload
def pad_or_trim[N: IntVar, L: IntVar](
    array: np.ndarray[[N]], length: Int[L] = N_SAMPLES, *, axis: int = -1
) -> np.ndarray[[L]]: ...
@overload
def pad_or_trim[D0: IntVar, D1: IntVar, L: IntVar](
    array: Tensor[[D0, D1]],
    length: Int[L] = N_SAMPLES,
    *,
    axis: Literal[-1] = -1,
) -> Tensor[[D0, L]]: ...
@overload
def pad_or_trim[D0: IntVar, D1: IntVar, L: IntVar](
    array: np.ndarray[[D0, D1]],
    length: Int[L] = N_SAMPLES,
    *,
    axis: Literal[-1] = -1,
) -> np.ndarray[[D0, L]]: ...
@overload
def pad_or_trim(
    array: Tensor, length: int = N_SAMPLES, *, axis: int = -1
) -> Tensor: ...
@overload
def pad_or_trim(
    array: np.ndarray, length: int = N_SAMPLES, *, axis: int = -1
) -> np.ndarray: ...


def pad_or_trim(
    array: Tensor | np.ndarray, length: int = N_SAMPLES, *, axis: int = -1
) -> Tensor | np.ndarray:
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        # torch.is_tensor narrows at runtime but the stub returns plain bool.
        array = cast("Tensor", array)
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        # torch.is_tensor narrows at runtime but the stub returns plain bool.
        array = cast("np.ndarray", array)
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters[NM: IntVar](
    device: Union[str, torch.device], n_mels: Int[NM]
) -> Tensor[[NM, 201]]:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        # Boundary: np.load/torch.from_numpy are unshaped in the stubs. The
        # stored filterbank has shape (n_mels, 1 + n_fft // 2) = (NM, 201)
        # for n_fft=400, per the docstring above.
        return cast(
            "Tensor[[NM, 201]]", torch.from_numpy(f[f"mel_{n_mels}"]).to(device)
        )


def log_mel_spectrogram[N: IntVar, NM: IntVar, NF: IntVar](
    audio: Union[str, np.ndarray[[N]], Tensor[[N]]],
    n_mels: Int[NM] = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> Tensor[[NM, NF]]:
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 and 128 are supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (n_mels, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    # torch.is_tensor narrows at runtime but the stub returns plain bool, and
    # torch.from_numpy is unshaped in the stubs; the 1D waveform shape is kept.
    audio = cast("Tensor[[N]]", audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    # Boundary: F.pad with a dynamic width and the strided STFT framing are
    # gradual in the stubs. A onesided n_fft=400 STFT has 201 frequency bins;
    # the frame count NF is output-only. Precision is recovered by inference
    # downstream of this cast (filters @ magnitudes).
    magnitudes = cast("Tensor[[201, NF]]", stft[..., :-1].abs() ** 2)

    # lru_cache erases the Int binding, so re-establish NM from the argument.
    filters = cast("Tensor[[NM, 201]]", mel_filters(audio.device, n_mels))
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
