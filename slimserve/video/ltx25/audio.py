# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 audio path: latent -> mel (causal 2-D conv VAE) -> 16 kHz stereo
(BigVGAN-v2 vocoder, anti-aliased SnakeBeta) -> 48 kHz (bandwidth extension:
causal mel of the 16 kHz signal -> second generator's residual on top of a
Hann-sinc 3x resample).

Loads `vae/ltx-2.5-audio-vae-bf16.safetensors` directly (PyTorch conv layouts
are transposed to MLX's channels-last at load). Reads against Lightricks'
`ltx_core/model/audio_vae/{audio_vae,vocoder}.py`.

Precision: everything runs fp32. Upstream forces the vocoder + BWE to fp32
(bf16 through 108 sequential convolutions degrades spectral metrics 40-90%);
the whole path is 0.18 B parameters and a fraction of a second, so there is
no operand-dtype lever here worth any error.

Token geometry: 16 kHz, mel hop 160, latent time downsample 4 -> 25 latent
frames per second; each frame is one DiT token of 8 channels x 16 mel bins.
T latent frames decode to 4T - 3 mel frames -> (4T - 3) * 160 samples at
16 kHz -> x3 at 48 kHz stereo.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from slimserve.video.ltx25 import checkpoints

G = mx.float32
LATENT_CHANNELS, LATENT_MEL_BINS = 8, 16
MEL_SAMPLE_RATE, MEL_HOP, LATENT_DOWNSAMPLE = 16000, 160, 4
LATENTS_PER_SECOND = MEL_SAMPLE_RATE / MEL_HOP / LATENT_DOWNSAMPLE  # 25.0


# ---- geometry helpers for the pipeline ------------------------------------
def token_count(num_frames: int, fps: float = 24.0) -> int:
    """Audio tokens for a clip (121 frames @ 24 fps -> 126)."""
    return round(num_frames / fps * LATENTS_PER_SECOND)


def positions(num_tokens: int) -> mx.array:
    """(1, T, 1) fp32 token midpoints in seconds (causal mel timing)."""
    idx = np.arange(num_tokens + 1, dtype=np.float32)
    t = (
        np.maximum(idx * LATENT_DOWNSAMPLE + 1 - LATENT_DOWNSAMPLE, 0.0)
        * MEL_HOP
        / MEL_SAMPLE_RATE
    )
    return mx.array(((t[:-1] + t[1:]) / 2.0).astype(np.float32))[None, :, None]


def patchify(latent: mx.array) -> mx.array:
    """(B, 8, T, 16) -> (B, T, 128)."""
    b, c, t, f = latent.shape
    return latent.transpose(0, 2, 1, 3).reshape(b, t, c * f)


def unpatchify(tokens: mx.array) -> mx.array:
    """(B, T, 128) -> (B, 8, T, 16)."""
    b, t, _ = tokens.shape
    return tokens.reshape(b, t, LATENT_CHANNELS, LATENT_MEL_BINS).transpose(0, 2, 1, 3)


# ---- weights --------------------------------------------------------------
def load_weights(
    root: Path | None = None, encoder: bool = False
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    """fp32 weights in MLX layouts, upstream names. Encoder only on request."""
    path = checkpoints.path_of("audio-vae", root)
    cfg = checkpoints.read_header(path).config()
    raw = checkpoints.load_raw(path)
    w: dict[str, mx.array] = {}
    for name in list(raw):
        a = raw.pop(name)
        if name.startswith("audio_vae.encoder.") and not encoder:
            continue
        if name.endswith("inverse_basis"):
            continue
        a = a.astype(G)
        if a.ndim == 4:  # Conv2d (O, I, H, W) -> (O, H, W, I)
            a = a.transpose(0, 2, 3, 1)
        elif a.ndim == 3:
            # ConvTranspose1d (I, O, K) -> (O, K, I)
            # Conv1d / filters / STFT basis (O, I, K) -> (O, K, I)
            a = a.transpose(1, 2, 0) if ".ups." in name else a.transpose(0, 2, 1)
        w[name] = a
    mx.eval(w)
    return w, cfg


# ---- audio VAE (2-D causal conv; H = time is the causal axis) -------------
def _pixel_norm(x: mx.array) -> mx.array:
    return mx.fast.rms_norm(x, None, 1e-6)


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


class _VAE:
    def __init__(self, w: dict[str, mx.array]):
        self.w = w

    def conv(self, p: str, x: mx.array, stride: int = 1) -> mx.array:
        """`<p>.conv`: 3x3 pads time on the past side only, frequency symmetrically."""
        k = self.w[p + ".conv.weight"]
        if k.shape[1] > 1:
            x = mx.pad(x, [(0, 0), (k.shape[1] - 1, 0), (1, 1), (0, 0)])
        return mx.conv2d(x, k, stride=stride) + self.w[p + ".conv.bias"]

    def res(self, p: str, x: mx.array) -> mx.array:
        h = self.conv(p + ".conv1", _silu(_pixel_norm(x)))
        h = self.conv(p + ".conv2", _silu(_pixel_norm(h)))
        if p + ".nin_shortcut.conv.weight" in self.w:
            x = self.conv(p + ".nin_shortcut", x)
        return x + h

    def mid(self, p: str, x: mx.array) -> mx.array:
        return self.res(p + ".mid.block_2", self.res(p + ".mid.block_1", x))


def decode_mel(w: dict[str, mx.array], latent: mx.array) -> mx.array:
    """Normalized latent (B, 8, T, 16) -> log-mel (B, 2, 4T - 3, 64)."""
    v, p = _VAE(w), "audio_vae.decoder"
    b, c, t, f = latent.shape
    x = patchify(latent.astype(G))
    x = (
        x * w["audio_vae.per_channel_statistics.std-of-means"]
        + w["audio_vae.per_channel_statistics.mean-of-means"]
    )
    x = x.reshape(b, t, c, f).transpose(
        0, 1, 3, 2
    )  # (B, T, 16, 8): time, freq, channels
    x = v.mid(p, v.conv(p + ".conv_in", x))
    level = 0
    while f"{p}.up.{level + 1}.block.0.conv1.conv.weight" in w:
        level += 1
    for i in range(level, -1, -1):
        j = 0
        while f"{p}.up.{i}.block.{j}.conv1.conv.weight" in w:
            x = v.res(f"{p}.up.{i}.block.{j}", x)
            j += 1
        if f"{p}.up.{i}.upsample.conv.conv.weight" in w:
            x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)
            x = v.conv(f"{p}.up.{i}.upsample.conv", x)[
                :, 1:
            ]  # causal: drop the first frame
    x = v.conv(p + ".conv_out", _silu(_pixel_norm(x)))
    return x.transpose(0, 3, 1, 2)


def encode_mel(w: dict[str, mx.array], mel: mx.array) -> mx.array:
    """log-mel (B, 2, T', 64) -> normalized latent (B, 8, T, 16) (posterior mean)."""
    v, p = _VAE(w), "audio_vae.encoder"
    x = v.conv(p + ".conv_in", mel.astype(G).transpose(0, 2, 3, 1))
    i = 0
    while f"{p}.down.{i}.block.0.conv1.conv.weight" in w:
        j = 0
        while f"{p}.down.{i}.block.{j}.conv1.conv.weight" in w:
            x = v.res(f"{p}.down.{i}.block.{j}", x)
            j += 1
        if f"{p}.down.{i}.downsample.conv.weight" in w:
            x = mx.pad(x, [(0, 0), (2, 0), (0, 1), (0, 0)])
            x = (
                mx.conv2d(x, w[f"{p}.down.{i}.downsample.conv.weight"], stride=2)
                + w[f"{p}.down.{i}.downsample.conv.bias"]
            )
        i += 1
    x = v.conv(p + ".conv_out", _silu(_pixel_norm(v.mid(p, x))))
    b, t = x.shape[0], x.shape[1]
    x = x[..., :LATENT_CHANNELS].transpose(0, 1, 3, 2).reshape(b, t, -1)
    x = (x - w["audio_vae.per_channel_statistics.mean-of-means"]) / w[
        "audio_vae.per_channel_statistics.std-of-means"
    ]
    return unpatchify(x)


# ---- BigVGAN-v2 generator -------------------------------------------------
def _edge_pad(x: mx.array, left: int, right: int) -> mx.array:
    parts = [mx.repeat(x[:, :1], left, axis=1)] if left else []
    parts.append(x)
    if right:
        parts.append(mx.repeat(x[:, -1:], right, axis=1))
    return mx.concatenate(parts, axis=1)


def _sinc_upsample(
    x: mx.array, kernel: mx.array, ratio: int, pad: int, left: int, right: int
) -> mx.array:
    """Upstream UpSample1d on (N, T, 1): replicate-pad the samples, transposed
    conv with the (symmetric) sinc kernel (1, K, 1), scale, crop
    -> (N, T * ratio, 1). Reference form, kept for the parity script."""
    y = mx.conv_transpose1d(_edge_pad(x, pad, pad), kernel, stride=ratio) * float(ratio)
    return y[:, left : y.shape[1] - right]


# The anti-aliasing filters are 12 taps on one channel. MLX's conv1d and
# conv_transpose1d spend ~2.5-6 ms per call on them regardless of size
# (dispatch-bound: 200 calls per clip), so each filter is compiled once into a
# shifted multiply-add chain with its taps as constants: 0.9 ms a call and
# agreement to 1e-6 (perf/ltx25_metal_campaign.md section 15).
_FIR_CACHE: dict[tuple, Callable] = {}


def _taps(kernel: mx.array) -> tuple[float, ...]:
    return tuple(float(v) for v in np.array(kernel).reshape(-1))


def _upsample2(
    x: mx.array, kernel: mx.array, pad: int, left: int, right: int
) -> mx.array:
    """== _sinc_upsample(x, kernel, 2, pad, left, right), as polyphase FIRs."""
    taps = _taps(kernel)
    key = ("up", taps)
    if key not in _FIR_CACHE:
        # transposed conv, stride 2: y[2n + ph] = sum_j x[n - j] k[2j + ph]
        phases = [taps[ph::2] for ph in (0, 1)]
        length = max(len(ph) for ph in phases)
        # correlation form: y_ph[n] = sum_j rows[ph][j] * xpp[n + j]
        rows = [tuple(reversed(ph)) + (0.0,) * (length - len(ph)) for ph in phases]

        @mx.compile
        def fir(xpp: mx.array) -> mx.array:
            n = xpp.shape[1] - (length - 1)
            outs = [
                sum(c * xpp[:, j : j + n] for j, c in enumerate(row) if c != 0.0)
                for row in rows
            ]
            return mx.concatenate(outs, axis=-1).reshape(xpp.shape[0], 2 * n, 1) * 2.0

        _FIR_CACHE[key] = (fir, length)
    fir, length = _FIR_CACHE[key]
    xp = _edge_pad(x, pad, pad)
    xpp = mx.concatenate(
        [mx.zeros((xp.shape[0], length - 1, 1), dtype=xp.dtype), xp], axis=1
    )
    y = fir(xpp)  # == the transposed conv's output, minus its last k - 2 samples
    full = 2 * xp.shape[1] + kernel.shape[1] - 2
    return y[:, left : full - right]


def _downsample2(x: mx.array, kernel: mx.array, left: int, right: int) -> mx.array:
    """== conv1d(_edge_pad(x, left, right), kernel, stride=2)."""
    taps = _taps(kernel)
    key = ("down", taps)
    fir = _FIR_CACHE.get(key)
    if fir is None:

        @mx.compile
        def fir(xp: mx.array) -> mx.array:
            n = (xp.shape[1] - len(taps)) // 2 + 1
            return sum(
                c * xp[:, j : j + 2 * (n - 1) + 1 : 2] for j, c in enumerate(taps)
            )

        _FIR_CACHE[key] = fir
    return fir(_edge_pad(x, left, right))


@mx.compile
def _snake(h: mx.array, alpha: mx.array, inv_beta: mx.array) -> mx.array:
    return h + inv_beta * mx.square(mx.sin(alpha * h))


class _Generator:
    def __init__(self, w: dict[str, mx.array], prefix: str, cfg: dict[str, Any]):
        self.w, self.p = w, prefix
        self.rates = cfg["upsample_rates"]
        self.kernels = cfg["upsample_kernel_sizes"]
        self.res_kernels = cfg["resblock_kernel_sizes"]
        self.dilations = cfg["resblock_dilation_sizes"]

    def act(self, p: str, x: mx.array) -> mx.array:
        """Anti-aliased SnakeBeta: 2x upsample, x + sin^2(a x) / b, 2x downsample."""
        w = self.w
        b, t, c = x.shape
        up, down = w[p + ".upsample.filter"], w[p + ".downsample.lowpass.filter"]
        k = up.shape[1]
        pad = k // 2 - 1
        h = x.transpose(0, 2, 1).reshape(b * c, t, 1)
        h = _upsample2(h, up, pad, pad * 2 + (k - 2) // 2, pad * 2 + (k - 1) // 2)
        h = h.reshape(b, c, 2 * t).transpose(0, 2, 1)
        alpha, beta = mx.exp(w[p + ".act.alpha"]), mx.exp(w[p + ".act.beta"])
        h = _snake(h, alpha, 1.0 / (beta + 1e-9))
        k = down.shape[1]
        h = h.transpose(0, 2, 1).reshape(b * c, 2 * t, 1)
        h = _downsample2(h, down, k // 2 - (1 - k % 2), k // 2)
        return h.reshape(b, c, t).transpose(0, 2, 1)

    def conv(self, p: str, x: mx.array, dilation: int = 1) -> mx.array:
        k = self.w[p + ".weight"]
        y = mx.conv1d(x, k, padding=dilation * (k.shape[1] - 1) // 2, dilation=dilation)
        bias = self.w.get(p + ".bias")
        return y if bias is None else y + bias

    def __call__(self, mel: mx.array) -> mx.array:
        """(B, T, 2 * 64) -> (B, T * prod(rates), 2), no final activation."""
        w, p = self.w, self.p
        x = self.conv(p + ".conv_pre", mel)
        for i, (rate, kernel) in enumerate(zip(self.rates, self.kernels)):
            x = mx.conv_transpose1d(
                x, w[f"{p}.ups.{i}.weight"], stride=rate, padding=(kernel - rate) // 2
            )
            x = x + w[f"{p}.ups.{i}.bias"]
            acc = None
            for j, dils in enumerate(self.dilations):
                rp = f"{p}.resblocks.{i * len(self.res_kernels) + j}"
                h = x
                for n, d in enumerate(dils):
                    y = self.conv(
                        f"{rp}.convs1.{n}", self.act(f"{rp}.acts1.{n}", h), dilation=d
                    )
                    h = h + self.conv(
                        f"{rp}.convs2.{n}", self.act(f"{rp}.acts2.{n}", y)
                    )
                acc = h if acc is None else acc + h
            x = acc / len(self.res_kernels)
            mx.eval(x)
        return self.conv(p + ".conv_post", self.act(p + ".act_post", x))


def _resample(x: mx.array, ratio: int) -> mx.array:
    """Hann-windowed sinc upsample (upstream UpSample1d, window_type="hann"),
    (N, T) -> (N, T * ratio)."""
    rolloff, lpfw = 0.99, 6
    width = math.ceil(lpfw / rolloff)
    k = 2 * width * ratio + 1
    t = (np.arange(k, dtype=np.float64) / ratio - width) * rolloff
    window = np.cos(np.clip(t, -lpfw, lpfw) * np.pi / lpfw / 2) ** 2
    kernel = mx.array((np.sinc(t) * window * rolloff / ratio).astype(np.float32))[
        None, :, None
    ]
    return _sinc_upsample(
        x[:, :, None], kernel, ratio, width, 2 * width * ratio, k - ratio
    )[..., 0]


class Vocoder:
    """mel (B, 2, T, 64) -> waveform (B, 2, T * 160 * 3) at 48 kHz, in [-1, 1]."""

    def __init__(self, w: dict[str, mx.array], cfg: dict[str, Any]):
        self.w = w
        self.base_cfg, self.bwe_cfg = cfg["vocoder"], cfg["bwe"]
        self.base = _Generator(w, "vocoder.vocoder", self.base_cfg)
        self.bwe = _Generator(w, "vocoder.bwe_generator", self.bwe_cfg)
        self.sample_rate = self.bwe_cfg["output_sampling_rate"]

    @staticmethod
    def _fold(mel: mx.array) -> mx.array:
        b, c, t, m = mel.shape
        return mel.transpose(0, 1, 3, 2).reshape(b, c * m, t).transpose(0, 2, 1)

    def _log_mel(self, wav: mx.array) -> mx.array:
        """Causal STFT (left pad n_fft - hop) with the checkpoint's bases.
        (N, T) -> (N, T', 64)."""
        n_fft, hop = self.bwe_cfg["n_fft"], self.bwe_cfg["hop_length"]
        x = mx.pad(wav[:, :, None], [(0, 0), (n_fft - hop, 0), (0, 0)])
        spec = mx.conv1d(
            x, self.w["vocoder.mel_stft.stft_fn.forward_basis"], stride=hop
        )
        bins = n_fft // 2 + 1
        power = mx.square(spec[..., :bins]) + mx.square(spec[..., bins:])
        return mx.log(
            mx.maximum(mx.sqrt(power) @ self.w["vocoder.mel_stft.mel_basis"].T, 1e-5)
        )

    def __call__(self, mel: mx.array) -> mx.array:
        mel = mel.astype(G)
        b, c = mel.shape[:2]
        x = self.base(self._fold(mel))
        # use_tanh_at_final is false in the 2.5 checkpoint: the 16 kHz stage clamps.
        x = (
            mx.tanh(x)
            if self.base_cfg.get("use_tanh_at_final", True)
            else mx.clip(x, -1.0, 1.0)
        )
        x = x.transpose(0, 2, 1)  # (B, 2, T16)
        length = x.shape[-1]
        ratio = self.sample_rate // self.bwe_cfg["input_sampling_rate"]
        hop = self.bwe_cfg["hop_length"]
        if length % hop:
            x = mx.pad(x, [(0, 0), (0, 0), (0, hop - length % hop)])
        flat = x.reshape(b * c, -1)
        mx.eval(flat)
        bwe_mel = self._log_mel(flat)
        bwe_mel = bwe_mel.reshape(b, c, bwe_mel.shape[1], -1)
        residual = self.bwe(self._fold(bwe_mel)).transpose(0, 2, 1)
        skip = _resample(flat, ratio).reshape(b, c, -1)
        n = min(skip.shape[-1], residual.shape[-1])
        return mx.clip(skip[..., :n] + residual[..., :n], -1.0, 1.0)[
            ..., : length * ratio
        ]


# ---- encoder-side mel (audio-conditioned pipelines) -----------------------
def _slaney_filterbank(
    sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float
) -> np.ndarray:
    to_mel = lambda f: np.where(
        f < 1000.0,
        3.0 * f / 200.0,
        15.0 + 27.0 * np.log(np.maximum(f, 1e-10) / 1000.0) / np.log(6.4),
    )  # noqa: E731
    to_hz = lambda m: np.where(
        m < 15.0, 200.0 * m / 3.0, 1000.0 * np.exp((m - 15.0) * np.log(6.4) / 27.0)
    )  # noqa: E731
    hz = to_hz(
        np.linspace(
            float(to_mel(np.float64(f_min))),
            float(to_mel(np.float64(f_max))),
            n_mels + 2,
        )
    )
    freqs = np.linspace(0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, freqs.shape[0]))
    for i in range(n_mels):
        lo, mid, hi = hz[i : i + 3]
        fb[i] = np.maximum(0, (freqs - lo) / (mid - lo)) * (freqs <= mid) + np.maximum(
            0, (hi - freqs) / (hi - mid)
        ) * (freqs > mid)
        fb[i] *= 2.0 / (hi - lo)
    return fb.astype(np.float32)


def waveform_to_mel(waveform: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """16 kHz waveform (C, T) -> log-mel (1, C, T', 64) as the VAE encoder expects:
    upstream AudioProcessor.waveform_to_mel, a torchaudio MelSpectrogram with
    f_min 0, f_max sr / 2, Slaney scale and norm, power 1, Hann window, centered
    reflect padding, then log(clamp(mel, 1e-5)). Resampling to 16 kHz is the
    caller's job (`media.resample_sinc`)."""
    pre = cfg["audio_vae"]["preprocessing"]
    sr, n_fft, hop = (
        pre["audio"]["sampling_rate"],
        pre["stft"]["filter_length"],
        pre["stft"]["hop_length"],
    )
    fb = _slaney_filterbank(sr, n_fft, pre["mel"]["n_mel_channels"], 0.0, sr / 2.0)
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    out = []
    for ch in np.asarray(waveform, dtype=np.float32):
        padded = np.pad(ch, n_fft // 2, mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop] * window
        out.append(
            np.log(
                np.maximum(np.abs(np.fft.rfft(frames)).astype(np.float32) @ fb.T, 1e-5)
            )
        )
    return np.stack(out)[None]


# ---- public ---------------------------------------------------------------
class AudioDecoder:
    """Audio VAE decoder + vocoder + BWE. `decode` takes the DiT's audio tokens."""

    sample_rate = 48000
    channels = 2

    def __init__(self, root: Path | None = None, encoder: bool = False):
        self.root, self.with_encoder = root, encoder
        self.w: dict[str, mx.array] | None = None
        self.cfg: dict[str, Any] = {}
        self.vocoder: Vocoder | None = None

    def load(self) -> AudioDecoder:
        if self.w is None:
            self.w, self.cfg = load_weights(self.root, encoder=self.with_encoder)
            self.vocoder = Vocoder(self.w, self.cfg["vocoder"])
            self.sample_rate = self.vocoder.sample_rate
        return self

    def unload(self) -> None:
        self.w, self.vocoder = None, None

    def decode_mx(self, tokens: mx.array) -> mx.array:
        """(B, T, 128) normalized tokens -> (B, 2, samples) fp32 at 48 kHz."""
        self.load()
        mel = decode_mel(self.w, unpatchify(tokens))
        mx.eval(mel)
        return self.vocoder(mel)

    def decode(self, tokens: mx.array) -> tuple[np.ndarray, int]:
        """(1, T, 128) -> (float32 (2, (4T - 3) * 480), 48000). Not trimmed to the
        video duration: upstream muxes the full waveform."""
        wav = self.decode_mx(tokens)
        mx.eval(wav)
        return np.array(wav[0]), self.sample_rate

    def encode(self, waveform: np.ndarray, sample_rate: int = 16000) -> mx.array:
        """(channels, T) float waveform at `sample_rate` -> normalized tokens
        (1, T_latent, 128): upstream encode_audio (resample to 16 kHz, log-mel,
        the encoder's posterior mean, per-channel normalization)."""
        if not self.with_encoder:
            raise RuntimeError("AudioDecoder(encoder=True) is required to encode")
        self.load()
        from slimserve.video.ltx25 import media

        wav = media.resample_sinc(
            np.asarray(waveform, dtype=np.float32), sample_rate, 16000
        )
        return patchify(encode_mel(self.w, mx.array(waveform_to_mel(wav, self.cfg))))
