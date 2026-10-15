# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 diffusion video decoder (NADiffusionDecoder), Lightricks' default and
recommended decoder: sharper faces, textures and text than the conv decoder.

`vae/ltx-2.5-video-vae-bf16.safetensors`. Four deterministic stages of 3-D
neighborhood attention (kernels 3x7x7, 3x7x7, 3x5x5, 3x5x5) with pixel-shuffle
upsamples take the latent to pixel/4 resolution; eight diffusion blocks with an
11x11x11 window, AdaLN from the timestep and the stage-4 feature as context,
then run one x0 step from pure noise at t = 1. Reads against Lightricks'
`ltx_core/model/video_vae/diffusion_decoder/`; weights keep upstream names.

Neighborhood attention (NATTEN semantics: a window of fixed size centred on the
query and clamped inside the volume) runs as a Metal kernel, one thread per
(query, head), online softmax in fp32. The block-gather MLX formulation the
baseline uses is kept as `na3d_mlx` for validation only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from slimserve.video.ltx25 import checkpoints

G = mx.float32
Kernel = tuple[int, int, int]
NOISE_SEED_OFFSET = 30000
MAX_ELEMENTS = 1 << 31
W_CHUNKS = (
    4  # upstream's chunked_eager: the stage-5 width is processed in 4 haloed slabs
)
CHUNK_TOKENS = (
    3_300_000  # attention tokens per temporal chunk (a 768x512x121 clip runs whole)
)


@dataclass(frozen=True)
class DiffVAEConfig:
    in_channels: int
    out_channels: int
    patch: int
    head_dim: int
    channels: tuple[int, ...]
    depths: tuple[int, ...]
    kernels: tuple[Kernel, ...]  # four deterministic stages
    upsamples: tuple[tuple[Kernel, int], ...]  # (stride, channel reduction)
    stage5_kernel: Kernel
    timestep_scale: float
    t_hidden: int = 384
    t_freq: int = 256

    @classmethod
    def from_header(cls, header: checkpoints.Header) -> DiffVAEConfig:
        d = header.config()["vae"]["decoder"]
        if d.get("_class_name") != "NADiffusionDecoder":
            raise ValueError(f"not a diffusion decoder: {d.get('_class_name')!r}")
        return cls(
            in_channels=d["in_channels"],
            out_channels=d["out_channels"],
            patch=d["patch_size"],
            head_dim=d["head_dim"],
            channels=tuple(d["stage_channels"]),
            depths=tuple(d["stage_depths"]),
            kernels=tuple(tuple(k) for k in d["stage_kernels"][:4]),
            upsamples=tuple((tuple(s), r) for s, r in d["upsamples"]),
            stage5_kernel=tuple(d["stage5_kernel"]),
            timestep_scale=float(d.get("timestep_scale_multiplier", 1000.0)),
        )

    def cumulative_strides(self) -> list[Kernel]:
        out: list[Kernel] = [(1, 1, 1)]
        for (st, sh, sw), _ in self.upsamples:
            t, h, w = out[-1]
            out.append((t * st, h * sh, w * sw))
        return out

    def min_latent_shape(self) -> Kernel:
        strides = self.cumulative_strides()
        kernels = [*self.kernels, self.stage5_kernel]
        return tuple(
            max(math.ceil(k[a] / s[a]) for k, s in zip(kernels, strides))
            for a in range(3)
        )

    @property
    def ghost_frames(self) -> int:
        return (self.kernels[0][0] // 2) * 2


# ---- neighborhood attention --------------------------------------------------
def window_starts(length: int, kernel: int) -> mx.array:
    k_eff = min(kernel, length)
    return mx.clip(mx.arange(length) - k_eff // 2, 0, length - k_eff).astype(mx.int32)


def na3d_mlx(q: mx.array, k: mx.array, v: mx.array, kernel: Kernel) -> mx.array:
    """Reference: exact windows via a gather per query block.

    q, k, v (1, T, H, W, heads, hd).
    """
    _, t, h, w, heads, hd = q.shape
    out = mx.zeros_like(q)
    starts = [np.array(window_starts(n, kk)) for n, kk in zip((t, h, w), kernel)]
    keff = [min(kk, n) for kk, n in zip(kernel, (t, h, w))]
    for ti in range(t):
        st = starts[0][ti]
        for hi in range(h):
            sh = starts[1][hi]
            sw = starts[2]  # (w,)
            ks = mx.stack(
                [
                    k[0, st : st + keff[0], sh : sh + keff[1], s : s + keff[2]].reshape(
                        -1, heads, hd
                    )
                    for s in sw
                ]
            )  # (w, L, heads, hd)
            vs = mx.stack(
                [
                    v[0, st : st + keff[0], sh : sh + keff[1], s : s + keff[2]].reshape(
                        -1, heads, hd
                    )
                    for s in sw
                ]
            )
            qs = q[0, ti, hi]  # (w, heads, hd)
            scores = mx.einsum("wnd,wlnd->wnl", qs.astype(G), ks.astype(G))
            probs = mx.softmax(scores, axis=-1)
            out[0, ti, hi] = mx.einsum("wnl,wlnd->wnd", probs, vs.astype(G)).astype(
                q.dtype
            )
        mx.eval(out)
    return out


_NA_KERNELS: dict[tuple, object] = {}

# lanes per query group: each owns HD / LANES dims; loads are 128-byte coalesced
LANES = 8
# adjacent-in-W queries served by one lane group: their windows overlap in all
# but QW-1 columns, so one key/value row load feeds QW online softmaxes. 2 is
# 1.29x over 1 at the stage-5 slab; 3 spills registers (ledger section 22).
QW = 2

# Shared prologue: the query grid is (Tq, H, Wq) at offset (TOFF, WOFF) inside
# the key/value volume (T, H, W), so a slab's halo frames and columns feed the
# windows without being computed as queries themselves.
_NA_PROLOGUE = """
    const uint T = shape[0], H = shape[1], W = shape[2], NH = shape[3];
    const uint Tq = shape[4], Wq = shape[5], TOFF = shape[6], WOFF = shape[7];
    const uint gid = thread_position_in_grid.x;
    const uint lane = gid % LANES;
    const uint grp = gid / LANES;
    const uint groups_w = (Wq + QW - 1) / QW;
    const uint total = Tq * H * groups_w * NH;
    const bool live = grp < total;
    const uint head = live ? grp % NH : 0;
    const uint g = live ? grp / NH : 0;
    const uint gw = g % groups_w;
    const uint hi = (g / groups_w) % H;
    const uint tq = g / (groups_w * H);
    const int ti = (int)(tq + TOFF);
    const uint stride_tok = NH * HD;
    const uint stride_q = NH * HD;
    float qr[QW][DL];
    float m[QW], l[QW];
    float acc[QW][DL];
    int wi[QW];
    for (uint i = 0; i < QW; ++i) {
        const uint wq = min(gw * QW + i, Wq - 1);
        wi[i] = (int)(wq + WOFF);
        const size_t qtok = ((size_t)tq * H + hi) * Wq + wq;
        const size_t qoff = qtok * stride_q + head * HD + lane * DL;
        for (uint d = 0; d < DL; ++d) qr[i][d] = (float)q[qoff + d];
        m[i] = -1.0e30f; l[i] = 0.0f;  // finite: a masked first key stays well defined
        for (uint d = 0; d < DL; ++d) acc[i][d] = 0.0f;
    }
"""

_NA_UPDATE = """
                float kr[DL], vr[DL];
                for (uint d = 0; d < DL; ++d) {
                    kr[d] = (float)KB[off + d]; vr[d] = (float)VB[off + d];
                }
                for (uint i = 0; i < QW; ++i) {
                    float s = 0.0f;
                    for (uint d = 0; d < DL; ++d) s += qr[i][d] * kr[d];
                    SHUFFLES
                    const bool vis = VIS;
                    const float m_new = vis ? max(m[i], s) : m[i];
                    const float corr = exp(m[i] - m_new);
                    const float p = vis ? exp(s - m_new) : 0.0f;
                    l[i] = l[i] * corr + p;
                    for (uint d = 0; d < DL; ++d)
                        acc[i][d] = acc[i][d] * corr + p * vr[d];
                    m[i] = m_new;
                }
"""

_NA_EPILOGUE = """
    if (live) {
        for (uint i = 0; i < QW; ++i) {
            const uint wq = gw * QW + i;
            if (wq >= Wq) break;
            const size_t qtok = ((size_t)tq * H + hi) * Wq + wq;
            const size_t base = qtok * stride_q + head * HD + lane * DL;
            const float inv = 1.0f / l[i];
            for (uint d = 0; d < DL; ++d) out[base + d] = (T_IN)(acc[i][d] * inv);
        }
    }
"""

# Plain decode: NATTEN semantics, the window shifted inside the volume.
_NA_SOURCE = (
    _NA_PROLOGUE
    + """
    const int kt = min((int)KT, (int)T);
    const int kh = min((int)KH, (int)H);
    const int kw = min((int)KW, (int)W);
    const int t0 = clamp(ti - kt / 2, 0, (int)T - kt);
    const int h0 = clamp((int)hi - kh / 2, 0, (int)H - kh);
    int w0[QW];
    for (uint i = 0; i < QW; ++i) w0[i] = clamp(wi[i] - kw / 2, 0, (int)W - kw);
    const int wlo = w0[0], whi = w0[QW - 1] + kw;
    for (int a = 0; a < kt; ++a) {
        for (int b = 0; b < kh; ++b) {
            const size_t row = ((size_t)(t0 + a) * H + (h0 + b)) * W;
            for (int c = wlo; c < whi; ++c) {
                const size_t off = (row + c) * stride_tok + head * HD + lane * DL;
"""
    + _NA_UPDATE.replace("KB", "k")
    .replace("VB", "v")
    .replace("VIS", "(c >= w0[i]) && (c < w0[i] + kw)")
    + """
            }
        }
    }
"""
    + _NA_EPILOGUE
)

# Joint (video + keyframe) NA, upstream joint_eager / joint_triton semantics:
# the window is CENTRED on the query and taps outside the volume are masked
# ("clamp-and-mask, not NATTEN's inward shift"), unlike the plain decode.
# (k, v) is the query's own volume (T frames, Kt window), (kb, vb) a stack of
# frames reached through slots[tq * S + s] (-1 = empty), Kh x Kw at (hi, wi).
KEYFRAME_SLOTS = 2

_NA_JOINT_SOURCE = (
    _NA_PROLOGUE
    + """
    const uint S = shape[8];
    const int t_lo = ti - (int)(KT / 2), h_lo = (int)hi - (int)(KH / 2);
    const int ta = max(t_lo, 0), tb = min(t_lo + (int)KT, (int)T);
    const int ha = max(h_lo, 0), hb = min(h_lo + (int)KH, (int)H);
    int wa[QW], wb[QW];
    for (uint i = 0; i < QW; ++i) {
        wa[i] = max(wi[i] - (int)(KW / 2), 0);
        wb[i] = min(wi[i] - (int)(KW / 2) + (int)KW, (int)W);
    }
    const int wlo = wa[0], whi = wb[QW - 1];
    for (int a = ta; a < tb; ++a) {
        for (int b = ha; b < hb; ++b) {
            const size_t row = ((size_t)a * H + b) * W;
            for (int c = wlo; c < whi; ++c) {
                const size_t off = (row + c) * stride_tok + head * HD + lane * DL;
"""
    + _NA_UPDATE.replace("KB", "k")
    .replace("VB", "v")
    .replace("VIS", "(c >= wa[i]) && (c < wb[i])")
    + """
            }
        }
    }
    for (uint si = 0; si < S; ++si) {
        const int pl = slots[tq * S + si];
        if (pl < 0) continue;
        for (int b = ha; b < hb; ++b) {
            const size_t row = ((size_t)pl * H + b) * W;
            for (int c = wlo; c < whi; ++c) {
                const size_t off = (row + c) * stride_tok + head * HD + lane * DL;
"""
    + _NA_UPDATE.replace("KB", "kb")
    .replace("VB", "vb")
    .replace("VIS", "(c >= wa[i]) && (c < wb[i])")
    + """
            }
        }
    }
"""
    + _NA_EPILOGUE
)


def _na_kernel(kind: str, kernel: Kernel, hd: int, dtype, lanes: int, qw: int):
    key = (kind, kernel, hd, str(dtype), lanes, qw)
    kern = _NA_KERNELS.get(key)
    if kern is None:
        src = (_NA_SOURCE if kind == "plain" else _NA_JOINT_SOURCE).replace(
            "SHUFFLES",
            "".join(
                f"s += simd_shuffle_xor(s, {o});\n" for o in (4, 2, 1) if o < lanes
            ),
        )
        for name, val in (
            ("KT", kernel[0]),
            ("KH", kernel[1]),
            ("KW", kernel[2]),
            ("HD", hd),
            ("LANES", lanes),
            ("DL", hd // lanes),
            ("QW", qw),
        ):
            src = src.replace(name, str(val))
        dt_tag = "f16" if dtype == mx.float16 else "f32"
        names = ["q", "k", "v"] + (["kb", "vb", "slots"] if kind == "joint" else [])
        kern = mx.fast.metal_kernel(
            name=f"na3d_{kind}_{kernel[0]}x{kernel[1]}x{kernel[2]}_d{hd}_l{lanes}_q{qw}_{dt_tag}",
            input_names=names + ["shape"],
            output_names=["out"],
            source=src,
        )
        _NA_KERNELS[key] = kern
    return kern


def _launch(kern, inputs, q, shape_list, lanes, qw):
    _, tq, h, wq, heads, _ = q.shape
    groups = tq * h * ((wq + qw - 1) // qw) * heads
    total = groups * lanes
    shape = mx.array(shape_list, dtype=mx.uint32)
    # MLX 0.32.2: a metal_kernel launched behind still-running MLX ops read
    # incomplete inputs 4 times out of 5 on the stage-4 volume of a 10 s clip
    # (NaN output); mx.eval on the inputs did not prevent it, a synchronize
    # does (ledger section 18). Costs a pipeline drain per call, ~1 ms.
    mx.synchronize()
    (out,) = kern(
        inputs=inputs + [shape],
        template=[("T_IN", q.dtype)],
        grid=((total + 255) // 256 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )
    mx.eval(out)
    mx.synchronize()
    return out


def na3d(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    kernel: Kernel,
    lanes: int = LANES,
    qw: int = QW,
    offset: tuple[int, int] = (0, 0),
) -> mx.array:
    """Metal neighborhood attention (NATTEN semantics). k, v (1, T, H, W, heads,
    hd) in fp16 or fp32; q (1, Tq, H, Wq, heads, hd) is the query block that
    sits at frame `offset[0]` and column `offset[1]` of the k/v volume (a slab
    core; the halo frames and columns are keys only).

    Measured on the stage-5 shape (ledger sections 18 and 22): 8 lanes per
    query, 2 queries per lane group."""
    _, t, h, w, heads, hd = k.shape
    _, tq, hq, wq, _, _ = q.shape
    if hq != h or tq + offset[0] > t or wq + offset[1] > w:
        raise ValueError(f"query block {q.shape} at {offset} outside k/v {k.shape}")
    kern = _na_kernel("plain", kernel, hd, q.dtype, lanes, qw)
    return _launch(kern, [q, k, v], q, [t, h, w, heads, tq, wq, *offset], lanes, qw)


def na3d_joint(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    kb: mx.array,
    vb: mx.array,
    slots: mx.array,
    kernel: Kernel,
    lanes: int = LANES,
    qw: int = QW,
    offset: tuple[int, int] = (0, 0),
) -> mx.array:
    """One softmax over the query's own window in (k, v) (1, T, H, W, heads, hd)
    and the Kh x Kw window on each slot-indexed frame of (kb, vb)
    (1, P, H, W, heads, hd); `slots` (Tq, S) int32 per query frame, -1 for an
    empty slot. `q` and `offset` as in `na3d`.

    The video pass calls this with the volume as (k, v) and the planes as (kb,
    vb); the plane pass with the planes as (k, v) under a Kt = 1 kernel and the
    nearest video frames as (kb, vb)."""
    _, t, h, w, heads, hd = k.shape
    _, tq, hq, wq, _, _ = q.shape
    if hq != h or tq + offset[0] > t or wq + offset[1] > w:
        raise ValueError(f"query block {q.shape} at {offset} outside k/v {k.shape}")
    if slots.ndim != 2 or slots.shape[0] != tq:
        raise ValueError(f"slots {slots.shape} for {tq} query frames")
    kern = _na_kernel("joint", kernel, hd, q.dtype, lanes, qw)
    return _launch(
        kern,
        [q, k, v, kb, vb, slots.astype(mx.int32)],
        q,
        [t, h, w, heads, tq, wq, *offset, slots.shape[1]],
        lanes,
        qw,
    )


def nearest_slots(
    query_times: np.ndarray, candidate_times: np.ndarray, n: int = KEYFRAME_SLOTS
) -> np.ndarray:
    """(Q, n) candidate indices ranked by (|dt|, index); -1 pads when there are
    fewer than n candidates. Upstream `_nearest_slots` (stable argsort)."""
    q = np.asarray(query_times, dtype=np.float32)
    c = np.asarray(candidate_times, dtype=np.float32)
    out = np.full((len(q), n), -1, dtype=np.int32)
    if len(c) == 0:
        return out
    order = np.argsort(np.abs(q[:, None] - c[None, :]), axis=1, kind="stable")
    take = min(n, len(c))
    out[:, :take] = order[:, :take]
    return out


def keyframe_stage_times(pixel_frames: np.ndarray, remaining_stride: int) -> np.ndarray:
    """Chunk-centre time of each keyframe in a stage's temporal units: a stage
    whose remaining temporal upsampling is r has cells of r pixel frames except
    cell 0 (the causal first frame), so t_s(0) = 0 and t_s(f) = (f + (r-1)/2) / r.
    Video frame j at that stage sits at exactly j, so the two streams share one
    origin. Upstream `keyframe_stage_times`."""
    f = np.asarray(pixel_frames, dtype=np.float32)
    t = (f + (remaining_stride - 1) / 2) / remaining_stride
    return np.where(f == 0, 0.0, t).astype(np.float32)


# ---- model ---------------------------------------------------------------------
def _rms(x: mx.array, weight: mx.array, eps: float = 1e-6) -> mx.array:
    # fast.rms_norm accumulates in fp32 whatever the input dtype; no casts needed
    return mx.fast.rms_norm(x, weight.astype(x.dtype), eps)


def _rope_split(hd: int) -> tuple[int, int, int]:
    d_t = (hd // 4) // 2 * 2
    d_hw = (hd - d_t) // 2
    return d_t, d_hw, d_hw


def _inv_freqs(dim: int) -> mx.array:
    return mx.array(
        (1.0 / 10000.0 ** (np.arange(0, dim, 2, dtype=np.float64) / dim)).astype(
            np.float32
        )
    )


def _rotate(x: mx.array, angle: mx.array) -> mx.array:
    xe, xo = x[..., 0::2], x[..., 1::2]
    cos, sin = mx.cos(angle), mx.sin(angle)
    return mx.stack([xe * cos - xo * sin, xe * sin + xo * cos], axis=-1).reshape(
        x.shape
    )


_ROPE_FNS: dict[tuple, object] = {}


def _axial_rope(x: mx.array, t_pos, h_pos, w_pos, split, inv) -> mx.array:
    """Axial RoPE (t, h, w) on (1, T, H, W, heads, hd); compiled per split so the
    slicing, trig and rotation run as one fused pass instead of ~30 small ops."""
    d_t, d_h, _ = split
    fn = _ROPE_FNS.get(split)
    if fn is None:

        @mx.compile
        def fn(x, t_pos, h_pos, w_pos, inv_t, inv_h, inv_w):
            xf = x.astype(G)
            parts = [xf[..., :d_t], xf[..., d_t : d_t + d_h], xf[..., d_t + d_h :]]
            angles = [
                (t_pos[:, None] * inv_t[None, :])[None, :, None, None, None, :],
                (h_pos[:, None] * inv_h[None, :])[None, None, :, None, None, :],
                (w_pos[:, None] * inv_w[None, :])[None, None, None, :, None, :],
            ]
            return mx.concatenate(
                [_rotate(p, a) for p, a in zip(parts, angles)], axis=-1
            ).astype(x.dtype)

        _ROPE_FNS[split] = fn
    return fn(x, t_pos, h_pos, w_pos, *inv)


def _pixel_shuffle(x: mx.array, stride: Kernel) -> mx.array:
    b, t, h, w, cp = x.shape
    p1, p2, p3 = stride
    c = cp // (p1 * p2 * p3)
    x = x.reshape(b, t, h, w, c, p1, p2, p3).transpose(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(b, t * p1, h * p2, w * p3, c)


def _patchify(x: mx.array, p: int) -> mx.array:
    b, c, f, h, w = x.shape
    x = x.reshape(b, c, f, h // p, p, w // p, p).transpose(0, 2, 3, 5, 1, 6, 4)
    return x.reshape(b, f, h // p, w // p, c * p * p)


def _unpatchify(tokens: mx.array, p: int, c: int) -> mx.array:
    b, f, hq, wr, _ = tokens.shape
    x = tokens.reshape(b, f, hq, wr, c, p, p).transpose(0, 4, 1, 2, 6, 3, 5)
    return x.reshape(b, c, f, hq * p, wr * p)


@mx.compile
def _silu_mul(g: mx.array, u: mx.array) -> mx.array:
    return g * mx.sigmoid(g) * u


def _timestep_embedding(t: mx.array, dim: int) -> mx.array:
    half = dim // 2
    freqs = mx.array(
        np.exp(-math.log(10000.0) * np.arange(half, dtype=np.float32) / half)
    )
    args = t.astype(G).reshape(-1, 1) * freqs.reshape(1, -1)
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


# fp16 operands and stream, untiled: 768x512x121 peaks at 16.3 GiB, 1536x1024x121 at
# 35.3 GiB (2026-10-02); slope 133 B per output pixel-frame, intercept ~10 GiB.
PEAK_BASE_BYTES = 10 << 30
PEAK_BYTES_PER_PIXEL_FRAME = 140


def estimate_peak_bytes(latent_shape: tuple[int, ...]) -> int:
    _, _, f, h, w = latent_shape
    return PEAK_BASE_BYTES + PEAK_BYTES_PER_PIXEL_FRAME * (8 * f - 7) * (32 * h) * (
        32 * w
    )


class DiffusionVAE:
    """`operand` is the GEMM / attention dtype, `stream` the residual dtype.

    Upstream runs this decoder entirely in bf16. fp16 operands with an fp16
    stream measure 64.5 dB / max 1 level against an all-fp32 run (fp32 stream:
    65.4 dB), at 7.5 GiB instead of 10.2 for 768x512x49; both are defaults.
    """

    def __init__(
        self,
        root: Path | None = None,
        operand: mx.Dtype = mx.float16,
        stream: mx.Dtype = mx.float16,
        attention: str = "metal",
    ):
        self.path = checkpoints.path_of("video-vae-diffusion", root)
        self.operand, self.stream, self.attention = operand, stream, attention
        self.w: dict[str, mx.array] = {}
        self.cfg: DiffVAEConfig | None = None

    def load(self) -> DiffusionVAE:
        if self.w:
            return self
        header = checkpoints.read_header(self.path)
        self.cfg = DiffVAEConfig.from_header(header)
        raw = checkpoints.load_raw(self.path)
        w = {}
        for name in list(raw):
            if not name.startswith(("decoder.", "per_channel_statistics.")):
                raw.pop(name)
                continue
            a = raw.pop(name)
            glue = name.endswith(
                ("norm.weight", "norm1.weight", "norm2.weight", "norm_out.weight")
            ) or (
                "scale_shift_table" in name
                or "per_channel" in name
                or "type_emb" in name
            )
            w[name] = a.astype(G if glue else self.operand)
        mx.eval(w)
        self.w = w
        self.split = _rope_split(self.cfg.head_dim)
        self.inv = tuple(_inv_freqs(d) for d in self.split)
        return self

    # ---- pieces ----
    def _lin(self, name: str, x: mx.array) -> mx.array:
        wt, b = self.w[name + ".weight"], self.w.get(name + ".bias")
        if (x.size // x.shape[-1]) * wt.shape[0] >= MAX_ELEMENTS:
            # MLX 0.32.2 slices/splits of arrays past 2^31 elements return wrong
            # data for the tail (section 18); callers chunk before getting here.
            raise ValueError(f"{name}: output would exceed 2^31 elements; chunk it")
        y = x.astype(self.operand) @ wt.T
        return (y if b is None else y + b).astype(self.stream)

    def _qkv(
        self,
        p: str,
        y: mx.array,
        t_pos: mx.array | None = None,
        w_pos: mx.array | None = None,
        core: tuple[int, int, int, int] | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """y (1, T, H, W, C) already normalised/modulated -> q (pre-scaled), k, v
        (1, T, H, W, heads, hd) in the operand dtype, RoPE applied at `t_pos`
        (integer frame indices for video, fractional stage times for planes).
        With `core` = (t0, t1, w0, w1) q is projected for that block only (the
        slab's halo frames and columns are keys, never queries)."""
        cfg = self.cfg
        b, t, h, w, c = y.shape
        heads, hd = c // cfg.head_dim, cfg.head_dim
        wq, bq = self.w[p + ".qkv.weight"], self.w[p + ".qkv.bias"]
        t_pos = mx.arange(t).astype(G) if t_pos is None else t_pos
        h_pos = mx.arange(h).astype(G)
        w_pos = mx.arange(w).astype(G) if w_pos is None else w_pos

        def project(i: int, src: mx.array) -> mx.array:
            # one projection per output rather than one split: the fused output
            # passes 2^31 elements on long clips, where mx.split returns garbage
            return (
                (
                    src.astype(self.operand) @ wq[i * c : (i + 1) * c].T
                    + bq[i * c : (i + 1) * c]
                )
                .astype(self.stream)
                .reshape(b, src.shape[1], h, src.shape[3], heads, hd)
            )

        if core is None:
            yq, tq_pos, wq_pos = y, t_pos, w_pos
        else:
            t0, t1, w0, w1 = core
            yq, tq_pos, wq_pos = y[:, t0:t1, :, w0:w1], t_pos[t0:t1], w_pos[w0:w1]
        q = _rms(project(0, yq), self.w[p + ".q_norm.weight"]) * hd**-0.5
        q = _axial_rope(q, tq_pos, h_pos, wq_pos, self.split, self.inv)
        k = _rms(project(1, y), self.w[p + ".k_norm.weight"])
        k = _axial_rope(k, t_pos, h_pos, w_pos, self.split, self.inv)
        v = project(2, y)
        q, k, v = (a.astype(self.operand) for a in (q, k, v))
        mx.eval(q, k, v)  # the fp32 RoPE temporaries die here
        return q, k, v

    def _attn(
        self,
        p: str,
        y: mx.array,
        kernel: Kernel,
        w_pos: mx.array | None = None,
        t_pos: mx.array | None = None,
        planes: tuple[mx.array, mx.array] | None = None,
        core: tuple[int, int, int, int] | None = None,
    ) -> mx.array:
        """y (1, T, H, W, C) already normalised/modulated -> attention output
        projected (stream dtype), for the `core` = (t0, t1, w0, w1) query block
        only (default: all of y). With `planes` = (plane input (1, P, H, W, C)
        prepared the same way, plane times) the video queries also see their
        nearest keyframe planes (joint softmax)."""
        b, t, h, w, c = y.shape
        t0, t1, w0, w1 = core if core is not None else (0, t, 0, w)
        q, k, v = self._qkv(p, y, t_pos, w_pos, core)
        del y
        if planes is None:
            if self.attention == "metal":
                o = na3d(q, k, v, kernel, offset=(t0, w0))
            else:
                o = na3d_mlx(q, k, v, kernel)[:, t0:t1, :, w0:w1]
        else:
            py, times = planes
            _, kb, vb = self._qkv(p, py, times, w_pos)
            t_np = np.arange(t) if t_pos is None else np.array(t_pos)
            slots = mx.array(nearest_slots(t_np[t0:t1], np.array(times)))
            o = na3d_joint(q, k, v, kb, vb, slots, kernel, offset=(t0, w0))
            del kb, vb
        mx.eval(o)
        del q, k, v
        y = self._lin(p + ".proj", o.reshape(b, t1 - t0, h, w1 - w0, c))
        mx.eval(y)
        return y

    def _plane_attn(
        self,
        p: str,
        py: mx.array,
        times: mx.array,
        frames_of,
        n_frames: int,
        kernel: Kernel,
        w_slabs: bool = False,
    ) -> mx.array:
        """The plane queries' pass: each plane attends to the Kh x Kw window on
        itself (a Kt = 1 kernel over the plane stack) and on its nearest video
        frames. `frames_of(indices)` returns those frames' prepared input
        (1, n, H, W, C) from the pre-attention stream; only the frames some
        plane points at are projected. With `w_slabs` (stage 5) both streams
        are cut into upstream's W slabs, as chunked/attn.py does, so plane
        queries at the volume edge see the same replicated halo the video does."""
        b, n_planes, h, w, c = py.shape
        slots = nearest_slots(np.array(times), np.arange(n_frames))
        wanted = np.unique(slots[slots >= 0])
        remap = {int(f): i for i, f in enumerate(wanted)}
        local = mx.array(
            np.vectorize(lambda s: remap[int(s)] if s >= 0 else -1)(slots).astype(
                np.int32
            )
        )
        fy, ft = frames_of(wanted), mx.array(wanted).astype(G)
        k1 = (1, kernel[1], kernel[2])

        def one(pbuf, fbuf, w_pos):
            q, k, v = self._qkv(p, pbuf, times, w_pos)
            _, kb, vb = self._qkv(p, fbuf, ft, w_pos)
            o = na3d_joint(q, k, v, kb, vb, local, k1)
            del q, k, v, kb, vb
            return o

        if not w_slabs:
            o = one(py, fy, None)
        else:
            halo = kernel[2] // 2
            frame_slabs = [buf for buf, _, _ in self._w_slabs(fy, halo)]
            outs = []
            for j, (pbuf, w_pos, n) in enumerate(self._w_slabs(py, halo)):
                oj = one(pbuf, frame_slabs[j], w_pos)[:, :, :, halo : halo + n]
                mx.eval(oj)
                outs.append(oj)
            o = mx.concatenate(outs, axis=3)
        y = self._lin(p + ".proj", o.reshape(b, n_planes, h, w, c))
        mx.eval(y)
        return y

    def _t_chunks(self, x: mx.array, kernel: Kernel):
        """(core start, core end, slab start, slab end) temporal chunks whose
        slabs carry a full halo, so windows clamp exactly as in the whole volume."""
        t_total, h, w = x.shape[1:4]
        halo, kt = kernel[0] // 2, min(kernel[0], x.shape[1])
        per_frame = h * w
        frames = max(1, CHUNK_TOKENS // per_frame)
        step = math.ceil(t_total / math.ceil(t_total / frames))
        for c0 in range(0, t_total, step):
            c1 = min(t_total, c0 + step)
            s0, s1 = max(0, c0 - halo), min(t_total, c1 + halo)
            if s1 - s0 < kt:
                s0, s1 = (max(0, s1 - kt), s1) if s0 > 0 else (0, min(t_total, kt))
            yield c0, c1, s0, s1

    def _chunked_attention(
        self,
        p: str,
        x: mx.array,
        kernel: Kernel,
        pre,
        w_slabs: bool,
        planes: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """Attention over temporal chunks (and upstream's W slabs for stage 5);
        `pre` maps a slab of x to its normalised/modulated input. `planes` =
        (prepared plane input, plane times) adds the keyframe planes to every
        video query's softmax."""
        cores = []
        for c0, c1, s0, s1 in self._t_chunks(x, kernel):
            xs = x[:, s0:s1]
            t_pos = mx.arange(s0, s1).astype(G)
            if w_slabs:
                halo = kernel[2] // 2
                outs = []
                plane_slabs = (
                    None
                    if planes is None
                    else [buf for buf, _, _ in self._w_slabs(planes[0], halo)]
                )
                for i, (buf, w_pos, n) in enumerate(self._w_slabs(xs, halo)):
                    pl = None if planes is None else (plane_slabs[i], planes[1])
                    o = self._attn(
                        p,
                        pre(buf),
                        kernel,
                        w_pos=w_pos,
                        t_pos=t_pos,
                        planes=pl,
                        core=(c0 - s0, c1 - s0, halo, halo + n),
                    )
                    del buf
                    mx.eval(o)
                    outs.append(o)
                core = mx.concatenate(outs, axis=3)
                del outs
            else:
                core = self._attn(
                    p,
                    pre(xs),
                    kernel,
                    t_pos=t_pos,
                    planes=planes,
                    core=(c0 - s0, c1 - s0, 0, xs.shape[3]),
                )
            mx.eval(core)
            cores.append(core)
        return cores[0] if len(cores) == 1 else mx.concatenate(cores, axis=1)

    def _swiglu(self, p: str, x: mx.array, chunks: int = 4) -> mx.array:
        """Per-token MLP in temporal chunks: the 4x hidden state stays in the
        operand dtype and never exists for the whole volume at once."""
        wg, wu, wd = (self.w[f"{p}.{n}.weight"] for n in ("w_gate", "w_up", "w_down"))
        outs = []
        step = max(1, math.ceil(x.shape[1] / chunks))
        for t0 in range(0, x.shape[1], step):
            xo = x[:, t0 : t0 + step].astype(self.operand)
            h = _silu_mul(xo @ wg.T, xo @ wu.T)
            y = (h @ wd.T).astype(self.stream)
            mx.eval(y)
            outs.append(y)
        return mx.concatenate(outs, axis=1)

    def _det_block(
        self,
        p: str,
        x: mx.array,
        kernel: Kernel,
        planes: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array | tuple[mx.array, mx.array]:
        """Pre-norm NA -> SwiGLU with residuals. With `planes` = (plane stream,
        plane times) both streams run the block with shared weights and meet in
        the joint softmax (upstream NABlock.forward_with_keyframes)."""
        n1 = self.w[p + ".norm1.weight"]
        n2 = self.w[p + ".norm2.weight"]
        if planes is None:
            x = x + self._chunked_attention(
                p + ".attn", x, kernel, lambda b: _rms(b, n1), w_slabs=False
            )
            x = x + self._swiglu(p + ".mlp", _rms(x, n2))
            mx.eval(x)
            return x
        px, times = planes
        py = _rms(px, n1)
        plane_out = self._plane_attn(
            p + ".attn",
            py,
            times,
            lambda idx: _rms(x[:, mx.array(idx)], n1),
            x.shape[1],
            kernel,
        )
        x = x + self._chunked_attention(
            p + ".attn",
            x,
            kernel,
            lambda b: _rms(b, n1),
            w_slabs=False,
            planes=(py, times),
        )
        px = px + plane_out
        x = x + self._swiglu(p + ".mlp", _rms(x, n2))
        px = px + self._swiglu(p + ".mlp", _rms(px, n2), chunks=1)
        mx.eval(x, px)
        return x, px

    def _upsample(self, i: int, x: mx.array, drop_leading: bool) -> mx.array:
        stride, _ = self.cfg.upsamples[i]
        y = _pixel_shuffle(self._lin(f"decoder.upsamples.{i}.proj", x), stride)
        return y[:, 1:] if (stride[0] == 2 and drop_leading) else y

    def _upsample_planes(self, i: int, px: mx.array) -> mx.array:
        """Spatial-only upsample of the plane stack (1, P, H, W, C): each plane
        is its own T = 1 clip, so a temporal stride of 2 yields two frames of
        which the leading one is dropped (phase 1), keeping P invariant
        (upstream `upsample_keyframe_planes`)."""
        stride, _ = self.cfg.upsamples[i]
        y = _pixel_shuffle(self._lin(f"decoder.upsamples.{i}.proj", px), stride)
        return y[:, 1::2] if stride[0] == 2 else y

    def _w_slabs(self, x: mx.array, halo: int):
        """Upstream build_w_slabs: W_CHUNKS slabs with a `halo` on each side."""
        width = x.shape[3]
        chunk = math.ceil(width / W_CHUNKS)
        extent = chunk + 2 * halo
        left = None
        for i in range(W_CHUNKS):
            cs, ce = i * chunk, min(width, (i + 1) * chunk)
            core = x[:, :, :, cs:ce]
            n = ce - cs
            has_right = i < W_CHUNKS - 1
            if left is None:
                lp = (
                    mx.repeat(core[:, :, :, :1], halo, axis=3)
                    if n
                    else mx.zeros_like(x[:, :, :, :halo])
                )
            else:
                lp = (
                    left
                    if left.shape[3] >= halo
                    else mx.concatenate(
                        [mx.zeros_like(x[:, :, :, : halo - left.shape[3]]), left],
                        axis=3,
                    )
                )
            parts = [lp, core]
            filled = 0
            if has_right:
                right = x[:, :, :, ce : min(width, ce + halo)]
                filled = right.shape[3]
                if filled:
                    parts.append(right)
            missing = extent - (halo + n + filled)
            if missing > 0:
                fill = core[:, :, :, -1:] if n else mx.zeros_like(x[:, :, :, :1])
                parts.append(mx.repeat(fill, missing, axis=3))
            buf = mx.concatenate(parts, axis=3)
            if has_right:
                left = x[:, :, :, ce - min(halo, n) : ce]
            yield buf, mx.arange(extent).astype(G) + float(cs - halo), n

    def _chunk_bounds(
        self, t_total: int, h: int, w: int, kernel: Kernel
    ) -> list[tuple[int, int]]:
        frames = max(1, CHUNK_TOKENS // (h * w))
        step = math.ceil(t_total / math.ceil(t_total / frames))
        return [(c0, min(t_total, c0 + step)) for c0 in range(0, t_total, step)]

    def _slab(
        self,
        chunks: list[mx.array],
        bounds: list[tuple[int, int]],
        i: int,
        kernel: Kernel,
    ):
        """Chunk i plus its temporal halo gathered from the neighbouring chunks:
        (slab, core offset, slab start frame). Exact: windows are centred and
        clamped against the volume, and every core query finds the same keys."""
        t_total = bounds[-1][1]
        halo, kt = kernel[0] // 2, min(kernel[0], t_total)
        c0, c1 = bounds[i]
        s0, s1 = max(0, c0 - halo), min(t_total, c1 + halo)
        if s1 - s0 < kt:
            s0, s1 = (max(0, s1 - kt), s1) if s0 > 0 else (0, min(t_total, kt))
        parts = []
        for (b0, b1), ch in zip(bounds, chunks):
            lo, hi = max(b0, s0), min(b1, s1)
            if lo < hi:
                parts.append(ch[:, lo - b0 : hi - b0])
        return (
            (parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)),
            c0 - s0,
            s0,
        )

    def _diff_block(
        self,
        p: str,
        chunks: list[mx.array],
        bounds: list[tuple[int, int]],
        context: list[mx.array],
        mod: list[mx.array],
        planes: tuple[mx.array, mx.array, mx.array] | None = None,
    ) -> list[mx.array] | tuple[list[mx.array], mx.array]:
        """One stage-5 block over the chunk-resident residual stream. With
        `planes` = (plane stream, plane context, plane times) the keyframe pixel
        stream runs the same block (own context injection, joint softmax, shared
        MLP; upstream forward_combined_with_keyframes)."""
        table = self.w[p + ".scale_shift_table"]
        pm = [m + table[i].reshape(1, 1, 1, 1, -1) for i, m in enumerate(mod)]
        scale_msa, shift_msa, scale_mlp, shift_mlp = pm[0], pm[1], pm[3], pm[4]
        n1, n2 = self.w[p + ".norm1.weight"], self.w[p + ".norm2.weight"]
        kernel = self.cfg.stage5_kernel
        halo_w = kernel[2] // 2

        def pre(b):
            return _rms(b, n1) * (1 + scale_msa) + shift_msa

        # 1. context injection, per chunk
        for i, (c0, c1) in enumerate(bounds):
            chunks[i] = (chunks[i] + self._lin(p + ".context_proj", context[i])).astype(
                self.stream
            )
            mx.eval(chunks[i])
        plane_slabs = plane_out = None
        if planes is not None:
            px, pctx, times = planes
            px = (px + self._lin(p + ".context_proj", pctx)).astype(self.stream)
            py = pre(px)
            n_frames = bounds[-1][1]

            def frames_of(idx):
                parts = []
                for f in idx:
                    f = int(f)
                    ci = next(i for i, (c0, c1) in enumerate(bounds) if c0 <= f < c1)
                    parts.append(
                        chunks[ci][:, f - bounds[ci][0] : f - bounds[ci][0] + 1]
                    )
                return pre(mx.concatenate(parts, axis=1))

            plane_out = self._plane_attn(
                p + ".attn", py, times, frames_of, n_frames, kernel, w_slabs=True
            )
            plane_slabs = [buf for buf, _, _ in self._w_slabs(py, halo_w)]
        # 2. attention: read slabs from the pre-attention stream, write the new stream
        new = []
        for i, (c0, c1) in enumerate(bounds):
            slab, off, s0 = self._slab(chunks, bounds, i, kernel)
            t_pos = mx.arange(s0, s0 + slab.shape[1]).astype(G)
            outs = []
            for j, (buf, w_pos, n) in enumerate(self._w_slabs(slab, halo_w)):
                y = pre(buf)
                del buf
                pl = None if planes is None else (plane_slabs[j], times)
                o = self._attn(
                    p + ".attn",
                    y,
                    kernel,
                    w_pos=w_pos,
                    t_pos=t_pos,
                    planes=pl,
                    core=(off, off + (c1 - c0), halo_w, halo_w + n),
                )
                del y
                mx.eval(o)
                outs.append(o)
            del slab
            core = (chunks[i] + mx.concatenate(outs, axis=3)).astype(self.stream)
            del outs
            mx.eval(core)
            new.append(core)
        chunks = new
        # 3. MLP, per chunk
        for i in range(len(chunks)):
            chunks[i] = chunks[i] + self._swiglu(
                p + ".mlp", _rms(chunks[i], n2) * (1 + scale_mlp) + shift_mlp
            )
            mx.eval(chunks[i])
        if planes is None:
            return chunks
        px = px + plane_out
        px = px + self._swiglu(
            p + ".mlp", _rms(px, n2) * (1 + scale_mlp) + shift_mlp, chunks=1
        )
        mx.eval(px)
        return chunks, px

    # ---- decode ----
    def denormalize(self, z: mx.array) -> mx.array:
        s = self.w["per_channel_statistics.std-of-means"].reshape(1, -1, 1, 1, 1)
        m = self.w["per_channel_statistics.mean-of-means"].reshape(1, -1, 1, 1, 1)
        return z.astype(G) * s + m

    def _pad_to_floor(self, latent: mx.array):
        f_min, h_min, w_min = self.cfg.min_latent_shape()
        _, _, f, h, w = latent.shape
        t_pad, h_need, w_need = max(f_min - f, 0), max(h_min - h, 0), max(w_min - w, 0)
        h_b, w_b = h_need // 2, w_need // 2
        if t_pad:
            latent = mx.concatenate(
                [latent, mx.repeat(latent[:, :, -1:], t_pad, axis=2)], axis=2
            )
        if h_need:
            latent = mx.concatenate(
                [
                    mx.repeat(latent[:, :, :, :1], h_b, 3),
                    latent,
                    mx.repeat(latent[:, :, :, -1:], h_need - h_b, 3),
                ],
                axis=3,
            )
        if w_need:
            latent = mx.concatenate(
                [
                    mx.repeat(latent[..., :1], w_b, 4),
                    latent,
                    mx.repeat(latent[..., -1:], w_need - w_b, 4),
                ],
                axis=4,
            )
        return latent, h_b, w_b

    def remaining_time_strides(self) -> tuple[int, ...]:
        """Temporal upsampling still to come at each stage input, plus 1 for
        stage 5: (8, 8, 4, 2, 1) for the production ladder."""
        strides = [st for (st, _, _), _ in self.cfg.upsamples]
        return tuple(math.prod(strides[i:]) for i in range(len(strides))) + (1,)

    def stages_1_to_4(
        self,
        latent_padded: mx.array,
        tap=None,
        keyframes: tuple[mx.array, np.ndarray] | None = None,
    ) -> mx.array | tuple[mx.array, mx.array]:
        """Deterministic stages. `keyframes` = (plane latents (1, C, P, H, W)
        padded like the video latent, pixel frame index per plane) runs the
        dual stream and also returns the planes' stage-4 feature."""
        cfg = self.cfg
        z = self.denormalize(latent_padded)
        z = mx.concatenate(
            [z, mx.repeat(z[:, :, -1:], cfg.ghost_frames, axis=2)], axis=2
        )
        x = self._lin("decoder.conv_in", z.transpose(0, 2, 3, 4, 1))
        px = frames = None
        if keyframes is not None:
            planes, frames = keyframes
            # the keyframe tag is the only plane-specific weight: added to the
            # un-normalised latents right before the shared conv_in
            pz = self.denormalize(planes).transpose(0, 2, 3, 4, 1) + self.w[
                "decoder.type_emb"
            ].reshape(1, 1, 1, 1, -1)
            px = self._lin("decoder.conv_in", pz)
        remaining = self.remaining_time_strides()
        for s in range(4):
            times = (
                None
                if px is None
                else mx.array(keyframe_stage_times(frames, remaining[s]))
            )
            for i in range(cfg.depths[s]):
                if px is None:
                    x = self._det_block(
                        f"decoder.det_stages.{s}.{i}", x, cfg.kernels[s]
                    )
                else:
                    x, px = self._det_block(
                        f"decoder.det_stages.{s}.{i}", x, cfg.kernels[s], (px, times)
                    )
            if tap:
                tap(f"s{s + 1}.out", x)
                if px is not None:
                    tap(f"s{s + 1}.planes", px)
            if s < 3:
                x = self._upsample(s, x, drop_leading=True)
                if px is not None:
                    px = self._upsample_planes(s, px)
        # ghost crop: the replicated frames fed stage 1-4's windows and are dropped here
        strides = cfg.cumulative_strides()[3]
        keep = min(
            x.shape[1],
            max(
                x.shape[1] - cfg.ghost_frames * strides[0],
                math.ceil(cfg.stage5_kernel[0] / 2),
            ),
        )
        return x[:, :keep] if px is None else (x[:, :keep], px)

    def stage_5(
        self,
        x_t: mx.array,
        feat: mx.array,
        t: float = 1.0,
        tap=None,
        keyframes: tuple[mx.array, mx.array, np.ndarray] | None = None,
    ) -> mx.array:
        """One x0 step. `keyframes` = (plane noise (1, C_out, P, H_px, W_px),
        plane stage-4 feature, pixel frame index per plane): the keyframe pixel
        stream runs alongside so the joint attention reads planes at the noise
        level it was trained on; only the video pixels are returned."""
        cfg = self.cfg
        (st3, _, _), _ = cfg.upsamples[3]
        te = _timestep_embedding(mx.array([t * cfg.timestep_scale]), cfg.t_freq)
        te = self._lin("decoder.t_embedder.mlp.0", te)
        te = self._lin("decoder.t_embedder.mlp.2", te * mx.sigmoid(te))
        mod = self._lin("decoder.shared_adaln.proj", te * mx.sigmoid(te))
        mod = [m.reshape(1, 1, 1, 1, -1) for m in mx.split(mod, 7, axis=-1)]
        tokens = _patchify(x_t, cfg.patch)
        bounds = self._chunk_bounds(
            tokens.shape[1], tokens.shape[2], tokens.shape[3], cfg.stage5_kernel
        )
        chunks = []
        for c0, c1 in bounds:
            ch = self._lin("decoder.conv_in_x_t", tokens[:, c0:c1])
            mx.eval(ch)
            chunks.append(ch)
        del tokens, x_t
        # Context frames come from upsample 3 of the stage-4 feature (temporal
        # stride 2, leading frame dropped), computed per stage-5 chunk: the
        # whole context passes 2^31 elements on long clips.
        context = []
        for c0, c1 in bounds:
            if (
                st3 == 2
            ):  # context frame j is shuffled frame j + 1, i.e. feat frame (j + 1) // 2
                a, bnd = (c0 + 1) // 2, min(feat.shape[1], (c1 + 1) // 2 + 1)
                piece = _pixel_shuffle(
                    self._lin("decoder.upsamples.3.proj", feat[:, a:bnd]),
                    cfg.upsamples[3][0],
                )[:, c0 + 1 - 2 * a : c1 + 1 - 2 * a]
            else:
                piece = _pixel_shuffle(
                    self._lin("decoder.upsamples.3.proj", feat[:, c0:c1]),
                    cfg.upsamples[3][0],
                )
            piece = piece.astype(self.operand)
            mx.eval(piece)
            if piece.shape[1] != c1 - c0:
                raise RuntimeError(f"context chunk {piece.shape} for frames {c0}:{c1}")
            context.append(piece)
        del feat
        planes = None
        if keyframes is not None:
            p_noise, p_feat, frames = keyframes
            # plane context: upsample 3 of the planes' stage-4 feature, phase 1
            pctx = _pixel_shuffle(
                self._lin("decoder.upsamples.3.proj", p_feat), cfg.upsamples[3][0]
            )
            pctx = (pctx[:, 1::2] if st3 == 2 else pctx).astype(self.operand)
            px = self._lin("decoder.conv_in_x_t", _patchify(p_noise, cfg.patch))
            # at stage 5 the remaining stride is 1: plane times are pixel frames
            planes = (px, pctx, mx.array(keyframe_stage_times(frames, 1)))
            mx.eval(px, pctx)
        for i in range(cfg.depths[4]):
            if planes is None:
                chunks = self._diff_block(
                    f"decoder.diff_blocks.{i}", chunks, bounds, context, mod
                )
            else:
                chunks, px = self._diff_block(
                    f"decoder.diff_blocks.{i}", chunks, bounds, context, mod, planes
                )
                planes = (px, planes[1], planes[2])
            if tap:
                tap(
                    f"s5.b{i}.out",
                    chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=1),
                )
        del context, planes
        pixels = []
        for ch in chunks:
            px = _unpatchify(
                self._lin(
                    "decoder.conv_out", _rms(ch, self.w["decoder.norm_out.weight"])
                ),
                cfg.patch,
                cfg.out_channels,
            )
            mx.eval(px)
            pixels.append(px)
        return pixels[0] if len(pixels) == 1 else mx.concatenate(pixels, axis=2)

    def _pad_planes(self, planes: mx.array, h_b: int, w_b: int, h: int, w: int):
        """The spatial padding `_pad_to_floor` gave the video latent, applied to
        the plane stack (1, C, P, h, w): both streams must share one geometry
        or every plane is offset from the video."""
        _, _, _, ph, pw = planes.shape
        if (ph, pw) != (h, w):
            raise ValueError(f"plane latents {planes.shape} vs video {(h, w)}")
        _, h_min, w_min = self.cfg.min_latent_shape()
        h_need, w_need = max(h_min - h, 0), max(w_min - w, 0)
        if h_need:
            planes = mx.concatenate(
                [
                    mx.repeat(planes[:, :, :, :1], h_b, 3),
                    planes,
                    mx.repeat(planes[:, :, :, -1:], h_need - h_b, 3),
                ],
                axis=3,
            )
        if w_need:
            planes = mx.concatenate(
                [
                    mx.repeat(planes[..., :1], w_b, 4),
                    planes,
                    mx.repeat(planes[..., -1:], w_need - w_b, 4),
                ],
                axis=4,
            )
        return planes

    def decode_raw(
        self,
        latent: mx.array,
        seed: int = 0,
        tap=None,
        keyframes: tuple[mx.array, list[int]] | None = None,
    ) -> mx.array:
        """Normalized latent (1, 128, F, H, W)
        -> pixels (1, 3, 8F-7, 32H, 32W) fp32 in ~[-1, 1].

        `keyframes` = (normalized plane latents (1, 128, P, H, W), one latent
        frame per keyframe, each encoded/generated as a standalone one-frame
        clip; its pixel frame index per plane) runs upstream's keyframe-aware
        decode (DFR always decodes this way)."""
        self.load()
        cfg = self.cfg
        _, _, f, h, w = latent.shape
        padded, h_b, w_b = self._pad_to_floor(latent.astype(G))
        f_px_total = (f - 1) * cfg.cumulative_strides()[4][0] + 1
        if keyframes is not None:
            planes, frames = keyframes
            frames = np.asarray(frames, dtype=np.int64)
            if planes.shape[2] != len(frames) or planes.shape[2] == 0:
                raise ValueError(
                    f"{planes.shape[2]} planes for {len(frames)} frame indices"
                )
            if frames.min() < 0 or frames.max() >= f_px_total:
                raise ValueError(f"keyframe frames {frames} outside 0..{f_px_total}")
            planes = self._pad_planes(planes.astype(G), h_b, w_b, h, w)
            feat, p_feat = self.stages_1_to_4(padded, tap, (planes, frames))
        else:
            feat = self.stages_1_to_4(padded, tap)
        t4, h4, w4 = feat.shape[1:4]
        (st, sh, sw), _ = cfg.upsamples[3]
        canvas = (t4 * st - 1, h4 * sh * cfg.patch, w4 * sw * cfg.patch)
        noise = mx.random.normal(
            (1, cfg.out_channels, *canvas), key=mx.random.key(seed + NOISE_SEED_OFFSET)
        )
        kf = None
        if keyframes is not None:
            # the plane stream draws its own noise, after the video's (upstream
            # draws both from one generator in that order)
            p_noise = mx.random.normal(
                (1, cfg.out_channels, len(frames), canvas[1], canvas[2]),
                key=mx.random.key(seed + NOISE_SEED_OFFSET + 1),
            )
            kf = (p_noise.astype(self.stream), p_feat, frames)
        pixels = self.stage_5(noise.astype(self.stream), feat, 1.0, tap, kf)
        t_scale = cfg.cumulative_strides()[4][0]
        s_scale = cfg.cumulative_strides()[4][1] * cfg.patch
        f_px, h_px, w_px = (f - 1) * t_scale + 1, h * s_scale, w * s_scale
        return pixels[
            :,
            :,
            :f_px,
            h_b * s_scale : h_b * s_scale + h_px,
            w_b * s_scale : w_b * s_scale + w_px,
        ].astype(G)
