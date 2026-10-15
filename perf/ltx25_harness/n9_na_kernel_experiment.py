"""N9: neighborhood-attention kernel variants on the production stage-5 slab.

Variant `qw`: each thread group of LANES lanes serves QW adjacent queries along
W. Their windows overlap in all but QW-1 columns, so one loaded key/value row
serves all QW queries (the current kernel reloads it once per query). Online
softmax state per query stays in registers. Exact NATTEN clamping per query.

Usage: PYTHONPATH=<worktree> python n9_na_kernel_experiment.py [T H W] [QW...]
Default slab 33x256x106 (a stage-5 temporal chunk x W slab at 1536x1024),
4 heads x 64, kernel 11x11x11. Reports TF/s and max error vs the shipped kernel.
Run through gpu_run.py --need-gb 20."""

import sys
import time

import mlx.core as mx
import numpy as np

from slimserve.video.ltx25 import diffvae

mx.set_cache_limit(16 << 30)
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
T, H, W = (int(x) for x in pos[:3]) if len(pos) >= 3 else (33, 256, 106)
QWS = [int(x) for x in pos[3:]] or [2]
LANES_LIST = [8, 4]
NH, HD = 4, 64
KERNEL = (11, 11, 11)

_SRC = """
    // LANES lanes per group of QW adjacent-in-W queries of one head; lane j owns
    // dims [j*DL, (j+1)*DL) of every query in the group. One key row load feeds
    // all QW online softmaxes.
    const uint T = shape[0], H = shape[1], W = shape[2], NH = shape[3];
    const uint gid = thread_position_in_grid.x;
    const uint lane = gid % LANES;
    const uint grp = gid / LANES;
    const uint groups_w = (W + QW - 1) / QW;
    const uint total = T * H * groups_w * NH;
    const bool live = grp < total;
    const uint head = live ? grp % NH : 0;
    const uint g = live ? grp / NH : 0;
    const uint gw = g % groups_w;
    const uint hi = (g / groups_w) % H;
    const uint ti = g / (groups_w * H);
    const int kt = min((int)KT, (int)T);
    const int kh = min((int)KH, (int)H);
    const int kw = min((int)KW, (int)W);
    const int t0 = clamp((int)ti - kt / 2, 0, (int)T - kt);
    const int h0 = clamp((int)hi - kh / 2, 0, (int)H - kh);
    const uint stride_tok = NH * HD;
    const uint wbase = gw * QW;
    int w0[QW];
    float qr[QW][DL];
    float m[QW], l[QW];
    float acc[QW][DL];
    for (uint i = 0; i < QW; ++i) {
        const int wi = min((int)(wbase + i), (int)W - 1);
        w0[i] = clamp(wi - kw / 2, 0, (int)W - kw);
        const size_t tok = ((size_t)ti * H + hi) * W + wi;
        const size_t qoff = tok * stride_tok + head * HD + lane * DL;
        for (uint d = 0; d < DL; ++d) qr[i][d] = (float)q[qoff + d];
        m[i] = -1.0e30f; l[i] = 0.0f;  // finite: a masked first key must not make exp(-inf - -inf)
        for (uint d = 0; d < DL; ++d) acc[i][d] = 0.0f;
    }
    const int wlo = w0[0], whi = w0[QW - 1] + kw;  // union of the QW windows
    for (int a = 0; a < kt; ++a) {
        for (int b = 0; b < kh; ++b) {
            const size_t row = ((size_t)(t0 + a) * H + (h0 + b)) * W;
            for (int c = wlo; c < whi; ++c) {
                const size_t off = (row + c) * stride_tok + head * HD + lane * DL;
                float kr[DL], vr[DL];
                for (uint d = 0; d < DL; ++d) { kr[d] = (float)k[off + d]; vr[d] = (float)v[off + d]; }
                for (uint i = 0; i < QW; ++i) {
                    float s = 0.0f;
                    for (uint d = 0; d < DL; ++d) s += qr[i][d] * kr[d];
                    SHUFFLES
                    const bool vis = (c >= w0[i]) && (c < w0[i] + kw);
                    const float m_new = vis ? max(m[i], s) : m[i];
                    const float corr = exp(m[i] - m_new);
                    const float p = vis ? exp(s - m_new) : 0.0f;
                    l[i] = l[i] * corr + p;
                    for (uint d = 0; d < DL; ++d) acc[i][d] = acc[i][d] * corr + p * vr[d];
                    m[i] = m_new;
                }
            }
        }
    }
    if (live) {
        for (uint i = 0; i < QW; ++i) {
            const uint wi = wbase + i;
            if (wi >= W) break;
            const size_t tok = ((size_t)ti * H + hi) * W + wi;
            const size_t base = tok * stride_tok + head * HD + lane * DL;
            const float inv = 1.0f / l[i];
            for (uint d = 0; d < DL; ++d) out[base + d] = (T_IN)(acc[i][d] * inv);
        }
    }
"""

_KERNELS = {}


def na3d_qw(q, k, v, kernel, qw, lanes=8):
    _, t, h, w, heads, hd = q.shape
    key = (kernel, hd, qw, lanes)
    kern = _KERNELS.get(key)
    if kern is None:
        src = _SRC.replace(
            "SHUFFLES",
            "".join(f"s += simd_shuffle_xor(s, {o});\n" for o in (4, 2, 1) if o < lanes),
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
        kern = mx.fast.metal_kernel(
            name=f"na3d_qw{qw}_l{lanes}",
            input_names=["q", "k", "v", "shape"],
            output_names=["out"],
            source=src,
        )
        _KERNELS[key] = kern
    groups = t * h * ((w + qw - 1) // qw) * heads
    total = groups * lanes
    shape = mx.array([t, h, w, heads], dtype=mx.uint32)
    mx.synchronize()
    (out,) = kern(
        inputs=[q, k, v, shape],
        template=[("T_IN", q.dtype)],
        grid=((total + 255) // 256 * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )
    mx.eval(out)
    mx.synchronize()
    return out


def bench(fn, n=3):
    fn()
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    mx.synchronize()
    return (time.perf_counter() - t) / n


rng = np.random.default_rng(0)
mk = lambda: mx.array(
    (rng.standard_normal((1, T, H, W, NH, HD)) * 0.3).astype(np.float16)
)
q, k, v = mk(), mk(), mk()
keys = min(KERNEL[0], T) * min(KERNEL[1], H) * min(KERNEL[2], W)
flops = 2 * 2 * T * H * W * NH * keys * HD
ref = diffvae.na3d(q, k, v, KERNEL)
t_ref = bench(lambda: diffvae.na3d(q, k, v, KERNEL))
print(f"slab {T}x{H}x{W}, {NH}x{HD}, kernel {KERNEL}: shipped {t_ref * 1e3:.0f} ms, {flops / t_ref / 1e12:.2f} TF/s")
for lanes in LANES_LIST:
    for qw in QWS:
        out = na3d_qw(q, k, v, KERNEL, qw, lanes)
        err = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
        t_qw = bench(lambda: na3d_qw(q, k, v, KERNEL, qw, lanes))
        print(
            f"  lanes={lanes} QW={qw}: {t_qw * 1e3:.0f} ms, {flops / t_qw / 1e12:.2f} TF/s,"
            f" x{t_ref / t_qw:.2f}, max |diff| vs shipped {err:.2e}"
        )
