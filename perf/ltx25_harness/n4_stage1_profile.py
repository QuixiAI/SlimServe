"""Per-op profile of the stage-1 forward (1,536 video tokens) and per-op TF/s, plus a synthetic
dev-batch (4 x 1,536) and a 24,576-token HD forward, all with the same instrumentation as n3_forward_profile.
Usage: PYTHONPATH=<worktree> python n4_stage1_profile.py  (gpu_run.py --need-gb 48)"""
import os, pickle, time, collections, mlx.core as mx
from slimserve.video.ltx25 import checkpoints, dit as D, sampling
from paths import SCRATCH
weights, _, tcfg = checkpoints.load_dit("distilled")
m = D.LTX25DiT(weights, D.DiTConfig.from_checkpoint(tcfg))
d = pickle.load(open(f"{SCRATCH}/gate/out/stage2_inputs.pkl", "rb"))["kw"]
vt, at = mx.array(d["video_text_embeds"][1]), mx.array(d["audio_text_embeds"][1])
def inputs(f, h, w, b=1):
    rep = lambda x: mx.repeat(x, b, axis=0)
    n_a = sampling.audio_token_count(8 * f - 7, 24.0)
    st = sampling.noised_state((b, f * h * w, 128), rep(sampling.video_positions(f, h, w, 24.0)), 1, sigma=0.9, tokens_per_frame=h * w)
    au = sampling.noised_state((b, n_a, 128), rep(sampling.audio_positions(n_a)), 2, sigma=0.9)
    return (st.latent, au.latent, mx.full((b,), 0.9), rep(vt), rep(at), st.positions, au.positions), st.keyframes_mask
acc = collections.defaultdict(lambda: [0.0, 0, 0.0])
def cls(name):
    n = name.split(".", 2)[-1] if name.startswith("transformer_blocks") else "top." + name
    for a, b in (("audio_to_video_attn", "a2v"), ("video_to_audio_attn", "v2a")): n = n.replace(a, b)
    return n
def timed(key, fn, ins, flops=0.0):
    mx.eval(*ins); mx.synchronize(); t = time.perf_counter(); y = fn(); mx.eval(y); mx.synchronize()
    r = acc[key]; r[0] += time.perf_counter() - t; r[1] += 1; r[2] += flops; return y
_lin = m.lin
def lin(name, x):
    w = m.w.get(name + ".weight"); n, k = (w.shape if w is not None else (m.split_k[name][0].shape[0], 16384))
    return timed("lin " + cls(name), lambda: _lin(name, x), [x], 2.0 * (x.size // x.shape[-1]) * k * n)
_sdpa = mx.fast.scaled_dot_product_attention
def sdpa(q, k, v, **kw):
    return timed(f"sdpa q{q.shape[2]} k{k.shape[2]} d{q.shape[3]}", lambda: _sdpa(q, k, v, **kw), [q, k, v], 4.0 * q.shape[0] * q.shape[1] * q.shape[2] * k.shape[2] * q.shape[3])
for label, (f, h, w, b) in (("stage1 1x1536", (16, 8, 12, 1)), ("dev batch 4x1536", (16, 8, 12, 4))):
    args, kf = inputs(f, h, w, b)
    m.lin, mx.fast.scaled_dot_product_attention = _lin, _sdpa
    v, a = m(*args, video_keyframes_mask=kf); mx.eval(v, a)
    mx.synchronize(); t = time.perf_counter(); v, a = m(*args, video_keyframes_mask=kf); mx.eval(v, a); mx.synchronize(); clean = time.perf_counter() - t
    acc.clear(); m.lin, mx.fast.scaled_dot_product_attention = lin, sdpa
    mx.synchronize(); t = time.perf_counter(); v, a = m(*args, video_keyframes_mask=kf); mx.eval(v, a); mx.synchronize(); prof = time.perf_counter() - t
    tracked = sum(r[0] for r in acc.values()); flops = sum(r[2] for r in acc.values())
    print(f"\n## {label}: clean {clean:.2f} s, tracked ops {tracked:.2f} s of profiled {prof:.2f}, nominal {flops/1e12:.1f} TFLOP -> {flops/clean/1e12:.1f} TF/s effective, {flops/20.8e12:.2f} s at peak")
    print("| op | calls | s | share of clean | TF/s |\n| --- | ---: | ---: | ---: | ---: |")
    for k, r in sorted(acc.items(), key=lambda kv: -kv[1][0])[:14]:
        print(f"| {k} | {r[1]} | {r[0]:.3f} | {100*r[0]/clean:.1f}% | {r[2]/r[0]/1e12:.1f} |")
