"""N0/N1 gate: load the official DiT with the engine loader, run the engine forward on the captured
stage-2 inputs (gate/out/stage2_inputs.pkl, written by forward_gate3.py capture), compare against the
fp32 truth and the runner's fp16-operand shim.
Usage: PYTHONPATH=<worktree> python n1_forward_gate.py [distilled|dev] [OUT.npz]"""
import os, sys, time, pickle, numpy as np, mlx.core as mx
from slimserve.video.ltx25 import checkpoints
from slimserve.video.ltx25.dit import LTX25DiT, DiTConfig
from paths import SCRATCH
variant = sys.argv[1] if len(sys.argv) > 1 else "distilled"
out = sys.argv[2] if len(sys.argv) > 2 else None
GATE = f"{SCRATCH}/gate/out"
t0 = time.perf_counter()
weights, connectors, tcfg = checkpoints.load_dit(variant)
mx.synchronize(); t_load = time.perf_counter() - t0
dts = {}
for k, a in weights.items(): dts[str(a.dtype)] = dts.get(str(a.dtype), 0) + a.nbytes
print(f"[n1] load {t_load:.1f} s  dit tensors={len(weights)} connector tensors={len(connectors)} "
      f"bytes by dtype={ {k: round(v/2**30, 2) for k, v in dts.items()} } GiB  active={mx.get_active_memory()/2**30:.1f} peak={mx.get_peak_memory()/2**30:.1f} GiB", flush=True)
model = LTX25DiT(weights, DiTConfig.from_checkpoint(tcfg))
d = pickle.load(open(f"{GATE}/stage2_inputs.pkl", "rb"))["kw"]
g = lambda k: None if d.get(k) is None else mx.array(d[k][1])
args = dict(video_latent=g("video_latent"), audio_latent=g("audio_latent"), timestep=g("timestep"),
            video_text=g("video_text_embeds"), audio_text=g("audio_text_embeds"),
            video_positions=g("video_positions"), audio_positions=g("audio_positions"),
            video_keyframes_mask=g("video_keyframes_mask"))
def run():
    mx.synchronize(); t = time.perf_counter(); v, a = model(**args); mx.eval(v, a); mx.synchronize()
    return time.perf_counter() - t, np.array(v), np.array(a)
_, v0, a0 = run(); mx.reset_peak_memory()
times = []
for _ in range(3):
    t, v, a = run(); times.append(t)
print(f"[n1] forward median {np.median(times):.2f} s runs {['%.2f' % x for x in times]} "
      f"deterministic={bool((v == v0).all() and (a == a0).all())} peak={mx.get_peak_memory()/2**30:.1f} GiB", flush=True)
def cmp(name, ref):
    r = np.load(f"{GATE}/{ref}")
    for key, x in (("video", v), ("audio", a)):
        y = r[key].astype(np.float64); x64 = x.astype(np.float64)
        rel = np.linalg.norm(x64 - y) / np.linalg.norm(y)
        cos = (x64 * y).sum() / (np.linalg.norm(x64) * np.linalg.norm(y))
        print(f"[n1] vs {name:28s} {key}: rel-L2 {rel:.5f} cos {cos:.6f}", flush=True)
cmp("fp32 truth", "g3_truth_fp32.npz")
cmp("runner fp16-operand shim", "g3_fp16ops_fp32glue.npz")
cmp("runner as-is (bf16 entry)", "g3_runner_asis_bf16.npz")
if out: np.savez(out, video=v, audio=a)
