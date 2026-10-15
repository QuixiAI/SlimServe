"""N5: runtime distilled LoRA on the dev DiT vs the distilled checkpoint, same captured stage-2 inputs.
Also checks runtime low-rank == fused weights on a few linears, and attach/detach restores the base exactly."""
import os, pickle, time, numpy as np, mlx.core as mx
from slimserve.video.ltx25.pipeline import LTX25Engine
from paths import SCRATCH
eng = LTX25Engine(variant="dev"); dit = eng.load_dit(); lora = eng.load_distilled_lora()
d = pickle.load(open(f"{SCRATCH}/gate/out/stage2_inputs.pkl", "rb"))["kw"]
g = lambda k: None if d.get(k) is None else mx.array(d[k][1])
args = dict(video_latent=g("video_latent"), audio_latent=g("audio_latent"), timestep=g("timestep"), video_text=g("video_text_embeds"),
            audio_text=g("audio_text_embeds"), video_positions=g("video_positions"), audio_positions=g("audio_positions"), video_keyframes_mask=g("video_keyframes_mask"))
def run():
    mx.synchronize(); t = time.perf_counter(); v, a = dit(**args); mx.eval(v, a); mx.synchronize(); return time.perf_counter() - t, np.array(v), np.array(a)
ref = np.load(f"{SCRATCH}/n1/n1_splitk.npz")
def cmp(name, x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    print(f"[lora] {name:44s} rel-L2 {np.linalg.norm(x-y)/np.linalg.norm(y):.4f} cos {(x*y).sum()/np.linalg.norm(x)/np.linalg.norm(y):.6f}", flush=True)
t0, v_dev, a_dev = run(); t0, v_dev, a_dev = run()
lora.attach(dit); t1, v_l, a_l = run(); t1, v_l, a_l = run()
print(f"[lora] dev forward {t0:.2f} s; dev + runtime LoRA {t1:.2f} s; adapter pairs {len(lora.pairs)}")
cmp("dev (no LoRA) video vs distilled", v_dev, ref["video"]); cmp("dev + LoRA video vs distilled", v_l, ref["video"])
cmp("dev (no LoRA) audio vs distilled", a_dev, ref["audio"]); cmp("dev + LoRA audio vs distilled", a_l, ref["audio"])
# runtime low-rank vs fused weight on single linears (fp32 reference)
for name in ("transformer_blocks.10.attn1.to_q", "transformer_blocks.10.ff.net.0.proj", "transformer_blocks.10.attn2.to_out.0"):
    a, b = lora.pairs[name]; w = dit.w[name + ".weight"]; bias = dit.w.get(name + ".bias")
    x = (mx.random.normal((512, w.shape[1])) * 0.5).astype(mx.float16)
    y = dit.lin(name, x).astype(mx.float32)
    wf = w.astype(mx.float32) + b.astype(mx.float32) @ a.astype(mx.float32)
    yf = x.astype(mx.float32) @ wf.T + (0 if bias is None else bias.astype(mx.float32))
    print(f"[lora] {name}: runtime vs fused fp32 rel {float(mx.linalg.norm(y - yf) / mx.linalg.norm(yf)):.2e}; |BA|/|W| {float(mx.linalg.norm(wf - w.astype(mx.float32)) / mx.linalg.norm(w.astype(mx.float32))):.3f}")
lora.detach(dit); _, v_b, _ = run(); print("[lora] detach restores base exactly:", bool((v_b == v_dev).all()), "lora dict empty:", not dit.lora)
