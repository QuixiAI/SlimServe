"""Clean per-forward timing for one dtype (one process per dtype, no retained copies).
Usage: LTX_DIT_DTYPE=<dt> python forward_speed.py MODEL_DIR TOKENS"""
import os, sys, time, gc
import numpy as np, mlx.core as mx
from mlx.utils import tree_map
import ltx_core_mlx.model.transformer.model as dit_mod
from ltx_core_mlx.model.transformer.model import LTXModel
from ltx_pipelines_mlx.distilled import DistilledPipeline
MODEL, TOKENS = sys.argv[1], int(sys.argv[2])
dt = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[os.environ["LTX_DIT_DTYPE"]]
PROMPT = "A red fox trotting through a snowy pine forest at dawn, soft golden light, gentle camera dolly forward, birds chirping"
class Captured(Exception): pass
cap = {}; _orig = LTXModel.__call__
def _capture(self, *a, **k):
    vl = k.get("video_latent", a[0] if a else None)
    if vl is not None and vl.shape[1] == TOKENS and "kw" not in cap:
        mx.eval(*[x for x in list(a)+list(k.values()) if isinstance(x, mx.array)]); cap.update(args=a, kw=k, model=self); raise Captured()
    return _orig(self, *a, **k)
LTXModel.__call__ = _capture
pipe = DistilledPipeline(MODEL, low_memory=True, low_ram_streaming=False); mx.random.seed(42)
try: pipe.generate_two_stage(PROMPT, height=512, width=768, num_frames=121, frame_rate=24.0, seed=42)
except Captured: pass
LTXModel.__call__ = _orig
model, a, k = cap["model"], cap["args"], cap["kw"]; cap.clear()
def cast_tree(o):
    if isinstance(o, mx.array): return o.astype(dt) if o.dtype in (mx.bfloat16, mx.float16) else o
    if isinstance(o, tuple): return tuple(cast_tree(x) for x in o)
    if isinstance(o, list): return [cast_tree(x) for x in o]
    if isinstance(o, dict): return {kk: cast_tree(v) for kk, v in o.items()}
    return o
model.update(tree_map(lambda p: p.astype(dt) if p.dtype in (mx.bfloat16, mx.float16) else p, model.parameters()))
mx.eval(model.parameters()); gc.collect(); mx.clear_cache()
a, k = cast_tree(a), cast_tree(k); dit_mod._DIT_DTYPE = dt
v, au = model(*a, **k); mx.eval(v, au); mx.synchronize()
times = []
for _ in range(4):
    mx.synchronize(); t = time.perf_counter(); v, au = model(*a, **k); mx.eval(v, au); mx.synchronize(); times.append(time.perf_counter() - t)
print(f"[speed] {os.environ['LTX_DIT_DTYPE']} @ {TOKENS} tok: median {np.median(times):.2f} s/forward  runs {['%.2f'%x for x in times]}  active_mem={mx.get_active_memory()/2**30:.1f} GiB peak={mx.get_peak_memory()/2**30:.1f} GiB", flush=True)
