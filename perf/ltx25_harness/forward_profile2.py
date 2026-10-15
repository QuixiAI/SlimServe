"""Per-op-class profile of one real stage-2 DiT forward, in bf16 and fp16.
Usage: python forward_profile.py MODEL_DIR TOKENS"""
import os, sys, time
import numpy as np, mlx.core as mx
from mlx.utils import tree_map
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_hook2 as profile_hook
if os.environ.get('LTX_OPCAST'):
    import operand_cast; operand_cast.install({'fp16': mx.float16, 'bf16': mx.bfloat16}[os.environ['LTX_OPCAST']])
import ltx_core_mlx.model.transformer.model as dit_mod
from ltx_core_mlx.model.transformer.model import LTXModel
from ltx_pipelines_mlx.distilled import DistilledPipeline
MODEL, TOKENS = sys.argv[1], int(sys.argv[2])
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
model, a, k = cap["model"], cap["args"], cap["kw"]
def cast_tree(o, dt):
    if isinstance(o, mx.array): return o.astype(dt) if o.dtype in (mx.bfloat16, mx.float16) else o
    if isinstance(o, tuple): return tuple(cast_tree(x, dt) for x in o)
    if isinstance(o, list): return [cast_tree(x, dt) for x in o]
    if isinstance(o, dict): return {kk: cast_tree(v, dt) for kk, v in o.items()}
    return o
orig_params = model.parameters()
profile_hook.install()
for name, dt in ((os.environ.get("LTX_OPCAST","bf16"), {"fp16": mx.float16, "bf16": mx.bfloat16}[os.environ.get("LTX_OPCAST","bf16")]),):
    model.update(tree_map(lambda p: p.astype(dt) if p.dtype in (mx.bfloat16, mx.float16) else p, orig_params)); mx.eval(model.parameters())
    dit_mod._DIT_DTYPE = dt; ca, ck = cast_tree(a, dt), cast_tree(k, dt)
    v, au = model(*ca, **ck); mx.eval(v, au); mx.synchronize()
    profile_hook.acc.clear()
    mx.synchronize(); t = time.perf_counter(); v, au = model(*ca, **ck); mx.eval(v, au); mx.synchronize(); tot = time.perf_counter() - t
    print(f"\n[profile] {name} forward @ {TOKENS} tokens: {tot:.2f} s total (eval-synchronized, inflated)", flush=True)
    profile_hook.report(tot)
