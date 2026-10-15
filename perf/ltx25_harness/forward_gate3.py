"""Identical-input per-forward gate. Mode 'capture': run the pipeline, capture the first stage-2 forward's
inputs to a pickle. Mode 'replay': load the DiT only, load the pickled inputs, run under the given config.
Usage: LTX_DIT_DTYPE=<entry> python forward_gate3.py MODEL capture|replay WEIGHT_DTYPE CAST(0|1) OUT.npz"""
import os, sys, time, gc, pickle, numpy as np, mlx.core as mx
from mlx.utils import tree_map
import ltx_core_mlx.model.transformer.model as dit_mod
from ltx_core_mlx.model.transformer.model import LTXModel
from ltx_pipelines_mlx.distilled import DistilledPipeline
MODEL, MODE, WD, CAST, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1", sys.argv[5]
TOKENS = 6144; INP = "out/stage2_inputs.pkl"
wdt = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}[WD]
if CAST:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import operand_cast; operand_cast.install(wdt)
def to_np(o):
    if isinstance(o, mx.array): return ("__mx__", np.array(o.astype(mx.float32)) if o.dtype in (mx.bfloat16, mx.float16) else np.array(o), str(o.dtype))
    if isinstance(o, tuple): return tuple(to_np(x) for x in o)
    if isinstance(o, list): return [to_np(x) for x in o]
    if isinstance(o, dict): return {k: to_np(v) for k, v in o.items()}
    return o
DT = {"mlx.core.bfloat16": mx.bfloat16, "mlx.core.float16": mx.float16, "mlx.core.float32": mx.float32, "mlx.core.int32": mx.int32, "mlx.core.int64": mx.int64, "mlx.core.bool_": mx.bool_, "mlx.core.uint32": mx.uint32}
def to_mx(o):
    if isinstance(o, tuple) and len(o) == 3 and o[0] == "__mx__": return mx.array(o[1]).astype(DT[o[2]])
    if isinstance(o, tuple): return tuple(to_mx(x) for x in o)
    if isinstance(o, list): return [to_mx(x) for x in o]
    if isinstance(o, dict): return {k: to_mx(v) for k, v in o.items()}
    return o
pipe = DistilledPipeline(MODEL, low_memory=True, low_ram_streaming=False)
if MODE == "capture":
    class Captured(Exception): pass
    cap = {}; _orig = LTXModel.__call__
    def _capture(self, *a, **k):
        vl = k.get("video_latent", a[0] if a else None)
        if vl is not None and vl.shape[1] == TOKENS and "kw" not in cap:
            mx.eval(*[x for x in list(a)+list(k.values()) if isinstance(x, mx.array)]); cap.update(args=a, kw=k, model=self); raise Captured()
        return _orig(self, *a, **k)
    LTXModel.__call__ = _capture; mx.random.seed(42)
    try: pipe.generate_two_stage("A red fox trotting through a snowy pine forest at dawn, soft golden light, gentle camera dolly forward, birds chirping", height=512, width=768, num_frames=121, frame_rate=24.0, seed=42)
    except Captured: pass
    LTXModel.__call__ = _orig
    model, a, k = cap["model"], cap["args"], cap["kw"]
    bad = [kk for kk, v in k.items() if v is not None and not isinstance(v, (mx.array, tuple, list, dict, str, int, float, bool))]
    print("[gate3] non-array kwargs:", bad, flush=True)
    pickle.dump({"args": to_np(a), "kw": to_np({kk: v for kk, v in k.items() if kk not in bad})}, open(INP, "wb")); print("[gate3] inputs saved", flush=True)
else:
    pipe.load(); model = pipe.dit
    d = pickle.load(open(INP, "rb")); a, k = to_mx(d["args"]), to_mx(d["kw"])
model.update(tree_map(lambda p: p.astype(wdt) if p.dtype in (mx.bfloat16, mx.float16) else p, model.parameters()))
mx.eval(model.parameters()); gc.collect(); mx.clear_cache()
v, au = model(*a, **k); mx.eval(v, au); mx.synchronize()
times = []
for _ in range(3):
    mx.synchronize(); t = time.perf_counter(); v, au = model(*a, **k); mx.eval(v, au); mx.synchronize(); times.append(time.perf_counter() - t)
np.savez(OUT, video=np.array(v.astype(mx.float32)), audio=np.array(au.astype(mx.float32)))
print(f"[gate3] entry={os.environ.get('LTX_DIT_DTYPE')} weights={WD} operand_cast={CAST}: median {np.median(times):.2f} s/forward runs {['%.2f'%x for x in times]} peak={mx.get_peak_memory()/2**30:.1f} GiB", flush=True)
