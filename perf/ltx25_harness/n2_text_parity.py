"""N2 text-path parity. One process per mode (never two Gemma copies in one process).
  ref  MODEL_DIR OUT.npz            runner as shipped (bf16 pack) on the prompts
  ours OPERAND STREAM OUT.npz       engine; OPERAND/STREAM in fp16|bf16|fp32
  cmp  A.npz B.npz                  rel-L2 / cosine of A against B
Usage: PYTHONPATH=<worktree> python n2_text_parity.py <mode> ..."""
import sys, time, numpy as np, mlx.core as mx
PROMPTS = {
    "fox": "A red fox trotting through a snowy pine forest at dawn, soft golden light, gentle camera dolly forward, birds chirping",
    "long": ("A weathered fisherman in a yellow oilskin coat stands at the bow of a small wooden boat as it cuts through "
             "choppy grey water under a heavy overcast sky. He hauls a dripping net over the gunwale hand over hand, "
             "silver fish thrashing in the mesh, while gulls wheel and cry overhead. The camera starts wide on the boat "
             "and slowly pushes in to a close-up of his hands, rope burns and salt on the knuckles, then tilts up to his "
             "face as he squints toward a lighthouse on the far headland. Sound of waves slapping the hull, the creak of "
             "wet rope, a distant foghorn, and the man muttering to himself in a low gravelly voice: \"Not a bad morning "
             "after all.\" Cinematic, shallow depth of field, cold blue-green palette with the coat as the only warm colour."),
}
mode = sys.argv[1]
if mode == "ref":
    from ltx_pipelines_mlx.distilled import DistilledPipeline
    pipe = DistilledPipeline(sys.argv[2], low_memory=True, low_ram_streaming=False)
    out = {}
    for name, p in PROMPTS.items():
        t = time.perf_counter(); v, a = pipe._encode_text(p); mx.eval(v, a); dt = time.perf_counter() - t
        ids, mask = pipe.text_encoder.tokenize(p)
        out[f"{name}_video"], out[f"{name}_audio"] = np.array(v.astype(mx.float32)), np.array(a.astype(mx.float32))
        out[f"{name}_ids"] = np.array(ids)[0][np.array(mask)[0] > 0]
        print(f"[text-ref] {name}: tokens={int(np.array(mask).sum())} {dt:.1f} s shapes {v.shape} {a.shape} dtype {v.dtype}", flush=True)
    np.savez(sys.argv[3], **out)
elif mode == "ours":
    from slimserve.video.ltx25.text import TextEncoder
    DT = {"fp16": mx.float16, "bf16": mx.bfloat16, "fp32": mx.float32}
    enc = TextEncoder(operand=DT[sys.argv[2]], stream=DT[sys.argv[3]])
    t = time.perf_counter(); enc.load(); mx.synchronize(); print(f"[text] load {time.perf_counter()-t:.1f} s active={mx.get_active_memory()/2**30:.1f} GiB", flush=True)
    out = {}
    for name, p in PROMPTS.items():
        enc.max_abs = []
        t = time.perf_counter(); v, a = enc.encode(p); mx.eval(v, a); dt = time.perf_counter() - t
        peaks = sorted(((float(m), n) for n, m in enc.max_abs), reverse=True)[:4]
        out[f"{name}_video"], out[f"{name}_audio"] = np.array(v), np.array(a)
        out[f"{name}_ids"] = np.array(enc.tokenize(p))
        print(f"[text] {name}: tokens={len(enc.tokenize(p))} {dt:.2f} s nan={bool(np.isnan(out[name+'_video']).any())} largest GEMM outputs {peaks}", flush=True)
    enc.max_abs = None
    t = time.perf_counter(); enc.encode(PROMPTS["fox"]); print(f"[text] warm fox encode {time.perf_counter()-t:.2f} s peak={mx.get_peak_memory()/2**30:.1f} GiB", flush=True)
    np.savez(sys.argv[4], **out)
else:
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    for k in a.files:
        if k.endswith("_ids"):
            print(f"[cmp] {k}: identical={a[k].shape == b[k].shape and bool((a[k] == b[k]).all())} n={len(a[k])}"); continue
        x, y = a[k].astype(np.float64), b[k].astype(np.float64)
        print(f"[cmp] {k}: rel-L2 {np.linalg.norm(x-y)/np.linalg.norm(y):.5f} cos {(x*y).sum()/(np.linalg.norm(x)*np.linalg.norm(y)):.6f}")
