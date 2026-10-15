"""Scan bf16 safetensors for fp16-range risk. Usage: python weight_scan.py FILE [FILE...]"""
import sys, numpy as np
from safetensors import safe_open
FP16_MAX, FP16_MIN_NORMAL = 65504.0, 6.1035e-5
for path in sys.argv[1:]:
    n = over = sub = 0; worst = (0.0, ""); tiny = (1e9, "")
    with safe_open(path, "np") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if t.dtype != np.dtype("bfloat16") and str(t.dtype) not in ("bfloat16",):
                try: t = t.astype(np.float32)
                except Exception: continue
            else: t = t.astype(np.float32)
            a = np.abs(t); m = float(a.max()) if a.size else 0.0
            n += 1; over += int(m > FP16_MAX)
            nz = a[a > 0]
            frac_sub = float((nz < FP16_MIN_NORMAL).mean()) if nz.size else 0.0
            sub += int(frac_sub > 0.01)
            if m > worst[0]: worst = (m, k)
            if nz.size and float(nz.min()) < tiny[0]: tiny = (float(nz.min()), k)
    print(f"{path}\n  tensors={n} over_fp16_max={over} tensors_with_>1%_subnormal={sub}\n  max|w|={worst[0]:.3g} ({worst[1]})\n  min nonzero |w|={tiny[0]:.3g} ({tiny[1]})")
