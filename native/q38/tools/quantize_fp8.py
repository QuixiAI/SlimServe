#!/usr/bin/env python3
"""Round selected projections of a BF16 q38 pack to FP8 (E4M3, one fp32 scale
per output row: the pack's `fp8ch` format) and write them as a SIDECAR pack
holding only those projections, for mixpack.py:

  quantize_fp8.py --src bf16.q38w --out down-fp8.q38w --only "L*.down;lm_head"
  mixpack.py --out mixed.q38w --mix "L*.down=down-fp8.q38w;lm_head=down-fp8.q38w;*=gptq.q38w"

Plain round-to-nearest with the row scale max|w| / 448. On held-out
activations this is already about 0.0004 relative output error (GPTQ NVFP4:
0.001 to 0.003), so no calibration is spent on the 8-bit classes. The sidecar
cannot be the last (base) rule of a mix: it has no norms or embeddings.

CPU only.
"""
import argparse
import fnmatch
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_hf import Writer  # noqa: E402
from q38ref import E4M3, Pack  # noqa: E402

F = np.float32
POS = E4M3[np.arange(0, 0x7F, dtype=np.uint8)].astype(F)  # increasing, code == index
EDGES = (POS[1:] + POS[:-1]) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="BF16 pack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", required=True, help="';'-separated globs of projections to write")
    ap.add_argument("--rows", type=int, default=1024, help="rows per work chunk")
    a = ap.parse_args()
    src = Pack(os.path.expanduser(a.src))
    pats = a.only.split(";")
    wr = Writer(os.path.expanduser(a.out))
    projs, t0 = {}, time.time()
    for name, p in src.meta["projections"].items():
        if not any(fnmatch.fnmatchcase(name, pat) for pat in pats):
            continue
        N, K = p["N"], p["K"]
        codes = np.empty((N, K), dtype=np.uint8)
        scale = np.empty((N,), dtype=F)
        num = den = 0.0
        for r in range(0, N, a.rows):
            w = src.proj_rows(name, r, min(N, r + a.rows))
            s = np.maximum(np.abs(w).max(axis=1), 1e-30) / 448.0
            aw = np.abs(w) / s[:, None]
            idx = np.searchsorted(EDGES, aw).astype(np.uint8)
            codes[r : r + w.shape[0]] = idx | ((w < 0).astype(np.uint8) << 7)
            scale[r : r + w.shape[0]] = s
            q = np.where(w < 0, -POS[idx], POS[idx]) * s[:, None]
            num += float(((q - w).astype(np.float64) ** 2).sum())
            den += float((w.astype(np.float64) ** 2).sum())
        wr.add(name + ".w", "u8", [N, K], [codes])
        wr.add(name + ".scale", "f32", [N], [scale])
        projs[name] = {"fmt": "fp8ch", "N": N, "K": K, "rel_err_w": num / den}
        print(f"{name:>14}  N {N:6d} K {K:5d}  weight err {num / den:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    wr.finish({"projections": projs, "source": f"{src.meta.get('source')} -> fp8ch rtn sidecar ({a.only})"})
    print(f"wrote {a.out}: {len(projs)} projections")


if __name__ == "__main__":
    main()
