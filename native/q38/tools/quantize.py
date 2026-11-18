#!/usr/bin/env python3
"""Quantize a BF16 q38 pack's projections to NVFP4 (group-16 E2M1 nibbles,
E4M3 group scales, one fp32 global multiplier per projection).

  quantize.py --src bf16.q38w --out all-nvfp4.q38w [--jobs 16]

Every projection except the tiny dense `ba` heads is quantized, so the output
is the "everything at 4.5 bits" corner; mixed recipes are composed afterwards
from several packs (see mixpack.py) and judged by kld.py.

Per group of 16 weights the scale is searched over the E4M3 codes around
max|w|/6 for the smallest squared error after rounding to the E2M1 grid, which
matters because E4M3 scales only have 3 mantissa bits. With --imatrix the
error is weighted per input column by the calibration second moment E[x^2]
(the activation-aware objective llama.cpp's imatrix and AWQ-style methods
use), so directions the model actually excites are rounded more carefully.

CPU only.
"""
import argparse
import fnmatch
import json
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_hf import PAGE, Writer  # noqa: E402
from q38ref import E4M3, Pack  # noqa: E402

F = np.float32
E2M1_MAG = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=F)
E2M1_EDGES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=F)
# Positive E4M3 codes in increasing value order (0x00..0x7e; 0x7f is NaN).
POS_CODES = np.arange(0, 0x7F, dtype=np.uint8)
POS_VALS = E4M3[POS_CODES]
ROWS_PER_TASK = 1024
SEARCH = (-2, -1, 0, 1, 2)

_src = None


def _init(path):
    global _src
    _src = Pack(path)


def quant_rows(args):
    """Quantize rows [r0, r1) of projection `name`. Returns (packed, scales)."""
    name, r0, r1, g, colw = args
    w = _src.proj_rows(name, r0, r1)
    rows, K = w.shape
    G = K // 16
    wg = w.reshape(rows, G, 16)
    a = np.abs(wg)
    target = a.max(-1) / 6.0 / g  # ideal group scale in global units
    base = np.clip(np.searchsorted(POS_VALS, target), 1, len(POS_VALS) - 1)
    cw = None if colw is None else colw.reshape(1, G, 16)
    best_err = np.full((rows, G), np.inf, dtype=F)
    best_code = np.zeros((rows, G), dtype=np.uint8)
    best_idx = np.zeros((rows, G, 16), dtype=np.uint8)
    for d in SEARCH:
        ci = np.clip(base + d, 1, len(POS_VALS) - 1)
        s = (POS_VALS[ci] * g)[:, :, None]
        idx = np.searchsorted(E2M1_EDGES, a / s).astype(np.uint8)
        diff = a - E2M1_MAG[idx] * s
        diff *= diff
        if cw is not None:
            diff *= cw
        err = diff.sum(-1)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_code = np.where(better, POS_CODES[ci], best_code)
        best_idx = np.where(better[:, :, None], idx, best_idx)
    nib = (best_idx | ((wg < 0).astype(np.uint8) << 3)).reshape(rows, K)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    return packed, best_code, float(best_err.sum()), float((w.astype(np.float64) ** 2 * (1 if colw is None else colw)).sum())


def global_scale(src, name, N):
    """Global multiplier so the largest group scale lands at the top of E4M3."""
    mx = 0.0
    for r in range(0, N, 4096):
        mx = max(mx, float(np.abs(src.proj_rows(name, r, min(N, r + 4096))).max()))
    return np.float32(mx / 6.0 / 448.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--imatrix", help=".npz of per-projection input second moments (kld.py --imatrix-out)")
    ap.add_argument("--only", help="substring filter on projection names (smoke)")
    ap.add_argument("--keep", default="*.ba", help="';'-separated globs copied through unquantized")
    a = ap.parse_args()

    src = Pack(a.src)
    imat = np.load(a.imatrix) if a.imatrix else None
    wr = Writer(a.out)
    projs = {}
    t0 = time.time()
    names_done = set()
    with ProcessPoolExecutor(a.jobs, initializer=_init, initargs=(a.src,)) as pool:
        for name, p in src.meta["projections"].items():
            if a.only and a.only not in name:
                continue
            N, K = p["N"], p["K"]
            if any(fnmatch.fnmatchcase(name, pat) for pat in a.keep.split(";")) or p["fmt"] != "bf16":
                # copy through untouched
                for suffix in (".w", ".scale", ".gscale"):
                    if name + suffix in src.t:
                        e = src.t[name + suffix]
                        wr.add(name + suffix, e["dtype"], e["shape"],
                               [src.mm[src.base + e["offset"] : src.base + e["offset"] + e["nbytes"]]])
                projs[name] = p
                continue
            g = global_scale(src, name, N)
            colw = None
            if imat is not None and name in imat:
                colw = imat[name].astype(F)
                colw = colw / colw.mean()
            tasks = [(name, r, min(N, r + ROWS_PER_TASK), g, colw) for r in range(0, N, ROWS_PER_TASK)]
            packed, scales, err, tot = [], [], 0.0, 0.0
            for pk, sc, e_, t_ in pool.map(quant_rows, tasks):
                packed.append(pk)
                scales.append(sc)
                err += e_
                tot += t_
            wr.add(name + ".w", "u8", [N, K // 2], packed)
            wr.add(name + ".scale", "u8", [N, K // 16], scales)
            wr.add(name + ".gscale", "f32", [1], [np.array([g], dtype=F)])
            projs[name] = {"fmt": "nvfp4", "N": N, "K": K, "rel_err": err / max(tot, 1e-30)}
            names_done.add(name)
            if len(names_done) % 16 == 0 or name == "lm_head":
                print(f"{len(names_done)} projections  {name}  rel err {err / max(tot, 1e-30):.5f}  "
                      f"{time.time() - t0:.0f}s", flush=True)
        # everything that is not a projection operand copies through
        proj_tensors = {n + s for n in src.meta["projections"] for s in (".w", ".scale", ".gscale")}
        for tname, e in src.t.items():
            if tname in proj_tensors:
                continue
            if a.only:
                continue
            wr.add(tname, e["dtype"], e["shape"],
                   [src.mm[src.base + e["offset"] : src.base + e["offset"] + e["nbytes"]]])
    meta = dict(src.meta)
    meta["projections"] = projs
    meta["source"] = f"{src.meta.get('source')} -> q38 nvfp4" + (" (imatrix)" if imat is not None else "")
    meta["decode_read_bytes"] = sum(
        wr.tensors[n + ".w"]["nbytes"] + wr.tensors.get(n + ".scale", {"nbytes": 0})["nbytes"] for n in projs
    )
    wr.finish(meta)
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 1e9:.2f} GB, read per token "
          f"{meta['decode_read_bytes'] / 1e9:.2f} GB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
