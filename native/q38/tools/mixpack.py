#!/usr/bin/env python3
"""Compose a q38 pack from several packs, one source per projection.

  mixpack.py --out mixed.q38w --mix "L*.qkvz=fp8.q38w;lm_head=fp8.q38w;*=nv.q38w"

Rules are glob=pack, first match wins (same syntax as kld.py eval --mix).
Non-projection tensors come from the last rule's pack. The result is what
the engine loads, so a recipe can be scored on the GPU in seconds and
benchmarked for speed at the same time.
"""
import argparse
import fnmatch
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_hf import Writer  # noqa: E402
from q38ref import Pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mix", required=True)
    a = ap.parse_args()
    rules, packs = [], {}
    for part in a.mix.split(";"):
        pat, path = part.split("=", 1)
        path = os.path.expanduser(path)
        packs.setdefault(path, Pack(path))
        rules.append((pat, packs[path]))
    base = rules[-1][1]
    wr = Writer(os.path.expanduser(a.out))
    t0 = time.time()

    def put(pk, name):
        e = pk.t[name]
        wr.add(name, e["dtype"], e["shape"], [pk.mm[pk.base + e["offset"] : pk.base + e["offset"] + e["nbytes"]]])

    projs, counts = {}, {}
    for name in base.meta["projections"]:
        pk = next(p for pat, p in rules if fnmatch.fnmatchcase(name, pat))
        for suffix in (".w", ".b", ".scale", ".gscale"):
            if name + suffix in pk.t:
                put(pk, name + suffix)
        projs[name] = pk.meta["projections"][name]
        counts[projs[name]["fmt"]] = counts.get(projs[name]["fmt"], 0) + 1
    ptens = {n + s for n in base.meta["projections"] for s in (".w", ".b", ".scale", ".gscale")}
    for name in base.t:
        if name not in ptens:
            put(base, name)
    meta = dict(base.meta)
    meta["projections"] = projs
    meta["source"] = "mix: " + a.mix
    meta["decode_read_bytes"] = sum(
        sum(wr.tensors.get(n + sfx, {"nbytes": 0})["nbytes"] for sfx in (".w", ".b", ".scale")) for n in projs)
    wr.finish(meta)
    print(f"wrote {a.out}: read per token {meta['decode_read_bytes'] / 1e9:.2f} GB, formats {counts}, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
