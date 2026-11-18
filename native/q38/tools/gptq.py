#!/usr/bin/env python3
"""GPTQ quantization of a BF16 q38 pack to NVFP4, one calibrated forward pass.

  gptq.py --src bf16.q38w --set calib.json --out gptq-nvfp4.q38w

The CPU reference forward runs the calibration sequences through the model
layer by layer. At every projection the hook below sees the projection's real
input X for all calibration tokens (already produced by the quantized layers
upstream), builds the Hessian H = X^T X, and rounds the weights column by
column with optimal-brain-surgeon error feedback (GPTQ, Frantar et al. 2022):
the rounding error of each column is pushed onto the not-yet-rounded columns
so the projection's OUTPUT error is minimized, not the weight error. The
quantized projection then produces the activations the next layer calibrates
on.

NVFP4 specifics: weights are E2M1 nibbles times an E4M3 scale per group of 16
columns times one fp32 global multiplier. The group scale is searched (5 E4M3
codes around max/6) when the walk reaches the group, on the error-compensated
weights, then the 16 columns are rounded with feedback.

CPU only (torch for the linear algebra, on the CPU). The dense `ba` gate heads
stay BF16.
"""
import argparse
import fnmatch
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_hf import Writer  # noqa: E402
from q38ref import E4M3, Pack, forward  # noqa: E402

F = np.float32
MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
EDGES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
POS_VALS = torch.from_numpy(E4M3[np.arange(0, 0x7F, dtype=np.uint8)].copy())  # increasing
SEARCH = (-2, -1, 0, 1, 2)


def hessian_factor(H, damp):
    """Upper-triangular U with U^T U = H^-1, from the Cholesky factor of the
    index-reversed H (avoids forming H^-1)."""
    K = H.shape[0]
    d = torch.diagonal(H)
    dead = d <= 0
    d[dead] = 1.0
    d += damp * float(d.mean())
    L = torch.linalg.cholesky(torch.flip(H, (0, 1)))
    Li = torch.linalg.solve_triangular(L, torch.eye(K), upper=False)
    return torch.flip(Li, (0, 1)).contiguous(), dead


def gptq_nvfp4(W, U, dead, block=128):
    """W [N, K] float32 (modified in place), U upper factor. Returns the
    dequantized weights, nibble indices [N, K] uint8, scale codes [N, K/16]
    uint8 and the global multiplier."""
    N, K = W.shape
    W[:, dead] = 0
    g = float(W.abs().max()) / 6.0 / 448.0
    g = max(g, 1e-30)
    Q = torch.empty_like(W)
    nib = torch.empty((N, K), dtype=torch.uint8)
    codes = torch.empty((N, K // 16), dtype=torch.uint8)
    for b0 in range(0, K, block):
        b1 = min(K, b0 + block)
        Wb = W[:, b0:b1]  # view; updated in place inside the block
        Err = torch.empty((N, b1 - b0))
        for g0 in range(b0, b1, 16):
            wg = Wb[:, g0 - b0 : g0 - b0 + 16]
            a = wg.abs()
            target = a.amax(1) / 6.0 / g
            base = torch.searchsorted(POS_VALS, target).clamp_(1, len(POS_VALS) - 1)
            best_err = torch.full((N,), float("inf"))
            best_ci = base.clone()
            for d in SEARCH:
                ci = (base + d).clamp_(1, len(POS_VALS) - 1)
                s = (POS_VALS[ci] * g)[:, None]
                q = MAG[torch.bucketize(a / s, EDGES)] * s
                err = ((a - q) ** 2).sum(1)
                better = err < best_err
                best_err = torch.where(better, err, best_err)
                best_ci = torch.where(better, ci, best_ci)
            codes[:, g0 // 16] = best_ci.to(torch.uint8)  # POS code == table index
            s = POS_VALS[best_ci] * g
            for i in range(g0, g0 + 16):
                j = i - b0
                w = Wb[:, j]
                idx = torch.bucketize(w.abs() / s, EDGES)
                q = torch.where(w < 0, -MAG[idx], MAG[idx]) * s
                Q[:, i] = q
                nib[:, i] = (idx | ((w < 0).to(torch.int64) << 3)).to(torch.uint8)
                e = (w - q) / U[i, i]
                Err[:, j] = e
                if i + 1 < b1:
                    Wb[:, j + 1 :].addr_(e, U[i, i + 1 : b1], alpha=-1.0)
        if b1 < K:
            W[:, b1:].addmm_(Err, U[b0:b1, b1:], alpha=-1.0)
    return Q, nib, codes, np.float32(g)


UNI_CLIP = (1.0, 0.92, 0.85, 0.78, 0.7, 0.6)  # group scale candidates, as fractions of max|w|


def gptq_uniform(W, U, dead, bits, block=128):
    """Uniform planar formats u4 / u5 (kernels/q38_gemu.metal): `bits`-bit
    level index n per weight, value (n - L/2 + 0.5) * s with L = 2^bits, one
    scale per group of 16 as a u8 code c, s = g * 2^(-c/16). U None = plain
    rounding (no error feedback). Returns dequantized weights, n [N, K] uint8,
    codes [N, K/16] uint8 and g."""
    N, K = W.shape
    L = 1 << bits
    half = L / 2 - 0.5
    if dead is not None:
        W[:, dead] = 0
    g = max(float(W.abs().max()) / half, 1e-30)
    Q = torch.empty_like(W)
    nidx = torch.empty((N, K), dtype=torch.uint8)
    codes = torch.empty((N, K // 16), dtype=torch.uint8)

    def pick_scale(wg):
        mx = wg.abs().amax(1).clamp_min(1e-30)
        best_err = torch.full((N,), float("inf"))
        best_c = torch.zeros((N,))
        for k in UNI_CLIP:
            c = torch.round(-16.0 * torch.log2(mx * k / half / g)).clamp_(0, 255)
            sq = (g * torch.pow(2.0, -c / 16.0))[:, None]
            q = (torch.floor(wg / sq).clamp_(-L / 2, L / 2 - 1) + 0.5) * sq
            err = ((wg - q) ** 2).sum(1)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_c = torch.where(better, c, best_c)
        return best_c

    for b0 in range(0, K, block):
        b1 = min(K, b0 + block)
        Wb = W[:, b0:b1]
        Err = torch.empty((N, b1 - b0)) if U is not None else None
        for g0 in range(b0, b1, 16):
            wg = Wb[:, g0 - b0 : g0 - b0 + 16]
            c = pick_scale(wg)
            codes[:, g0 // 16] = c.to(torch.uint8)
            s = g * torch.pow(2.0, -c / 16.0)
            if U is None:
                n = torch.floor(wg / s[:, None]).clamp_(-L / 2, L / 2 - 1)
                Q[:, g0 : g0 + 16] = (n + 0.5) * s[:, None]
                nidx[:, g0 : g0 + 16] = (n + L / 2).to(torch.uint8)
                continue
            for i in range(g0, g0 + 16):
                j = i - b0
                w = Wb[:, j]
                n = torch.floor(w / s).clamp_(-L / 2, L / 2 - 1)
                q = (n + 0.5) * s
                Q[:, i] = q
                nidx[:, i] = (n + L / 2).to(torch.uint8)
                e = (w - q) / U[i, i]
                Err[:, j] = e
                if i + 1 < b1:
                    Wb[:, j + 1 :].addr_(e, U[i, i + 1 : b1], alpha=-1.0)
        if U is not None and b1 < K:
            W[:, b1:].addmm_(Err, U[b0:b1, b1:], alpha=-1.0)
    return Q, nidx, codes, np.float32(g)


def quantize(W, U, dead, fmt, block):
    """Round W [N, K] (modified in place) to `fmt`. Returns (Q, tensors), where
    tensors maps a pack suffix to (dtype, shape, array)."""
    N, K = W.shape
    if fmt == "nvfp4":
        if U is None:
            U, dead = torch.eye(K), torch.zeros(K, dtype=torch.bool)
        Q, nib, codes, g = gptq_nvfp4(W, U, dead, block)
        nb = nib.numpy()
        packed = (nb[:, 0::2] | (nb[:, 1::2] << 4)).astype(np.uint8)
        return Q, {".w": ("u8", [N, K // 2], packed), ".scale": ("u8", [N, K // 16], codes.numpy()),
                   ".gscale": ("f32", [1], np.array([g], dtype=F))}
    bits = {"u4": 4, "u5": 5}[fmt]
    Q, nidx, codes, g = gptq_uniform(W, U, dead, bits, block)
    n = nidx.numpy()
    n4 = n >> (bits - 4)
    out = {".w": ("u8", [N, K // 2], (n4[:, 0::2] | (n4[:, 1::2] << 4)).astype(np.uint8)),
           ".scale": ("u8", [N, K // 16], codes.numpy()), ".gscale": ("f32", [1], np.array([g], dtype=F))}
    if bits == 5:
        out[".b"] = ("u8", [N, K // 8], np.packbits(n & 1, axis=1, bitorder="little"))
    return Q, out


class GptqPack:
    """Looks like a Pack to the reference forward; quantizes on first use."""

    def __init__(self, src, wr, keep, damp, block, damp_map=(), fmt="nvfp4", fmt_map=(), second=None):
        self.src, self.wr, self.keep, self.damp, self.block = src, wr, keep, damp, block
        self.damp_map = list(damp_map)  # [(glob, damp)], first match wins
        self.fmt, self.fmt_map = fmt, list(fmt_map)
        self.second = second  # (Writer, fmt, projs): the same Hessians rounded to a second format
        self.rtn = False
        self.meta = src.meta
        self.projs = {}
        self.t0 = time.time()

    def raw(self, name, dtype):
        return self.src.raw(name, dtype)

    def f32(self, name):
        return self.src.f32(name)

    def _copy(self, name):
        for suffix in (".w", ".b", ".scale", ".gscale"):
            if name + suffix in self.src.t:
                e = self.src.t[name + suffix]
                blob = self.src.mm[self.src.base + e["offset"] : self.src.base + e["offset"] + e["nbytes"]]
                self.wr.add(name + suffix, e["dtype"], e["shape"], [blob])
                if self.second:
                    self.second[0].add(name + suffix, e["dtype"], e["shape"], [blob])
        self.projs[name] = self.src.meta["projections"][name]
        if self.second:
            self.second[2][name] = self.src.meta["projections"][name]

    def apply(self, name, x):
        p = self.src.meta["projections"][name]
        N, K = p["N"], p["K"]
        if any(fnmatch.fnmatchcase(name, pat) for pat in self.keep) or p["fmt"] != "bf16":
            self._copy(name)
            return None if self.rtn else self.src.apply(name, x)
        t = time.time()
        fmt = next((f for pat, f in self.fmt_map if fnmatch.fnmatchcase(name, pat)), self.fmt)
        X = None if self.rtn else torch.from_numpy(np.ascontiguousarray(x))
        U = dead = None
        if not self.rtn:
            H = (X.T @ X) / X.shape[0]
            damp = next((d for pat, d in self.damp_map if fnmatch.fnmatchcase(name, pat)), self.damp)
            U, dead = hessian_factor(H, damp)
            del H
        W = torch.from_numpy(self.src.proj_rows(name, 0, N).copy())
        W0 = W.clone() if (N * K < 3e8 or self.second) else None
        if self.second:
            wr2, fmt2, projs2 = self.second
            Q2, tens2 = quantize(W0.clone(), U, dead, fmt2, self.block)
            for sfx, (dt, shape, arr) in tens2.items():
                wr2.add(name + sfx, dt, shape, [arr])
            projs2[name] = {"fmt": fmt2, "N": N, "K": K,
                            "rel_err_w": float(((Q2 - W0) ** 2).sum() / (W0 ** 2).sum())}
            del Q2, tens2
        Q, tens = quantize(W, U, dead, fmt, self.block)
        del W, U
        for sfx, (dt, shape, arr) in tens.items():
            self.wr.add(name + sfx, dt, shape, [arr])
        del tens
        rel_w = None
        if W0 is not None:
            rel_w = float(((Q - W0) ** 2).sum() / (W0 ** 2).sum())
        if name == "lm_head" or self.rtn:
            out = np.zeros((1, 1), dtype=F)  # nothing downstream needs it
            rel_o = None
        else:
            out = (X @ Q.T).numpy()
            rel_o = None
            if W0 is not None:
                ref = X[:512] @ W0.T
                rel_o = float(((torch.from_numpy(out[:512]) - ref) ** 2).sum() / (ref ** 2).sum())
        self.projs[name] = {"fmt": fmt, "N": N, "K": K, "rel_err_w": rel_w, "rel_err_out": rel_o}
        print(f"{name:>14} {fmt:>5}  N {N:6d} K {K:5d}  weight err {rel_w if rel_w is None else round(rel_w, 5)}  "
              f"output err {rel_o if rel_o is None else round(rel_o, 5)}  {time.time() - t:.0f}s  "
              f"(total {time.time() - self.t0:.0f}s)", flush=True)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="BF16 pack")
    ap.add_argument("--set", help="calibration sequences (kld.py build-set)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", default="L*.ba", help="';'-separated globs copied through unquantized")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--damp-map", default="",
                    help="';'-separated glob=damp overrides, first match wins. Held-out output error "
                         "(2026-11-09, 19k calibration tokens): the K=17408 down projections overfit badly at "
                         "0.01 (no better than plain rounding) and want 0.5; gate_up and gdn_out want 0.2; "
                         "qkvz is best at 0.01-0.05")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--fmt", default="nvfp4", choices=["nvfp4", "u4", "u5"], help="target format")
    ap.add_argument("--fmt-map", default="", help="';'-separated glob=format overrides, first match wins")
    ap.add_argument("--out2", help="second pack: every projection rounded to --fmt2 against the same "
                                   "Hessians (activations propagate through the first pack's weights)")
    ap.add_argument("--fmt2", default="u5", choices=["nvfp4", "u4", "u5"])
    ap.add_argument("--rtn", action="store_true", help="plain rounding, no calibration pass (--set unused)")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--layers", type=int, default=0, help="smoke: only the first N layers, no pack written")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    src = Pack(os.path.expanduser(a.src))
    seqs = []
    if not a.rtn:
        seqs = json.load(open(os.path.expanduser(a.set)))["sequences"]
        if a.max_seqs:
            seqs = seqs[: a.max_seqs]
        print(f"calibrating on {len(seqs)} sequences, {sum(map(len, seqs))} tokens", flush=True)
    wr = Writer(os.path.expanduser(a.out))
    keep = a.keep.split(";")
    if a.layers:
        keep.append("lm_head")
    damp_map = [(kv.split("=")[0], float(kv.split("=")[1])) for kv in a.damp_map.split(";") if kv]
    fmt_map = [tuple(kv.split("=")) for kv in a.fmt_map.split(";") if kv]
    second = (Writer(os.path.expanduser(a.out2)), a.fmt2, {}) if a.out2 else None
    gp = GptqPack(src, wr, keep, a.damp, a.block, damp_map, a.fmt, fmt_map, second)
    gp.rtn = a.rtn
    if a.layers:
        gp.meta = dict(src.meta)
        gp.meta["layer_types"] = src.meta["layer_types"][: a.layers]
        forward(gp, seqs)
        os.remove(wr.tmp)
        return
    if a.rtn:
        for name in src.meta["projections"]:
            gp.apply(name, None)
    else:
        forward(gp, seqs)
    ptens = {n + sfx for n in src.meta["projections"] for sfx in (".w", ".b", ".scale", ".gscale")}
    how = "rtn" if a.rtn else f"gptq, {sum(map(len, seqs))} calibration tokens, damp {a.damp} {a.damp_map}"
    for w, projs, label in [(wr, gp.projs, f"{a.fmt} {a.fmt_map}")] + ([(second[0], second[2], second[1])] if second else []):
        for tname, e in src.t.items():
            if tname not in ptens:
                w.add(tname, e["dtype"], e["shape"],
                      [src.mm[src.base + e["offset"] : src.base + e["offset"] + e["nbytes"]]])
        meta = dict(src.meta)
        meta["projections"] = projs
        meta["source"] = f"{src.meta.get('source')} -> q38 {label} ({how})"
        meta["decode_read_bytes"] = sum(
            sum(w.tensors.get(n + sfx, {"nbytes": 0})["nbytes"] for sfx in (".w", ".b", ".scale")) for n in projs)
        w.finish(meta)
        print(f"wrote {w.path}: read per token {meta['decode_read_bytes'] / 1e9:.2f} GB, {time.time() - gp.t0:.0f}s")


if __name__ == "__main__":
    main()
