#!/usr/bin/env python3
"""CPU reference forward for Qwen3.8-27B from a q38 pack (numpy, float32).

The golden oracle for the native engine: teacher-forced logits and per-layer
residual streams for a token sequence, computed layer by layer over the whole
sequence so each projection is dequantized once. Nothing touches the GPU.

  q38ref.py --pack P --prompt "text"            top-5 next tokens per position tail
  q38ref.py --pack P --tokens ids.txt --save out.npz   logits + hidden dumps

Math mirrors the serving stack (see perf/qwen38_metal_design.md and the muse
emit in csrc/quixicore/tm_metal/qc_metal_serving.mm):
  layer:  h = gemma_norm(res); res += mixer(h); h2 = gemma_norm(res);
          res += down(silu(gate(h2)) * up(h2))
  final:  logits = lm_head(gemma_norm(res))
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

F = np.float32


def e4m3_lut():
    lut = np.zeros(256, dtype=F)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0:
            v = (m / 8.0) * 2.0 ** -6
        elif e == 15 and m == 7:
            v = 0.0  # NaN encoding; real checkpoints never contain it
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        lut[b] = s * v
    return lut


E4M3 = e4m3_lut()
_mag = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=F)
E2M1 = np.concatenate([_mag, -_mag])


class Pack:
    def __init__(self, path):
        f = open(path, "rb")
        assert f.read(8) == b"Q38PACK1", "not a q38 pack"
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
        self.meta = h["meta"]
        self.t = h["tensors"]
        self.base = 16 + n
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def raw(self, name, dtype):
        e = self.t[name]
        a = self.base + e["offset"]
        return np.frombuffer(self.mm[a : a + e["nbytes"]], dtype=dtype).reshape(e["shape"])

    def f32(self, name):
        e = self.t[name]
        if e["dtype"] == "f32":
            return self.raw(name, "<f4").astype(F)
        if e["dtype"] == "bf16":
            u = self.raw(name, "<u2").astype(np.uint32) << 16
            return u.view(F)
        raise ValueError(name)

    def proj_rows(self, name, r0, r1):
        """Dequantized rows [r0, r1) of projection `name` as float32 [rows, K]."""
        p = self.meta["projections"][name]
        K = p["K"]
        if p["fmt"] == "fp8ch":
            w = E4M3[self.raw(name + ".w", np.uint8)[r0:r1]]
            return w * self.raw(name + ".scale", "<f4")[r0:r1, None]
        if p["fmt"] == "nvfp4":
            pk = self.raw(name + ".w", np.uint8)[r0:r1]
            rows = pk.shape[0]
            v = np.empty((rows, K), dtype=F)
            v[:, 0::2] = E2M1[pk & 0x0F]
            v[:, 1::2] = E2M1[pk >> 4]
            sc = E4M3[self.raw(name + ".scale", np.uint8)[r0:r1]]
            v = v.reshape(rows, K // 16, 16) * sc[:, :, None]
            return v.reshape(rows, K) * self.raw(name + ".gscale", "<f4")[0]
        if p["fmt"] in ("u4", "u5"):
            # uniform planar: value = (n - L/2 + 0.5) * gscale * 2^(-c/16)
            pk = self.raw(name + ".w", np.uint8)[r0:r1]
            rows = pk.shape[0]
            n = np.empty((rows, K), dtype=np.int32)
            n[:, 0::2] = pk & 0x0F
            n[:, 1::2] = pk >> 4
            L = 16
            if p["fmt"] == "u5":
                bits = np.unpackbits(self.raw(name + ".b", np.uint8)[r0:r1], axis=1, bitorder="little")
                n = 2 * n + bits
                L = 32
            sc = np.exp2(-self.raw(name + ".scale", np.uint8)[r0:r1].astype(F) / 16.0)
            v = (n.astype(F) - (L / 2 - 0.5)).reshape(rows, K // 16, 16) * sc[:, :, None]
            return (v.reshape(rows, K) * self.raw(name + ".gscale", "<f4")[0]).astype(F)
        return self.f32(name + ".w")[r0:r1]

    def apply(self, name, x, chunk=8192):
        """x [T, K] -> [T, N] through projection `name`."""
        N = self.meta["projections"][name]["N"]
        out = np.empty((x.shape[0], N), dtype=F)
        for r in range(0, N, chunk):
            w = self.proj_rows(name, r, min(N, r + chunk))
            out[:, r : r + w.shape[0]] = x @ w.T
        return out


def gemma_norm(x, w, eps):
    return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * (1.0 + w)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gdn_mixer(pk, o, h, m, segs):
    g = m["gdn"]
    Hk, Hv, Dk, Dv, ks = g["k_heads"], g["v_heads"], g["k_dim"], g["v_dim"], g["conv_kernel"]
    qkvz = pk.apply(o + "qkvz", h)
    ba = pk.apply(o + "ba", h)
    conv_ch = 2 * Hk * Dk + Hv * Dv
    w = pk.f32(o + "conv_w")  # [conv_ch, ks]
    A_log, dt_bias = pk.f32(o + "A_log"), pk.f32(o + "dt_bias")
    nw = pk.f32(o + "gdn_norm")  # [Dv], plain weight (not 1 + w)
    eps = 1e-6 / Dk
    rep = Hv // Hk
    gated_all = np.empty((h.shape[0], Hv * Dv), dtype=F)
    for s0, s1 in segs:
        T = s1 - s0
        mixed, z = qkvz[s0:s1, :conv_ch], qkvz[s0:s1, conv_ch:]
        b, a = ba[s0:s1, :Hv], ba[s0:s1, Hv:]
        # causal short conv over time: out[t] = sum_j w[:, j] * x[t - (ks-1) + j]
        xp = np.concatenate([np.zeros((ks - 1, conv_ch), dtype=F), mixed], 0)
        conv = np.zeros_like(mixed)
        for j in range(ks):
            conv += xp[j : j + T] * w[:, j]
        conv = silu(conv)
        q = conv[:, : Hk * Dk].reshape(T, Hk, Dk)
        k = conv[:, Hk * Dk : 2 * Hk * Dk].reshape(T, Hk, Dk)
        v = conv[:, 2 * Hk * Dk :].reshape(T, Hv, Dv)
        q = q * (Dk ** -1.0) / np.sqrt((q * q).sum(-1, keepdims=True) / Dk + eps)
        k = k * (Dk ** -0.5) / np.sqrt((k * k).sum(-1, keepdims=True) / Dk + eps)
        alpha = a + dt_bias
        softplus = np.where(alpha > 20.0, alpha, np.log1p(np.exp(np.minimum(alpha, 20.0))))
        decay = np.exp(-np.exp(A_log) * softplus)  # [T, Hv]
        beta = sigmoid(b)
        S = np.zeros((Hv, Dv, Dk), dtype=F)
        y = np.empty((T, Hv, Dv), dtype=F)
        for t in range(T):
            kt = np.repeat(k[t], rep, axis=0)  # [Hv, Dk]
            qt = np.repeat(q[t], rep, axis=0)
            S *= decay[t][:, None, None]
            kv_mem = np.matmul(S, kt[:, :, None])[:, :, 0]
            delta = (v[t] - kv_mem) * beta[t][:, None]
            S += delta[:, :, None] * kt[:, None, :]
            y[t] = np.matmul(S, qt[:, :, None])[:, :, 0]
        inv = 1.0 / np.sqrt((y * y).mean(-1, keepdims=True) + m["eps"])
        gated = y * inv * nw * silu(z.reshape(T, Hv, Dv))
        gated_all[s0:s1] = gated.reshape(T, Hv * Dv)
    return pk.apply(o + "gdn_out", gated_all)


def attn_mixer(pk, o, h, m, segs):
    a = m["attn"]
    Hq, Hkv, D, R = a["heads"], a["kv_heads"], a["head_dim"], a["rot_dim"]
    N = h.shape[0]
    qkv = pk.apply(o + "qkv", h)
    qg = qkv[:, : Hq * 2 * D].reshape(N, Hq, 2, D)
    q_all = gemma_norm(qg[:, :, 0], pk.f32(o + "q_norm"), m["eps"])
    gate = qg[:, :, 1]
    k_all = gemma_norm(qkv[:, Hq * 2 * D : Hq * 2 * D + Hkv * D].reshape(N, Hkv, D),
                       pk.f32(o + "k_norm"), m["eps"])
    v_all = qkv[:, Hq * 2 * D + Hkv * D :].reshape(N, Hkv, D)
    R2 = R // 2
    inv_freq = 1.0 / (a["rope_theta"] ** (np.arange(0, R, 2, dtype=np.float64) / R))
    grp = Hq // Hkv
    out = np.empty((N, Hq, D), dtype=F)
    for s0, s1 in segs:
        T = s1 - s0
        ang = np.arange(T, dtype=np.float64)[:, None] * inv_freq[None]
        c, sn = np.cos(ang).astype(F)[:, None], np.sin(ang).astype(F)[:, None]

        def rope(x):
            x = x.copy()
            x1, x2 = x[..., :R2].copy(), x[..., R2:R].copy()
            x[..., :R2] = x1 * c - x2 * sn
            x[..., R2:R] = x2 * c + x1 * sn
            return x

        q, k, v = rope(q_all[s0:s1]), rope(k_all[s0:s1]), v_all[s0:s1]
        mask = np.triu(np.full((T, T), -np.inf, dtype=F), 1)
        for hq in range(Hq):
            hk = hq // grp
            sc = (q[:, hq] @ k[:, hk].T) * (D ** -0.5) + mask
            sc = sc - sc.max(-1, keepdims=True)
            p = np.exp(sc)
            p /= p.sum(-1, keepdims=True)
            out[s0:s1, hq] = p @ v[:, hk]
    if a["output_gate"]:
        out = out * sigmoid(gate)
    return pk.apply(o + "o", out.reshape(N, Hq * D))


def forward(pk, tokens, hidden_out=None, log=None):
    """Teacher-forced forward. `tokens` is one sequence or a list of sequences;
    several sequences share each projection pass (weights are decoded once per
    layer) and are kept causally independent. Returns (final, logits) over the
    concatenated rows, or lists split per sequence when a list was given."""
    many = len(tokens) > 0 and isinstance(tokens[0], (list, tuple, np.ndarray))
    seqs = [list(t) for t in tokens] if many else [list(tokens)]
    segs, n = [], 0
    for sq in seqs:
        segs.append((n, n + len(sq)))
        n += len(sq)
    flat = np.concatenate([np.asarray(sq, dtype=np.int64) for sq in seqs])
    m = pk.meta
    emb = pk.raw("embed", "<u2")
    res = (emb[flat].astype(np.uint32) << 16).view(F).copy()
    for l, kind in enumerate(m["layer_types"]):
        o = f"L{l}."
        h = gemma_norm(res, pk.f32(o + "in_norm"), m["eps"])
        mix = gdn_mixer(pk, o, h, m, segs) if kind == "gdn" else attn_mixer(pk, o, h, m, segs)
        res = res + mix
        h2 = gemma_norm(res, pk.f32(o + "post_norm"), m["eps"])
        gu = pk.apply(o + "gate_up", h2)
        I = m["inter"]
        res = res + pk.apply(o + "down", silu(gu[:, :I]) * gu[:, I:])
        if hidden_out is not None:
            hidden_out.append(res.copy())
        if log:
            log(l, res)
    final = gemma_norm(res, pk.f32("final_norm"), m["eps"])
    logits = pk.apply("lm_head", final)
    if many:
        return [final[a:b] for a, b in segs], [logits[a:b] for a, b in segs]
    return final, logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=os.path.expanduser("~/models/q38/qwen38-27b-nvfp4.q38w"))
    ap.add_argument("--tokenizer", default=os.path.expanduser("~/models/Qwen3.8-27B-NVFP4/tokenizer.json"))
    ap.add_argument("--prompt")
    ap.add_argument("--tokens", help="file of whitespace-separated token ids")
    ap.add_argument("--save", help="write logits/final/hidden to this .npz")
    ap.add_argument("--hidden", action="store_true", help="also save per-layer residual streams")
    a = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(a.tokenizer)
    if a.tokens:
        ids = [int(x) for x in open(a.tokens).read().split()]
    else:
        ids = tok.encode(a.prompt, add_special_tokens=False).ids
    pk = Pack(a.pack)
    t0 = time.time()
    hidden = [] if a.hidden else None

    def log(l, res):
        if l % 8 == 7:
            print(f"  layer {l + 1}  {time.time() - t0:.0f}s  |res| {np.abs(res).mean():.3f}", file=sys.stderr, flush=True)

    final, logits = forward(pk, ids, hidden_out=hidden, log=log)
    print(f"{len(ids)} tokens, {time.time() - t0:.0f}s")
    for t in range(max(0, len(ids) - 6), len(ids)):
        top = np.argsort(-logits[t])[:5]
        print(f"pos {t} {tok.decode([ids[t]])!r:>14} -> " + "  ".join(f"{tok.decode([int(i)])!r}:{logits[t, i]:.2f}" for i in top))
    if a.save:
        out = {"tokens": np.array(ids, dtype=np.int32), "logits": logits, "final": final}
        if hidden is not None:
            out["hidden"] = np.stack(hidden)
        np.savez(a.save, **out)
        print("saved", a.save)


if __name__ == "__main__":
    main()
