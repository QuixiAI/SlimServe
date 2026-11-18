#!/usr/bin/env python3
"""Compare a q38_lab --dump against the CPU reference (.npz from q38ref.py).

  parity.py ref.npz dump.bin              logits of the last step's rows
  parity.py ref.npz dump.bin --cap L      residual stream after layer L

The dump holds the LAST step's rows, so they are matched against the tail
rows of the reference sequence.
"""
import argparse

import numpy as np


def stats(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return cos, float(np.abs(a - b).max()), float(np.abs(a).max())


def kl(p_logits, q_logits):
    p = p_logits.astype(np.float64)
    q = q_logits.astype(np.float64)
    p -= p.max(-1, keepdims=True)
    q -= q.max(-1, keepdims=True)
    lp = p - np.log(np.exp(p).sum(-1, keepdims=True))
    lq = q - np.log(np.exp(q).sum(-1, keepdims=True))
    return (np.exp(lp) * (lp - lq)).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("dump")
    ap.add_argument("--cap", type=int, default=0)
    a = ap.parse_args()
    ref = np.load(a.ref)
    raw = np.fromfile(a.dump, dtype=np.float32)
    if a.cap:
        want = ref["hidden"][a.cap - 1]
        got = raw.reshape(-1, want.shape[1])
        want = want[-got.shape[0]:]
        for r in range(got.shape[0]):
            cos, mad, amax = stats(want[r], got[r])
            print(f"layer {a.cap} row {r}: cos {cos:.6f}  max|d| {mad:.4f}  (ref max {amax:.2f})")
        return
    want = ref["logits"]
    got = raw.reshape(-1, want.shape[1])
    want = want[-got.shape[0]:]
    d = kl(want, got)
    for r in range(got.shape[0]):
        cos, mad, _ = stats(want[r], got[r])
        print(f"row {r}: cos {cos:.6f}  max|d| {mad:.3f}  KL(ref||engine) {d[r]:.5f}  "
              f"top1 ref {int(want[r].argmax())} engine {int(got[r].argmax())}")
    print(f"top-1 agreement {int((want.argmax(-1) == got.argmax(-1)).sum())}/{got.shape[0]}, mean KL {d.mean():.5f}")


if __name__ == "__main__":
    main()
