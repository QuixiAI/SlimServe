#!/usr/bin/env python3
"""Quant quality on the CPU: teacher-forced KL divergence and top-1 agreement
of a candidate weight recipe against the BF16 reference.

  kld.py build-set --out set.json                 token sequences from local text
  kld.py ref --pack bf16.q38w --set set.json --out ref.npz [--imatrix-out im.npz]
  kld.py eval --ref ref.npz --set set.json --mix "*=cand.q38w"
  kld.py eval --ref ref.npz --set set.json --mix "*.gate_up=a.q38w;*.down=a.q38w;*=bf16.q38w"
  kld.py engine --ref ref.npz --set set.json --pack cand.q38w     same score from the GPU engine

A mix is an ordered list of glob=pack rules over projection names (L12.qkvz,
L3.qkv, lm_head, ...); the first match supplies that projection, so any
per-tensor bit allocation can be scored without writing a new pack. Non
projection tensors (norms, conv, embeddings) come from the last rule's pack.

KL is KL(reference || candidate) in nats, averaged over every position.
"""
import argparse
import fnmatch
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q38ref import Pack, forward  # noqa: E402


class Mix:
    def __init__(self, spec):
        self.rules = []
        packs = {}
        for part in spec.split(";"):
            pat, path = part.split("=", 1)
            path = os.path.expanduser(path)
            if path not in packs:
                packs[path] = Pack(path)
            self.rules.append((pat, packs[path]))
        self.base = self.rules[-1][1]
        self.meta = self.base.meta
        self.moments = None
        self.read_bytes = 0
        for name in self.meta["projections"]:
            pk = self.route(name)
            self.read_bytes += pk.t[name + ".w"]["nbytes"] + pk.t.get(name + ".scale", {"nbytes": 0})["nbytes"]

    def route(self, name):
        for pat, pk in self.rules:
            if fnmatch.fnmatchcase(name, pat):
                return pk
        raise KeyError(f"no rule matches projection {name}")

    def raw(self, name, dtype):
        return self.base.raw(name, dtype)

    def f32(self, name):
        return self.base.f32(name)

    def apply(self, name, x):
        if self.moments is not None:
            m = self.moments.setdefault(name, [np.zeros(x.shape[1], dtype=np.float64), 0])
            m[0] += (x.astype(np.float64) ** 2).sum(0)
            m[1] += x.shape[0]
        return self.route(name).apply(name, x)


def log_softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def run(mix, seqs, label):
    t0 = time.time()

    def log(l, res):
        if l % 16 == 15:
            print(f"  {label}: layer {l + 1}/64  {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    _, logits = forward(mix, seqs, log=log)
    return logits


CHAT = [
    ("You are a helpful assistant.",
     "Explain why the sky is blue in two short paragraphs.",
     "Sunlight looks white but is a mix of every color. As it passes through the atmosphere it "
     "collides with air molecules, and the shorter the wavelength, the more strongly it is "
     "scattered in every direction.\n\nBlue light is scattered roughly ten times more than red, "
     "so it reaches your eyes from all over the sky rather than only from the direction of the "
     "sun. Violet is scattered even more, but the sun emits less of it and our eyes are less "
     "sensitive to it, so the sky reads as blue."),
    ("You are a careful coding assistant.",
     "Write a Python function that merges two sorted lists into one sorted list.",
     "```python\ndef merge_sorted(a, b):\n    out = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n"
     "        if a[i] <= b[j]:\n            out.append(a[i])\n            i += 1\n        else:\n"
     "            out.append(b[j])\n            j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n"
     "    return out\n```\n\nIt walks both lists once, so it runs in O(len(a) + len(b)) time."),
    ("You are a helpful assistant.",
     "A train leaves at 9:40 and travels 150 km at 60 km/h, then waits 20 minutes, then travels "
     "90 km at 45 km/h. When does it arrive?",
     "The first leg takes 150 / 60 = 2.5 hours, so the train reaches the stop at 12:10. After a "
     "20 minute wait it leaves at 12:30. The second leg takes 90 / 45 = 2 hours, so it arrives "
     "at 14:30."),
]


def build_set(a):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.expanduser(a.tokenizer))
    seqs, labels = [], []
    for system, user, answer in ([] if a.no_chat else CHAT):
        text = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
                f"<|im_start|>assistant\n<think>\n\n</think>\n\n{answer}<|im_end|>\n")
        seqs.append(tok.encode(text, add_special_tokens=False).ids[: a.max_len])
        labels.append("chat")
    for path in a.files or []:
        text = open(os.path.expanduser(path), errors="ignore").read()
        ids = tok.encode(text[a.skip_chars : a.skip_chars + a.max_len * 8], add_special_tokens=False).ids
        seqs.append(ids[: a.max_len])
        labels.append(os.path.basename(path))
    json.dump({"sequences": seqs, "labels": labels}, open(a.out, "w"))
    print(f"wrote {a.out}: {len(seqs)} sequences, {sum(map(len, seqs))} tokens "
          f"({', '.join(f'{l}:{len(s)}' for l, s in zip(labels, seqs))})")


def cmd_ref(a):
    seqs = json.load(open(a.set))["sequences"]
    mix = Mix(f"*={a.pack}")
    if a.imatrix_out:
        mix.moments = {}
    logits = run(mix, seqs, "reference")
    np.savez(a.out, **{f"s{i}": l.astype(np.float16) for i, l in enumerate(logits)},
             argmax=np.concatenate([l.argmax(-1) for l in logits]))
    print("saved", a.out)
    if a.imatrix_out:
        np.savez(a.imatrix_out, **{k: (v[0] / v[1]).astype(np.float32) for k, v in mix.moments.items()})
        print("saved", a.imatrix_out)


def cmd_eval(a):
    seqs = json.load(open(a.set))["sequences"]
    labels = json.load(open(a.set))["labels"]
    ref = np.load(a.ref)
    mix = Mix(a.mix)
    logits = run(mix, seqs, "candidate")
    kls, agree, n = [], 0, 0
    for i, l in enumerate(logits):
        r = ref[f"s{i}"].astype(np.float32)
        lp, lq = log_softmax(r), log_softmax(l)
        kl = (np.exp(lp) * (lp - lq)).sum(-1)
        same = int((r.argmax(-1) == l.argmax(-1)).sum())
        kls.append(kl)
        agree += same
        n += len(kl)
        print(f"  {labels[i]:>24}: mean KL {kl.mean():.5f}  p99 {np.quantile(kl, 0.99):.4f}  "
              f"top-1 {same}/{len(kl)}")
    allkl = np.concatenate(kls)
    result = {
        "mix": a.mix,
        "tokens": n,
        "mean_kl": float(allkl.mean()),
        "median_kl": float(np.median(allkl)),
        "p99_kl": float(np.quantile(allkl, 0.99)),
        "top1": agree / n,
        "read_gb": mix.read_bytes / 1e9,
    }
    print(json.dumps(result))
    if a.log:
        with open(a.log, "a") as f:
            f.write(json.dumps(result) + "\n")


def cmd_engine(a):
    """Score a pack by running the native engine on the GPU (seconds, not minutes)."""
    import subprocess
    import tempfile

    data = json.load(open(a.set))
    seqs, labels = data["sequences"], data["labels"]
    ref = np.load(a.ref)
    kls, agree, n = [], 0, 0
    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/.local/scratch")) as td:
        for i, seq in enumerate(seqs):
            tf, lf = os.path.join(td, "t.tokens"), os.path.join(td, "l.bin")
            open(tf, "w").write(" ".join(map(str, seq)))
            subprocess.run([os.path.expanduser(a.lab), "--pack", os.path.expanduser(a.pack),
                            "--metallib", a.metallib, "--tokens", tf, "--chunk", str(a.chunk),
                            "--dump-all", lf] + (["--big-head", str(a.big_head)] if a.big_head else []),
                           check=True, stdout=subprocess.DEVNULL)
            raw = np.fromfile(lf, dtype="<u2")
            l = (raw.astype(np.uint32) << 16).view(np.float32).reshape(len(seq), -1)
            r = ref[f"s{i}"].astype(np.float32)
            lo = max(a.score_from, a.big_head)  # rows before this are not scored (a big head has no logits)
            if lo >= len(seq):
                continue
            l, r = l[lo:], r[lo:]
            lp, lq = log_softmax(r), log_softmax(l)
            kl = (np.exp(lp) * (lp - lq)).sum(-1)
            same = int((r.argmax(-1) == l.argmax(-1)).sum())
            kls.append(kl)
            agree += same
            n += len(kl)
            print(f"  {labels[i]:>24}: mean KL {kl.mean():.5f}  p99 {np.quantile(kl, 0.99):.4f}  "
                  f"top-1 {same}/{len(kl)}")
    allkl = np.concatenate(kls)
    result = {"engine_pack": a.pack, "chunk": a.chunk, "big_head": a.big_head, "score_from": a.score_from,
              "tokens": n, "mean_kl": float(allkl.mean()),
              "median_kl": float(np.median(allkl)), "p99_kl": float(np.quantile(allkl, 0.99)),
              "top1": agree / n}
    print(json.dumps(result))
    if a.log:
        with open(a.log, "a") as f:
            f.write(json.dumps(result) + "\n")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-set")
    b.add_argument("--out", required=True)
    b.add_argument("--tokenizer", default="~/models/Qwen3.8-27B-NVFP4/tokenizer.json")
    b.add_argument("--files", nargs="*")
    b.add_argument("--max-len", type=int, default=192)
    b.add_argument("--skip-chars", type=int, default=0)
    b.add_argument("--no-chat", action="store_true", help="leave out the built-in chat samples")
    r = sub.add_parser("ref")
    r.add_argument("--pack", required=True)
    r.add_argument("--set", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--imatrix-out")
    e = sub.add_parser("eval")
    e.add_argument("--ref", required=True)
    e.add_argument("--set", required=True)
    e.add_argument("--mix", required=True)
    e.add_argument("--log")
    g = sub.add_parser("engine")
    here = os.path.dirname(os.path.abspath(__file__))
    g.add_argument("--ref", required=True)
    g.add_argument("--set", required=True)
    g.add_argument("--pack", required=True)
    bd = os.environ.get("Q38_BUILD") or os.path.join(here, "..", "build")
    g.add_argument("--lab", default=os.path.join(bd, "q38_lab"))
    g.add_argument("--metallib", default=os.path.join(bd, "q38_kernels.metallib")
                   + ":" + os.path.join(bd, "q38_native.metallib"))
    g.add_argument("--chunk", type=int, default=8)
    g.add_argument("--big-head", type=int, default=0,
                   help="feed the first N tokens of every sequence as 64-row prefill chunks")
    g.add_argument("--score-from", type=int, default=0, help="score rows from this index on")
    g.add_argument("--log")
    a = ap.parse_args()
    {"build-set": build_set, "ref": cmd_ref, "eval": cmd_eval, "engine": cmd_engine}[a.cmd](a)


if __name__ == "__main__":
    main()
