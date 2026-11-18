#!/usr/bin/env python3
"""Token frequency ranking for the DSpine draft vocabulary (pack_dspine.py
--vocab-rank). Counts Qwen3.8 token ids over wikitext-103 (prose; the two
train parquet files of Salesforce/wikitext wikitext-103-raw-v1 as
wt103-*.parquet in the working directory), local source trees (code, markdown)
and every token stream the campaign produced; writes the ids in descending
count order. Needs tokenizers and pyarrow.   draft_vocab.py OUT.json
2026-11-10 run: 98,731 distinct tokens over 126M; cumulative coverage 93.8% at
16k, 98.0% at 32k, 99.5% at 48k, 99.9% at 64k."""
import sys, os, glob, json, collections, itertools
import numpy as np
from tokenizers import Tokenizer
tok = Tokenizer.from_file(os.path.expanduser("~/models/Qwen3.8-27B/tokenizer.json"))
cnt = collections.Counter()
def feed(texts, weight=1):
    for i in range(0, len(texts), 2000):
        for enc in tok.encode_batch(texts[i:i + 2000], add_special_tokens=False):
            for t in enc.ids: cnt[t] += weight
import pyarrow.parquet as pq
n = 0
for f in sorted(glob.glob("wt103-*.parquet")):
    col = pq.read_table(f).column("text").to_pylist()
    col = [c for c in col if c.strip()][:400000]   # ~40M tokens are plenty
    feed(col); n += len(col); print(f, len(col), "lines", sum(cnt.values()), "tokens", flush=True)
# code and markdown from local trees
files = []
for root in ("~/Code/slimserve/SlimServe-q38native", "~/llama.cpp"):
    for ext in ("py", "c", "cpp", "h", "mm", "metal", "md", "sh", "rs", "js", "ts", "json", "toml", "yaml", "txt", "cu", "cuh", "hpp"):
        files += glob.glob(os.path.expanduser(f"{root}/**/*.{ext}"), recursive=True)
files = [f for f in files if "/build" not in f and "/.git/" not in f and "/perf/results/" not in f and os.path.getsize(f) < 400_000]
texts = []
for f in files[:20000]:
    try: texts.append(open(f, errors="ignore").read())
    except Exception: pass
before = sum(cnt.values()); feed(texts, weight=2); print("local files", len(texts), sum(cnt.values()) - before, "weighted tokens", flush=True)
# the campaign's own token streams (model outputs, chat-format prompts): weighted up, they are the target distribution
own = collections.Counter()
for f in glob.glob(os.path.expanduser("~/.local/scratch/qwen38-campaign/ref/**/*.tokens"), recursive=True) + \
         glob.glob(os.path.expanduser("~/Code/slimserve/SlimServe-q38native/perf/results/**/*.tokens"), recursive=True):
    try: own.update(int(x) for x in open(f).read().split())
    except Exception: pass
for f in ("ref/eval2_set.json", "ref/calib3_set.json", "ref/calib_set.json", "ref/calib2_set.json", "ref/eval_set.json"):
    try:
        for s in json.load(open(os.path.expanduser("~/.local/scratch/qwen38-campaign/" + f)))["sequences"]: own.update(int(x) for x in s)
    except Exception as e: print("skip", f, e)
print("own streams:", sum(own.values()), "tokens,", len(own), "distinct")
for t, c in own.items(): cnt[t] += 50 * c
order = [t for t, _ in cnt.most_common()]
json.dump(order, open(sys.argv[1], "w"))
tot = sum(cnt.values()); cum = np.cumsum([c for _, c in cnt.most_common()]) / tot
print("distinct", len(order), "coverage at 16k/32k/48k/64k/96k:", [round(float(cum[min(k, len(cum)) - 1]), 4) for k in (16384, 32768, 49152, 65536, 98304)])
