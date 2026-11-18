#!/usr/bin/env python3
"""Token files for q38_lab from chat prompts, through the model's own chat
template (thinking left on, the serving default).

  mkprompt.py --user "Write an essay about ..." --out essay.tokens
  mkprompt.py --detok out.tokens          print the text of a generated token file
"""
import argparse
import os

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="~/models/Qwen3.8-27B")
ap.add_argument("--system")
ap.add_argument("--user")
ap.add_argument("--out")
ap.add_argument("--detok")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(os.path.expanduser(a.model))
if a.detok:
    ids = [int(x) for x in open(a.detok).read().split()]
    print(tok.decode(ids))
else:
    msgs = ([{"role": "system", "content": a.system}] if a.system else []) + [
        {"role": "user", "content": a.user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    ids = tok.encode(text, add_special_tokens=False)
    open(a.out, "w").write(" ".join(map(str, ids)))
    print(f"{len(ids)} tokens -> {a.out}")
