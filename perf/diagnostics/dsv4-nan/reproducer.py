"""Minimal decode+prefill mixed-batch reproducer.

One spec-decode request (A) generates continuously while long prompts (B)
prefill against it. Every chunked-prefill step of a B request creates the
decode+prefill mixed batch (A's 6 verify rows + B's chunk row) that the
NaN row fingerprints implicate. Cycles fresh 24K-token prompts from the
unused tail of corpus5 so APC never shortcuts the prefills.
"""
import json
import sys
import threading
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "18050"
URL = f"http://127.0.0.1:{PORT}/v1/completions"
SC = ("/tmp/claude-1001/-home-ubuntu-SlimServe/"
      "1c975788-d502-4dff-93f8-4258ce64e11d/scratchpad/dualbench")

from vllm.tokenizers.registry import get_tokenizer  # noqa: E402

MODEL = ("/home/ubuntu/models/antirez-deepseek-v4-gguf/"
         "DeepSeek-V4-Flash-Layers37-42Q4KExperts-OtherExpertLayersIQ2XXSGateUp"
         "-Q2KDown-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-fixed-0731.gguf")
tok = get_tokenizer(MODEL)
ids = tok.encode(open(f"{SC}/corpus5.txt").read(), add_special_tokens=False)
SPARE_START = 260_000  # everything before was served in earlier runs
spare = ids[SPARE_START:]
print(f"spare tokens: {len(spare)}", flush=True)


def post(prompt, max_tokens, timeout=1200):
    body = json.dumps({
        "model": "DeepSeek-v4-Flash-0731", "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


DEADLINE = time.time() + 30 * 60
decode_rounds = 0


def decoder():
    """Request A: long spec-decode generations, back to back."""
    global decode_rounds
    prompt = tok.decode(spare[:1000]) + f' stream {threading.get_ident()}:'
    while time.time() < DEADLINE:
        out = post(prompt + f"\n\nround {decode_rounds}:", 4000)
        text = out["choices"][0]["text"]
        cpt = len(text) / max(1, out["usage"]["completion_tokens"])
        print(f"A round {decode_rounds}: {out['usage']['completion_tokens']} "
              f"tokens, chars/token {cpt:.2f}"
              + ("  <-- DEGENERATE" if cpt < 0.5 else ""), flush=True)
        decode_rounds += 1


threads = []
for _d in range(4):
    _t = threading.Thread(target=decoder, daemon=True)
    _t.start()
    threads.append(_t)
    time.sleep(3)  # stagger so verify windows interleave, not align
t = threads[0]
time.sleep(20)  # let A settle into steady decode

# B: cycle 24K-token prefills; each chunks into several mixed-batch steps.
PLEN = 24_000
n_prompts = (len(spare) - 2000) // PLEN
prefills = 0
while time.time() < DEADLINE and t.is_alive():
    region = spare[2000 + (prefills % n_prompts) * PLEN:][:PLEN]
    # vary the head so repeated cycles never share an APC prefix
    prompt = f"pass {prefills}: " + tok.decode(region)
    started = time.time()
    out = post(prompt, 4)
    prefills += 1
    if prefills % 10 == 0:
        print(f"B prefills: {prefills} (last {time.time()-started:.1f}s, "
              f"{out['usage']['prompt_tokens']} prompt tokens)", flush=True)

print(f"DONE: {prefills} prefills, {decode_rounds} decode rounds", flush=True)
