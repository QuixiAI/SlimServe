#!/usr/bin/env python3
"""Pack a DSpine block drafter (tencent/Qwen3.8-27B-DSpine-b16) into a sidecar
q38 pack.

  pack_dspine.py --src ~/models/Qwen3.8-27B-DSpine-b16 --target ~/models/Qwen3.8-27B \
                 --out ~/models/q38/dspine-b16-bf16.q38w
  quantize.py --src dspine-b16-bf16.q38w --out dspine-b16-nvfp4.q38w

The drafter is five full-attention decoder layers over [anchor, mask, ...]
blocks whose context K/V come from fc(concat of five target residuals), plus
the code bridge that passes each position's predicted code to its successor.

Projections (quantizable): D.fc, D<i>.qkv (rows q | k | v), D<i>.o,
D<i>.gate_up (rows gate | up), D<i>.down. The drafter's RMSNorms are the plain
`x * rsqrt(mean(x^2) + eps) * w`; their weights are stored as (w - 1) so the
engine's Gemma-style norm kernel, which multiplies by (1 + w), serves them
unchanged. Bridge tensors stay BF16.

Tables for the sampling tail (the reference's build_serial_tables), so a
drafted token costs two row lookups instead of two matrix products:
  D.tsv [V, 2R]  = (32 codebook) @ [G_p; A]^T   predecessor gate | value of a sender token
  D.tg  [V, R]   = (head * w_norm) @ W_last      a candidate's logit response to a message
  D.write_t [R, H] = W_last^T, D.G [R, R] = W_last^T W_last (fp32)
D.tg uses the target checkpoint's BF16 LM head (--target). The codebook
itself is not packed: the bridge only ever needs the anchor's row of D.tsv.
D.vocab (--vocab-rank, --draft-vocab): the token ids the drafter's LM head
covers, most frequent first.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_hf import Checkpoint, Writer, add_proj, bf16_to_f32  # noqa: E402


def f32_to_bf16(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype("<u2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", required=True, help="the target's Hugging Face checkpoint (for lm_head.weight)")
    ap.add_argument("--vocab-rank", help="token ids by descending frequency (tools/draft_vocab.py)")
    ap.add_argument("--draft-vocab", type=int, default=65536,
                    help="tokens of the ranking the drafter's LM head covers (multiple of 64; 0 = all)")
    a = ap.parse_args()
    src = os.path.expanduser(a.src)
    cfg = json.load(open(os.path.join(src, "config.json")))
    df = cfg["dflash_config"]
    br = df["selected_prefix_code_bridge"]
    nl = cfg["num_hidden_layers"]
    assert df["selected_prefix_decode_mode"] == "serial" and df["draft_block_causal"] is True
    assert df["selected_prefix_feature_write_mode"] == "independent" and df["selected_prefix_terminal"] is False
    ck = Checkpoint(src)
    wr = Writer(os.path.expanduser(a.out))
    projs = {}
    t0 = time.time()

    def copy(out_name, src_name, sub=None):
        raw, dt, shp = ck.raw(src_name)
        assert dt == "BF16", (src_name, dt)
        if sub is not None:  # leading-index slice of a stacked tensor
            per = int(np.prod(shp[1:])) * 2
            raw, shp = raw[sub * per : (sub + 1) * per], shp[1:]
        wr.add(out_name, "bf16", shp, [raw])

    def norm(out_name, src_name):
        raw, dt, shp = ck.raw(src_name)
        assert dt == "BF16"
        wr.add(out_name, "bf16", shp, [f32_to_bf16(bf16_to_f32(raw) - 1.0)])

    norm("D.hidden_norm", "hidden_norm.weight")
    norm("D.norm", "norm.weight")
    add_proj(wr, ck, "D.fc", ["fc"], projs)
    B = "selected_prefix_bridge."
    copy("D.gate_bias", B + "gate.bias")
    # packed [3R, C] = [gate (G_q); predecessor_gate (G_p); prev (A)]
    packed = [ck.raw(B + n)[0] for n in ("gate.weight", "predecessor_gate.weight", "prev.weight")]
    wr.add("D.packed", "bf16", [3 * br["rank"], br["code_size"]], packed)
    assert not np.frombuffer(ck.raw(B + "out.weight")[0], dtype="<u2").any(), "terminal projection must be zero"
    for i in range(nl):
        L, o = f"layers.{i}.", f"D{i}."
        norm(o + "in_norm", L + "input_layernorm.weight")
        norm(o + "post_norm", L + "post_attention_layernorm.weight")
        norm(o + "q_norm", L + "self_attn.q_norm.weight")
        norm(o + "k_norm", L + "self_attn.k_norm.weight")
        sa = L + "self_attn."
        add_proj(wr, ck, o + "qkv", [sa + "q_proj", sa + "k_proj", sa + "v_proj"], projs)
        add_proj(wr, ck, o + "o", [sa + "o_proj"], projs)
        add_proj(wr, ck, o + "gate_up", [L + "mlp.gate_proj", L + "mlp.up_proj"], projs)
        add_proj(wr, ck, o + "down", [L + "mlp.down_proj"], projs)
        if i == nl - 1:
            copy(o + "read", B + "read_weight")
        else:
            copy(o + "read", B + "early_read_weights", sub=i)
        copy(o + "write", B + "feature_write_weights", sub=i)
    # sampling-tail tables
    R, C = br["rank"], br["code_size"]
    f32 = lambda name: bf16_to_f32(ck.raw(name)[0])
    pv = np.concatenate([f32(B + "predecessor_gate.weight"), f32(B + "prev.weight")]).reshape(2 * R, C)
    cb_raw, _, cb_shape = ck.raw(B + "codebook")
    V = cb_shape[0]
    H = cfg["hidden_size"]
    w_last = f32(B + "feature_write_weights").reshape(nl, H, R)[nl - 1]
    nw = f32("norm.weight")
    tk = Checkpoint(os.path.expanduser(a.target))
    head_name = next(n for n in tk.entries if n.endswith("lm_head.weight"))
    head_raw, hdt, hshape = tk.raw(head_name)
    assert hdt == "BF16" and hshape == [V, H], (hdt, hshape)
    tsv, tg = [], []
    for r0 in range(0, V, 8192):
        r1 = min(V, r0 + 8192)
        code = bf16_to_f32(f32_to_bf16(32.0 * bf16_to_f32(cb_raw[r0 * C * 2 : r1 * C * 2])).tobytes())
        tsv.append(f32_to_bf16(code.reshape(r1 - r0, C) @ pv.T))
        hd = bf16_to_f32(head_raw[r0 * H * 2 : r1 * H * 2]).reshape(r1 - r0, H)
        tg.append(f32_to_bf16(bf16_to_f32(f32_to_bf16(hd * nw).tobytes()).reshape(r1 - r0, H) @ w_last))
        print(f"tables: {r1}/{V} rows  {time.time() - t0:.0f}s", end="\r", file=sys.stderr, flush=True)
    wr.add("D.tsv", "bf16", [V, 2 * R], tsv)
    wr.add("D.tg", "bf16", [V, R], tg)
    wr.add("D.write_t", "bf16", [R, H], [f32_to_bf16(np.ascontiguousarray(w_last.T))])
    wr.add("D.G", "f32", [R, R], [np.ascontiguousarray((w_last.T.astype(np.float64) @ w_last.astype(np.float64)).astype(np.float32))])
    # Draft vocabulary: the drafter proposes from the most frequent tokens only,
    # so its LM head pass reads a quarter of the rows (the engine falls back to
    # the full head while a sequence is using many tokens outside the list).
    nv = 0
    if a.vocab_rank and a.draft_vocab:
        rank = json.load(open(os.path.expanduser(a.vocab_rank)))
        nv = a.draft_vocab
        assert nv % 64 == 0 and len(rank) >= nv and len(set(rank[:nv])) == nv and max(rank[:nv]) < V
        wr.add("D.vocab", "u8", [nv * 4], [np.array(rank[:nv], dtype="<i4")])
    read = sum(wr.tensors[n + ".w"]["nbytes"] + wr.tensors.get(n + ".scale", {"nbytes": 0})["nbytes"]
               for n in projs)
    wr.finish({"source": os.path.basename(src), "kind": "dspine", "tables": 1, "draft_vocab": nv, "projections": projs,
               "decode_read_bytes": read, "layers": nl, "block_size": cfg["block_size"],
               "mask_token": df["mask_token_id"], "taps": df["target_layer_ids"],
               "heads": cfg["num_attention_heads"], "kv_heads": cfg["num_key_value_heads"],
               "head_dim": cfg["head_dim"], "rope_theta": cfg["rope_theta"], "eps": cfg["rms_norm_eps"],
               "code_size": br["code_size"], "rank": br["rank"], "bridge_eps": br["eps"],
               "head_topk": df["selected_prefix_last_write_refine"]["head_topk"],
               "window": cfg.get("sliding_window") or 0})
    print(f"wrote {a.out}: {read / 1e9:.2f} GB of projections, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
