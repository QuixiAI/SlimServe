#!/usr/bin/env python3
"""Pack the checkpoint's multi-token-prediction head into a sidecar q38 pack.

  pack_mtp.py --src ~/models/Qwen3.8-27B --out ~/models/q38/qwen38-27b-mtp-bf16.q38w

The MTP head is one gated full-attention decoder layer in front of the shared
LM head: fc([norm_e(embed(next token)) | norm_h(final hidden)]) -> layer ->
norm. It is kept out of the main pack so the main pack's recipe and the draft
head's recipe can change independently (quantize.py works on this file too).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pack_hf import Checkpoint, Writer, add_proj, bf16_to_f32  # noqa: E402,F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = os.path.expanduser(a.src)
    t = json.load(open(os.path.join(src, "config.json")))["text_config"]
    assert t["mtp_num_hidden_layers"] == 1 and not t["mtp_use_dedicated_embeddings"]
    ck = Checkpoint(src)
    wr = Writer(os.path.expanduser(a.out))
    projs = {}
    t0 = time.time()

    def copy(out_name, src_name):
        raw, dt, shp = ck.raw(src_name)
        assert dt == "BF16", (src_name, dt)
        wr.add(out_name, "bf16", shp, [raw])

    L = "mtp.layers.0."
    copy("MTP.pre_e", "mtp.pre_fc_norm_embedding.weight")
    copy("MTP.pre_h", "mtp.pre_fc_norm_hidden.weight")
    copy("MTP.in_norm", L + "input_layernorm.weight")
    copy("MTP.post_norm", L + "post_attention_layernorm.weight")
    copy("MTP.q_norm", L + "self_attn.q_norm.weight")
    copy("MTP.k_norm", L + "self_attn.k_norm.weight")
    copy("MTP.norm", "mtp.norm.weight")
    add_proj(wr, ck, "MTP.fc", ["mtp.fc"], projs)
    sa = L + "self_attn."
    add_proj(wr, ck, "MTP.qkv", [sa + "q_proj", sa + "k_proj", sa + "v_proj"], projs)
    add_proj(wr, ck, "MTP.o", [sa + "o_proj"], projs)
    add_proj(wr, ck, "MTP.gate_up", [L + "mlp.gate_proj", L + "mlp.up_proj"], projs)
    add_proj(wr, ck, "MTP.down", [L + "mlp.down_proj"], projs)
    read = sum(wr.tensors[n + ".w"]["nbytes"] + wr.tensors.get(n + ".scale", {"nbytes": 0})["nbytes"]
               for n in projs)
    wr.finish({"source": os.path.basename(src) + " mtp", "kind": "mtp", "projections": projs,
               "decode_read_bytes": read})
    print(f"wrote {a.out}: {read / 1e9:.2f} GB of projections, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
