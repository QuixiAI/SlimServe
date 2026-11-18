#!/usr/bin/env python3
"""Pack a Hugging Face Qwen3.8-27B checkpoint into the q38 engine's weight container.

Works for the BF16 original (every projection stays bf16) and for
compressed-tensors checkpoints such as unsloth/Qwen3.8-27B-NVFP4.

The container is what the native engine memory-maps: every tensor starts on a
16 KiB page boundary and is already in the layout the Metal kernels read, so
load is one mmap plus no-copy MTLBuffers.

  Q38PACK1 | u64 header_len | JSON header (space padded) | page-aligned data

Projection formats (the `fmt` field):
  fp8ch  w: u8 [N, K] raw E4M3 bytes, scale: f32 [N]       (per-row multiplier)
  nvfp4  w: u8 [N, K/2] packed E2M1 nibbles, scale: u8 [N, K/16] raw E4M3,
         gscale: f32 [1] true multiplier (checkpoint stores the divisor)
  bf16   w: bf16 [N, K]

Fusions done here, matching the serving stack's merged linears:
  qkvz    = rows [in_proj_qkv | in_proj_z]
  ba      = rows [in_proj_b | in_proj_a]
  gate_up = rows [gate_proj | up_proj]
  qkv     = rows [q_proj (per-head q|gate) | k_proj | v_proj]

Pure CPU and disk; loads nothing onto the GPU.
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

PAGE = 16384
MAGIC = b"Q38PACK1"
LM = "model.language_model."


class Checkpoint:
    """A Hugging Face safetensors checkpoint: one file or an indexed shard set."""

    def __init__(self, src_dir):
        idx = os.path.join(src_dir, "model.safetensors.index.json")
        files = ["model.safetensors"]
        if os.path.exists(idx):
            files = sorted(set(json.load(open(idx))["weight_map"].values()))
        self.entries = {}
        for fn in files:
            path = os.path.join(src_dir, fn)
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
            header.pop("__metadata__", None)
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            for name, e in header.items():
                self.entries[name] = (mm, 8 + n, e)

    def raw(self, name):
        mm, base, e = self.entries[name]
        a, b = e["data_offsets"]
        return mm[base + a : base + b], e["dtype"], e["shape"]

    def has(self, name):
        return name in self.entries


def bf16_to_f32(raw):
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32)


class Writer:
    def __init__(self, path):
        self.path = path
        self.tmp = path + ".tmp"
        self.f = open(self.tmp, "wb")
        self.tensors = {}
        self.pos = 0  # relative to the data region
        self.sizes = 0

    def add(self, name, dtype, shape, chunks):
        """chunks: iterable of bytes-like, concatenated."""
        pad = (-self.pos) % PAGE
        if pad:
            self.f.write(b"\0" * pad)
            self.pos += pad
        start = self.pos
        for c in chunks:
            b = memoryview(np.ascontiguousarray(c)).cast("B")
            step = 1 << 28
            for i in range(0, len(b), step):
                self.f.write(b[i : i + step])
            self.pos += len(b)
        nbytes = self.pos - start
        item = {"f32": 4, "bf16": 2, "u8": 1}[dtype]
        want = item * int(np.prod(shape))
        if nbytes != want:
            raise SystemExit(f"{name}: wrote {nbytes} bytes, shape {shape} {dtype} wants {want}")
        self.tensors[name] = {"dtype": dtype, "shape": list(shape), "offset": start, "nbytes": nbytes}

    def finish(self, meta):
        pad = (-self.pos) % PAGE
        if pad:
            self.f.write(b"\0" * pad)
        self.f.close()
        header = json.dumps({"meta": meta, "tensors": self.tensors}, separators=(",", ":")).encode()
        total = len(MAGIC) + 8 + len(header)
        header += b" " * ((-total) % PAGE)
        with open(self.path + ".hdr", "wb") as h:
            h.write(MAGIC)
            h.write(struct.pack("<Q", len(header)))
            h.write(header)
        # header + data -> final file (data copied once more; simple and safe)
        with open(self.path + ".hdr", "ab") as out, open(self.tmp, "rb") as src:
            while True:
                buf = src.read(1 << 28)
                if not buf:
                    break
                out.write(buf)
        os.replace(self.path + ".hdr", self.path)
        os.remove(self.tmp)


def proj_parts(ck, names):
    """Describe a fused projection from checkpoint linears `names` (same K).

    Returns (fmt, N, K, w_chunks, scale_chunks, gscale or None).
    """
    fmts = []
    for n in names:
        if ck.has(n + ".weight_packed"):
            fmts.append("nvfp4")
        elif ck.has(n + ".weight_scale"):
            fmts.append("fp8ch")
        else:
            fmts.append("bf16")
    if len(set(fmts)) != 1:
        raise SystemExit(f"mixed formats across fused shards {names}: {fmts}")
    fmt = fmts[0]
    w, s, N, K, g = [], [], 0, None, None
    for n in names:
        if fmt == "nvfp4":
            raw, dt, shp = ck.raw(n + ".weight_packed")
            assert dt == "U8", dt
            k = shp[1] * 2
            sraw, sdt, sshp = ck.raw(n + ".weight_scale")
            assert sdt == "F8_E4M3" and sshp == [shp[0], k // 16], (sdt, sshp)
            graw, gdt, _ = ck.raw(n + ".weight_global_scale")
            div = float(np.frombuffer(graw, dtype="<f4")[0])
            if g is not None and g != div:
                raise SystemExit(f"{names}: fused shards carry different NVFP4 global scales")
            g = div
            w.append(raw)
            s.append(sraw)
        elif fmt == "fp8ch":
            raw, dt, shp = ck.raw(n + ".weight")
            assert dt == "F8_E4M3", dt
            k = shp[1]
            sraw, sdt, sshp = ck.raw(n + ".weight_scale")
            assert sdt == "BF16" and sshp == [shp[0], 1], (sdt, sshp)
            w.append(raw)
            s.append(bf16_to_f32(sraw))
        else:
            raw, dt, shp = ck.raw(n + ".weight")
            assert dt == "BF16", dt
            k = shp[1]
            w.append(raw)
        if K is None:
            K = k
        assert K == k, (names, K, k)
        N += shp[0]
    gscale = None if g is None else np.array([1.0 / g], dtype=np.float32)
    return fmt, N, K, w, s, gscale


def add_proj(wr, ck, out_name, names, projs):
    fmt, N, K, w, s, g = proj_parts(ck, names)
    if fmt == "nvfp4":
        wr.add(out_name + ".w", "u8", [N, K // 2], w)
        wr.add(out_name + ".scale", "u8", [N, K // 16], s)
        wr.add(out_name + ".gscale", "f32", [1], [g])
    elif fmt == "fp8ch":
        wr.add(out_name + ".w", "u8", [N, K], w)
        wr.add(out_name + ".scale", "f32", [N], s)
    else:
        wr.add(out_name + ".w", "bf16", [N, K], w)
    projs[out_name] = {"fmt": fmt, "N": N, "K": K}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="checkpoint directory")
    ap.add_argument("--out", required=True, help="output .q38w path")
    ap.add_argument("--source-name", default=None, help="recorded in the header")
    ap.add_argument("--layers", type=int, default=0, help="pack only the first N layers (smoke)")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.src, "config.json")))
    t = cfg["text_config"]
    ck = Checkpoint(a.src)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    wr = Writer(a.out)
    projs = {}
    n_layers = a.layers or t["num_hidden_layers"]
    layer_types = t["layer_types"][:n_layers]
    t0 = time.time()

    def copy(out_name, src_name, want_dtype, shape=None):
        raw, dt, shp = ck.raw(src_name)
        assert dt == "BF16", (src_name, dt)
        shape = shape or shp
        if want_dtype == "f32":
            wr.add(out_name, "f32", shape, [bf16_to_f32(raw)])
        else:
            wr.add(out_name, "bf16", shape, [raw])

    copy("embed", LM + "embed_tokens.weight", "bf16")
    copy("final_norm", LM + "norm.weight", "bf16")
    add_proj(wr, ck, "lm_head", ["lm_head"], projs)

    for l, kind in enumerate(layer_types):
        p = f"{LM}layers.{l}."
        o = f"L{l}."
        copy(o + "in_norm", p + "input_layernorm.weight", "bf16")
        copy(o + "post_norm", p + "post_attention_layernorm.weight", "bf16")
        if kind == "linear_attention":
            la = p + "linear_attn."
            add_proj(wr, ck, o + "qkvz", [la + "in_proj_qkv", la + "in_proj_z"], projs)
            add_proj(wr, ck, o + "ba", [la + "in_proj_b", la + "in_proj_a"], projs)
            add_proj(wr, ck, o + "gdn_out", [la + "out_proj"], projs)
            conv_dim = t["linear_num_key_heads"] * t["linear_key_head_dim"] * 2 + \
                t["linear_num_value_heads"] * t["linear_value_head_dim"]
            copy(o + "conv_w", la + "conv1d.weight", "f32", [conv_dim, t["linear_conv_kernel_dim"]])
            copy(o + "A_log", la + "A_log", "f32")
            copy(o + "dt_bias", la + "dt_bias", "f32")
            copy(o + "gdn_norm", la + "norm.weight", "bf16")
        else:
            sa = p + "self_attn."
            add_proj(wr, ck, o + "qkv", [sa + "q_proj", sa + "k_proj", sa + "v_proj"], projs)
            add_proj(wr, ck, o + "o", [sa + "o_proj"], projs)
            copy(o + "q_norm", sa + "q_norm.weight", "bf16")
            copy(o + "k_norm", sa + "k_norm.weight", "bf16")
        add_proj(wr, ck, o + "gate_up", [p + "mlp.gate_proj", p + "mlp.up_proj"], projs)
        add_proj(wr, ck, o + "down", [p + "mlp.down_proj"], projs)
        if l % 8 == 7 or l == n_layers - 1:
            print(f"layer {l + 1}/{n_layers}  {wr.pos / 1e9:.1f} GB  {time.time() - t0:.0f}s", flush=True)

    read_bytes = sum(
        wr.tensors[n + ".w"]["nbytes"] + wr.tensors.get(n + ".scale", {"nbytes": 0})["nbytes"]
        for n in projs
    )
    meta = {
        "source": a.source_name or os.path.basename(os.path.normpath(a.src)),
        "num_layers": n_layers,
        "layer_types": ["gdn" if k == "linear_attention" else "attn" for k in layer_types],
        "hidden": t["hidden_size"],
        "inter": t["intermediate_size"],
        "vocab": t["vocab_size"],
        "eps": t["rms_norm_eps"],
        "attn": {
            "heads": t["num_attention_heads"],
            "kv_heads": t["num_key_value_heads"],
            "head_dim": t["head_dim"],
            "rot_dim": int(t["head_dim"] * t["partial_rotary_factor"]),
            "rope_theta": t["rope_parameters"]["rope_theta"],
            "mrope_section": t["rope_parameters"]["mrope_section"],
            "mrope_interleaved": t["rope_parameters"]["mrope_interleaved"],
            "output_gate": t["attn_output_gate"],
        },
        "gdn": {
            "k_heads": t["linear_num_key_heads"],
            "v_heads": t["linear_num_value_heads"],
            "k_dim": t["linear_key_head_dim"],
            "v_dim": t["linear_value_head_dim"],
            "conv_kernel": t["linear_conv_kernel_dim"],
        },
        "eos_token_ids": [248046, 248044],
        "max_position_embeddings": t["max_position_embeddings"],
        "projections": projs,
        "decode_read_bytes": read_bytes,
    }
    wr.finish(meta)
    print(f"wrote {a.out}: {os.path.getsize(a.out) / 1e9:.2f} GB, {len(wr.tensors)} tensors, "
          f"projection bytes read per token {read_bytes / 1e9:.2f} GB, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
