# SPDX-License-Identifier: Apache-2.0
"""Prompt -> DiT text conditioning for LTX-2.5.

Gemma-4 12B (the LTX fine-tune, text tower only) -> all 49 hidden states ->
per-token RMS norm -> aggregate projection (188160 -> 4096 video / 2048 audio)
-> an 8-layer 1-D connector per modality with 128 learnable registers.

Files: the text-encoder safetensors carries the tower, the projection and the
tokenizer (`tokenizer_json`, a U8 tensor); the two connectors ship inside the
DiT file. Weight names are upstream's.

Precision: same policy as the DiT. `operand` is the GEMM/attention dtype
(fp16), `stream` the residual/norm dtype (fp32). The runner computes both in
bf16; operand=stream=bfloat16 reproduces it and is how the port was checked.

Padding: upstream left-pads the prompt to 1024 tokens and runs the tower on
all of them. Real tokens never attend to padding (masked) and RoPE is
relative, so the tower runs on the real tokens only; padded positions are
zeroed by the mask and then replaced by registers either way.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from slimserve.video.ltx25 import checkpoints
from slimserve.video.ltx25.dit import apply_rope, rope_tables

MAX_TOKENS = 1024
_CONNECTORS = (
    ("video", "video_embeddings_connector"),
    ("audio", "audio_embeddings_connector"),
)


def _rms(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    x = x.astype(mx.float32)
    y = x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
    return y if weight is None else y * weight


def _gelu(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


class TextEncoder:
    def __init__(
        self,
        root: Path | None = None,
        variant: str = "distilled",
        operand: mx.Dtype = mx.float16,
        stream: mx.Dtype = mx.float32,
    ):
        self.root, self.variant = root, variant
        self.O, self.S = operand, stream
        self.w: dict[str, mx.array] | None = None  # tower + projection
        self.c: dict[str, mx.array] | None = None  # connectors
        self.tokenizer = None
        self.cfg: dict | None = None
        self.max_abs = None  # set to [] to record max|.| of every GEMM output

    # ---- loading ----------------------------------------------------------
    def load_tokenizer(self) -> None:
        if self.tokenizer is not None:
            return
        from tokenizers import Tokenizer

        raw = checkpoints.load_raw(checkpoints.path_of("text-encoder", self.root))
        self.tokenizer = Tokenizer.from_str(
            bytes(np.array(raw["tokenizer_json"])).decode("utf-8")
        )
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()

    def load(self) -> None:
        if self.w is not None:
            return
        path = checkpoints.path_of("text-encoder", self.root)
        self.cfg = json.loads(checkpoints.read_header(path).metadata["gemma_config"])[
            "text_config"
        ]
        t = self.cfg
        if (
            t["num_kv_shared_layers"]
            or t["use_bidirectional_attention"] == "all"
            or not t["attention_k_eq_v"]
        ):
            raise ValueError("unsupported Gemma-4 text config")
        self.load_tokenizer()
        raw = checkpoints.load_raw(path)
        # vision_model / audio_projector / multi_modal_projector are not on the text
        # path.
        # Pop from the lazy dicts: a surviving reference to the bf16 source keeps
        # it resident next to the cast copy (the load peaked at 2x before this).
        keep = {
            k: raw.pop(k)
            for k in list(raw)
            if k.startswith(("model.", "text_embedding_projection."))
        }
        del raw
        self.w = self._cast(keep)
        dit = checkpoints.load_raw(
            checkpoints.path_of(f"dit-{self.variant}", self.root)
        )
        n = len(checkpoints.DIT_PREFIX)
        conn = {
            k[n:]: dit.pop(k)
            for k in list(dit)
            if k[n:].startswith(checkpoints.CONNECTOR_PREFIXES)
        }
        del dit
        self.c = self._cast(conn)

    def _cast(self, tensors: dict[str, mx.array]) -> dict[str, mx.array]:
        out = checkpoints.cast_operands(tensors, self.O)
        for k in out:  # norm weights and scalars are glue
            if (
                out[k].ndim == 1
                and k.endswith(("norm.weight", "layer_scalar"))
                and ".attn1." not in k
            ):
                out[k] = out[k].astype(mx.float32)
        return out

    def unload(self) -> None:
        self.w = self.c = None
        mx.clear_cache()

    # ---- pieces -----------------------------------------------------------
    def tokenize(self, prompt: str) -> list[int]:
        """`<bos>` + prompt, truncated from the head to MAX_TOKENS.

        The bundled Gemma-4 tokenizer's post-processor adds no special tokens,
        so upstream's LTXGemmaTokenizer prepends BOS itself. Gemma's hidden
        states depend on it at every position, and the aggregate projection and
        connectors were trained on BOS-prefixed states; without it the text
        conditioning is off-distribution (the Mac port has this bug).
        """
        self.load_tokenizer()
        ids = self.tokenizer.encode(prompt.strip()).ids
        bos = self.tokenizer.token_to_id("<bos>")
        if bos is None:
            raise RuntimeError("tokenizer has no <bos>; the encode path requires it")
        if not ids or ids[0] != bos:
            ids = [bos, *ids]
        return ids[:MAX_TOKENS]

    def _lin(self, w: dict[str, mx.array], name: str, x: mx.array) -> mx.array:
        wt, b = w[name + ".weight"], w.get(name + ".bias")
        y = x @ wt.T if b is None else mx.addmm(b, x, wt.T)
        if self.max_abs is not None:
            self.max_abs.append((name, mx.max(mx.abs(y.astype(mx.float32)))))
        return y

    # ---- Gemma-4 tower ----------------------------------------------------
    def _rotary(self, sliding: bool, pos: mx.array):
        """(head_dim, cos, sin) for the layer flavour at fp32 positions `pos`;
        cos/sin are (1, 1, n, head_dim / 2) in the operand dtype."""
        t, op = self.cfg, self.O
        hd = t["head_dim"] if sliding else t["global_head_dim"]
        rp = t["rope_parameters"]["sliding_attention" if sliding else "full_attention"]
        half = hd // 2
        angles = half if sliding else int(rp["partial_rotary_factor"] * hd // 2)
        inv = 1.0 / mx.power(
            mx.array(rp["rope_theta"], dtype=mx.float32),
            mx.arange(0, 2 * angles, 2, dtype=mx.float32) / hd,
        )
        if angles < half:
            inv = mx.concatenate([inv, mx.zeros((half - angles,), dtype=mx.float32)])
        f = pos[:, None] * inv[None, :]
        return hd, mx.cos(f)[None, None].astype(op), mx.sin(f)[None, None].astype(op)

    def _causal_mask(self, n: int, sliding: bool) -> mx.array:
        q_idx, k_idx = mx.arange(n)[:, None], mx.arange(n)[None, :]
        visible = k_idx <= q_idx
        if sliding:
            visible = mx.logical_and(
                visible, k_idx > q_idx - self.cfg["sliding_window"]
            )
        return mx.where(
            visible, mx.array(0.0, dtype=self.O), mx.array(-mx.inf, dtype=self.O)
        )

    @staticmethod
    def _rope(
        x, cos, sin
    ):  # rotate_half form: [a, b] -> [a cos - b sin, b cos + a sin]
        half = x.shape[-1] // 2
        a, b = x[..., :half], x[..., half:]
        return mx.concatenate([a * cos - b * sin, b * cos + a * sin], axis=-1)

    def _layer(self, i: int, h: mx.array, flavors: dict, masks: dict) -> mx.array:
        """One decoder layer on the stream `h` (1, n, D)."""
        w, t, op, S = self.w, self.cfg, self.O, self.S
        eps, heads = t["rms_norm_eps"], t["num_attention_heads"]
        kind = t["layer_types"][i]
        sliding = kind == "sliding_attention"
        p = f"model.layers.{i}"
        n = h.shape[1]
        hd, cos, sin = flavors[sliding]
        x = _rms(h, w[p + ".input_layernorm.weight"], eps).astype(op)
        q = self._lin(w, p + ".self_attn.q_proj", x).reshape(1, n, heads, hd)
        k = self._lin(w, p + ".self_attn.k_proj", x)
        kv_heads = k.shape[-1] // hd
        k = k.reshape(1, n, kv_heads, hd)
        vkey = p + ".self_attn.v_proj"
        v = (
            self._lin(w, vkey, x).reshape(1, n, kv_heads, hd)
            if vkey + ".weight" in w
            else k
        )
        q = self._rope(
            _rms(q, w[p + ".self_attn.q_norm.weight"], eps)
            .astype(op)
            .transpose(0, 2, 1, 3),
            cos,
            sin,
        )
        k = self._rope(
            _rms(k, w[p + ".self_attn.k_norm.weight"], eps)
            .astype(op)
            .transpose(0, 2, 1, 3),
            cos,
            sin,
        )
        v = _rms(v, None, eps).astype(op).transpose(0, 2, 1, 3)
        mask = masks[sliding]
        length = k.shape[2]
        if kv_heads != heads:
            rep = heads // kv_heads
            k = mx.broadcast_to(k[:, :, None], (1, kv_heads, rep, length, hd)).reshape(
                1, heads, length, hd
            )
            v = mx.broadcast_to(v[:, :, None], (1, kv_heads, rep, length, hd)).reshape(
                1, heads, length, hd
            )
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0, mask=mask)
        y = self._lin(
            w,
            p + ".self_attn.o_proj",
            y.transpose(0, 2, 1, 3).reshape(1, n, heads * hd),
        )
        h = h + _rms(y, w[p + ".post_attention_layernorm.weight"], eps).astype(S)
        x = _rms(h, w[p + ".pre_feedforward_layernorm.weight"], eps).astype(op)
        y = _gelu(self._lin(w, p + ".mlp.gate_proj", x)) * self._lin(
            w, p + ".mlp.up_proj", x
        )
        y = self._lin(w, p + ".mlp.down_proj", y)
        return (
            h + _rms(y, w[p + ".post_feedforward_layernorm.weight"], eps).astype(S)
        ) * w[p + ".layer_scalar"].astype(S)

    def _embed(self, ids: list[int]) -> mx.array:
        t, S = self.cfg, self.S
        # Upstream holds embed_scale in the model dtype, bf16: sqrt(3840) rounds to
        # 62.0.
        scale = mx.array(math.sqrt(t["hidden_size"])).astype(mx.bfloat16).astype(S)
        return (
            self.w["model.embed_tokens.weight"][mx.array(ids)][None].astype(S) * scale
        )

    def _tower(self, ids: list[int], pad: int) -> list[mx.array]:
        """Hidden states (embeddings + every layer; the last one final-normed), stream
        dtype."""
        w, t = self.w, self.cfg
        n = len(ids)
        h = self._embed(ids)
        pos = mx.arange(pad, pad + n).astype(mx.float32)
        flavors = {True: self._rotary(True, pos), False: self._rotary(False, pos)}
        masks = {True: self._causal_mask(n, True), False: self._causal_mask(n, False)}
        states = [h]
        for i in range(len(t["layer_types"])):
            h = self._layer(i, h, flavors, masks)
            mx.eval(h)
            states.append(h)
        states[-1] = _rms(states[-1], w["model.norm.weight"], t["rms_norm_eps"]).astype(
            self.S
        )
        return states

    def _connector(self, name: str, x: mx.array, n_valid: int) -> mx.array:
        """x: (1, n_valid, dim) stream dtype -> (1, MAX_TOKENS, dim) fp32."""
        c, op, S = self.c, self.O, self.S
        reg = c[name + ".learnable_registers"].astype(S)
        heads = c[f"{name}.transformer_1d_blocks.0.attn1.to_gate_logits.weight"].shape[
            0
        ]
        dim = x.shape[-1]
        tiled = mx.tile(reg, (MAX_TOKENS // reg.shape[0], 1))
        x = mx.concatenate([x[0], tiled[n_valid:]], axis=0)[
            None
        ]  # prompt first, registers fill the padding
        pos = mx.arange(MAX_TOKENS).astype(mx.float32)[None, :, None]
        cos, sin = (
            r.astype(op) for r in rope_tables(pos, dim, heads, (4096,), 10000.0)
        )
        hd = dim // heads
        split = lambda z: z.reshape(1, MAX_TOKENS, heads, hd).transpose(0, 2, 1, 3)  # noqa: E731
        i = 0
        while f"{name}.transformer_1d_blocks.{i}.attn1.to_q.weight" in c:
            p = f"{name}.transformer_1d_blocks.{i}"
            z = mx.fast.rms_norm(x, None, 1e-6).astype(op)
            # Upstream Attention(norm_eps=1e-6); the Mac port's nn.RMSNorm default
            # (1e-5) was a port deviation.
            q = mx.fast.rms_norm(
                self._lin(c, p + ".attn1.to_q", z), c[p + ".attn1.q_norm.weight"], 1e-6
            )
            k = mx.fast.rms_norm(
                self._lin(c, p + ".attn1.to_k", z), c[p + ".attn1.k_norm.weight"], 1e-6
            )
            v = self._lin(c, p + ".attn1.to_v", z)
            y = mx.fast.scaled_dot_product_attention(
                apply_rope(split(q), cos, sin),
                apply_rope(split(k), cos, sin),
                split(v),
                scale=hd**-0.5,
            )
            gate = 2.0 * mx.sigmoid(self._lin(c, p + ".attn1.to_gate_logits", z))
            y = (
                (y * gate.transpose(0, 2, 1)[..., None])
                .transpose(0, 2, 1, 3)
                .reshape(1, MAX_TOKENS, dim)
            )
            x = x + self._lin(c, p + ".attn1.to_out.0", y).astype(S)
            z = mx.fast.rms_norm(x, None, 1e-6).astype(op)
            x = x + self._lin(
                c, p + ".ff.net.2", _gelu(self._lin(c, p + ".ff.net.0.proj", z))
            ).astype(S)
            mx.eval(x)
            i += 1
        return mx.fast.rms_norm(x, None, 1e-6).astype(mx.float32)

    # ---- public -----------------------------------------------------------
    def encode(self, prompt: str) -> tuple[mx.array, mx.array]:
        """(video_context (1, 1024, 4096), audio_context (1, 1024, 2048)), fp32.

        No attention mask is needed downstream: padding is replaced by
        registers, every one of the 1024 context tokens is attended.
        """
        self.load()
        ids = self.tokenize(prompt)
        n = len(ids)
        pad = MAX_TOKENS - n
        states = self._tower(ids, pad)
        enc = mx.stack(states, axis=-1)  # (1, n, D, L)
        enc = enc * mx.rsqrt(mx.mean(enc * enc, axis=2, keepdims=True) + 1e-6)
        flat = enc.reshape(1, n, -1)
        out = []
        for (_, cname), d in zip(_CONNECTORS, (4096, 2048)):
            head = "video" if d == 4096 else "audio"
            x = (flat * math.sqrt(d / self.cfg["hidden_size"])).astype(self.O)
            x = self._lin(
                self.w, f"text_embedding_projection.{head}_aggregate_embed", x
            ).astype(self.S)
            out.append(self._connector(cname, x, n))
        mx.eval(out)
        return out[0], out[1]
