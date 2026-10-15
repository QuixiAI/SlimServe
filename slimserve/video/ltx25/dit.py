# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 audio+video diffusion transformer forward (AVTransformer3DModel).

48 joint blocks; per block: video and audio self-attention (split RoPE,
qk RMSNorm, per-head gate), text cross-attention with prompt AdaLN,
bidirectional audio<->video cross-attention with their own AdaLN and gates,
and a GELU feed-forward per modality.

Precision (perf/ltx25_metal_campaign.md section 9): the residual stream, the
norms, the AdaLN tables and modulation, RoPE tables and x0 recovery are fp32;
every GEMM and attention operand is fp16. Latents, text embeddings and the
timestep enter in fp32. `F` below is the operand dtype, `G` the glue dtype.

Weights are addressed by their upstream names (see checkpoints.load_dit), so
this file reads against Lightricks' `ltx_core/model/transformer/`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

F = mx.float16
G = mx.float32
SPLIT_K_MIN = 16384
SPLIT_K_CHUNK = 2048


@dataclass(frozen=True)
class DiTConfig:
    num_layers: int = 48
    video_dim: int = 4096
    audio_dim: int = 2048
    heads: int = 32
    audio_heads: int = 32
    timestep_dim: int = 256
    timestep_scale: float = 1000.0
    av_ca_timestep_scale: float = 1000.0
    rope_theta: float = 10000.0
    max_pos: tuple[int, ...] = (20, 2048, 2048)
    audio_max_pos: tuple[int, ...] = (20,)
    norm_eps: float = 1e-6

    @classmethod
    def from_checkpoint(cls, t: dict[str, Any]) -> DiTConfig:
        unsupported = {
            "rope_type": "split",
            "qk_norm": "rms_norm",
            "apply_gated_attention": True,
            "cross_attention_adaln": True,
            "av_cross_ada_norm": True,
            "frequencies_precision": "float64",
            "share_ff": False,
            "ff_bias": False,
        }
        for key, want in unsupported.items():
            if t.get(key) != want:
                raise ValueError(
                    f"LTX-2.5 DiT config {key}={t.get(key)!r}, "
                    f"engine implements {want!r}"
                )
        return cls(
            num_layers=t["num_layers"],
            video_dim=t["cross_attention_dim"],
            audio_dim=t["audio_cross_attention_dim"],
            heads=t["num_attention_heads"],
            audio_heads=t["audio_num_attention_heads"],
            timestep_scale=float(t["timestep_scale_multiplier"]),
            av_ca_timestep_scale=float(t["av_ca_timestep_scale_multiplier"]),
            rope_theta=float(t["positional_embedding_theta"]),
            max_pos=tuple(t["positional_embedding_max_pos"]),
            audio_max_pos=tuple(t["audio_positional_embedding_max_pos"]),
            norm_eps=float(t["norm_eps"]),
        )


def timestep_embedding(t: mx.array, dim: int) -> mx.array:
    """Sinusoidal [cos, sin] embedding of a 1-D fp32 timestep array."""
    half = dim // 2
    freqs = mx.exp(-math.log(10000.0) * mx.arange(half).astype(G) / half)
    args = t.astype(G)[:, None] * freqs[None, :]
    return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


def rope_tables(
    positions: mx.array,
    inner_dim: int,
    heads: int,
    max_pos: tuple[int, ...],
    theta: float,
) -> tuple[mx.array, mx.array]:
    """Split-RoPE cos/sin, shape (B, heads, N, head_dim / 2), fp32.

    Log-spaced frequency grid computed in fp64 (the checkpoint's
    `frequencies_precision`), fractional positions in fp32, as upstream.
    """
    b, n, axes = positions.shape
    num_freqs = inner_dim // (2 * axes)
    grid = theta ** np.linspace(0.0, 1.0, num_freqs, dtype=np.float64) * (math.pi / 2.0)
    grid = mx.array(grid.astype(np.float32))
    frac = positions.astype(G) / mx.array(list(max_pos[:axes]), dtype=G)
    scaled = grid * (frac[..., None] * 2.0 - 1.0)  # (B, N, axes, num_freqs)
    freqs = scaled.transpose(0, 1, 3, 2).reshape(b, n, -1)
    pad = inner_dim // 2 - freqs.shape[-1]
    if pad > 0:
        freqs = mx.concatenate([mx.zeros((b, n, pad), dtype=G), freqs], axis=-1)
    half = inner_dim // (2 * heads)
    cos = mx.cos(freqs).reshape(b, n, heads, half).transpose(0, 2, 1, 3)
    sin = mx.sin(freqs).reshape(b, n, heads, half).transpose(0, 2, 1, 3)
    return cos, sin


def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


# Elementwise chains, compiled so MLX emits one fused kernel pass for each
# instead of one pass (and one temporary) per arithmetic op.
@mx.compile
def _modulate(xn: mx.array, scale: mx.array, shift: mx.array) -> mx.array:
    return (xn * (1.0 + scale) + shift).astype(F)


@mx.compile
def _gated_residual(x: mx.array, y: mx.array, gate: mx.array) -> mx.array:
    return x + y * gate


@mx.compile
def _gelu_tanh(h: mx.array) -> mx.array:
    return 0.5 * h * (1.0 + mx.tanh(0.7978845608028654 * (h + 0.044715 * h * h * h)))


@mx.compile
def _head_gate(out: mx.array, logits: mx.array) -> mx.array:
    return out * (2.0 * mx.sigmoid(logits)).transpose(0, 2, 1)[..., None]


def _rows(x: mx.array, reps: mx.array | None) -> mx.array:
    return x if reps is None else mx.take(x, reps, axis=0)


def _select_rows(inputs: dict, rows: list[int], batch: int) -> dict:
    """The same inputs restricted to batch rows `rows` (arrays whose leading
    axis is the batch; text stays as its distinct rows, `text_rows` is
    re-indexed; stg masks are sliced)."""
    idx = mx.array(rows)
    out = {}
    for key, val in inputs.items():
        if key in ("video_text", "audio_text") or val is None:
            out[key] = val
        elif key == "stg":
            out[key] = {k: mx.take(m, idx) for k, m in val.items()}
        elif isinstance(val, mx.array) and val.ndim >= 1 and val.shape[0] == batch:
            out[key] = mx.take(val, idx, axis=0)
        else:
            out[key] = val
    return out


class StepCache:
    """First-block cache (the FBCache formulation): a sampler step whose block-0
    residual (hidden after block 0 minus the embedded input) has moved less
    than `threshold` in relative L1, accumulated since the last computed step,
    skips blocks 1..N-1 and reuses that step's residual over block 0. Output-
    changing; one per sampler loop. `force` marks a step that must compute."""

    def __init__(self, threshold: float):
        self.threshold = float(threshold)
        self.ref: mx.array | None = None
        self.resid: tuple[mx.array, mx.array] | None = None
        self.acc = 0.0
        self.force = False
        self.computed = 0
        self.skipped = 0
        self.history: list[tuple[float, bool]] = []  # (rel-L1, skipped) per step
        self._signal: mx.array | None = None

    def hit(self, signal: mx.array) -> bool:
        self._signal = signal
        if self.ref is None:
            return False
        rel = float(
            (mx.mean(mx.abs(signal - self.ref)) / mx.mean(mx.abs(self.ref))).item()
        )
        skip = not self.force and self.acc + rel < self.threshold
        self.history.append((rel, skip))
        if skip:
            self.acc += rel
            self.skipped += 1
        return skip

    def reuse(self, v1: mx.array, a1: mx.array | None) -> tuple[mx.array, mx.array]:
        rv, ra = self.resid
        return v1 + rv, None if a1 is None else a1 + ra

    def store(self, v1, a1, v, a) -> None:
        self.ref, self.acc, self.computed = self._signal, 0.0, self.computed + 1
        self.resid = (v - v1, None if a is None else a - a1)
        mx.eval(self.ref, *(r for r in self.resid if r is not None))


def attention_tiles(
    f: int,
    h: int,
    w: int,
    tiles: tuple[int, int],
    halo: int,
    total: int | None = None,
) -> tuple[list[mx.array], list[mx.array], mx.array]:
    """Index sets for tiled video self-attention: the (f, h, w) token grid is cut
    into tiles[0] x tiles[1] spatial tiles (every frame); a tile's queries attend
    to the keys of its own rows and columns widened by `halo` latent cells.
    Tokens past f*h*w (DFR slots, reference tokens) are keys of every tile and
    attend to everything. Returns (query indices, key indices) per tile and the
    permutation that puts the concatenated tile outputs back in token order."""
    n = f * h * w
    total = n if total is None else total
    grid = np.arange(n).reshape(f, h, w)
    extra = np.arange(n, total)
    th, tw = tiles
    hb = [round(i * h / th) for i in range(th + 1)]
    wb = [round(i * w / tw) for i in range(tw + 1)]
    queries, keys = [], []
    for i in range(th):
        for j in range(tw):
            q = grid[:, hb[i] : hb[i + 1], wb[j] : wb[j + 1]].reshape(-1)
            k = grid[
                :,
                max(hb[i] - halo, 0) : min(hb[i + 1] + halo, h),
                max(wb[j] - halo, 0) : min(wb[j + 1] + halo, w),
            ].reshape(-1)
            queries.append(q)
            keys.append(np.concatenate([k, extra]))
    if len(extra):
        queries.append(extra)
        keys.append(np.arange(total))
    order = np.concatenate(queries)
    inverse = np.argsort(order).astype(np.int32)
    return (
        [mx.array(q.astype(np.int32)) for q in queries],
        [mx.array(k.astype(np.int32)) for k in keys],
        mx.array(inverse),
    )


class Modulation:
    """One AdaLN head's output: (rows, P, dim) fp32, optionally per-token.

    With per-token timesteps (conditioning tokens at a different sigma) the
    timestep takes a handful of distinct values, so the head runs on the
    distinct rows only and each parameter is gathered at its point of use.
    """

    __slots__ = ("rows", "inverse", "batch", "tokens")

    def __init__(
        self, rows: mx.array, inverse: mx.array | None, batch: int, tokens: int
    ):
        self.rows, self.inverse, self.batch, self.tokens = rows, inverse, batch, tokens

    def get(self, i: int, table_row: mx.array | None = None) -> mx.array:
        p = self.rows[:, i, :]
        if table_row is not None:
            p = p + table_row
        if self.inverse is None:
            return p[:, None, :]
        return mx.take(p, self.inverse, axis=0).reshape(self.batch, self.tokens, -1)


class LTX25DiT:
    """Functional forward over the official weight dict."""

    def __init__(
        self, weights: dict[str, mx.array], config: DiTConfig, eval_every: int = 8
    ):
        self.w = weights
        self.cfg = config
        self.eval_every = eval_every
        # MLX's fp16 GEMM drops from ~19 to 10 TF/s (6 at 24k rows) once K
        # reaches 16384 (the video FFN's second linear). Summing K-chunks keeps
        # every GEMM in the fast regime: 79 -> 45 ms at 6,144 rows, 526 -> 172
        # ms at 24,576. The chunks replace the original, so memory is unchanged.
        self.split_k: dict[str, list[mx.array]] = {}
        for name in [
            n
            for n, a in weights.items()
            if n.endswith(".weight") and a.ndim == 2 and a.shape[1] >= SPLIT_K_MIN
        ]:
            a = weights.pop(name)
            parts = [
                mx.contiguous(a[:, i : i + SPLIT_K_CHUNK])
                for i in range(0, a.shape[1], SPLIT_K_CHUNK)
            ]
            mx.eval(parts)
            self.split_k[name[: -len(".weight")]] = parts
        # name -> [(A, B, scale)]: runtime low-rank terms, y += scale * (x A^T) B^T.
        self.lora: dict[str, list[tuple[mx.array, mx.array, float]]] = {}

    # ---- primitives -------------------------------------------------------
    def lin(self, name: str, x: mx.array) -> mx.array:
        """fp16 GEMM. `x` must already be fp16 (cast once by the caller)."""
        parts = self.split_k.get(name)
        if parts is not None:
            y = x[..., :SPLIT_K_CHUNK] @ parts[0].T
            for i, part in enumerate(parts[1:], 1):
                y = y + x[..., i * SPLIT_K_CHUNK : (i + 1) * SPLIT_K_CHUNK] @ part.T
            b = self.w.get(name + ".bias")
            if b is not None:
                y = y + b
        else:
            w = self.w[name + ".weight"]
            b = self.w.get(name + ".bias")
            y = x @ w.T if b is None else mx.addmm(b, x, w.T)
        for a_mat, b_mat, scale in self.lora.get(name, ()):
            y = y + ((x @ a_mat.T) @ b_mat.T) * scale
        return y

    def _adaln(
        self, name: str, t_emb: mx.array, num_params: int, dim: int
    ) -> tuple[mx.array, mx.array]:
        e = self.lin(f"{name}.emb.timestep_embedder.linear_1", t_emb.astype(F))
        e = self.lin(f"{name}.emb.timestep_embedder.linear_2", e * mx.sigmoid(e))
        p = self.lin(f"{name}.linear", e * mx.sigmoid(e))
        return p.astype(G).reshape(-1, num_params, dim), e.astype(G)

    def _modulation(
        self,
        names: list[tuple[str, int, int]],
        t: mx.array,
        per_token: mx.array | None,
        scale: float,
    ) -> list[tuple[Modulation, Modulation]]:
        """Run AdaLN heads on one timestep source. Returns (params, embedded) per
        head."""
        if per_token is None:
            emb = timestep_embedding(t * scale, self.cfg.timestep_dim)
            inverse, b, n = None, int(t.shape[0]), 1
        else:
            b, n = per_token.shape
            uniq, inv = np.unique(
                np.array(per_token.astype(G)).reshape(-1), return_inverse=True
            )
            emb = timestep_embedding(mx.array(uniq) * scale, self.cfg.timestep_dim)
            inverse = mx.array(inv.astype(np.int32))
        out = []
        for name, num_params, dim in names:
            p, e = self._adaln(name, emb, num_params, dim)
            out.append(
                (Modulation(p, inverse, b, n), Modulation(e[:, None, :], inverse, b, n))
            )
        return out

    def _attn(
        self,
        p: str,
        x: mx.array,
        ctx: mx.array | None = None,
        rope_q: tuple[mx.array, mx.array] | None = None,
        rope_k: tuple[mx.array, mx.array] | None = None,
        mask: mx.array | None = None,
        skip: mx.array | None = None,
        ctx_rows: mx.array | None = None,
        tiles: tuple[list[mx.array], list[mx.array], mx.array] | None = None,
    ) -> mx.array:
        """x, ctx are fp16. Returns fp16 (B, N, out_dim).

        `ctx` may hold only the distinct context rows, with `ctx_rows` mapping
        each batch row to one of them: K and V are projected once per distinct
        row and gathered (the dev pipeline's three conditional-text passes).
        `tiles` (from `attention_tiles`) restricts each query tile to its own
        keys: one attention call per tile, outputs put back in token order.
        """
        w = self.w
        heads = w[p + ".to_gate_logits.weight"].shape[0]
        b = x.shape[0]
        kv = x if ctx is None else ctx
        q = mx.fast.rms_norm(
            self.lin(p + ".to_q", x), w[p + ".q_norm.weight"], self.cfg.norm_eps
        )
        k = mx.fast.rms_norm(
            self.lin(p + ".to_k", kv), w[p + ".k_norm.weight"], self.cfg.norm_eps
        )
        v = self.lin(p + ".to_v", kv)
        hd = q.shape[-1] // heads
        q = q.reshape(b, -1, heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(kv.shape[0], -1, heads, hd).transpose(0, 2, 1, 3)
        v = v.reshape(kv.shape[0], -1, heads, hd).transpose(0, 2, 1, 3)
        if ctx_rows is not None:
            k, v = mx.take(k, ctx_rows, axis=0), mx.take(v, ctx_rows, axis=0)
        if rope_q is not None:
            q = apply_rope(q, *rope_q)
            k = apply_rope(k, *(rope_k or rope_q))
        if mask is not None and mask.dtype != mx.bool_:
            mask = mask.astype(F)
        if tiles is not None and mask is None:
            qi, ki, inverse = tiles
            parts = [
                mx.fast.scaled_dot_product_attention(
                    mx.take(q, i, axis=2),
                    mx.take(k, j, axis=2),
                    mx.take(v, j, axis=2),
                    scale=hd**-0.5,
                )
                for i, j in zip(qi, ki)
            ]
            out = mx.take(mx.concatenate(parts, axis=2), inverse, axis=2)
        else:
            out = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=hd**-0.5, mask=mask
            )
        if skip is not None:  # STG: perturbed samples pass the values through
            out = out * skip + v * (1.0 - skip)
        out = _head_gate(
            out, self.lin(p + ".to_gate_logits", x)
        )  # logits (B, N, heads)
        out = out.transpose(0, 2, 1, 3).reshape(b, -1, heads * hd)
        return self.lin(p + ".to_out.0", out)

    def _ff(self, p: str, x: mx.array) -> mx.array:
        # gelu-approximate (tanh form), as the checkpoint's activation_fn.
        return self.lin(p + ".net.2", _gelu_tanh(self.lin(p + ".net.0.proj", x)))

    def _norm(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.cfg.norm_eps)

    # ---- block ------------------------------------------------------------
    def _block(
        self, i: int, v: mx.array, a: mx.array, s: dict[str, Any]
    ) -> tuple[mx.array, mx.array]:
        w = self.w
        p = f"transformer_blocks.{i}"
        vt, at = w[p + ".scale_shift_table"], w[p + ".audio_scale_shift_table"]
        vm, am = s["video_mod"], s["audio_mod"]
        stg = s["stg"]
        # A modality a guider skips this step (upstream Modality.enabled False,
        # BasicTransformerBlock run_vx / run_ax) keeps its hidden state as it
        # entered the block: no self-attention, text cross-attention, cross
        # update or feed-forward for it, while the other modality still reads
        # it through the cross-attention.
        run_v, run_a = s["run_video"], s["run_audio"]
        has_v, has_a = s["has_video"], s["has_audio"]

        def skip(kind: str, like_ndim: int) -> mx.array | None:
            m = stg.get((kind, i)) if stg else None
            return None if m is None else m.reshape([-1] + [1] * (like_ndim - 1))

        # 1. video self-attention (table rows 0..2 = shift, scale, gate)
        if run_v:
            x = _modulate(self._norm(v), vm.get(1, vt[1]), vm.get(0, vt[0]))
            sk = skip("video_self", 4)
            y = self._attn(
                p + ".attn1",
                x,
                rope_q=s["video_rope"],
                mask=s["video_mask"],
                skip=None if sk is None else sk.astype(F),
                tiles=s["video_tiles"],
            )
            v = _gated_residual(v, y, vm.get(2, vt[2]))

        # 2. audio self-attention
        if run_a:
            x = _modulate(self._norm(a), am.get(1, at[1]), am.get(0, at[0]))
            sk = skip("audio_self", 4)
            y = self._attn(
                p + ".audio_attn1",
                x,
                rope_q=s["audio_rope"],
                mask=s["audio_mask"],
                skip=None if sk is None else sk.astype(F),
            )
            a = _gated_residual(a, y, am.get(2, at[2]))

        # 3. video text cross-attention (rows 6..8; prompt table modulates the text)
        if run_v and s["video_text"] is not None:
            x = _modulate(self._norm(v), vm.get(7, vt[7]), vm.get(6, vt[6]))
            pt, pm = w[p + ".prompt_scale_shift_table"], s["video_prompt_mod"]
            text = (
                s["video_text"] * (1.0 + _rows(pm.get(1, pt[1]), s["text_reps"]))
                + _rows(pm.get(0, pt[0]), s["text_reps"])
            ).astype(F)
            y = self._attn(
                p + ".attn2",
                x,
                ctx=text,
                mask=s["video_cross_mask"],
                ctx_rows=s["text_rows"],
            )
            v = _gated_residual(v, y, vm.get(8, vt[8]))

        # 4. audio text cross-attention
        if run_a and s["audio_text"] is not None:
            x = _modulate(self._norm(a), am.get(7, at[7]), am.get(6, at[6]))
            pt, pm = w[p + ".audio_prompt_scale_shift_table"], s["audio_prompt_mod"]
            text = (
                s["audio_text"] * (1.0 + _rows(pm.get(1, pt[1]), s["text_reps"]))
                + _rows(pm.get(0, pt[0]), s["text_reps"])
            ).astype(F)
            y = self._attn(p + ".audio_attn2", x, ctx=text, ctx_rows=s["text_rows"])
            a = _gated_residual(a, y, am.get(8, at[8]))

        # 5-6. audio<->video cross attention; both directions read the same norms.
        # Table rows: 0 scale_a2v, 1 shift_a2v, 2 scale_v2a, 3 shift_v2a, 4 gate.
        cvt, cat = (
            w[p + ".scale_shift_table_a2v_ca_video"],
            w[p + ".scale_shift_table_a2v_ca_audio"],
        )
        cvm, cam = s["av_video_mod"], s["av_audio_mod"]
        vn, an = (
            (None if v is None else self._norm(v)),
            (None if a is None else self._norm(a)),
        )
        v_new = v
        if run_v and has_a:
            vq = _modulate(vn, cvm.get(0, cvt[0]), cvm.get(1, cvt[1]))
            akv = _modulate(an, cam.get(0, cat[0]), cam.get(1, cat[1]))
            y = self._attn(
                p + ".audio_to_video_attn",
                vq,
                ctx=akv,
                rope_q=s["video_cross_rope"],
                rope_k=s["audio_cross_rope"],
            )
            y = y * s["a2v_gate_mod"].get(0, cvt[4])
            sk = skip("a2v", 3)
            v_new = v + (y if sk is None else y * sk)

        if run_a and has_v:
            aq = _modulate(an, cam.get(2, cat[2]), cam.get(3, cat[3]))
            vkv = _modulate(vn, cvm.get(2, cvt[2]), cvm.get(3, cvt[3]))
            y = self._attn(
                p + ".video_to_audio_attn",
                aq,
                ctx=vkv,
                rope_q=s["audio_cross_rope"],
                rope_k=s["video_cross_rope"],
            )
            y = y * s["v2a_gate_mod"].get(0, cat[4])
            sk = skip("v2a", 3)
            a = a + (y if sk is None else y * sk)
        v = v_new

        # 7-8. feed-forward (rows 3..5)
        if run_v:
            x = _modulate(self._norm(v), vm.get(4, vt[4]), vm.get(3, vt[3]))
            v = _gated_residual(v, self._ff(p + ".ff", x), vm.get(5, vt[5]))
        if run_a:
            x = _modulate(self._norm(a), am.get(4, at[4]), am.get(3, at[3]))
            a = _gated_residual(a, self._ff(p + ".audio_ff", x), am.get(5, at[5]))
        return v, a

    # ---- model ------------------------------------------------------------
    def __call__(
        self,
        video_latent: mx.array | None,
        audio_latent: mx.array | None,
        timestep: mx.array,
        video_text: mx.array | None,
        audio_text: mx.array | None,
        video_positions: mx.array | None,
        audio_positions: mx.array | None,
        video_keyframes_mask: mx.array | None = None,
        video_timesteps: mx.array | None = None,
        audio_timesteps: mx.array | None = None,
        video_sigma: mx.array | None = None,
        audio_sigma: mx.array | None = None,
        video_attention_mask: mx.array | None = None,
        audio_attention_mask: mx.array | None = None,
        video_cross_attention_mask: mx.array | None = None,
        stg: dict[tuple[str, int], mx.array] | None = None,
        text_rows: mx.array | None = None,
        share_from: tuple[int, int, int] | None = None,
        video_tiles: tuple[list[mx.array], list[mx.array], mx.array] | None = None,
        step_cache: StepCache | None = None,
        run_video: bool = True,
        run_audio: bool = True,
    ) -> tuple[mx.array | None, mx.array | None]:
        """Velocity prediction. All inputs fp32; returns fp32 (video, audio).

        `stg` maps (kind, block) -> (B,) keep-mask (1 keep, 0 skip) with kind in
        video_self / audio_self / a2v / v2a. `text_rows` lets `video_text` /
        `audio_text` carry only their distinct rows, indexed per batch row.
        `share_from = (block, src, dst)`: batch row `dst` is identical to row
        `src` until `block` (an STG pass skips attention only from that block
        on), so rows other than `dst` run blocks 0..block-1 and `dst` is
        forked from `src`'s hidden state there. Exact up to GEMM row-count
        kernel selection. `video_tiles` and `step_cache` are the fast tier
        (tiled video self-attention, the first-block step cache); both change
        the output. `run_video` / `run_audio` False: that modality is a guider's
        skipped step (upstream Modality.enabled): its stream is left as its
        patchified input through every block, the other modality still
        cross-attends to it, and its output is None.
        """
        if not (run_video or run_audio):
            raise ValueError("at least one modality must run")
        if video_latent is None and audio_latent is None:
            raise ValueError("a forward needs video or audio")
        if video_latent is None and (share_from is not None or video_tiles is not None):
            raise ValueError("an audio-only forward takes no STG fork or video tiles")
        inputs = dict(
            video_latent=video_latent,
            audio_latent=audio_latent,
            timestep=timestep,
            video_text=video_text,
            audio_text=audio_text,
            video_positions=video_positions,
            audio_positions=audio_positions,
            video_keyframes_mask=video_keyframes_mask,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
            video_sigma=video_sigma,
            audio_sigma=audio_sigma,
            video_attention_mask=video_attention_mask,
            audio_attention_mask=audio_attention_mask,
            video_cross_attention_mask=video_cross_attention_mask,
            stg=stg,
            text_rows=text_rows,
            video_tiles=video_tiles,
            run_video=run_video,
            run_audio=run_audio,
        )
        fork_block = self.cfg.num_layers
        if share_from is not None and share_from[0] == 0:
            share_from = None
        if share_from is not None:
            fork_block, src, dst = share_from
            b = (audio_latent if video_latent is None else video_latent).shape[0]
            keep = [r for r in range(b) if r != dst]
            v, a, state, _, _ = self._prepare(**_select_rows(inputs, keep, b))
            src_pos = keep.index(src)

            def expand(x: mx.array) -> mx.array:  # fork row dst from row src
                return mx.concatenate(
                    [x[:dst], x[src_pos : src_pos + 1], x[dst:]], axis=0
                )

        else:
            v, a, state, video_emb, audio_emb = self._prepare(**inputs)
            expand = lambda x: x  # noqa: E731
        v_in = v
        v, a = self._block(0, v, a, state)
        v1 = a1 = None
        if step_cache is not None and (not run_video or v is None):
            step_cache = None  # the cache keys on the video residual
        if step_cache is not None:
            v1, a1 = expand(v), expand(a)
            if step_cache.hit(v - v_in):
                v, a = step_cache.reuse(v1, a1)
                if share_from is not None:
                    _, _, _, video_emb, audio_emb = self._prepare(**inputs)
                return self._out_both(v, a, video_emb, audio_emb, run_video, run_audio)
        for i in range(1, self.cfg.num_layers):
            if i == fork_block:
                v, a = expand(v), expand(a)
                _, _, state, video_emb, audio_emb = self._prepare(**inputs)
            v, a = self._block(i, v, a, state)
            # Bound each Metal command buffer (GPU watchdog) and the live graph
            # without stalling: the GPU runs this group while Python builds the
            # next one.
            if self.eval_every and (i + 1) % self.eval_every == 0:
                mx.async_eval(v, a)
        if step_cache is not None:
            step_cache.store(v1, a1, v, a)
        return self._out_both(v, a, video_emb, audio_emb, run_video, run_audio)

    def _out_both(
        self, v, a, video_emb, audio_emb, run_video=True, run_audio=True
    ) -> tuple[mx.array | None, mx.array | None]:
        w = self.w
        return (
            self._out(v, video_emb, w["scale_shift_table"], "proj_out")
            if run_video and v is not None
            else None,
            self._out(a, audio_emb, w["audio_scale_shift_table"], "audio_proj_out")
            if run_audio and a is not None
            else None,
        )

    def _prepare(
        self,
        video_latent,
        audio_latent,
        timestep,
        video_text,
        audio_text,
        video_positions,
        audio_positions,
        video_keyframes_mask=None,
        video_timesteps=None,
        audio_timesteps=None,
        video_sigma=None,
        audio_sigma=None,
        video_attention_mask=None,
        audio_attention_mask=None,
        video_cross_attention_mask=None,
        stg=None,
        text_rows=None,
        video_tiles=None,
        run_video=True,
        run_audio=True,
    ):
        """Patchify, AdaLN heads, RoPE tables: everything the block loop reads."""
        cfg, w = self.cfg, self.w
        vd, ad = cfg.video_dim, cfg.audio_dim
        timestep = timestep.astype(G)

        no_video = video_latent is None  # an audio-only forward (T2A)
        no_audio = audio_latent is None  # a video-only forward (alpha-gen)
        if no_video:
            v = None
        else:
            v = self.lin("patchify_proj", video_latent.astype(F)).astype(G)
            if video_keyframes_mask is not None:
                v = v + (video_keyframes_mask > 0).astype(G) * w[
                    "keyframes_abs_pos_embedding"
                ].astype(G)
        a = (
            None
            if no_audio
            else self.lin("audio_patchify_proj", audio_latent.astype(F)).astype(G)
        )

        ts = cfg.timestep_scale
        (video_mod, video_emb), (av_video_mod, _) = self._modulation(
            [("adaln_single", 9, vd), ("av_ca_video_scale_shift_adaln_single", 4, vd)],
            timestep,
            video_timesteps,
            ts,
        )
        (audio_mod, audio_emb), (av_audio_mod, _) = self._modulation(
            [
                ("audio_adaln_single", 9, ad),
                ("av_ca_audio_scale_shift_adaln_single", 4, ad),
            ],
            timestep,
            audio_timesteps,
            ts,
        )
        # Prompt AdaLN reads its own modality's sigma; each cross gate reads the
        # other modality's sigma at the av_ca scale. Always scalar per sample.
        vs = timestep if video_sigma is None else video_sigma.astype(G)
        as_ = timestep if audio_sigma is None else audio_sigma.astype(G)
        ((video_prompt_mod, _),) = self._modulation(
            [("prompt_adaln_single", 2, vd)], vs, None, ts
        )
        ((audio_prompt_mod, _),) = self._modulation(
            [("audio_prompt_adaln_single", 2, ad)], as_, None, ts
        )
        ((a2v_gate_mod, _),) = self._modulation(
            [("av_ca_a2v_gate_adaln_single", 1, vd)],
            as_,
            None,
            cfg.av_ca_timestep_scale,
        )
        ((v2a_gate_mod, _),) = self._modulation(
            [("av_ca_v2a_gate_adaln_single", 1, ad)], vs, None, cfg.av_ca_timestep_scale
        )

        heads = cfg.heads
        cross_inner = w["transformer_blocks.0.audio_to_video_attn.to_q.weight"].shape[0]
        cross_max = (max(cfg.max_pos[0], cfg.audio_max_pos[0]),)
        rope = lambda pos, inner, h, mp: tuple(  # noqa: E731
            t.astype(F) for t in rope_tables(pos, inner, h, mp, cfg.rope_theta)
        )
        state = {
            "video_mod": video_mod,
            "audio_mod": audio_mod,
            "video_prompt_mod": video_prompt_mod,
            "audio_prompt_mod": audio_prompt_mod,
            "av_video_mod": av_video_mod,
            "av_audio_mod": av_audio_mod,
            "a2v_gate_mod": a2v_gate_mod,
            "v2a_gate_mod": v2a_gate_mod,
            "video_text": None if video_text is None else video_text.astype(G),
            "audio_text": None if audio_text is None else audio_text.astype(G),
            "video_rope": None
            if no_video
            else rope(video_positions, vd, heads, cfg.max_pos),
            "audio_rope": None
            if no_audio
            else rope(audio_positions, ad, cfg.audio_heads, cfg.audio_max_pos),
            "video_cross_rope": None
            if no_video
            else rope(video_positions[:, :, 0:1], cross_inner, heads, cross_max),
            "audio_cross_rope": None
            if no_audio
            else rope(audio_positions[:, :, 0:1], cross_inner, heads, cross_max),
            "video_mask": video_attention_mask,
            "audio_mask": audio_attention_mask,
            "video_cross_mask": video_cross_attention_mask,
            "stg": stg,
            "text_rows": text_rows,
            "video_tiles": video_tiles,
            "run_video": run_video and not no_video,
            "run_audio": run_audio and not no_audio,
            "has_video": not no_video,
            "has_audio": not no_audio,
            # one batch row standing for each distinct text row, for the prompt AdaLN
            "text_reps": None
            if text_rows is None
            else mx.array(
                [
                    list(np.array(text_rows)).index(u)
                    for u in range(int(text_rows.max()) + 1)
                ]
            ),
        }
        mx.eval(
            *(() if no_video else (state["video_rope"], state["video_cross_rope"])),
            *(() if no_audio else (state["audio_rope"], state["audio_cross_rope"])),
        )
        return v, a, state, video_emb, audio_emb

    def _out(
        self, x: mx.array, emb: Modulation, table: mx.array, proj: str
    ) -> mx.array:
        e = emb.get(0)
        x = mx.fast.layer_norm(x, None, None, self.cfg.norm_eps)
        x = x * (1.0 + (table[1] + e)) + (table[0] + e)
        return self.lin(proj, x.astype(F)).astype(G)


def x0_from_velocity(x_t: mx.array, v: mx.array, sigma: mx.array) -> mx.array:
    """x0 = x_t - sigma * v in fp32; sigma is (B,) or per-token (B, N)."""
    s = sigma.astype(G)
    s = s[:, None, None] if s.ndim == 1 else s[:, :, None]
    return x_t.astype(G) - s * v.astype(G)
