# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 duration head: the clip length a prompt implies.

Upstream `ltx_core/duration_head/duration_head.py` (DurationHead) and
`ltx_pipelines/utils/blocks.py` (DurationPredictor): a few-MB regression head
on the text encoder's connector outputs. Each modality is projected to 256
dims and tagged with a learnable embedding, one learnable query cross-attends
the concatenated tokens (4 heads), and a GELU MLP emits log-seconds. The
pipelines call it when a request names no frame count: the prediction is
clamped to [1 s, 20 s] and snapped to the VAE's 8k + 1 frame grid.

fp32 throughout: the head is tiny and upstream runs it in the model dtype.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from slimserve.video.ltx25 import checkpoints
from slimserve.video.ltx25.sampling import VIDEO_TEMPORAL_SCALE

MIN_SECONDS, MAX_SECONDS = 1.0, 20.0
_PREFIX = "duration_head."


def snap_frames_to_grid(frames: int) -> int:
    """Round down to the nearest 8k + 1 (the causal VAE's temporal grid)."""
    if frames < 1:
        raise ValueError(f"frames must be >= 1, got {frames}")
    return (frames - 1) // VIDEO_TEMPORAL_SCALE * VIDEO_TEMPORAL_SCALE + 1


def seconds_to_frames(
    seconds: float, fps: float, min_frames: int = 1, max_frames: int = 1024
) -> int:
    """Upstream seconds_to_clamped_num_frames: round, clamp, snap down to the
    grid; if snapping undershoots the minimum, snap up instead."""
    raw = max(min_frames, min(round(seconds * fps), max_frames))
    frames = snap_frames_to_grid(raw)
    if frames < min_frames:
        t = VIDEO_TEMPORAL_SCALE
        frames = min(-(-(min_frames - 1) // t) * t + 1, max_frames)
    return frames


class DurationHead:
    def __init__(self, root: Path | None = None):
        self.root = root
        self.w: dict[str, mx.array] | None = None

    def load(self) -> DurationHead:
        if self.w is None:
            raw = checkpoints.load_raw(checkpoints.path_of("duration-head", self.root))
            self.w = {
                k[len(_PREFIX) :]: v.astype(mx.float32)
                for k, v in raw.items()
                if k.startswith(_PREFIX)
            }
            if "mlp_out.weight" not in self.w:
                raise ValueError("no DurationHead weights in the duration-head file")
            mx.eval(self.w)
        return self

    @staticmethod
    def _lin(w: dict[str, mx.array], name: str, x: mx.array) -> mx.array:
        return mx.addmm(w[name + ".bias"], x, w[name + ".weight"].T)

    def seconds(
        self, video_tokens: mx.array | None, audio_tokens: mx.array | None
    ) -> float:
        """Predicted duration in seconds from the connector outputs
        ((1, N, 4096) and / or (1, N, 2048), any dtype)."""
        if video_tokens is None and audio_tokens is None:
            raise ValueError("the duration head needs video and / or audio tokens")
        w = self.load().w
        groups = []
        if video_tokens is not None:
            groups.append(
                self._lin(w, "video_input_proj", video_tokens.astype(mx.float32))
                + w["video_modality_emb"]
            )
        if audio_tokens is not None:
            groups.append(
                self._lin(w, "audio_input_proj", audio_tokens.astype(mx.float32))
                + w["audio_modality_emb"]
            )
        tokens = mx.concatenate(groups, axis=1)  # (1, T, 256)
        hidden = w["attention_pooler.query_tokens"].shape[-1]
        heads = 4
        hd = hidden // heads
        # torch MultiheadAttention: packed in_proj [q; k; v], scaled dot product,
        # out_proj
        wq, wk, wv = mx.split(w["attention_pooler.cross_attn.in_proj_weight"], 3, 0)
        bq, bk, bv = mx.split(w["attention_pooler.cross_attn.in_proj_bias"], 3, 0)
        q = mx.addmm(bq, w["attention_pooler.query_tokens"][None], wq.T)  # (1, Q, 256)
        k = mx.addmm(bk, tokens, wk.T)
        v = mx.addmm(bv, tokens, wv.T)
        split = lambda z: z.reshape(1, -1, heads, hd).transpose(0, 2, 1, 3)  # noqa: E731
        pooled = mx.fast.scaled_dot_product_attention(
            split(q), split(k), split(v), scale=hd**-0.5
        )
        pooled = pooled.transpose(0, 2, 1, 3).reshape(1, -1, hidden)
        pooled = self._lin(w, "attention_pooler.cross_attn.out_proj", pooled)
        h = self._lin(w, "mlp_hidden", pooled.reshape(1, -1))
        h = 0.5 * h * (1.0 + mx.tanh(0.7978845608028654 * (h + 0.044715 * h * h * h)))
        log_seconds = self._lin(w, "mlp_out", h)
        return float(mx.exp(log_seconds).item())

    def num_frames(
        self,
        video_tokens: mx.array | None,
        audio_tokens: mx.array | None,
        fps: float,
        min_seconds: float = MIN_SECONDS,
        max_seconds: float = MAX_SECONDS,
    ) -> tuple[int, float]:
        """(frames on the 8k + 1 grid, raw predicted seconds), clamped to
        [min_seconds, max_seconds] as upstream's DurationPredictor."""
        seconds = self.seconds(video_tokens, audio_tokens)
        frames = seconds_to_frames(
            seconds,
            fps,
            min_frames=round(min_seconds * fps),
            max_frames=round(max_seconds * fps),
        )
        return frames, seconds
