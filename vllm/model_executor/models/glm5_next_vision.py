# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash vision tower (``glm5_next_vision``).

Qwen2-VL-family ViT (Conv3d patch embed, 2D rotary over (h, w) with
NeoX rotate-half on head_dim 64, packed varlen attention per image) with
GLM-5.3 specifics from the reference ``Glm5NextVisionModel``: RMSNorm
q/k norms per head, SwiGLU-limit clamped MLPs with biases, no absolute
position embedding, and a post-layernorm -> 2x2 Conv2d downsample ->
gated merger (proj + LayerNorm + GELU + clamped SwiGLU) producing one
LLM-width (4096) embedding per 2x2 patch window.
"""

from __future__ import annotations

from functools import lru_cache

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.distributed import parallel_state
from vllm.distributed import utils as dist_utils
from vllm.model_executor.layers.attention import MMEncoderAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
from vllm.model_executor.models.vision import get_vit_attn_backend


class Glm5NextVisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.head_dim = dist_utils.divide(embed_dim, num_heads)
        self.num_heads_local = dist_utils.divide(num_heads, self.tp_size)
        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
        )
        self.proj = RowParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads_local,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
            prefix=f"{prefix}.attn",
        )
        self.apply_rotary_emb = ApplyRotaryEmb(enforce_enable=True)

    def forward(
        self,
        x: torch.Tensor,  # [s, b, d]
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,
        sequence_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        x, _ = self.qkv(x)
        seq_len, batch_size, _ = x.shape
        qkv = einops.rearrange(
            x,
            "s b (three head head_dim) -> b s three head head_dim",
            three=3,
            head=self.num_heads_local,
        )
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        qk = torch.stack([q, k], dim=0).reshape(
            2 * batch_size, seq_len, self.num_heads_local, self.head_dim
        ).contiguous()
        qk = self.apply_rotary_emb(qk, rotary_pos_emb_cos, rotary_pos_emb_sin)
        q, k = qk.view(
            2, batch_size, seq_len, self.num_heads_local, self.head_dim
        ).unbind(dim=0)
        out = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )
        out = einops.rearrange(out, "b s h d -> s b (h d)", b=batch_size)
        out, _ = self.proj(out.contiguous())
        return out


class _ClampedSwiGLU(nn.Module):
    """gate/up projections with the GLM-5.3 clamp: gate <= limit,
    |up| <= limit, then silu(gate) * up -> down."""

    def __init__(
        self,
        dim: int,
        hidden: int,
        limit: float,
        bias: bool,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.limit = limit
        self.gate_up_proj = MergedColumnParallelLinear(
            dim, [hidden, hidden], bias=bias, quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            hidden, dim, bias=bias, quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gu, _ = self.gate_up_proj(x)
        gate, up = gu.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        out, _ = self.down_proj(F.silu(gate) * up)
        return out


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, vision_config, quant_config, prefix: str) -> None:
        super().__init__()
        dim = vision_config.hidden_size
        eps = vision_config.rms_norm_eps
        self.norm1 = RMSNorm(dim, eps=eps)
        self.norm2 = RMSNorm(dim, eps=eps)
        self.attn = Glm5NextVisionAttention(
            dim, vision_config.num_heads, eps, quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = _ClampedSwiGLU(
            dim, vision_config.intermediate_size, vision_config.swiglu_limit,
            bias=vision_config.attention_bias, quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(self, x, cu_seqlens, cos, sin, max_seqlen, sequence_lengths):
        x = x + self.attn(
            self.norm1(x), cu_seqlens, cos, sin, max_seqlen, sequence_lengths
        )
        x = x + self.mlp(self.norm2(x))
        return x


class Glm5NextVisionPatchMerger(nn.Module):
    def __init__(self, vision_config, quant_config, prefix: str) -> None:
        super().__init__()
        dim = vision_config.out_hidden_size
        self.proj = ReplicatedLinear(
            dim, dim, bias=False, quant_config=quant_config,
            prefix=f"{prefix}.proj",
        )
        self.post_projection_norm = nn.LayerNorm(dim)
        self.mlp = _ClampedSwiGLU(
            dim, vision_config.projection_intermediate_size,
            vision_config.swiglu_limit, bias=False, quant_config=quant_config,
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.proj(x)
        x = F.gelu(self.post_projection_norm(x))
        return self.mlp(x)


class Glm5NextVisionTransformer(nn.Module):
    def __init__(self, vision_config, quant_config, prefix: str = "") -> None:
        super().__init__()
        self.config = vision_config
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.patch_size = vision_config.patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size
        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv3d(
            vision_config.in_channels,
            self.hidden_size,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            stride=(self.temporal_patch_size, self.patch_size, self.patch_size),
            bias=True,
        )
        self.rotary_pos_emb = get_rope(
            head_size=self.head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )
        self.blocks = nn.ModuleList(
            [
                Glm5NextVisionBlock(
                    vision_config, quant_config, prefix=f"{prefix}.blocks.{i}"
                )
                for i in range(vision_config.depth)
            ]
        )
        self.post_layernorm = RMSNorm(
            self.hidden_size, eps=vision_config.rms_norm_eps
        )
        self.downsample = nn.Conv2d(
            self.hidden_size,
            self.out_hidden_size,
            kernel_size=self.spatial_merge_size,
            stride=self.spatial_merge_size,
            bias=True,
        )
        self.merger = Glm5NextVisionPatchMerger(
            vision_config, quant_config, prefix=f"{prefix}.merger"
        )
        self.attn_backend = get_vit_attn_backend(
            head_size=self.head_dim, dtype=torch.get_default_dtype()
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, merge: int) -> torch.Tensor:
        hpos = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        hpos = hpos.reshape(h // merge, merge, w // merge, merge)
        hpos = hpos.transpose(0, 2, 1, 3).flatten()
        wpos = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos = wpos.reshape(h // merge, merge, w // merge, merge)
        wpos = wpos.transpose(0, 2, 1, 3).flatten()
        return torch.from_numpy(np.stack([hpos, wpos], axis=-1))

    def rot_pos_emb(self, grid_thw: list[list[int]]):
        max_grid = max(max(h, w) for _, h, w in grid_thw)
        pos_ids = torch.cat(
            [
                self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
                for t, h, w in grid_thw
            ],
            dim=0,
        ).to(self.device, non_blocking=True)
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid)
        return cos[pos_ids].flatten(1), sin[pos_ids].flatten(1)

    def forward(
        self, x: torch.Tensor, grid_thw: torch.Tensor | list[list[int]]
    ) -> torch.Tensor:
        grid = grid_thw.tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw
        x = x.to(device=self.device, dtype=self.dtype)
        x = x.view(
            -1, self.config.in_channels, self.temporal_patch_size,
            self.patch_size, self.patch_size,
        )
        h = self.patch_embed.proj(x).view(-1, self.hidden_size)

        cos, sin = self.rot_pos_emb(grid)
        g = np.array(grid, dtype=np.int32)
        per_frame = g[:, 1] * g[:, 2]
        cu = np.repeat(per_frame, g[:, 0]).cumsum(dtype=np.int32)
        cu = np.concatenate([np.zeros(1, dtype=np.int32), cu])
        sequence_lengths = MMEncoderAttention.maybe_compute_seq_lens(
            self.attn_backend, cu, self.device
        )
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu),
            dtype=torch.int32,
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend, cu, self.hidden_size, self.tp_size, self.device
        )

        h = h.unsqueeze(1)  # [s, b=1, d]
        for blk in self.blocks:
            h = blk(h, cu_seqlens, cos, sin, max_seqlen, sequence_lengths)
        h = self.post_layernorm(h.squeeze(1))
        m = self.spatial_merge_size
        h = h.view(-1, m, m, self.hidden_size).permute(0, 3, 1, 2)
        h = self.downsample(h).view(-1, self.out_hidden_size)
        return self.merger(h)
