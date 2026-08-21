# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5-family vision stack for the Qwen3.8-27B GGUF campaign.

The tower (`Qwen3_VisionTransformer` upstream) is vendored from upstream
vLLM ``qwen3_vl.py`` and trimmed to what the shipped mmproj actually
contains (verified dump in perf/qwen38_metal_design.md):

- 27 pre-LN blocks: LayerNorm(eps 1e-6, weight+bias), fused qkv with bias
  (1152 -> 3456), SDPA over 16 heads of 72 dims with 2D half-rotary
  vision RoPE, LayerNorm, gelu(tanh) MLP 1152 -> 4304 -> 1152 (no gate).
- Patch embed 16px, temporal_patch 2 (a still image is duplicated across
  the two temporal taps, exactly llama.cpp's
  ``conv2d(w0, img) + conv2d(w1, img)``), run as one Linear over
  flattened [C, T, P, P] patches.
- Learned 48x48 position table, bilinearly interpolated (align-corners,
  matching both upstream's linspace interpolation and llama.cpp's
  ``GGML_SCALE_FLAG_ALIGN_CORNERS``) to the actual grid, added in
  spatial-merge order.
- 2x2 spatial merge -> LayerNorm (the GGUF's ``v.post_ln`` *is* the
  merger norm, see llama.cpp's conversion/qwen3vl.py mapping
  ``visual.merger.norm -> V_POST_NORM``) -> mm.0 (4608 -> 4608) -> GELU
  -> mm.2 (4608 -> 5120).

Removed relative to upstream: video pruning/EVS, deepstack
(``is_deepstack_layers`` is all-false in this artifact), CUDA-graph
encoder plumbing, TP/DP sharding (plain nn.Linear; this fork serves
single-device Metal), and the triton position-embed kernel.

The composite model (`Qwen3_5ForConditionalGeneration`) follows this
fork's Muse-Glimmer pattern: vision tower + projector feeding image
embeddings into the registered `Qwen3_5ForCausalLM` text model at
``image_token_id`` positions, with image-aware MRoPE positions.
"""

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    NestedTensors,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors

from .qwen3_5 import Qwen3_5ForCausalLM
from .utils import init_vllm_registered_model, maybe_prefix

logger = init_logger(__name__)

IMAGE_PLACEHOLDER = "<|image_pad|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"


# ---------------------------------------------------------------------------
# Preprocessing (HF Qwen2VLImageProcessor math, vendored: no torchvision in
# this environment). Patches are emitted flattened in spatial-merge order,
# each row laid out [C, T, P, P] to match the Linear patch embed.
# ---------------------------------------------------------------------------


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """HF qwen2_vl smart_resize: round to `factor`, clamp total pixels."""
    import math

    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class Qwen3_5ImageProcessor:
    """Single-image preprocessing to flattened temporal-merged patches.

    Mirrors HF's Qwen2VLImageProcessor for still images: smart-resize
    (bicubic), rescale + normalize, duplicate the frame across the two
    temporal taps, then flatten 16px patches in spatial-merge order.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, vision_config) -> None:
        v = vision_config
        self.patch_size = int(v.patch_size)
        self.merge_size = int(v.spatial_merge_size)
        self.temporal_patch_size = int(v.temporal_patch_size)
        image_size = int(getattr(v, "image_size", 768))
        factor = self.patch_size * self.merge_size
        # Trained envelope: the 48x48 position grid at image_size 768.
        self.max_pixels = int(getattr(v, "image_max_pixels", image_size * image_size))
        self.min_pixels = int(getattr(v, "image_min_pixels", 4 * factor * factor))
        self.image_mean = [float(x) for x in getattr(v, "image_mean", [0.5] * 3)]
        self.image_std = [float(x) for x in getattr(v, "image_std", [0.5] * 3)]

    def _patchify(self, img) -> tuple[np.ndarray, tuple[int, int, int]]:
        from PIL import Image

        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")
        width, height = img.size
        factor = self.patch_size * self.merge_size
        h_bar, w_bar = smart_resize(
            height, width, factor, self.min_pixels, self.max_pixels
        )
        img = img.resize((w_bar, h_bar), Image.Resampling.BICUBIC)

        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, C]
        mean = np.array(self.image_mean, dtype=np.float32)
        std = np.array(self.image_std, dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)  # [C, H, W]

        t, p, m = self.temporal_patch_size, self.patch_size, self.merge_size
        grid_h, grid_w = h_bar // p, w_bar // p
        # Still image: duplicate the frame across the temporal taps
        # (HF np.tile; llama.cpp applies both conv taps to the same frame).
        patches = np.tile(arr[np.newaxis], (t, 1, 1, 1))  # [T, C, H, W]
        patches = patches.reshape(1, t, 3, grid_h // m, m, p, grid_w // m, m, p)
        patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        flat = patches.reshape(grid_h * grid_w, 3 * t * p * p)
        return flat, (1, grid_h, grid_w)

    def preprocess(self, images, **kwargs) -> BatchFeature:
        if not isinstance(images, (list, tuple)):
            images = [images]
        flats, grids = [], []
        for img in images:
            flat, grid = self._patchify(img)
            flats.append(torch.from_numpy(flat))
            grids.append(grid)
        return BatchFeature(
            {
                "pixel_values": torch.cat(flats, dim=0),
                "image_grid_thw": torch.tensor(grids, dtype=torch.long),
            },
            tensor_type=None,
        )

    def __call__(self, images, **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


# ---------------------------------------------------------------------------
# Vision tower.
# ---------------------------------------------------------------------------


def _interp_pos_embed(table: torch.Tensor, h: int, w: int, side: int) -> torch.Tensor:
    """Bilinear align-corners interpolation of the position table.

    Vendored from upstream ``pos_embed_interpolate_native`` (t == 1),
    returning [h * w, hidden] in spatial-merge order (merge 2).
    """
    device = table.device
    h_idxs = torch.linspace(0, side - 1, h, dtype=torch.float32, device=device)
    w_idxs = torch.linspace(0, side - 1, w, dtype=torch.float32, device=device)

    h_floor = h_idxs.to(torch.long)
    w_floor = w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=side - 1)
    w_ceil = torch.clamp(w_floor + 1, max=side - 1)
    dh = h_idxs - h_floor
    dw = w_idxs - w_floor

    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = 1 - dh_grid - w01

    h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
    w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
    indices = (h_grid * side + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)

    embeds = table[indices].to(torch.float32) * weights
    combined = embeds.sum(dim=0)
    m = 2
    hidden = table.shape[1]
    combined = combined.reshape(h // m, m, w // m, m, hidden)
    combined = combined.permute(0, 2, 1, 3, 4).reshape(h * w, hidden)
    return combined.to(table.dtype)


@lru_cache(maxsize=256)
def _rot_pos_ids(h: int, w: int, merge: int) -> torch.Tensor:
    """(h, w) position ids per patch, in spatial-merge order. [L, 2]."""
    hpos = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
    wpos = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
    out = []
    for pos in (hpos, wpos):
        p = pos.reshape(h // merge, merge, w // merge, merge)
        out.append(p.transpose(0, 2, 1, 3).flatten())
    return torch.from_numpy(np.stack(out, axis=-1))


class Qwen3_5VisionPatchEmbed(nn.Module):
    """The two temporal conv taps as one Linear over [C, T, P, P] rows."""

    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(
            in_channels * temporal_patch_size * patch_size * patch_size,
            hidden_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.to(self.proj.weight.dtype))


class Qwen3_5VisionAttention(nn.Module):
    def __init__(self, hidden: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.qkv = nn.Linear(hidden, hidden * 3, bias=True)
        self.proj = nn.Linear(hidden, hidden, bias=True)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        # x: [L, hidden]; cos/sin: [L, head_dim] (already duplicated halves).
        L, d = x.shape
        qkv = self.qkv(x).view(L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)  # each [L, heads, head_dim]
        c = cos[:, None, :].to(q.dtype)
        s = sin[:, None, :].to(q.dtype)
        q = q * c + self._rotate_half(q) * s
        k = k * c + self._rotate_half(k) * s
        attn = F.scaled_dot_product_attention(
            q.permute(1, 0, 2)[None],
            k.permute(1, 0, 2)[None],
            v.permute(1, 0, 2)[None],
        )
        attn = attn[0].permute(1, 0, 2).reshape(L, d)
        return self.proj(attn)


class Qwen3_5VisionBlock(nn.Module):
    def __init__(
        self, hidden: int, num_heads: int, intermediate: int, eps: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, eps=eps)
        self.norm2 = nn.LayerNorm(hidden, eps=eps)
        self.attn = Qwen3_5VisionAttention(hidden, num_heads)
        self.mlp = nn.Module()
        self.mlp.linear_fc1 = nn.Linear(hidden, intermediate, bias=True)
        self.mlp.linear_fc2 = nn.Linear(intermediate, hidden, bias=True)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        h = self.mlp.linear_fc1(self.norm2(x))
        # hidden_act gelu_pytorch_tanh (GGUF `clip.use_gelu`; ggml's GELU is
        # also the tanh approximation).
        h = F.gelu(h, approximate="tanh")
        return x + self.mlp.linear_fc2(h)


class Qwen3_5VisionPatchMerger(nn.Module):
    """post_ln -> 2x2 concat -> mm.0 -> GELU -> mm.2.

    The norm is applied per 1152-dim token before the merge reshape, which
    is exactly clip.cpp's post_ln placement. GELU here follows upstream
    vLLM/HF (`nn.GELU()`, exact erf); llama.cpp's ggml GELU is the tanh
    approximation everywhere -- a deliberate approximation on their side.
    """

    def __init__(self, context_dim: int, merge: int, out_dim: int, eps: float) -> None:
        super().__init__()
        self.hidden_size = context_dim * merge * merge
        self.norm = nn.LayerNorm(context_dim, eps=eps)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, out_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class Qwen3_5VisionTransformer(nn.Module):
    def __init__(self, vision_config, norm_eps: float = 1e-6) -> None:
        super().__init__()
        v = vision_config
        self.hidden_size = v.hidden_size
        self.num_heads = v.num_heads
        self.patch_size = v.patch_size
        self.spatial_merge_size = v.spatial_merge_size
        self.temporal_patch_size = v.temporal_patch_size
        self.num_position_embeddings = v.num_position_embeddings
        self.num_grid_per_side = int(v.num_position_embeddings**0.5)
        self.head_dim = v.hidden_size // v.num_heads
        self.rope_theta = 10000.0

        self.patch_embed = Qwen3_5VisionPatchEmbed(
            v.patch_size, v.temporal_patch_size, v.in_channels, v.hidden_size
        )
        self.pos_embed = nn.Embedding(v.num_position_embeddings, v.hidden_size)
        self.blocks = nn.ModuleList(
            [
                Qwen3_5VisionBlock(
                    v.hidden_size, v.num_heads, v.intermediate_size, norm_eps
                )
                for _ in range(v.depth)
            ]
        )
        self.merger = Qwen3_5VisionPatchMerger(
            v.hidden_size, v.spatial_merge_size, v.out_hidden_size, norm_eps
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def _rope_cos_sin(self, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token cos/sin over the full head dim (halves duplicated).

        Matches ggml's GGML_ROPE_TYPE_VISION with sections [d/4]*4 over
        (y, x) positions and HF's apply_rotary_pos_emb_vision: dims
        [0, d/4) rotate by the h position, [d/4, d/2) by the w position,
        each paired with the dim d/2 higher; the frequency ladder restarts
        per axis.
        """
        device = self.device
        quarter = self.head_dim // 4
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(quarter, dtype=torch.float32, device=device) / quarter)
        )
        pos = _rot_pos_ids(h, w, self.spatial_merge_size).to(device)
        freqs_h = pos[:, 0:1].float() * inv_freq  # [L, d/4]
        freqs_w = pos[:, 1:2].float() * inv_freq  # [L, d/4]
        angles = torch.cat([freqs_h, freqs_w], dim=-1)  # [L, d/2]
        angles = torch.cat([angles, angles], dim=-1)  # [L, d]
        return angles.cos(), angles.sin()

    def forward(
        self, pixel_values: torch.Tensor, grid_thw: list[tuple[int, int, int]]
    ) -> list[torch.Tensor]:
        """pixel_values: [total_patches, C*T*P*P] flat across images.

        Returns one [tokens_i, out_hidden] tensor per image.
        """
        x = self.patch_embed(pixel_values.to(self.device))
        outputs: list[torch.Tensor] = []
        start = 0
        for t, h, w in grid_thw:
            assert t == 1, f"still images only (grid_t={t})"
            length = h * w
            seq = x[start : start + length]
            start += length
            pos = _interp_pos_embed(self.pos_embed.weight, h, w, self.num_grid_per_side)
            seq = seq + pos.to(seq.dtype)
            cos, sin = self._rope_cos_sin(h, w)
            for blk in self.blocks:
                seq = blk(seq, cos, sin)
            outputs.append(self.merger(seq))
        return outputs


# ---------------------------------------------------------------------------
# Multimodal processor plumbing (Muse-Glimmer pattern).
# ---------------------------------------------------------------------------


class Qwen3_5Processor:
    """HF-processor facade over the GGUF tokenizer and image processor."""

    def __init__(self, tokenizer, image_processor, image_token_id: int) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_token_id = image_token_id

    def __call__(
        self,
        text=None,
        images=None,
        return_tensors=None,
        **kwargs,
    ) -> BatchFeature:
        data: dict[str, Any] = {}
        if text is not None:
            if isinstance(text, str):
                text = [text]
            encodings = [self.tokenizer(t, add_special_tokens=False) for t in text]
            data["input_ids"] = [e["input_ids"] for e in encodings]
        if images:
            data.update(self.image_processor.preprocess(images))
        return BatchFeature(data, tensor_type=None)


class Qwen3_5ProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.model_config.hf_config

    def get_hf_processor(self, **kwargs):
        tokenizer = self.get_tokenizer()
        config = self.get_hf_config()
        image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if image_token_id != config.image_token_id:
            raise ValueError(
                f"tokenizer maps {IMAGE_PLACEHOLDER!r} to {image_token_id}, "
                f"config says {config.image_token_id}"
            )
        return Qwen3_5Processor(
            tokenizer,
            Qwen3_5ImageProcessor(config.vision_config),
            image_token_id,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}


class Qwen3_5DummyInputsBuilder(BaseDummyInputsBuilder[Qwen3_5ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        placeholder = VISION_START + IMAGE_PLACEHOLDER + VISION_END
        return placeholder * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options=None,
    ):
        size = int(getattr(self.info.get_hf_config().vision_config, "image_size", 768))
        return {
            "image": self._get_dummy_images(
                width=size, height=size, num_images=mm_counts.get("image", 0)
            )
        }


class Qwen3_5MultiModalProcessor(BaseMultiModalProcessor[Qwen3_5ProcessingInfo]):
    def _hf_processor_applies_updates(self, *args, **kwargs) -> bool:
        # The facade processor tokenizes the prompt as-is; the single
        # <|image_pad|> per image is expanded here via PromptReplacement.
        return False

    def _call_hf_processor(self, *args, **kwargs) -> BatchFeature:
        # Pure passthrough. Overriding matters anyway: the base
        # `_apply_hf_processor_mm_only` (pre-tokenized prompts, profiling)
        # otherwise calls `call_hf_processor_mm_only`, which needs
        # ProcessorMixin._merge_kwargs machinery the runtime-built facade
        # does not have; with an override it routes through dummy text and
        # the facade's plain __call__.
        return super()._call_hf_processor(*args, **kwargs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        grid_thw = hf_inputs.get("image_grid_thw", torch.empty(0, 3, dtype=torch.long))
        sizes = grid_thw.prod(-1) if isinstance(grid_thw, torch.Tensor) else None
        if sizes is None:
            sizes = torch.tensor([int(np.prod(g)) for g in grid_thw], dtype=torch.long)
        return dict(
            pixel_values=MultiModalFieldConfig.flat_from_sizes("image", sizes),
            image_grid_thw=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        processor = self.info.get_hf_processor()
        image_token_id = processor.image_token_id
        merge_length = processor.image_processor.merge_size**2

        def get_replacement(item_idx: int):
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item["image_grid_thw"].data
            num_tokens = int(grid_thw.prod()) // merge_length
            return [image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            )
        ]


# ---------------------------------------------------------------------------
# The composite model.
# ---------------------------------------------------------------------------


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3_5MultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3_5DummyInputsBuilder,
)
class Qwen3_5ForConditionalGeneration(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsEagle3,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return VISION_START + IMAGE_PLACEHOLDER + VISION_END
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.visual = Qwen3_5VisionTransformer(
            config.vision_config,
            norm_eps=getattr(config.vision_config, "layer_norm_eps", 1e-6),
        )
        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=config,
            architectures=["Qwen3_5ForCausalLM"],
            prefix=maybe_prefix(prefix, "language_model"),
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    # -- Text/hybrid plumbing delegated to the registered text model. The
    # engine resolves these on the architecture class, so they must exist
    # here (the GDN layers live inside language_model).
    get_mamba_state_dtype_from_config = (
        Qwen3_5ForCausalLM.get_mamba_state_dtype_from_config
    )
    get_mamba_state_shape_from_config = (
        Qwen3_5ForCausalLM.get_mamba_state_shape_from_config
    )
    get_mamba_state_copy_func = Qwen3_5ForCausalLM.get_mamba_state_copy_func

    def get_language_model(self) -> nn.Module:
        return self.language_model

    def embed_multimodal(self, **kwargs: object) -> NestedTensors | None:
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return None
        grid_thw = kwargs.get("image_grid_thw")
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.cat(
                [p.reshape(-1, p.shape[-1]) for p in pixel_values], dim=0
            )
        else:
            pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
        if isinstance(grid_thw, (list, tuple)):
            grid_list = [
                tuple(int(v) for v in g.reshape(-1).tolist()) for g in grid_thw
            ]
        else:
            grid_list = [
                tuple(int(v) for v in row) for row in grid_thw.reshape(-1, 3).tolist()
            ]
        pixel_values = pixel_values.to(self.visual.dtype)
        return self.visual(pixel_values, grid_list)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.embed_input_ids(input_ids)
        if multimodal_embeddings and is_multimodal is not None:
            flat = torch.cat([e.to(inputs_embeds.dtype) for e in multimodal_embeddings])
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds[is_multimodal] = flat.to(inputs_embeds.device)
        return inputs_embeds

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[object],
    ) -> tuple[torch.Tensor, int]:
        """Image-aware T/H/W positions (upstream Qwen3VL logic, image-only).

        Text tokens advance all three dims together; each image's tokens
        get a constant t and an (h, w) grid, and the stream resumes after
        the grid's max. The text config's interleaved mrope_section
        [11, 11, 10] consumes these downstream.
        """
        merge = self.config.vision_config.spatial_merge_size
        pos_list: list[np.ndarray] = []
        st = 0
        features = sorted(mm_features, key=lambda f: f.mm_position.offset)
        for f in features:
            if f.modality != "image":
                raise ValueError(
                    f"unsupported modality {f.modality!r} for Qwen3.5 mrope"
                )
            t, h, w = (int(v) for v in f.data["image_grid_thw"].data.reshape(-1))
            gh, gw = h // merge, w // merge
            offset = int(f.mm_position.offset)
            text_len = offset - st
            st_idx = int(pos_list[-1].max()) + 1 if pos_list else 0
            if text_len > 0:
                pos_list.append(
                    np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
                )
                st_idx += text_len
            grid = np.indices((max(t, 1), gh, gw)).reshape(3, -1)
            pos_list.append(grid + st_idx)
            st = offset + gh * gw * max(t, 1)
        if st < len(input_tokens):
            st_idx = int(pos_list[-1].max()) + 1 if pos_list else 0
            text_len = len(input_tokens) - st
            pos_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )
        llm_positions = np.concatenate(pos_list, axis=1).reshape(3, -1)
        assert llm_positions.shape[1] == len(input_tokens), (
            llm_positions.shape,
            len(input_tokens),
        )
        delta = int(llm_positions.max()) + 1 - len(input_tokens)
        return torch.from_numpy(llm_positions.copy()), delta

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # DFlash aux capture happens inside the text model.
        self.language_model.set_aux_hidden_state_layers(layers)

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        return self.language_model.get_eagle3_aux_hidden_state_layers()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        lang_weights = []
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if name.startswith("visual."):
                param = params_dict[name]
                param.data.copy_(weight.to(param.dtype))
                loaded.add(name)
            elif name.startswith("language_model."):
                lang_weights.append((name[len("language_model.") :], weight))
            else:
                lang_weights.append((name, weight))
        loaded.update(
            "language_model." + n
            for n in self.language_model.load_weights(lang_weights)
        )
        return loaded
