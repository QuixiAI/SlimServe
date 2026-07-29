# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Final, Generic, Literal, Protocol, TypeAlias, TypeVar

import torch
from transformers import PretrainedConfig

from vllm.config import MultiModalConfig, get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)

_C = TypeVar("_C", bound=PretrainedConfig)


class _RootConfig(Protocol[_C]):
    vision_config: _C


class VisionEncoderInfo(ABC, Generic[_C]):
    def __init__(self, hf_config: _RootConfig[_C]) -> None:
        super().__init__()

        self.hf_config = hf_config
        self.vision_config = hf_config.vision_config

    @abstractmethod
    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_image_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_patch_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_patch_grid_length(self) -> int:
        raise NotImplementedError


class VisionLanguageConfig(Protocol):
    vision_config: Final[PretrainedConfig]




def _get_vit_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    *,
    attn_backend_override: AttentionBackendEnum | None = None,
) -> AttentionBackendEnum:
    """
    Get the available attention backend for Vision Transformer.
    """
    return current_platform.get_vit_attn_backend(
        head_size,
        dtype,
        backend=attn_backend_override,
    )


def get_vit_attn_backend(
    head_size: int,
    dtype: torch.dtype,
) -> AttentionBackendEnum:
    """
    Get the attention backend for Vision Transformer.
    """
    mm_cfg = get_multimodal_config()
    attn_backend_override = (
        mm_cfg.mm_encoder_attn_backend if mm_cfg is not None else None
    )
    return _get_vit_attn_backend(
        head_size,
        dtype,
        attn_backend_override=attn_backend_override,
    )


def get_multimodal_config() -> MultiModalConfig | None:
    """Return the current ``MultiModalConfig``, or ``None`` when no engine
    config context is active (e.g., during unit tests) or when the current
    ``model_config`` does not carry a ``multimodal_config`` (e.g., minimal
    stubs used in tests)."""
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None or vllm_config.model_config is None:
        return None
    return getattr(vllm_config.model_config, "multimodal_config", None)


def get_fp8_padded_hidden_size(num_heads: int, head_dim: int) -> int | None:
    """Return the padded hidden size for FP8 ViT encoder attention, or
    ``None`` when FP8 is not enabled.

    cuDNN FP8 prefill attention requires ``head_dim`` to be a multiple of
    16. For non-aligned ``head_dim`` (e.g. 72), Q/K/V are padded to the
    nearest multiple of 16.
    """
    mm_cfg = get_multimodal_config()
    if mm_cfg is None or mm_cfg.mm_encoder_attn_dtype != "fp8":
        return None
    return num_heads * round_up(head_dim, 16)


def is_vit_use_data_parallel(num_heads: int | None = None) -> bool:
    """
    Get the tensor parallel type for Vision Transformer.
    """
    mm_cfg = get_multimodal_config()
    can_split = (
        num_heads % get_tensor_model_parallel_world_size() == 0
        if num_heads is not None
        else None
    )
    if num_heads is not None and not can_split:
        logger.warning_once(
            "The number of vision attention heads is not divisible by "
            "the tensor parallel size. Falling back to data parallelism "
            "for the vision encoder."
        )
        return True
    return mm_cfg is not None and mm_cfg.mm_encoder_tp_mode == "data"


VisionFeatureSelectStrategyStr = Literal["class", "default", "full"]

VisionFeatureSelectStrategy: TypeAlias = (
    VisionFeatureSelectStrategyStr | Callable[[torch.Tensor], torch.Tensor]
)


def _get_vision_feature_selector(
    strategy: VisionFeatureSelectStrategy | str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if callable(strategy):
        return strategy

    # https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/models/clip/modeling_clip.py#L762
    if strategy == "class":
        return lambda feats: feats[:, :1, :]

    # https://github.com/huggingface/transformers/blob/4a02bc7004285bdb12cc033e87ad2578ce2fa900/src/transformers/models/llava/modeling_llava.py#L196
    if strategy == "default":
        return lambda feats: feats[:, 1:, :]

    if strategy == "full":
        return lambda feats: feats

    raise ValueError(f"Unexpected feature select strategy: {strategy!r}")


def get_num_selected_vision_tokens(
    num_vision_tokens: int,
    strategy: VisionFeatureSelectStrategy | str,
) -> int:
    if callable(strategy):
        dummy_features = torch.empty(1, num_vision_tokens, 64)  # [B, L, D]
        dummy_selected_features = strategy(dummy_features)
        return dummy_selected_features.shape[1]

    if strategy == "class":
        return 1

    if strategy == "default":
        return num_vision_tokens - 1

    if strategy == "full":
        return num_vision_tokens

    raise ValueError(f"Unexpected feature select strategy: {strategy!r}")






def get_load_balance_assignment(
    sizes: list[int],
    num_gpus: int = 2,
) -> tuple[list[int], list[int], list[int]]:
    """
    Generate load balancing assignment and metadata
    for distributing data across GPUs.
    The load is determined by the total image sizes,
    not the number of images.

    Args:
        sizes: The size of each image
        num_gpus: Number of GPUs to balance across

    Returns:
        shuffle_indices:
            Indices to reorder data for balanced loading
        gpu_sample_counts:
            Number of samples assigned to each GPU
        grouped_sizes_per_gpu:
            Total size assigned to each GPU

    Example:
        ```
        sizes = [1000, 100, 200, 50]
        num_gpus = 2
        ```

    """

    n_samples = len(sizes)

    # Handle edge cases
    if n_samples == 0:
        return [], [0] * num_gpus, [0] * num_gpus

    # Use greedy algorithm - balance by total size, not sample count
    gpu_assignments = [list[int]() for _ in range(num_gpus)]
    gpu_loads = [0] * num_gpus  # This tracks total SIZE, not sample count

    # Sort indices by size (largest first for better load balancing)
    # sizes = [1000, 100, 200, 50]
    # large_to_small_indices = [0, 2, 1, 3]
    large_to_small_indices = sorted(
        range(n_samples), key=lambda i: sizes[i], reverse=True
    )

    for idx in large_to_small_indices:
        # Find GPU with minimum current load (by total size)
        min_gpu = min(range(num_gpus), key=lambda i: gpu_loads[i])
        gpu_assignments[min_gpu].append(idx)
        gpu_loads[min_gpu] += sizes[idx]

    # Create shuffle indices and counts
    shuffle_indices = list[int]()
    gpu_sample_counts = list[int]()
    for gpu_id in range(num_gpus):
        # GPU_0 = [1000] = [0]
        # GPU_1 = [200, 100, 50] = [2, 1, 3]
        # shuffle_indices = [0, 2, 1, 3]
        shuffle_indices.extend(gpu_assignments[gpu_id])
        # GPU_0 = [1]
        # GPU_1 = [3]
        # gpu_sample_counts = [1, 3]
        gpu_sample_counts.append(len(gpu_assignments[gpu_id]))

    return (shuffle_indices, gpu_sample_counts, gpu_loads)


def run_dp_sharded_mrope_vision_model(
    vision_model: torch.nn.Module,
    pixel_values: torch.Tensor,
    grid_thw_list: list[list[int]],
    *,
    rope_type: Literal["rope_3d", "rope_2d"],
) -> tuple[torch.Tensor, ...]:
    """Run a vision model with data parallelism (DP) sharding.
    The function will shard the input image tensor on the
    first dimension and run the vision model.
    This function is used to run the vision model with mrope.

    Args:
        vision_model (torch.nn.Module): Vision model.
        pixel_values (torch.Tensor): Image/Video input tensor.
        grid_thw_list: List of grid dimensions for each image
        rope_type: Type of rope used in the vision model.
                   Different rope types have different dimension to do ViT.
                   "rope_3d" for 3D rope (e.g., Qwen2.5-VL)
                   "rope_2d" for 2D rope (e.g., Kimi-VL)
    Returns:
        torch.Tensor: Output image embeddings

    Example:
        ```
        vision_model.out_hidden_size = 64
        vision_model.spatial_merge_size = 2
        pixel_values.shape = (1350, channel)
        grid_thw_list = [[1, 10, 100], [1, 10, 10], [1, 10, 20], [1, 50]]
        tp_size = 2
        ```

    """
    tp_size = get_tensor_model_parallel_world_size()

    # GPU_0 tp_rank_local = 0
    # GPU_1 tp_rank_local = 1
    tp_rank_local = get_tensor_model_parallel_rank()

    # patches_per_image = [1000, 100, 200, 50]
    patches_per_image = [math.prod(grid_thw) for grid_thw in grid_thw_list]
    # patches_per_image = [0, 1000, 1100, 1300, 1350]
    cum_patches_per_image = [0, *itertools.accumulate(patches_per_image)]

    # Get load balancing assignment with all metadata
    # image_to_tp_rank = [0, 2, 1, 3]
    # gpu_sample_counts = [1, 3]
    # grouped_pixel_values_len = [1000, 350]
    (image_to_tp_rank, gpu_sample_counts, grouped_pixel_values_len) = (
        get_load_balance_assignment(patches_per_image, tp_size)
    )

    # cu_gpu_sample_counts = [0, 1, 4]
    cum_gpu_sample_counts = [0, *itertools.accumulate(gpu_sample_counts)]

    # GPU_0 image_idxs_local = [0]
    # GPU_1 image_idxs_local = [2, 1, 3]
    image_idxs_local = image_to_tp_rank[
        cum_gpu_sample_counts[tp_rank_local] : cum_gpu_sample_counts[tp_rank_local + 1]
    ]

    # Get the pixel values for the local images based on the image_idxs_local
    if len(image_idxs_local) > 0:
        pixel_values_local = torch.cat(
            [
                pixel_values[cum_patches_per_image[i] : cum_patches_per_image[i + 1]]
                for i in image_idxs_local
            ]
        )
    else:
        # Handle case where this rank has no images
        pixel_values_local = torch.empty(
            (0, pixel_values.shape[1]),
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )
    # embed_dim_reduction_factor = 2 * 2
    if rope_type == "rope_2d":
        embed_dim_reduction_factor = (
            vision_model.merge_kernel_size[0] * vision_model.merge_kernel_size[1]
        )
    else:
        embed_dim_reduction_factor = (
            vision_model.spatial_merge_size * vision_model.spatial_merge_size
        )

    # Find the max length across all ranks
    # The output embedding of every DP rank has to be
    # padded to this length for tensor_model_parallel_all_gather
    # to work
    max_len_per_rank = max(grouped_pixel_values_len) // embed_dim_reduction_factor
    local_grid_thw_list = [grid_thw_list[i] for i in image_idxs_local]

    # Run the vision model on the local pixel_values_local
    if rope_type == "rope_2d":
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(
                pixel_values_local, torch.tensor(local_grid_thw_list)
            )
            if isinstance(image_embeds_local, list):
                image_embeds_local = torch.cat(image_embeds_local, dim=0)
        else:
            out_dim = getattr(vision_model.config, "hidden_size", None)
            image_embeds_local = torch.empty(
                (0, embed_dim_reduction_factor, out_dim),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )
    else:
        if pixel_values_local.shape[0] > 0:
            image_embeds_local = vision_model(pixel_values_local, local_grid_thw_list)
        else:
            # Handle empty case
            image_embeds_local = torch.empty(
                (0, vision_model.out_hidden_size),
                device=pixel_values.device,
                dtype=pixel_values.dtype,
            )

    # Pad the output based on max_len_per_rank
    # for tensor_model_parallel_all_gather to work
    current_len = image_embeds_local.shape[0]
    if current_len < max_len_per_rank:
        padding_size = max_len_per_rank - current_len
        if rope_type == "rope_2d":
            padding = torch.empty(
                (
                    padding_size,
                    image_embeds_local.shape[1],
                    image_embeds_local.shape[2],
                ),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        else:
            padding = torch.empty(
                (padding_size, image_embeds_local.shape[1]),
                dtype=image_embeds_local.dtype,
                device=image_embeds_local.device,
            )
        image_embeds_local_padded = torch.cat([image_embeds_local, padding], dim=0)
    else:
        image_embeds_local_padded = image_embeds_local

    # Do all_gather to collect embeddings from all ranks
    gathered_embeds = tensor_model_parallel_all_gather(image_embeds_local_padded, dim=0)

    # Remove padding and reconstruct per-rank embeddings
    rank_embeddings = list[torch.Tensor]()
    for rank in range(tp_size):
        start_idx = rank * max_len_per_rank
        end_idx = start_idx + (
            grouped_pixel_values_len[rank] // embed_dim_reduction_factor
        )
        rank_embeddings.append(gathered_embeds[start_idx:end_idx])

    patches_per_output_image = [
        (patch_size // embed_dim_reduction_factor) for patch_size in patches_per_image
    ]

    # Reconstruct embeddings in the original order
    original_order_embeddings = [None] * len(grid_thw_list)
    current_idx = 0
    for rank in range(tp_size):
        count = gpu_sample_counts[rank]
        if count > 0:
            # Get images assigned to this rank in shuffled order
            # GPU_0 = image_idxs_local  [0]
            # GPU_1 = image_idxs_local  [2, 1, 3]
            rank_images = image_to_tp_rank[current_idx : current_idx + count]

            rank_embed = rank_embeddings[rank]
            # Split rank embeddings back to individual images
            embed_start = 0
            for img_idx in rank_images:
                img_patches = patches_per_output_image[img_idx]
                original_order_embeddings[img_idx] = rank_embed[
                    embed_start : embed_start + img_patches
                ]
                embed_start += img_patches
            current_idx += count
    out_embeddings = tuple(
        embed for embed in original_order_embeddings if embed is not None
    )
    assert len(out_embeddings) == len(original_order_embeddings), (
        "Found unassigned embeddings"
    )
    return out_embeddings


def get_llm_pos_ids_for_vision(
    start_idx: int,
    vision_idx: int,
    spatial_merge_size: int,
    t_index: list[int],
    grid_hs: torch.Tensor,
    grid_ws: torch.Tensor,
) -> torch.Tensor:
    llm_pos_ids_list = []
    llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
    llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
    h_index = (
        torch.arange(llm_grid_h)
        .view(1, -1, 1)
        .expand(len(t_index), -1, llm_grid_w)
        .flatten()
    )
    w_index = (
        torch.arange(llm_grid_w)
        .view(1, 1, -1)
        .expand(len(t_index), llm_grid_h, -1)
        .flatten()
    )
    t_index_tensor = (
        torch.Tensor(t_index)
        .to(llm_grid_h.device)
        .view(-1, 1)
        .expand(-1, llm_grid_h * llm_grid_w)
        .long()
        .flatten()
    )
    _llm_pos_ids = torch.stack([t_index_tensor, h_index, w_index])
    llm_pos_ids_list.append(_llm_pos_ids + start_idx)
    llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
    return llm_pos_ids
