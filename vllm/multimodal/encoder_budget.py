# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import vllm.envs as envs
from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.multimodal.registry import MultiModalRegistry
from vllm.utils.torch_utils import set_default_torch_num_threads
from vllm.v1.core.encoder_cache_manager import compute_mm_encoder_budget

logger = init_logger(__name__)

_BUDGET_CACHE_VERSION = 1
_SNAPSHOT_KEYS = {
    "mm_limits",
    "encoder_compute_budget",
    "encoder_cache_size",
    "mm_max_toks_per_item",
    "mm_max_items_per_prompt",
    "mm_max_items_per_batch",
    "skip_prompt_length_check",
}


def _is_valid_budget_snapshot(snapshot: object) -> bool:
    if not isinstance(snapshot, dict) or not _SNAPSHOT_KEYS.issubset(snapshot):
        return False

    for key in (
        "mm_limits",
        "mm_max_toks_per_item",
        "mm_max_items_per_prompt",
        "mm_max_items_per_batch",
    ):
        values = snapshot[key]
        if not isinstance(values, dict) or any(
            not isinstance(name, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for name, value in values.items()
        ):
            return False

    for key in ("encoder_compute_budget", "encoder_cache_size"):
        value = snapshot[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return False

    return isinstance(snapshot["skip_prompt_length_check"], bool)


def _budget_cache_fingerprint(vllm_config: VllmConfig) -> str:
    from vllm import __version__

    model_config = vllm_config.model_config
    scheduler_config = vllm_config.scheduler_config
    factors = {
        "version": __version__,
        "model": str(model_config.model),
        "model_config": model_config.compute_hash(),
        "max_model_len": model_config.max_model_len,
        "multimodal_config": repr(model_config.multimodal_config),
        "max_num_seqs": scheduler_config.max_num_seqs,
        "max_num_batched_tokens": scheduler_config.max_num_batched_tokens,
        "enable_chunked_prefill": scheduler_config.enable_chunked_prefill,
    }
    encoded = json.dumps(factors, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _budget_cache_path(vllm_config: VllmConfig) -> Path:
    fingerprint = _budget_cache_fingerprint(vllm_config)
    return Path(envs.VLLM_CACHE_ROOT) / "multimodal_budget" / f"{fingerprint}.json"


def maybe_restore_multimodal_budget_snapshot(vllm_config: VllmConfig) -> bool:
    """Restore a matching derived budget, returning whether it was found."""
    if (
        vllm_config.cache_config.kv_cache_memory_bytes is None
        or vllm_config.multimodal_budget_snapshot is not None
    ):
        return vllm_config.multimodal_budget_snapshot is not None

    cache_path = _budget_cache_path(vllm_config)
    try:
        with cache_path.open() as cache_file:
            payload = json.load(cache_file)
        snapshot = payload["snapshot"]
        if (
            payload["version"] != _BUDGET_CACHE_VERSION
            or payload["fingerprint"] != cache_path.stem
            or not _is_valid_budget_snapshot(snapshot)
        ):
            return False
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False

    vllm_config.multimodal_budget_snapshot = snapshot
    return True


def _save_multimodal_budget_snapshot(vllm_config: VllmConfig) -> None:
    if vllm_config.cache_config.kv_cache_memory_bytes is None:
        return

    snapshot = vllm_config.multimodal_budget_snapshot
    if snapshot is None:
        return

    cache_path = _budget_cache_path(vllm_config)
    temp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    payload = {
        "version": _BUDGET_CACHE_VERSION,
        "fingerprint": cache_path.stem,
        "snapshot": snapshot,
    }
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with temp_path.open("w") as cache_file:
            json.dump(payload, cache_file, separators=(",", ":"))
        os.replace(temp_path, cache_path)
    except (OSError, TypeError, ValueError):
        temp_path.unlink(missing_ok=True)


def get_mm_max_toks_per_item(
    model_config: ModelConfig,
    mm_registry: MultiModalRegistry,
    processor: BaseMultiModalProcessor,
    mm_counts: Mapping[str, int],
) -> Mapping[str, int]:
    """
    Get the maximum number of tokens per data item from each modality based
    on underlying model configuration.
    """
    max_tokens_per_item = processor.info.get_mm_max_tokens_per_item(
        seq_len=model_config.max_model_len,
        mm_counts=mm_counts,
    )
    if max_tokens_per_item is not None:
        return max_tokens_per_item

    mm_inputs = mm_registry.get_dummy_mm_inputs(
        model_config,
        mm_counts=mm_counts,
        processor=processor,
    )

    return {
        modality: sum(item.get_num_embeds() for item in placeholders)
        for modality, placeholders in mm_inputs["mm_placeholders"].items()
    }


class MultiModalBudget:
    """Helper class to calculate budget information for multi-modal models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        mm_registry: MultiModalRegistry,
        *,
        enable_cache: bool = True,
        use_cached_snapshot: bool = False,
        processor: BaseMultiModalProcessor | None = None,
    ) -> None:
        super().__init__()

        self.model_config = model_config = vllm_config.model_config
        self.scheduler_config = scheduler_config = vllm_config.scheduler_config

        self.max_model_len = model_config.max_model_len
        self.max_num_reqs = scheduler_config.max_num_seqs

        snapshot = vllm_config.multimodal_budget_snapshot
        if use_cached_snapshot and snapshot is not None:
            from vllm.utils.bootstamp import bootstamp

            bootstamp("multimodal budget: snapshot reused")
            self.cache = None
            self.processor = None
            self.mm_limits = snapshot["mm_limits"]
            self.encoder_compute_budget = snapshot["encoder_compute_budget"]
            self.encoder_cache_size = snapshot["encoder_cache_size"]
            self.mm_max_toks_per_item = snapshot["mm_max_toks_per_item"]
            self.mm_max_items_per_prompt = snapshot["mm_max_items_per_prompt"]
            self.mm_max_items_per_batch = snapshot["mm_max_items_per_batch"]
            self.skip_prompt_length_check = snapshot["skip_prompt_length_check"]
            return

        if use_cached_snapshot:
            from vllm.utils.bootstamp import bootstamp

            bootstamp("multimodal budget: snapshot unavailable")

        with set_default_torch_num_threads():  # Avoid hang during startup
            if processor is None:
                cache = (
                    mm_registry.processor_only_cache_from_config(vllm_config)
                    if enable_cache
                    else None
                )
                processor = mm_registry.create_processor(model_config, cache=cache)
            else:
                cache = processor.cache

            self.cache = cache
            self.processor = processor
            mm_config = model_config.get_multimodal_config()
            enable_mm_embeds = mm_config is not None and mm_config.enable_mm_embeds

            supported_mm_limits = processor.info.supported_mm_limits
            self.mm_limits = mm_limits = processor.info.allowed_mm_limits
            self.skip_prompt_length_check = processor.info.skip_prompt_length_check

            # Modalities that pass through the MM encoder tower
            tower_modalities = {
                modality
                for modality in supported_mm_limits
                if mm_limits.get(modality, 0) > 0
            }
            # Modalities that bypass the tower (pre-computed embeddings only)
            embed_only_modalities = {
                modality
                for modality in supported_mm_limits
                if enable_mm_embeds and mm_limits.get(modality, 0) == 0
            }

            active_modalities = tower_modalities | embed_only_modalities

            all_mm_max_toks_per_item = get_mm_max_toks_per_item(
                model_config,
                mm_registry,
                processor,
                mm_counts=dict.fromkeys(active_modalities, 1),
            )

        if embed_only_modalities:
            logger.info_once(
                "enable_mm_embeds is True; modalities handled as embedding-only: %s",
                tuple(embed_only_modalities),
            )

        # Some models (e.g., Qwen3Omni with use_audio_in_video=True) share
        # placeholders between modalities, so not all active modalities will
        # have their own entry in the returned dict. We filter to only include
        # modalities that have independent placeholder tokens.
        active_mm_max_toks_per_item = {
            modality: all_mm_max_toks_per_item[modality]
            for modality in active_modalities
            if modality in all_mm_max_toks_per_item
        }
        tower_mm_max_toks_per_item = {
            modality: active_mm_max_toks_per_item[modality]
            for modality in tower_modalities
            if modality in active_mm_max_toks_per_item
        }

        # Encoder budget is computed from all active modalities (including
        # embedding-only ones that need encoder cache space).
        encoder_compute_budget, encoder_cache_size = compute_mm_encoder_budget(
            scheduler_config,
            active_mm_max_toks_per_item,
        )

        self.encoder_compute_budget = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        mm_max_items_per_prompt = dict[str, int]()
        mm_max_items_per_batch = dict[str, int]()

        # Per-prompt/per-batch limits are only relevant for tower modalities
        # (embedding-only modalities don't go through the encoder tower).
        for modality, max_toks_per_item in tower_mm_max_toks_per_item.items():
            (
                mm_max_items_per_prompt[modality],
                mm_max_items_per_batch[modality],
            ) = self._get_max_items(modality, max_toks_per_item)

        self.mm_max_toks_per_item = tower_mm_max_toks_per_item
        self.mm_max_items_per_prompt: Mapping[str, int] = mm_max_items_per_prompt
        self.mm_max_items_per_batch: Mapping[str, int] = mm_max_items_per_batch
        vllm_config.multimodal_budget_snapshot = {
            "mm_limits": dict(self.mm_limits),
            "encoder_compute_budget": self.encoder_compute_budget,
            "encoder_cache_size": self.encoder_cache_size,
            "mm_max_toks_per_item": dict(self.mm_max_toks_per_item),
            "mm_max_items_per_prompt": dict(self.mm_max_items_per_prompt),
            "mm_max_items_per_batch": dict(self.mm_max_items_per_batch),
            "skip_prompt_length_check": self.skip_prompt_length_check,
        }
        _save_multimodal_budget_snapshot(vllm_config)

    def _get_max_items(
        self,
        modality: str,
        max_tokens_per_item: int,
    ) -> tuple[int, int]:
        if max_tokens_per_item == 0:
            return 0, 0

        # Check how many items of this modality can be supported by
        # the encoder budget.
        if (encoder_budget := self.get_encoder_budget()) == 0:
            return 0, 0

        max_encoder_items_per_batch = encoder_budget // max_tokens_per_item

        # Check how many items of this modality can be supported by
        # the decoder budget.
        mm_limit = self.mm_limits[modality]

        max_items_per_prompt = max(
            1,
            min(mm_limit, self.max_model_len // max_tokens_per_item),
        )

        scheduler_config = self.scheduler_config
        max_num_reqs = self.max_num_reqs

        if not scheduler_config.enable_chunked_prefill:
            max_num_reqs = min(
                max_num_reqs,
                scheduler_config.max_num_batched_tokens // max_tokens_per_item,
            )

        max_decoder_items_per_batch = max_num_reqs * max_items_per_prompt

        max_items_per_batch = max(
            1,
            min(max_encoder_items_per_batch, max_decoder_items_per_batch),
        )

        return max_items_per_prompt, max_items_per_batch

    def get_modality_with_max_tokens(self) -> str:
        mm_max_toks_per_item = self.mm_max_toks_per_item
        modality, _ = max(mm_max_toks_per_item.items(), key=lambda x: (x[1], x[0]))

        return modality

    def get_encoder_budget(self) -> int:
        return min(self.encoder_compute_budget, self.encoder_cache_size)

    def reset_cache(self) -> None:
        if self.cache is not None:
            self.cache.clear_cache()


def get_dummy_encoder_profile_inputs(
    mm_registry: MultiModalRegistry,
    budget: MultiModalBudget,
) -> list[tuple[str, MultiModalKwargsItem]]:
    if budget.get_encoder_budget() <= 0 or not budget.mm_max_toks_per_item:
        return []

    modality = budget.get_modality_with_max_tokens()
    max_items_per_batch = budget.mm_max_items_per_batch[modality]
    dummy_mm_inputs = mm_registry.get_dummy_mm_inputs(
        budget.model_config,
        mm_counts={modality: 1},
        processor=budget.processor,
    )
    dummy_mm_item = dummy_mm_inputs["mm_kwargs"][modality][0]
    assert dummy_mm_item is not None, "Dummy item should be generated"

    return [(modality, dummy_mm_item)] * max_items_per_batch
