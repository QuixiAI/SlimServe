# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proposer for the GLM-5.3-Flash MTP head.

The draft layer owns two KV cache layers that land in two scheduler groups:
the sparse NoPE MLA cache (``...self_attn.attn``, the main owner whose
group carries the proposer's block table and slot mapping) and the pooled
indexer's key cache (``...self_attn.indexer.k_cache``). The base proposer
assumes one group; this follows ``Qwen4ExpMTPProposer`` and builds each
owner's metadata from its own group's block table.
"""

from copy import copy

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.worker.utils import AttentionGroup

MAIN_LAYER_SUFFIX = ".self_attn.attn"


class Glm5NextMTPProposer(EagleProposer):
    """Speculative decoding proposer for GLM-5.3-Flash MTP."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner)
        self._per_group_block_tables: dict[int, torch.Tensor] = {}

    def model_returns_tuple(self) -> bool:
        """The draft returns (pre-norm hidden, recycled post-norm hidden)."""
        return True

    def set_per_group_block_table(self, gid: int, block_table: torch.Tensor) -> None:
        """Stage one scheduler group's block table for drafting."""
        self._per_group_block_tables[gid] = block_table

    def build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int = 0,
    ) -> tuple[list[object], dict[str, object]]:
        per_group_attn_metadata: list[object] = []
        per_layer_attn_metadata: dict[str, object] = {}
        common_by_gid: dict[int, CommonAttentionMetadata] = {}
        num_reqs = common_attn_metadata.num_reqs

        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            group_common = common_by_gid.get(gid)
            if group_common is None:
                if gid == self.kv_cache_gid:
                    group_common = common_attn_metadata
                else:
                    block_table = self._per_group_block_tables.get(gid)
                    assert block_table is not None, (
                        f"Missing GLM-5.3 draft block table for KV cache group {gid}"
                    )
                    group_common = copy(common_attn_metadata)
                    group_common.block_table_tensor = block_table[:num_reqs]
                common_by_gid[gid] = group_common

            attn_metadata = attn_group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=group_common,
                draft_index=draft_index,
            )
            per_group_attn_metadata.append(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        return per_group_attn_metadata, per_layer_attn_metadata

    def initialize_attn_backend(
        self,
        kv_cache_config: KVCacheConfig,
        kernel_block_sizes: list[int] | None = None,
    ) -> None:
        num_mtp_layers = getattr(
            self.draft_model_config.hf_config, "num_nextn_predict_layers", 1
        )
        if num_mtp_layers != 1:
            raise NotImplementedError(
                "GLM-5.3 MTP proposer only supports one MTP layer"
            )
        assert kernel_block_sizes is not None, (
            "GLM-5.3 MTP requires resolved kernel block sizes"
        )
        assert len(kernel_block_sizes) == len(kv_cache_config.kv_cache_groups), (
            "GLM-5.3 MTP requires one kernel block size per KV cache group"
        )

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        layer_to_gid, layer_to_spec = self._map_draft_layers_to_groups(kv_cache_config)
        main_layers = [n for n in layer_to_spec if n.endswith(MAIN_LAYER_SUFFIX)]
        assert len(main_layers) == 1, (
            "GLM-5.3 MTP requires exactly one sparse MLA cache owner, got "
            f"{sorted(main_layers)}"
        )
        self.kv_cache_gid = layer_to_gid[main_layers[0]]

        attention_groups: list[AttentionGroup] = []
        for layer_name in sorted(self._draft_attn_layer_names):
            attn_layer = all_attn_layers[layer_name]
            gid = layer_to_gid[layer_name]
            attn_group = AttentionGroup(
                backend=attn_layer.get_attn_backend(),
                layer_names=[layer_name],
                kv_cache_spec=layer_to_spec[layer_name],
                kv_cache_group_id=gid,
            )
            attn_group.create_metadata_builders(
                self.vllm_config,
                self.device,
                kernel_block_size=kernel_block_sizes[gid],
            )
            attention_groups.append(attn_group)

        # Main owner first so the base proposer's single-group assumptions
        # (block table, slot mapping) read the sparse MLA group.
        self.draft_attn_groups = sorted(
            attention_groups,
            key=lambda group: (
                group.kv_cache_group_id != self.kv_cache_gid,
                group.kv_cache_group_id,
                group.layer_names[0],
            ),
        )
        self.block_size = kernel_block_sizes[self.kv_cache_gid]

    def _map_draft_layers_to_groups(
        self,
        kv_cache_config: KVCacheConfig,
    ) -> tuple[dict[str, int], dict[str, KVCacheSpec]]:
        layer_to_gid: dict[str, int] = {}
        layer_to_spec: dict[str, KVCacheSpec] = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            group_spec = group.kv_cache_spec
            for layer_name in group.layer_names:
                if layer_name not in self._draft_attn_layer_names:
                    continue
                if isinstance(group_spec, UniformTypeKVCacheSpecs):
                    spec = group_spec.kv_cache_specs.get(layer_name)
                    assert spec is not None, (
                        f"GLM-5.3 draft cache group {gid} has no spec for {layer_name}"
                    )
                else:
                    spec = group_spec
                layer_to_gid[layer_name] = gid
                layer_to_spec[layer_name] = spec

        assert layer_to_spec.keys() == self._draft_attn_layer_names, (
            "GLM-5.3 draft KV cache configuration is missing layers: "
            f"{sorted(self._draft_attn_layer_names - layer_to_spec.keys())}"
        )
        return layer_to_gid, layer_to_spec


__all__ = ["Glm5NextMTPProposer"]
