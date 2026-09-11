# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM MTP metadata for independent MLA and pooled-indexer cache groups.

Follows the existing Qwen multi-owner proposer design. Unlike QSA, GLM's
indexer consumes the common slot mapping directly, so each group's mapping
must also be rebuilt using that group's block size and physical block IDs.
"""

from copy import copy

import torch

from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID, compute_new_slot_mapping
from vllm.v1.worker.utils import AttentionGroup


class Glm5NextMTPProposer(EagleProposer):
    def __init__(self, vllm_config, device, runner=None):
        super().__init__(vllm_config, device, runner)
        self._per_group_block_tables = {}
        self._group_slot_buffers = {}
        self._group_block_sizes = {}
        self._group_num_actual_tokens = 0

    def model_returns_tuple(self) -> bool:
        return True

    def set_per_group_block_table(self, gid, block_table):
        self._per_group_block_tables[gid] = block_table

    def initialize_attn_backend(self, kv_cache_config, kernel_block_sizes=None):
        assert self.draft_model_config.hf_config.num_nextn_predict_layers == 1
        assert kernel_block_sizes is not None
        assert len(kernel_block_sizes) == len(kv_cache_config.kv_cache_groups)
        layers = get_layers_from_vllm_config(
            self.vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )
        owners = {}
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in self._draft_attn_layer_names & set(group.layer_names):
                spec = group.kv_cache_spec
                if isinstance(spec, UniformTypeKVCacheSpecs):
                    spec = spec.kv_cache_specs[name]
                owners[name] = (gid, spec)
        assert owners.keys() == self._draft_attn_layer_names, "Missing GLM draft caches"
        main = [
            name
            for name in owners
            if layers[name].get_attn_backend().get_name() == "QUIXICORE_MLA_SPARSE"
        ]
        indexers = [
            name
            for name in owners
            if layers[name].get_attn_backend().get_name() == "GLM5_NEXT_INDEXER"
        ]
        assert len(main) == len(indexers) == 1 and len(owners) == 2, (
            "GLM MTP requires one MLA cache and one pooled-indexer cache"
        )
        self.kv_cache_gid = owners[main[0]][0]
        self.draft_attn_groups = []
        self._group_block_sizes = {}
        self._group_slot_buffers = {}
        self._group_num_actual_tokens = 0
        for name in main + indexers:
            gid, spec = owners[name]
            group = AttentionGroup(
                backend=layers[name].get_attn_backend(),
                layer_names=[name],
                kv_cache_spec=spec,
                kv_cache_group_id=gid,
            )
            group.create_metadata_builders(
                self.vllm_config,
                self.device,
                kernel_block_size=kernel_block_sizes[gid],
            )
            self.draft_attn_groups.append(group)
            self._group_block_sizes[gid] = kernel_block_sizes[gid]
            if gid not in self._group_slot_buffers:
                buf = (
                    self._slot_mapping_buffer
                    if gid == self.kv_cache_gid
                    else torch.empty_like(self._slot_mapping_buffer)
                )
                buf.fill_(PADDING_SLOT_ID)
                self._group_slot_buffers[gid] = buf
        self.block_size = self._group_block_sizes[self.kv_cache_gid]

    def build_per_group_and_layer_attn_metadata(self, common, draft_index=0):
        num_tokens = common.num_actual_tokens
        self._group_num_actual_tokens = num_tokens
        positions = self._get_positions(num_tokens)
        assert positions.ndim == 1, "GLM uses one-dimensional NoPE positions"
        # Preserve rejected/padded slots when recomputing another group's IDs.
        rejected = common.slot_mapping[:num_tokens] < 0
        common_by_gid = {}
        per_group, per_layer = [], {}
        for group in self.draft_attn_groups:
            gid = group.kv_cache_group_id
            cm = common_by_gid.get(gid)
            if cm is None:
                cm = copy(common)
                buf = self._group_slot_buffers[gid]
                if gid == self.kv_cache_gid:
                    if buf.data_ptr() != common.slot_mapping.data_ptr():
                        buf[:num_tokens].copy_(common.slot_mapping[:num_tokens])
                else:
                    table = self._per_group_block_tables.get(gid)
                    assert table is not None, f"Missing GLM draft block table: {gid}"
                    cm.block_table_tensor = table[: common.num_reqs]
                    slots = compute_new_slot_mapping(
                        cad=cm,
                        new_positions=positions,
                        is_rejected_token_mask=rejected,
                        block_size=self._group_block_sizes[gid],
                        num_new_tokens=0,
                        max_model_len=self.max_model_len,
                    )
                    buf[:num_tokens].copy_(slots)
                cm.slot_mapping = buf[:num_tokens]
                common_by_gid[gid] = cm
            metadata = group.get_metadata_builder().build_for_drafting(
                common_attn_metadata=cm, draft_index=draft_index
            )
            per_group.append(metadata)
            for name in group.layer_names:
                per_layer[name] = metadata
        return per_group, per_layer

    def _get_slot_mapping(self, num_tokens, slot_mapping=None):
        # Metadata construction above populated stable, separate buffers.
        # Do not broadcast the primary group's physical slots into the indexer.
        per_layer = {}
        for group in self.draft_attn_groups:
            buf = self._group_slot_buffers[group.kv_cache_group_id]
            buf[self._group_num_actual_tokens : num_tokens].fill_(PADDING_SLOT_ID)
            for name in group.layer_names:
                per_layer[name] = buf[:num_tokens]
        return per_layer
