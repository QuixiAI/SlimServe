# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proposer for the GLM-5.3-Flash MTP head.

The draft layer owns two KV cache layers that land in two scheduler groups:
the sparse NoPE MLA cache (``...self_attn.attn``, the primary owner whose
group carries the proposer's block table and slot mapping) and the pooled
indexer's key cache (``...self_attn.indexer.k_cache``). The base proposer
assumes one group and hands the drafter one slot mapping; the indexer's
rows would then be inserted at the MLA group's slot numbers. This builds on
``Step3p5MTPProposer``, which stages a block table AND a slot mapping per
group from the runner and recomputes the non-primary groups' slot mappings
per draft step from their own block tables.
"""

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer
from vllm.v1.spec_decode.utils import PADDING_SLOT_ID
from vllm.v1.worker.utils import AttentionGroup

MAIN_LAYER_SUFFIX = ".self_attn.attn"


class Glm5NextMTPProposer(Step3p5MTPProposer):
    """Speculative decoding proposer for GLM-5.3-Flash MTP."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ) -> None:
        super().__init__(vllm_config, device, runner)

    def model_returns_tuple(self) -> bool:
        """The draft returns (pre-norm hidden, recycled post-norm hidden)."""
        return True

    def _maybe_share_lm_head(self, target_language_model: torch.nn.Module) -> None:
        # Unlike Step3.5, the GLM-5.3 head has no lm_head of its own: the
        # checkpoint carries shared_head.norm only, the target's lm_head is
        # shared in (base behaviour).
        EagleProposer._maybe_share_lm_head(self, target_language_model)

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

        # Primary owner first: the base proposer reads the first group's
        # block table and slot mapping.
        self.draft_attn_groups = sorted(
            attention_groups,
            key=lambda group: (
                group.kv_cache_group_id != self.kv_cache_gid,
                group.kv_cache_group_id,
                group.layer_names[0],
            ),
        )
        self.block_size = kernel_block_sizes[self.kv_cache_gid]
        # The indexer group's block is not the MLA group's (2176 vs 1088
        # tokens on the rtx6000 record: hybrid page-size unification), so
        # the per-step slot recomputation below uses each group's own size.
        self._per_group_block_sizes = {
            g.kv_cache_group_id: kernel_block_sizes[g.kv_cache_group_id]
            for g in self.draft_attn_groups
        }

    def _update_positions_dependent_metadata(
        self,
        positions: torch.Tensor,
        common_attn_metadata,
        batch_size: int,
        input_batch_size: int,
        block_size: int,
    ) -> torch.Tensor:
        """Step3p5's recompute, with each non-primary group's own block size."""
        old_positions_1d = positions[0] if self.uses_mrope else positions
        positions = EagleProposer._update_positions_dependent_metadata(
            self,
            positions,
            common_attn_metadata,
            batch_size,
            input_batch_size,
            block_size,
        )
        self._per_group_slot_mappings[self.kv_cache_gid] = (
            common_attn_metadata.slot_mapping
        )
        new_positions_1d = positions[0] if self.uses_mrope else positions
        exceeds = old_positions_1d + 1 >= self.max_model_len
        for attn_group in self.draft_attn_groups:
            gid = attn_group.kv_cache_group_id
            if gid == self.kv_cache_gid:
                continue
            block_table = self._per_group_block_tables.get(gid)
            if block_table is None:
                continue
            gbs = self._per_group_block_sizes[gid]
            n_blocks = block_table.shape[1]
            bn = new_positions_1d // gbs
            bn.clamp_(max=n_blocks - 1)
            bn = bn.to(torch.long)
            block_ids = block_table[:batch_size].gather(1, bn.unsqueeze(1)).squeeze(1)
            sm = block_ids * gbs + (new_positions_1d % gbs)
            sm.masked_fill_(exceeds, PADDING_SLOT_ID)
            buf = self._slot_mapping_buffer_for(gid)
            buf[:batch_size].copy_(sm)
            if input_batch_size > batch_size:
                buf[batch_size:input_batch_size].fill_(PADDING_SLOT_ID)
            self._per_group_slot_mappings[gid] = buf[:batch_size]
        return positions

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
