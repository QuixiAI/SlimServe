# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metal align-mode state migration with GPU-resident request decisions."""

import torch

from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
    is_conv_state_dim_first,
)
from vllm.quixicore.ops import _qc
from vllm.v1.worker.mamba_utils import (
    _get_mamba_spec_for_layer,
    get_mamba_group_ids,
    get_mamba_groups,
    validate_mamba_state_copy_funcs,
)


class MetalMambaAlign:
    def __init__(self, config, model, kv_config, state_idx):
        groups = get_mamba_groups(kv_config)
        self.group_ids = get_mamba_group_ids(groups)
        funcs = model.get_mamba_state_copy_funcs({spec.mamba_type for spec in groups})
        validate_mamba_state_copy_funcs(groups, funcs)
        forward = config.compilation_config.static_forward_context
        self.states, self.groups, self.kinds = [], [], []
        for local_gid, gid in enumerate(self.group_ids):
            group = kv_config.kv_cache_groups[gid]
            for name in group.layer_names:
                spec = _get_mamba_spec_for_layer(group, name)
                states = forward[name].kv_cache
                copy_funcs = funcs[spec.mamba_type]
                assert len(states) == len(copy_funcs)
                for state, func in zip(states, copy_funcs):
                    if func is get_conv_copy_spec:
                        kind = 2 if is_conv_state_dim_first() else 1
                    else:
                        assert func is get_temporal_copy_spec
                        kind = 0
                    self.states.append(state)
                    self.groups.append(local_gid)
                    self.kinds.append(kind)
        self.src = torch.empty_like(state_idx)
        self.dst = torch.empty_like(state_idx)
        self.bias = torch.empty_like(state_idx)
        self.tables = []

    def run(
        self,
        mapping,
        state_idx,
        computed,
        query_start,
        accepted,
        block_size,
        *,
        post=False,
        block_tables=None,
    ):
        if block_tables is not None:
            self.tables = [block_tables[gid] for gid in self.group_ids]
        _qc().mamba_align(
            self.states,
            self.tables,
            self.groups,
            self.kinds,
            mapping.to(torch.int32),
            state_idx,
            computed,
            computed if query_start is None else query_start,
            accepted,
            self.src,
            self.dst,
            self.bias,
            block_size,
            post,
        )
