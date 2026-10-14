# SPDX-License-Identifier: Apache-2.0
"""Metadata routing/wiring tests; GPU arithmetic parity is a separate gate."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from benchmarks import glm5_next_kda_metadata_candidate as candidate
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices


@pytest.mark.parametrize(
    "prefills,decodes,spec,mask,eligible",
    [
        (1, 0, 0, None, True),
        (16, 0, 0, None, True),
        (0, 16, 0, None, False),
        (1, 8, 0, None, False),
        (1, 0, 1, None, False),
        (1, 0, 0, object(), False),
    ],
)
def test_only_pure_prefill_reuses_exact_table(prefills, decodes, spec, mask, eligible):
    table = object()
    metadata = SimpleNamespace(
        num_prefills=prefills,
        num_decodes=decodes,
        num_spec_decodes=spec,
        spec_sequence_masks=mask,
        chunk_indices=table,
    )
    assert candidate.pure_prefill_indices(metadata) is (table if eligible else None)


def test_mixed_table_does_not_describe_kda_rows():
    # One existing decode and two prefills. GDN peels the decode row off,
    # whereas current KDA's chunk path includes it in the non-spec batch.
    full = prepare_chunk_indices(torch.tensor([0, 1, 66, 67]), 64)
    prefill = prepare_chunk_indices(torch.tensor([0, 65, 66]), 64)
    assert full.tolist() == [[0, 0], [1, 0], [1, 1], [2, 0]]
    assert prefill.tolist() == [[0, 0], [0, 1], [1, 0]]


@pytest.mark.parametrize("lower_bound", [None, -5.0])
@pytest.mark.parametrize("normalize", [False, True])
def test_same_table_and_gate_parameters_reach_both_kernel_stages(
    monkeypatch, lower_bound, normalize
):
    q, k, v, raw = (torch.ones(1, 2, 1, 4) for _ in range(4))
    beta = torch.ones(1, 2, 1)
    a_log, bias = torch.ones(1), torch.ones(4)
    state = torch.ones(2, 1, 4, 4)
    cu = torch.tensor([0, 1, 2], dtype=torch.int32)
    chunks = prepare_chunk_indices(cu, 64)
    gate = Mock(return_value=(raw, beta))
    core = Mock(return_value=(v, state))
    norm = Mock(side_effect=lambda x: x)
    monkeypatch.setattr(candidate, "fused_kda_gate_chunk_cumsum", gate)
    monkeypatch.setattr(candidate, "_chunk_kda_fwd_with_cumulative_g", core)
    monkeypatch.setattr(candidate, "l2norm_fwd", norm)
    output, final = candidate.chunk_kda_precomputed(
        q=q,
        k=k,
        v=v,
        raw_g=raw,
        raw_beta=beta,
        A_log=a_log,
        g_bias=bias,
        cu_seqlens=cu,
        chunk_indices=chunks,
        initial_state=state,
        output_final_state=True,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=normalize,
    )
    assert output is v and final is state
    assert norm.call_count == (2 if normalize else 0)
    for call in [gate.call_args, core.call_args]:
        assert call.kwargs["chunk_indices"] is chunks
        assert call.kwargs["cu_seqlens"] is cu
        assert call.kwargs["chunk_size"] == 64
    assert gate.call_args.kwargs["lower_bound"] == lower_bound
    assert core.call_args.kwargs["safe_gate"] == (lower_bound is not None)
    assert core.call_args.kwargs["scale"] == 0.5
    assert core.call_args.kwargs["output_final_state"] is True
