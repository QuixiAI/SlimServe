# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-reference parity for GPU-resident Mamba boundary migration."""

import pytest
import torch

from vllm.quixicore.ops import _qc

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal")


@pytest.mark.parametrize("post", [False, True])
@pytest.mark.parametrize("dim_first", [False, True])
def test_align_matches_reference_with_padded_states_and_sentinels(post, dim_first):
    nblocks, width, channels, block_size = 16, 5, 7, 8
    shape = (channels, width) if dim_first else (width, channels)
    raw = torch.arange(nblocks * 64, dtype=torch.float32).view(nblocks, 64)
    conv = raw[:, : width * channels].view(nblocks, *shape)
    temporal_raw = raw.clone() + 10000
    temporal = temporal_raw[:, :32].view(nblocks, 4, 8)
    cpu_states = [conv.clone(), temporal.clone()]
    gpu_raw = [raw.to("mps"), temporal_raw.to("mps")]
    states = [
        gpu_raw[0][:, : width * channels].view(nblocks, *shape),
        gpu_raw[1][:, :32].view(nblocks, 4, 8),
    ]
    # Distinct group tables prove physical IDs are resolved per state group.
    tables = [
        torch.tensor(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [0, 0, 0, 0, 0, 0]],
            dtype=torch.int32,
        ),
        torch.tensor(
            [[6, 5, 4, 3, 2, 1], [12, 11, 10, 9, 8, 7], [0, 0, 0, 0, 0, 0]],
            dtype=torch.int32,
        ),
    ]
    mapping = torch.tensor([2, 0, -1], dtype=torch.int32)
    idx = torch.tensor([2 if post else 1, -1, 0, 3], dtype=torch.int32)
    accepted = torch.tensor([5, 1, 3, 9], dtype=torch.int32)
    computed = (
        torch.tensor([18, 0, 9, 100], dtype=torch.int32)
        if post
        else torch.tensor([16, 0, 8, 100], dtype=torch.int32)
    )
    starts = torch.tensor([0, 3, 5, 5], dtype=torch.int32)
    expected_idx, expected_accepted = idx.clone(), accepted.clone()
    for batch, req in enumerate(mapping.tolist()):
        if req < 0:
            continue
        source = int(idx[req])
        if post:
            running = int(computed[req] - accepted[req] + 1)
            aligned = int(computed[req]) // block_size * block_size
            target, shift = aligned // block_size - 1, aligned - running
            if aligned < running or target < 0:
                continue
            if source == target:
                expected_accepted[req] = 1
            if source == target and shift == 0:
                continue
        else:
            after = int(computed[req] + starts[batch + 1] - starts[batch])
            target = (after + block_size - 1) // block_size - 1
            expected_idx[req] = target
            shift = int(accepted[req]) - 1
            if source < 0 or source == target:
                continue
            expected_accepted[req] = 1
        src, dst = int(tables[0][batch, source]), int(tables[0][batch, target])
        if dim_first:
            cpu_states[0][dst, :, : width - shift] = cpu_states[0][
                src, :, shift:
            ].clone()
        else:
            cpu_states[0][dst, : width - shift] = cpu_states[0][src, shift:].clone()
        src = int(tables[1][batch, source + shift])
        dst = int(tables[1][batch, target])
        cpu_states[1][dst] = cpu_states[1][src].clone()
    gpu_idx, gpu_accepted = idx.to("mps"), accepted.to("mps")
    scratch = [torch.empty_like(gpu_idx) for _ in range(3)]
    _qc().mamba_align(
        states,
        [t.to("mps") for t in tables],
        [0, 1],
        [2 if dim_first else 1, 0],
        mapping.to("mps"),
        gpu_idx,
        computed.to("mps"),
        starts.to("mps"),
        gpu_accepted,
        *scratch,
        block_size,
        post,
    )
    assert torch.equal(gpu_idx.cpu(), expected_idx)
    assert torch.equal(gpu_accepted.cpu(), expected_accepted)
    for result, expected in zip(states, cpu_states):
        assert torch.equal(result.cpu(), expected)
    assert torch.equal(
        gpu_raw[0][:, width * channels :].cpu(), raw[:, width * channels :]
    )
    assert torch.equal(gpu_raw[1][:, 32:].cpu(), temporal_raw[:, 32:])
