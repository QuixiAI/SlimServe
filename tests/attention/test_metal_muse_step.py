# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Muse layers retain independent KV groups and request histories."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Metal")


@pytest.fixture
def qc():
    from vllm.quixicore.ops import _qc

    module = _qc()
    yield module
    torch.mps.synchronize()
    module.muse_step_clear()


def _q8(matrix):
    # Exact integer weights in GGML Q8_0: fp16 scale then 32 signed bytes.
    rows, width = matrix.shape
    packed = torch.zeros(rows, width // 32, 34, dtype=torch.uint8)
    packed[:, :, :2] = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
    packed[:, :, 2:] = matrix.to(torch.int8).view(torch.uint8).reshape(rows, -1, 32)
    return packed.reshape(rows, -1).to("mps")


@pytest.mark.parametrize("one_request", [False, True])
def test_fused_layers_match_cpu_with_distinct_tables_and_expanded_rows(qc, one_request):
    width, block_size, context, rows = 256, 64, 1200, 2
    pages = (context + block_size - 1) // block_size
    blocks = 4 * pages + 4
    # Physical pages interleave both layers, including their K and V planes.
    backing = torch.zeros(blocks, 2, 2, block_size, 1, width, dtype=torch.bfloat16)
    expected = torch.zeros(rows, width, dtype=torch.bfloat16)
    tables, lengths, slots = [], [], []
    layer_updates = []
    for layer in range(2):
        patterns = []
        table_rows = []
        for request in range(1 if one_request else rows):
            first = 1 + (layer * 2 + request) * pages
            table_rows.append(list(range(first, first + pages)))
            pattern = ((torch.arange(width) + layer) % (3 + layer) - 1).bfloat16()
            if request:
                pattern = -pattern
            patterns.append(pattern)
            backing[first : first + pages, layer, 1] = pattern
        table = torch.tensor(table_rows, dtype=torch.int32)
        seq = torch.tensor(
            [context - 1, context] if one_request else [context] * rows,
            dtype=torch.int32,
        )
        if one_request:
            table = table.expand(rows, -1)
        slot = torch.tensor(
            [
                int(table[r, (int(seq[r]) - 1) // block_size]) * block_size
                + (int(seq[r]) - 1) % block_size
                for r in range(rows)
            ],
            dtype=torch.int64,
        )
        tables.append(
            table.to("mps") if not one_request else table[:1].to("mps").expand(rows, -1)
        )
        lengths.append(seq.to("mps"))
        slots.append(slot.to("mps"))
        # Zero Q/K/V projections: attention is a uniform mean of past values;
        # sigmoid(0) gates it by 1/2, then identity O and post-attention RMS.
        update = []
        for r in range(rows):
            past = context - 2 if one_request else context - 1
            mean = (
                patterns[0 if one_request else r].float() * past / int(seq[r])
            ).bfloat16()
            gated = (mean.float() * 0.5).bfloat16()
            normed = (
                gated.float() * torch.rsqrt(gated.float().square().mean() + 1e-8)
            ).bfloat16()
            update.append(normed)
        layer_updates.append(torch.stack(update))

    gpu_backing = backing.to("mps")
    caches = [gpu_backing[:, i].permute(1, 0, 2, 3, 4) for i in range(2)]
    zeros = _q8(torch.zeros(width, width))
    identity = _q8(torch.eye(width))
    norm = torch.ones(width, dtype=torch.bfloat16, device="mps")
    qc.muse_step_init(
        2, width, 1, 1, width, width, 2048, 10000.0, 1e-6, 1e-8, rows, norm
    )
    for layer in range(2):
        qc.muse_step_layer(
            layer,
            False,
            [zeros] * 3,
            [8] * 3,
            zeros,
            8,
            identity,
            8,
            [zeros] * 2,
            [8] * 2,
            zeros,
            8,
            norm,
            norm,
            norm,
            norm,
            norm,
            norm,
            caches[layer],
        )
    actual = expected.to("mps")
    aux = torch.empty(1, rows, width, dtype=torch.bfloat16, device="mps")
    qc.muse_step_run_aux(
        actual,
        torch.tensor([context - 2, context - 1], dtype=torch.int32, device="mps"),
        tables,
        lengths,
        slots,
        aux,
        [1],
        context,
        one_request,
    )
    expected = (expected + layer_updates[0]).bfloat16()
    torch.testing.assert_close(aux[0].cpu(), expected, atol=0.02, rtol=0.02)
    expected = (expected + layer_updates[1]).bfloat16()
    torch.testing.assert_close(actual.cpu(), expected, atol=0.03, rtol=0.02)
    for layer in range(2):
        for slot in slots[layer].cpu().tolist():
            backing[slot // block_size, layer, :, slot % block_size] = 0
    torch.testing.assert_close(gpu_backing.cpu(), backing, atol=0, rtol=0)
