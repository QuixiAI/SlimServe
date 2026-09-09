# SPDX-License-Identifier: Apache-2.0
import torch

from benchmarks.kernels.benchmark_glm53_indexer_tiles import (
    BASELINE,
    configurations,
    oracle,
)


def test_indexer_geometries_include_production_and_no_duplicates():
    configs = configurations()
    assert BASELINE in configs
    assert len(configs) == len(set(configs)) == 30


def test_indexer_cpu_oracle_reduces_heads_after_relu_and_masks_visibility():
    query = torch.ones((2, 32, 128), dtype=torch.bfloat16)
    keys = torch.ones((1, 3, 128), dtype=torch.bfloat16)
    keys[:, 1] = -1
    weights = torch.full((2, 32), 0.5)
    visible = torch.tensor([4, 8])
    scores, valid = oracle(query, weights, keys, visible, [0, 1], [0, 1, 2])
    assert torch.equal(valid, torch.tensor([[True, False, False], [True, True, False]]))
    assert scores[:, 1].eq(0).all()
    torch.testing.assert_close(
        scores[:, 0], torch.full((2,), 16 * 128**0.5, dtype=torch.float64)
    )
