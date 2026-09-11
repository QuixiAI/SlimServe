# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from benchmarks.kernels import benchmark_glm53_mhc_storage as probe
from benchmarks.kernels.benchmark_glm53_mhc_storage import exact, lossless_bf16


@pytest.mark.parametrize("indices", [[], [-1], [90], [0, 0]])
def test_sanitizer_site_selection_rejects_invalid_indices(indices):
    with pytest.raises(ValueError, match="distinct indices"):
        probe.selected_check_sites(indices)


def test_sanitizer_site_selection_defaults_to_exhaustive_census():
    assert probe.selected_check_sites(None) == list(range(90))
    assert probe.selected_check_sites([0, 89]) == [0, 89]


def test_storage_conversion_requires_exact_checkpoint_values():
    value = torch.tensor([0.0, -1.0, 0.125, 1e-20], dtype=torch.bfloat16).float()
    assert torch.equal(lossless_bf16(value).float(), value)
    with pytest.raises(ValueError, match="alter checkpoint"):
        lossless_bf16(torch.tensor([0.1], dtype=torch.float32))


def test_storage_probe_rejects_nan_and_any_output_difference():
    with pytest.raises(ValueError):
        lossless_bf16(torch.tensor([float("nan")]))
    value = torch.ones(2, dtype=torch.bfloat16)
    exact([value], [value.clone()])
    with pytest.raises(AssertionError, match="bit-exact"):
        exact([value], [value + 0.01])
    with pytest.raises(AssertionError, match="bit-exact"):
        exact([value], [value.float()])


def test_exact_probe_checks_signed_zero_and_rejects_nan():
    for dtype in (torch.float32, torch.bfloat16):
        positive = torch.tensor([0.0], dtype=dtype)
        negative = torch.tensor([-0.0], dtype=dtype)
        assert torch.equal(positive, negative)
        with pytest.raises(AssertionError, match="signed zero"):
            exact([positive], [negative])
        nan = torch.tensor([float("nan")], dtype=dtype)
        with pytest.raises(AssertionError, match="bit-exact"):
            exact([nan], [nan.clone()])


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("banks", [2, 6])
def test_timing_banks_share_only_activations_when_requested(monkeypatch, shared, banks):
    seeds = []

    def fake_inputs(batch, seed):
        seeds.append(seed)
        return [torch.full((batch, 2), float(seed)) for _ in range(4)]

    monkeypatch.setattr(probe, "inputs", fake_inputs)
    sites = [[torch.ones(4, 2) * i, torch.ones(3), torch.ones(24)] for i in range(3)]
    narrow = [site[0].bfloat16() for site in sites]
    rows = probe.timing_rows(sites, narrow, 2, shared, banks)
    assert len(rows) == 3 * banks
    assert seeds == (
        list(range(5001, 5001 + banks * 3, 3))
        if shared
        else list(range(5001, 5001 + banks * 3))
    )
    assert [row[2] for row in rows] == [False, True, True] * banks
    for variant in (0, 1):
        assert len({row[variant][4].data_ptr() for row in rows}) == 3 * banks
        for index, row in enumerate(rows):
            expected = narrow[index % 3] if variant else sites[index % 3][0]
            assert torch.equal(row[variant][4], expected)
            assert row[variant][4].data_ptr() != expected.data_ptr()
        assert len({row[variant][0].data_ptr() for row in rows}) == (
            banks if shared else 3 * banks
        )
    for row in rows:
        assert all(a.data_ptr() == b.data_ptr() for a, b in zip(row[0][:4], row[1][:4]))


def test_timing_banks_reject_mismatched_site_counts():
    with pytest.raises(ValueError, match="site counts"):
        probe.timing_rows([[]], [], 1)
    with pytest.raises(ValueError, match="positive weight bank"):
        probe.timing_rows([], [], 1, weight_banks=0)
