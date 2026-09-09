# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the independent gate; original strict parity stays unchanged."""

import math

import pytest
import torch

from benchmarks.kernels.mhc_fp64_oracle import (
    accuracy_pair,
    direct_bf16_rne,
    ideal_outputs,
    layer_accuracy,
)


@pytest.mark.parametrize("negative", [False, True])
def test_every_finite_bf16_midpoint_and_its_fp64_neighbors(negative):
    bits = torch.arange(0x7F7F, dtype=torch.int32)
    lo = bits.to(torch.int16).view(torch.bfloat16).double()
    hi = (bits + 1).to(torch.int16).view(torch.bfloat16).double()
    middle = (lo + hi) / 2
    ties = torch.where((bits & 1) == 0, lo, hi)
    if negative:
        lo, hi, middle, ties = -hi, -lo, -middle, -ties
    values = torch.stack(
        [
            torch.nextafter(middle, torch.full_like(middle, -math.inf)),
            middle,
            torch.nextafter(middle, torch.full_like(middle, math.inf)),
        ]
    )
    wanted = torch.stack([lo, ties, hi]).bfloat16()
    assert torch.equal(
        direct_bf16_rne(values).view(torch.int16), wanted.view(torch.int16)
    )


def test_direct_rounding_preserves_signed_zero_and_extrema():
    biggest = torch.finfo(torch.bfloat16).max
    values = torch.tensor([0.0, -0.0, biggest, -biggest], dtype=torch.float64)
    assert torch.equal(
        direct_bf16_rne(values).view(torch.int16), values.bfloat16().view(torch.int16)
    )


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan, 3.4e38])
def test_direct_rounding_rejects_nonfinite_and_overflow(value):
    with pytest.raises(ValueError, match="finite BF16-range"):
        direct_bf16_rne(torch.tensor([value], dtype=torch.float64))


def test_direct_rounding_rejects_wrong_dtype():
    with pytest.raises(ValueError, match="CPU float64"):
        direct_bf16_rne(torch.ones(2))


def test_independent_boundary_favors_candidate_without_changing_old_parity():
    from benchmarks.kernels.benchmark_glm53_mhc_prefill_tc import check_outputs

    ideal = torch.tensor([[2.007812650812945]], dtype=torch.float64)
    budget = torch.full_like(ideal, 5e-5)
    old = torch.tensor([[2.0]], dtype=torch.bfloat16)
    new = torch.tensor([[2.015625]], dtype=torch.bfloat16)
    old_accuracy = layer_accuracy(old, ideal, budget)
    new_accuracy = layer_accuracy(new, ideal, budget)
    assert old_accuracy["violations"] == new_accuracy["violations"] == 0
    assert new_accuracy["not_ideally_rounded"] == 0
    assert old_accuracy["not_ideally_rounded"] == 1
    assert new_accuracy["squared_error"] < old_accuracy["squared_error"]
    prefix = [torch.zeros(1, dtype=torch.bfloat16), torch.ones(1), torch.ones(1)]
    with pytest.raises(AssertionError, match="0.0078125"):
        check_outputs([*prefix, old], [*prefix, new])
    bad = torch.tensor([[2.03125]], dtype=torch.bfloat16)
    assert layer_accuracy(bad, ideal, budget)["violations"] == 1


def test_layer_gate_rejects_one_ulp_away_from_an_exact_result():
    ideal = torch.tensor([[2.0]], dtype=torch.float64)
    assert (
        layer_accuracy(
            torch.tensor([[2.015625]], dtype=torch.bfloat16),
            ideal,
            torch.full_like(ideal, 5e-5),
        )["violations"]
        == 1
    )


def fixture(rows=2):
    r = torch.stack([torch.full((rows, 4096), float(i)) for i in range(1, 5)], 1)
    w = torch.zeros(24, 16384)
    scale = torch.zeros(3)
    base = torch.zeros(24)
    return r.bfloat16(), w.bfloat16(), scale, base


def test_full_equations_uniform_closed_form_and_error_budget():
    args = [t.double() for t in fixture()]
    post, comb, layer, budget = ideal_outputs(*args)
    torch.testing.assert_close(post, torch.ones_like(post), rtol=0, atol=0)
    torch.testing.assert_close(
        layer, torch.full_like(layer, 5.00001), rtol=0, atol=1e-14
    )
    scalar = 0.25 + 1e-6
    for i in range(20):
        if i:
            scalar /= 4 * scalar + 1e-6
        scalar /= 4 * scalar + 1e-6
    torch.testing.assert_close(comb, torch.full_like(comb, scalar), rtol=0, atol=0)
    assert (budget > 0).all()


def test_full_projection_uses_stream_and_output_coordinates():
    args = [t.double() for t in fixture()]
    r, w, scale, base = args
    # Nonuniform stream and output weights, with exact binary products.
    for mix in range(24):
        for stream in range(4):
            w[mix, stream * 4096 : (stream + 1) * 4096] = (
                (mix + 1) * (stream + 1) / 65536
            )
    scale[:] = 0.03
    post, _, layer, _ = ideal_outputs(*args)
    rms = math.sqrt(7.5 + 1e-5)
    pre = [
        1 / (1 + math.exp(-((mix + 1) * 30 / 16) / rms * 0.03)) + 1e-6
        for mix in range(4)
    ]
    wanted_layer = sum((i + 1) * pre[i] for i in range(4))
    torch.testing.assert_close(
        layer, torch.full_like(layer, wanted_layer), rtol=1e-14, atol=0
    )
    for mix in range(4, 8):
        wanted_post = 2 / (1 + math.exp(-((mix + 1) * 30 / 16) / rms * 0.03))
        torch.testing.assert_close(
            post[:, mix - 4],
            torch.full_like(post[:, mix - 4], wanted_post),
            rtol=1e-14,
            atol=0,
        )


def test_pair_checks_every_row_and_chunk_size_does_not_change_metrics():
    r, w, scale, base = fixture(3)
    post, comb, layer, _ = ideal_outputs(*(t.double() for t in (r, w, scale, base)))
    outputs = [post.float(), comb.float(), direct_bf16_rne(layer)]
    a = accuracy_pair(r, w, scale, base, outputs, outputs, chunk_rows=1)
    b = accuracy_pair(r, w, scale, base, outputs, outputs, chunk_rows=3)
    assert a["passed"] and b["passed"] and a["rows"] == 3
    assert a["candidate"]["elements"] == 3 * 4096
    assert a["candidate"]["normalized_rms"] == pytest.approx(
        b["candidate"]["normalized_rms"]
    )
    bad = [t.clone() for t in outputs]
    bad[-1][-1, -1] *= 1.25
    result = accuracy_pair(r, w, scale, base, outputs, bad, chunk_rows=1)
    assert not result["passed"]
    assert result["candidate"]["violations"] == 1


@pytest.mark.parametrize("index", [0, 1])
def test_pair_checks_both_fp32_outputs(index):
    r, w, scale, base = fixture()
    p, c, y, _ = ideal_outputs(*(t.double() for t in (r, w, scale, base)))
    outputs = [p.float(), c.float(), direct_bf16_rne(y)]
    bad = [t.clone() for t in outputs]
    bad[index].flatten()[-1] *= 1.1
    result = accuracy_pair(r, w, scale, base, outputs, bad)
    assert not result["passed"]
    assert (
        result["candidate"]["post_violations" if index == 0 else "comb_violations"] == 1
    )


def test_oracle_rejects_shape_nonfinite_and_dtype_changes():
    r, w, scale, base = fixture()
    args = [t.double() for t in (r, w, scale, base)]
    with pytest.raises(ValueError, match="shapes"):
        ideal_outputs(args[0][:, :3], *args[1:])
    args[1][0, 0] = math.nan
    with pytest.raises(ValueError, match="nonfinite"):
        ideal_outputs(*args)
    with pytest.raises(ValueError, match="CPU float64"):
        ideal_outputs(r, *args[1:])


def test_accuracy_census_keeps_original_failure_and_held_out_seeds():
    from benchmarks.kernels.check_glm53_mhc_tc_accuracy import case_plan

    plan = case_plan([64, 65, 128, 129, 7616], list(range(90)))
    assert len(plan) == 2700
    bad = [
        p
        for p in plan
        if p["site"] == 79
        and p["batch"] == 7616
        and p["magnitude"] == 1.0
        and p["fused"]
    ]
    assert len(bad) == 1 and bad[0]["eager_seed"] == 2240
    assert bad[0]["graph_seeds"] == [12238, 12239, 12240]
    assert len(case_plan([7616], [79])) == 6
    for batches, sites in [
        ([63], [0]),
        ([64, 64], [0]),
        ([64], [90]),
        ([64], [0, 0]),
        ([], [0]),
        ([64], []),
    ]:
        with pytest.raises(ValueError):
            case_plan(batches, sites)


def test_pair_requires_nonempty_rows_and_native_parameter_precision():
    r, w, s, b = fixture()
    with pytest.raises(ValueError, match="nonempty"):
        accuracy_pair(r[:0], w, s, b, [], [])
    with pytest.raises(ValueError, match="native FP32"):
        accuracy_pair(r, w, s.bfloat16(), b, [], [])
