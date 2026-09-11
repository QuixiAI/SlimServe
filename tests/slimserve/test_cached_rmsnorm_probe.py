# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from benchmarks.kernels.check_glm53_cached_rmsnorm import (
    INPLACE_ARGS,
    TRIPLE_ARGS,
    checked_config,
    compare,
    make_inputs,
    oracle,
    source_layout,
)


@pytest.mark.parametrize(
    "name", ["triton_red_fused_rms_norm_0", "triton_red_fused_rms_norm_1"]
)
@pytest.mark.parametrize(
    "args,expected", [(INPLACE_ARGS, "inplace"), (TRIPLE_ARGS, "triple")]
)
def test_layout_uses_signature_not_kernel_suffix(name, args, expected):
    assert source_layout(f"def {name}({','.join(args)}):\n    pass\n", name) == expected


def test_unknown_or_ambiguous_source_fails_closed():
    with pytest.raises(ValueError, match="unrecognized"):
        source_layout("def norm(x): pass", "norm")
    with pytest.raises(ValueError, match="exactly one"):
        source_layout("def other(x): pass", "norm")
    with pytest.raises(ValueError, match="exactly one"):
        source_layout("def norm(x): pass\ndef norm(x): pass", "norm")


@pytest.mark.parametrize("width,warps", [(1024, 8), (4096, 16)])
def test_only_recorded_configs_allowed(width, warps):
    cfg = dict(XBLOCK=1, R0_BLOCK=width, num_warps=warps, num_stages=1)
    assert checked_config(dict(cfg, time_taken_ms=53)) == cfg
    with pytest.raises(ValueError, match="unexpected"):
        checked_config(dict(cfg, XBLOCK=2))


def test_metrics_count_bits_rows_and_signed_zero():
    a = torch.tensor([[-2.0, -0.0], [1.0, 2.0]], dtype=torch.bfloat16)
    b = a.clone()
    b[0, 1] = 0.0
    result = compare(a, b)
    assert result["bit_mismatches"] == result["affected_rows"] == 1
    assert result["numeric_mismatches"] == result["max_bf16_ulp"] == 0
    b[1, 0] = torch.nextafter(b[1, 0], torch.tensor(float("inf"), dtype=b.dtype))
    result = compare(a, b)
    assert result["bit_mismatches"] == result["affected_rows"] == 2
    assert result["numeric_mismatches"] == result["max_bf16_ulp"] == 1
    assert result["max_abs"] == 1 / 128
    assert result["mean_abs"] == 1 / 512


@pytest.mark.parametrize("bad", [torch.float32, float("nan"), float("inf")])
def test_metrics_reject_wrong_dtype_or_nonfinite(bad):
    a = torch.ones(2, 4, dtype=torch.bfloat16)
    b = a.float() if bad is torch.float32 else torch.full_like(a, bad)
    with pytest.raises(ValueError):
        compare(a, b)


def test_oracle_and_seeded_inputs_are_independent_and_repeatable():
    x = make_inputs(2, 530901, 1.0)
    assert torch.equal(x, make_inputs(2, 530901, 1.0))
    assert not torch.equal(x, make_inputs(2, 531001, 1.0))
    weight = torch.ones(4096, dtype=torch.bfloat16)
    expected = (
        x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-5)
    ).bfloat16()
    assert torch.equal(oracle(x, weight), expected)


def test_metric_chunk_boundaries():
    a = torch.ones(257, 2, dtype=torch.bfloat16)
    b = a.clone()
    b[[0, 128, 256], 1] = 2
    result = compare(a, b)
    assert result["bit_mismatches"] == result["affected_rows"] == 3
    assert result["max_abs"] == 1
    assert result["mean_abs"] == 3 / 514
