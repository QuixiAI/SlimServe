# SPDX-License-Identifier: Apache-2.0
"""Independent CPU FP64 accuracy contract for the GLM53 mHC prefill probe.

This is NOT the original installed-kernel parity gate. See
perf/glm53-mhc-tc-accuracy-contract.md for the separately prescribed gates.
The fused residual must first pass exact installed-kernel bit parity; this
oracle evaluates the subsequent projection, normalization, gates and mix.
"""

import math

import torch

RTOL = 2e-5
ATOL = 2e-6
GAMMA4 = (4 * 2**-24) / (1 - 4 * 2**-24)


def direct_bf16_rne(value):
    """Round finite, BF16-range CPU FP64 values without FP32 double rounding.

    Torch's double->BF16 conversion is only the initial estimate. Its adjacent
    values bracket any double-rounding error; choose the nearest using FP64
    distances and the even BF16 low bit for exact ties. Preserve signed zero.
    Overflow is deliberately out of this oracle's scope, not silently clamped.
    """
    if value.device.type != "cpu" or value.dtype != torch.float64:
        raise ValueError("direct rounding requires CPU float64")
    if (
        not torch.isfinite(value).all()
        or (value.abs() > torch.finfo(torch.bfloat16).max).any()
    ):
        raise ValueError("direct rounding requires finite BF16-range values")
    estimate = value.bfloat16()
    rounded = estimate
    distance = (estimate.double() - value).abs()
    # Pairwise selection implements the same nearest/even ordering without
    # a strided three-way argmin over millions of hidden coordinates.
    for direction in (-math.inf, math.inf):
        neighbor = torch.nextafter(estimate, torch.full_like(estimate, direction))
        other_distance = (neighbor.double() - value).abs()
        even = (neighbor.view(torch.int16) & 1) == 0
        better = (other_distance < distance) | ((other_distance == distance) & even)
        rounded = torch.where(better, neighbor, rounded)
        distance = torch.minimum(distance, other_distance)
    return torch.where(value == 0, estimate, rounded)


def ideal_outputs(residual, fn, scale, base):
    """Model equations in FP64, independent of CUDA splits/tiles/reductions."""
    if (
        residual.ndim != 3
        or tuple(residual.shape[1:]) != (4, 4096)
        or tuple(fn.shape) != (24, 16384)
        or tuple(scale.shape) != (3,)
        or tuple(base.shape) != (24,)
    ):
        raise ValueError("invalid GLM53 mHC oracle shapes")
    for tensor in (residual, fn, scale, base):
        if tensor.device.type != "cpu" or tensor.dtype != torch.float64:
            raise ValueError("ideal equations require CPU float64 inputs")
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite oracle input")
    flat = residual.flatten(1)
    normalized = (flat @ fn.T) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + 1e-5
    )
    pre = torch.sigmoid(normalized[:, :4] * scale[0] + base[:4]) + 1e-6
    post = torch.sigmoid(normalized[:, 4:8] * scale[1] + base[4:8]) * 2.0
    logits = (normalized[:, 8:] * scale[2] + base[8:]).reshape(-1, 4, 4)
    comb = torch.softmax(logits, dim=-1) + 1e-6
    for iteration in range(20):
        if iteration:
            comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    products = pre.unsqueeze(-1) * residual
    layer_input = products.sum(1)
    # Propagate the existing FP32 relative/absolute coefficient contract
    # through the four-stream sum, plus its FP32 summation allowance. This is
    # an acceptance budget, not a proof of every possible CUDA input's error.
    allowance = (RTOL + GAMMA4) * products.abs().sum(1)
    allowance += ATOL * residual.abs().sum(1)
    return post, comb, layer_input, allowance


def layer_accuracy(observed, ideal, allowance):
    """BF16 error must not exceed optimal rounding error + FP32 budget."""
    if (
        observed.shape != ideal.shape
        or observed.dtype != torch.bfloat16
        or ideal.shape != allowance.shape
        or ideal.dtype != torch.float64
        or allowance.dtype != torch.float64
        or any(t.device.type != "cpu" for t in (observed, ideal, allowance))
    ):
        raise ValueError("invalid layer-input oracle shapes, dtypes or devices")
    if (
        not all(torch.isfinite(t).all() for t in (observed, ideal, allowance))
        or (allowance < 0).any()
    ):
        raise ValueError("nonfinite value or negative error budget")
    rounded = direct_bf16_rne(ideal)
    error = (observed.double() - ideal).abs()
    optimal = (rounded.double() - ideal).abs()
    excess = (error - optimal).clamp_min(0)
    # Allow only FP64 comparison roundoff at the already-declared boundary.
    comparison_slack = 8 * torch.finfo(torch.float64).eps * ideal.abs()
    violations = excess > allowance + comparison_slack
    return {
        "elements": ideal.numel(),
        "violations": int(violations.sum()),
        "max_excess_over_budget": float((excess - allowance).clamp_min(0).max()),
        "max_abs_error": float(error.max()),
        "squared_error": float(error.square().sum()),
        "ideal_squared_sum": float(ideal.square().sum()),
        "optimal_squared_error": float(optimal.square().sum()),
        "not_ideally_rounded": int(
            (observed.view(torch.int16) != rounded.view(torch.int16)).sum()
        ),
    }


def accuracy_pair(residual, fn, scale, base, reference, candidate, chunk_rows=256):
    """Check every supplied row against one shared CPU FP64 calculation.

    reference/candidate contain post, comb and BF16 layer_input, not residual.
    Graph callers can supply a documented row selection; eager census callers
    supply all rows. Chunking bounds memory, never subsamples hidden columns.
    """
    if chunk_rows < 1 or residual.dtype != torch.bfloat16 or fn.dtype != torch.bfloat16:
        raise ValueError("positive chunk size and lossless BF16 values required")
    if residual.ndim != 3 or not residual.shape[0]:
        raise ValueError("nonempty three-dimensional residual required")
    if scale.dtype != torch.float32 or base.dtype != torch.float32:
        raise ValueError("native FP32 scale/base required")
    batch = residual.shape[0]
    shapes = [(batch, 4), (batch, 4, 4), (batch, 4096)]
    for arm in (reference, candidate):
        if len(arm) != 3:
            raise ValueError("expected three output tensors")
        for tensor, shape, dtype in zip(
            arm, shapes, (torch.float32, torch.float32, torch.bfloat16)
        ):
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError("output shape or dtype changed")
    w, s, b = (t.detach().cpu().double() for t in (fn, scale, base))
    result = {
        arm: {
            "post_violations": 0,
            "comb_violations": 0,
            "post_max_abs_error": 0.0,
            "comb_max_abs_error": 0.0,
        }
        for arm in ("reference", "candidate")
    }
    sum_keys = {
        "elements",
        "violations",
        "squared_error",
        "ideal_squared_sum",
        "optimal_squared_error",
        "not_ideally_rounded",
    }
    for first in range(0, batch, chunk_rows):
        rows = slice(first, first + chunk_rows)
        expected = ideal_outputs(residual[rows].detach().cpu().double(), w, s, b)
        for name, outputs in (("reference", reference), ("candidate", candidate)):
            stats = result[name]
            for field, got, want in zip(("post", "comb"), outputs[:2], expected[:2]):
                got = got[rows].detach().cpu().double()
                if not torch.isfinite(got).all():
                    raise ValueError(f"nonfinite {name} {field}")
                error = (got - want).abs()
                stats[field + "_violations"] += int(
                    (error > RTOL * want.abs() + ATOL).sum()
                )
                stats[field + "_max_abs_error"] = max(
                    stats[field + "_max_abs_error"], float(error.max())
                )
            layer = layer_accuracy(outputs[2][rows].detach().cpu(), *expected[2:])
            for key, value in layer.items():
                stats[key] = (
                    stats.get(key, 0) + value
                    if key in sum_keys
                    else max(stats.get(key, 0), value)
                )
    for stats in result.values():
        stats["normalized_rms"] = math.sqrt(
            stats["squared_error"] / max(stats["ideal_squared_sum"], 1e-60)
        )
        stats["passed"] = not any(
            stats[k] for k in ("post_violations", "comb_violations", "violations")
        )
    result["candidate_rms_noninferior"] = result["candidate"]["normalized_rms"] <= (
        1.001 * result["reference"]["normalized_rms"] + 1e-7
    )
    result["passed"] = (
        all(result[k]["passed"] for k in ("reference", "candidate"))
        and result["candidate_rms_noninferior"]
    )
    result["rows"] = batch
    return result
