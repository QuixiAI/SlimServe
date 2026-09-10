# SPDX-License-Identifier: Apache-2.0
"""Independent CPU FP64 compact-pool semantics and optional GPU comparison.

The diagnostic does not import the old raw-cache scorer as its oracle.
Run the GPU comparison only after live serving releases the GPUs. Passing
implementation-parity tests alone does not establish these numeric gates.
"""

import json

import torch


def pool_keys_fp64(packed, ape):
    assert packed.device.type == ape.device.type == "cpu"
    assert packed.ndim == 2 and packed.shape[1] == 256 and ape.shape == (4, 128)
    count = packed.shape[0] // 4
    members = packed[: count * 4].double().reshape(count, 4, 256)
    probabilities = torch.softmax(members[:, :, 128:] + ape.double(), dim=1)
    return (probabilities * members[:, :, :128]).sum(dim=1)


def scores_fp64(pooled_bf16, query_bf16, weights_fp32):
    assert all(t.device.type == "cpu" for t in (pooled_bf16, query_bf16, weights_fp32))
    scores = (pooled_bf16.double() @ query_bf16.double().T) * 128**-0.5
    return (scores.clamp_min(0) * weights_fp32.double()).sum(dim=-1)


def error_record(actual, expected, *, rtol, atol):
    error = (actual.double() - expected.double()).abs()
    finite = torch.isfinite(error)
    failed = ~finite | (error > atol + rtol * expected.double().abs())
    finite_error = error[finite]
    return dict(
        max_abs_error=finite_error.max().item() if finite_error.numel() else None,
        nonfinite_values=(~finite).count_nonzero().item(),
        failing_values=failed.count_nonzero().item(),
        compared_values=error.numel(),
        rtol=rtol,
        atol=atol,
    )


@torch.no_grad()
def validate(bs, seed):
    from vllm.model_executor.layers.glm5_next_pool_cache import (
        cached_pool_logits,
        update_pool_cache,
    )

    torch.manual_seed(seed)
    length = max(4096, bs + 17)
    rows = torch.randn(length, 256, dtype=torch.bfloat16)
    ape = torch.randn(4, 128)
    count = length // 4
    pages = (length + bs - 1) // bs
    physical = torch.randperm(pages)
    backing = torch.full((pages, 2, bs, 64), 7.0, device="cuda", dtype=torch.bfloat16)
    cache = backing[:, 0]
    packed, offsets = rows.cuda(), ape.cuda()
    pos = torch.arange(length)
    slots = (physical[pos // bs] * bs + pos % bs).cuda()
    cursor = 0
    # Includes singleton phases, small chunks and page-spanning chunks.
    sizes = (1, 2, 5, 17, bs + 1)
    step = 0
    while cursor < length:
        end = min(length, cursor + sizes[step % len(sizes)])
        update_pool_cache(
            packed[cursor:end], slots[cursor:end], offsets, cache, singleton_fused=True
        )
        cursor = end
        step += 1
    host_cache = cache.cpu()
    stored = torch.stack(
        [
            host_cache[physical[(p * 4) // bs]].reshape(-1)[
                (p % (bs // 4)) * 128 : (p % (bs // 4) + 1) * 128
            ]
            for p in range(count)
        ]
    )
    mathematical_keys = pool_keys_fp64(rows, ape)
    semantic_keys = mathematical_keys.to(torch.bfloat16)
    keys = error_record(stored, semantic_keys, atol=0.002, rtol=0.002)
    # Preserve the original rounded-reference gate. Separately expose whether
    # a mismatch straddles a BF16 rounding midpoint; this is diagnostic evidence,
    # not permission to relax the gate or claim correctly rounded FP64 pooling.
    unrounded_keys = error_record(stored, mathematical_keys, atol=0.002, rtol=0.002)
    rounded_error = (stored.double() - semantic_keys.double()).abs()
    failed = rounded_error > 0.002 + 0.002 * semantic_keys.double().abs()
    rounding_examples = []
    for pool, channel in failed.nonzero().tolist()[:16]:
        value = mathematical_keys[pool, channel].item()
        actual = stored[pool, channel].item()
        rounded = semantic_keys[pool, channel].item()
        rounding_examples.append(
            dict(
                pool=pool,
                channel=channel,
                stored=actual,
                mathematical_fp64=value,
                rounded_fp64_bf16=rounded,
                stored_error_vs_math=abs(actual - value),
                rounded_error_vs_math=abs(rounded - value),
                midpoint_distance=abs(value - (actual + rounded) / 2),
            )
        )
    query = torch.randn(4, 32, 128, dtype=torch.bfloat16)
    weights = torch.randn(4, 32) * 32**-0.5
    out = torch.empty(4, count, device="cuda")
    cached_pool_logits(
        query.cuda(),
        weights.cuda(),
        cache,
        physical.int().cuda()[None],
        torch.zeros(4, device="cuda", dtype=torch.int32),
        torch.full((4,), length, device="cuda", dtype=torch.int32),
        out,
    )
    actual_scores = out.cpu()
    # Isolate scorer arithmetic from a possible BF16 key-rounding difference.
    expected_scores = torch.stack(
        [scores_fp64(stored, q, w) for q, w in zip(query, weights)]
    )
    scores = error_record(actual_scores, expected_scores, atol=1e-6, rtol=1e-5)
    assert (backing[:, 1] == 7).all(), "unrelated slab overwritten"
    return dict(
        bs=bs,
        seed=seed,
        length=length,
        pool_key_gate=keys,
        pool_key_unrounded_diagnostic=unrounded_keys,
        rounding_examples=rounding_examples,
        scorer_gate=scores,
        strict_pass=keys["failing_values"] == scores["failing_values"] == 0,
    )


if __name__ == "__main__":
    passed = True
    for bs in (64, 4608):
        for seed in (0, 17, 42):
            record = validate(bs, seed)
            print(json.dumps(record), flush=True)
            passed &= record["strict_pass"]
    raise SystemExit(0 if passed else 1)
