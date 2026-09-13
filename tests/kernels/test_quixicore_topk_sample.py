# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused top-k / top-p / Gumbel-max sampling (quixicore ``topk_sample``).

The reference is the path it replaces: the torch top-k/top-p mask followed
by ``gumbel_sample`` with the same seeds and positions. The fused kernel
ranks tokens by p / E instead of logit + G with G = -log E, so a rare
near-tie can resolve differently; membership in the retained set is exact.
"""

import numpy as np
import pytest
import torch

from vllm.quixicore.ops import quixicore_ops
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample

if not torch.cuda.is_available() or not quixicore_ops.has_topk_sample():
    pytest.skip("quixicore topk_sample binding unavailable", allow_module_level=True)

DEVICE = torch.device("cuda")
VOCAB = 154880  # GLM-5.3-Flash


def _batch(batch: int, vocab: int, seed: int, spread: float = 4.0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    logits = torch.randn(batch, vocab, device=DEVICE, generator=g) * spread
    top_k = torch.randint(1, 33, (batch,), device=DEVICE, generator=g).to(torch.int32)
    idx = torch.arange(batch, device=DEVICE, dtype=torch.int32)
    seeds = torch.randint(
        -(2**62), 2**62, (batch,), device=DEVICE, generator=g, dtype=torch.int64
    )
    pos = torch.randint(0, 4096, (batch,), device=DEVICE, generator=g).to(torch.int64)
    return logits, top_k, idx, seeds, pos


def _reference(logits, top_k, top_p, idx, seeds, pos, use_fp64):
    masked = apply_top_k_top_p_pytorch(logits.clone(), top_k, top_p)
    temperature = torch.ones(logits.shape[0], device=DEVICE)
    sampled = gumbel_sample(
        masked, idx, temperature, seeds, pos, apply_temperature=False, use_fp64=use_fp64
    )
    return masked, sampled


@pytest.mark.parametrize("vocab", [VOCAB, 2053])
@pytest.mark.parametrize("top_p", [None, 0.95, 0.3])
@pytest.mark.parametrize("use_fp64", [False, True])
def test_matches_mask_then_gumbel(vocab, top_p, use_fp64):
    batch = 16
    total = agree = 0
    for trial in range(7):
        logits, top_k, idx, seeds, pos = _batch(batch, vocab, seed=100 + trial)
        p = None if top_p is None else torch.full((batch,), top_p, device=DEVICE)
        masked, ref = _reference(logits, top_k, p, idx, seeds, pos, use_fp64)
        out = quixicore_ops.topk_sample(logits, top_k, p, idx, seeds, pos, use_fp64)
        assert out.dtype == torch.int64 and out.shape == (batch,)
        kept = masked.gather(1, out.view(-1, 1)).view(-1)
        assert torch.isfinite(kept).all(), "sampled a masked token"
        total += batch
        agree += int((out == ref).sum())
    assert agree >= 0.99 * total, f"{agree}/{total} rows agree with the Gumbel path"


def test_top_k_one_is_argmax():
    logits, _, idx, seeds, pos = _batch(8, VOCAB, seed=7)
    top_k = torch.ones(8, device=DEVICE, dtype=torch.int32)
    out = quixicore_ops.topk_sample(logits, top_k, None, idx, seeds, pos, False)
    torch.testing.assert_close(out, logits.argmax(dim=1))


def test_deterministic_and_seed_sensitive():
    logits, top_k, idx, seeds, pos = _batch(16, VOCAB, seed=11)
    top_k.fill_(32)
    a = quixicore_ops.topk_sample(logits, top_k, None, idx, seeds, pos, False)
    b = quixicore_ops.topk_sample(logits, top_k, None, idx, seeds, pos, False)
    torch.testing.assert_close(a, b)
    c = quixicore_ops.topk_sample(logits, top_k, None, idx, seeds, pos + 1, False)
    assert not torch.equal(a, c)


def test_ties_kept_and_nucleus_drops_low_ids():
    # Every logit equal: top-k keeps all ties; top-p 0.5 drops exactly the
    # lower-id half in the ascending (logit, id) order.
    batch, vocab = 4, 4096
    logits = torch.zeros(batch, vocab, device=DEVICE)
    top_k = torch.full((batch,), 4, device=DEVICE, dtype=torch.int32)
    idx = torch.arange(batch, device=DEVICE, dtype=torch.int32)
    seeds = torch.arange(batch, device=DEVICE, dtype=torch.int64) * 7919
    lows = 0
    for pos_base in range(64):
        pos = torch.full((batch,), pos_base, device=DEVICE, dtype=torch.int64)
        out = quixicore_ops.topk_sample(logits, top_k, None, idx, seeds, pos, False)
        assert bool((out >= 0).all()) and bool((out < vocab).all())
        lows += int((out < vocab // 2).sum())
        p = torch.full((batch,), 0.5, device=DEVICE)
        out = quixicore_ops.topk_sample(logits, top_k, p, idx, seeds, pos, False)
        assert bool((out >= vocab // 2).all()), "nucleus kept a low-id tie"
    assert lows > 0, "top-k without top-p should reach the low-id ties too"


def test_masked_rows_sample_inside_the_mask():
    batch = 8
    logits = torch.full((batch, VOCAB), float("-inf"), device=DEVICE)
    g = torch.Generator(device=DEVICE).manual_seed(3)
    allowed = torch.randint(0, VOCAB, (batch, 3), device=DEVICE, generator=g)
    logits.scatter_(1, allowed, torch.randn(batch, 3, device=DEVICE, generator=g))
    top_k = torch.full((batch,), 32, device=DEVICE, dtype=torch.int32)
    idx = torch.arange(batch, device=DEVICE, dtype=torch.int32)
    seeds = torch.arange(batch, device=DEVICE, dtype=torch.int64)
    for pos_base in range(16):
        pos = torch.full((batch,), pos_base, device=DEVICE, dtype=torch.int64)
        out = quixicore_ops.topk_sample(logits, top_k, None, idx, seeds, pos, False)
        assert bool((out.view(-1, 1) == allowed).any(dim=1).all())


def test_invalid_rows_return_in_range_ids():
    # Warmups feed uninitialized logits and -1 request slots.
    batch = 4
    logits = torch.full((batch, VOCAB), float("nan"), device=DEVICE)
    logits[1].fill_(float("-inf"))
    top_k = torch.full((batch,), 20, device=DEVICE, dtype=torch.int32)
    idx = torch.full((batch,), -1, device=DEVICE, dtype=torch.int32)
    seeds = torch.zeros(1, device=DEVICE, dtype=torch.int64)
    pos = torch.zeros(batch, device=DEVICE, dtype=torch.int64)
    p = torch.full((batch,), 0.95, device=DEVICE)
    out = quixicore_ops.topk_sample(logits, top_k, p, idx, seeds, pos, False)
    assert bool((out >= 0).all()) and bool((out < VOCAB).all())


def test_sampler_eligibility_rules():
    from vllm.v1.worker.gpu.sample import topk_sample
    from vllm.v1.worker.gpu.sample.states import SamplingStates

    states = SamplingStates(max_num_reqs=4, vocab_size=VOCAB)
    states.top_k.np[:] = [20, 32, 33, 20]
    states.temperature.np[:] = [1.0, 1.0, 1.0, 0.0]
    logits = torch.zeros(2, VOCAB, device=DEVICE)
    pos = torch.zeros(2, device=DEVICE, dtype=torch.int64)
    k = torch.zeros(2, device=DEVICE, dtype=torch.int32)
    ok = topk_sample.eligible(states, k, np.array([0, 1]), logits, pos, False)
    assert ok == topk_sample.enabled()
    assert not topk_sample.eligible(states, k, np.array([0, 2]), logits, pos, False)
    assert not topk_sample.eligible(states, k, np.array([0, 3]), logits, pos, False)
    assert not topk_sample.eligible(states, None, np.array([0, 1]), logits, pos, False)
    assert not topk_sample.eligible(states, k, np.array([0, 1]), logits, pos, True)
    assert not topk_sample.eligible(
        states, k, np.array([0, 1]), logits.to(torch.bfloat16), pos, False
    )
