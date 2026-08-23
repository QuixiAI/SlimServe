# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit references for the DFlash 2 drafter (Qwen3.8-27B campaign).

These are the CPU-only proofs the campaign's correctness rests on: the
two-tap dynamic convolution against a naive per-token/per-tap reference,
the greedy selector walk against a manual step-by-step walk, and the
sampled selector walk's (seed, position)-keyed determinism plus its sparse
k-way distribution contract for the rejection sampler.
"""

import pytest
import torch

from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2QwenDraftModel,
    _apply_two_tap_conv,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _rejection_sample_mps,
)


@pytest.mark.parametrize("side", [0, 1])
def test_two_tap_conv_matches_naive_reference(side: int) -> None:
    torch.manual_seed(0)
    nb, bs, hidden, groups, gs, taps = 3, 8, 64, 4, 16, 2
    tokens = nb * bs
    x = torch.randn(tokens, hidden)
    coeffs = torch.randn(tokens, 2, taps, groups)
    base = torch.randn(2, taps, hidden)

    got = _apply_two_tap_conv(x, coeffs, base, side, bs, gs)

    ref = torch.zeros_like(x)
    xb = x.view(nb, bs, hidden)
    for b in range(nb):
        for t in range(bs):
            for tap in range(taps):
                w = base[side, tap] + coeffs[b * bs + t, side, tap].repeat_interleave(
                    gs
                )
                src = xb[b, t - tap] if t - tap >= 0 else torch.zeros(hidden)
                ref[b * bs + t] += w * src
    assert torch.allclose(got, ref, atol=1e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple Metal"
)
@pytest.mark.parametrize("side", [0, 1])
def test_two_tap_conv_metal_matches_torch(side: int) -> None:
    torch.manual_seed(9)
    device = torch.device("mps")
    blocks, block_size, hidden, groups, group_size = 3, 8, 5120, 320, 16
    tokens = blocks * block_size
    x = torch.randn(tokens, hidden, device=device).to(torch.bfloat16)
    coeffs = torch.randn(tokens, 2, 2, groups, device=device).to(torch.bfloat16)
    base = torch.randn(2, 2, hidden, device=device).to(torch.bfloat16)

    got = _apply_two_tap_conv(x, coeffs, base, side, block_size, group_size)
    dyn = coeffs[:, side].repeat_interleave(group_size, dim=-1)
    ref = (base[side, 0] + dyn[:, 0]) * x
    shifted = x.view(blocks, block_size, hidden).roll(shifts=1, dims=1).clone()
    shifted[:, 0] = 0
    ref += (base[side, 1] + dyn[:, 1]) * shifted.view(tokens, hidden)
    error = (got.float() - ref.float()).abs().max()
    # The fused kernel may differ by one BF16 ULP because Metal can contract
    # the final multiply-add differently from the decomposed Torch graph.
    assert float(error / ref.float().abs().max()) < 1e-2


def _fake_drafter(vocab: int, rank: int, hidden: int, block: int, k: int):
    m = object.__new__(DFlash2QwenDraftModel)
    torch.nn.Module.__init__(m)

    class Cfg:
        block_size: int
        vocab_size: int

    cfg = Cfg()
    cfg.block_size = block
    cfg.vocab_size = vocab
    m.config = cfg
    m.selector_top_k = k
    m.selector_predecessor = torch.nn.Embedding(vocab, rank)
    m.selector_successor = torch.nn.Embedding(vocab, rank)
    weight = torch.randn(hidden, vocab)
    m.selector_hidden = lambda h: h[:, :rank]
    m.compute_logits = lambda h: h @ weight
    return m, weight


def test_greedy_walk_matches_manual_walk() -> None:
    torch.manual_seed(1)
    vocab, rank, hidden, block, k = 100, 8, 16, 4, 5
    m, W = _fake_drafter(vocab, rank, hidden, block, k)
    steps = block - 1
    hs = torch.randn(2 * steps, hidden)
    anchors = torch.tensor([3, 7])
    out = m.select_draft_path(hs, anchors)
    assert out.shape == (2, steps)

    logits = (hs @ W)[:steps]
    tv, ti = logits.topk(k, -1)
    gate = hs[:steps, :rank]
    prev = m.selector_predecessor(anchors[:1])[0]
    ref = []
    for pos in range(steps):
        succ = m.selector_successor(ti[pos])
        s = succ @ (prev * gate[pos]) + tv[pos]
        tok = ti[pos, s.argmax()]
        ref.append(int(tok))
        prev = m.selector_predecessor(tok)
    assert out[0].tolist() == ref


def test_sampled_walk_is_seed_keyed_and_sparse() -> None:
    torch.manual_seed(0)
    vocab, rank, hidden, steps, k, nb = 200, 8, 16, 7, 16, 3
    m, _ = _fake_drafter(vocab, rank, hidden, steps + 1, k)
    hs = torch.randn(nb * steps, hidden)
    anchors = torch.tensor([3, 7, 11])
    temp = torch.ones(nb)
    idx = torch.arange(nb)
    pos = torch.arange(steps).unsqueeze(0) + torch.tensor([[100], [250], [400]])

    def run(seeds):
        dl = torch.full((nb, steps, vocab), float("-inf"))
        out = m.select_draft_path_sampled(
            hs, anchors, temp, idx, dl, seeds=seeds, positions=pos
        )
        return out.clone(), dl.clone()

    a1, d1 = run(torch.tensor([42, 42, 42]))
    a2, d2 = run(torch.tensor([42, 42, 42]))
    b1, _ = run(torch.tensor([7, 7, 7]))
    assert torch.equal(a1, a2) and torch.equal(d1, d2)
    assert not torch.equal(a1, b1)
    # Exactly k finite entries per (block, position): the kept k-way
    # distribution the rejection sampler consumes.
    finite = torch.isfinite(d1).sum(dim=-1)
    assert bool((finite == k).all())
    # Every chosen token is one of its position's candidates.
    for b in range(nb):
        for p in range(steps):
            assert torch.isfinite(d1[b, p, a1[b, p]])


def test_vectorized_rejection_sampler_is_seeded_and_lossless() -> None:
    """The first emitted token follows p even when proposals come from q."""
    torch.manual_seed(4)
    num_reqs, vocab = 8192, 32
    p_logits = torch.randn(vocab) * 1.5
    q_logits = torch.randn(vocab) * 1.5
    p = p_logits.softmax(dim=-1)
    q = q_logits.softmax(dim=-1)

    # One proposal plus one bonus row per request.  The proposal is sampled
    # from q, so lossless rejection must leave the emitted-token marginal at
    # p and accept with probability sum_x min(p(x), q(x)).
    proposed = torch.multinomial(q.expand(num_reqs, -1), 1).squeeze(1)
    draft_sampled = torch.zeros(num_reqs * 2, dtype=torch.int64)
    draft_sampled[1::2] = proposed
    target_logits = p_logits.expand(num_reqs * 2, -1).contiguous()
    draft_logits = q_logits.expand(num_reqs, 1, -1).contiguous()
    cu_num_logits = torch.arange(0, 2 * num_reqs + 1, 2, dtype=torch.int32)
    positions = (torch.arange(num_reqs, dtype=torch.int64) + 100).repeat_interleave(2)
    idx_mapping = torch.arange(num_reqs, dtype=torch.int64)
    temperature = torch.ones(num_reqs)
    seeds = torch.arange(num_reqs, dtype=torch.int64) + 42

    args = (
        target_logits,
        draft_logits,
        draft_sampled,
        cu_num_logits,
        positions,
        idx_mapping,
        temperature,
        seeds,
        1,
        vocab,
    )
    sampled_a, counts_a = _rejection_sample_mps(*args)
    sampled_b, counts_b = _rejection_sample_mps(*args)
    assert torch.equal(sampled_a, sampled_b)
    assert torch.equal(counts_a, counts_b)

    empirical = torch.bincount(sampled_a[:, 0], minlength=vocab) / num_reqs
    total_variation = 0.5 * (empirical - p).abs().sum()
    assert float(total_variation) < 0.05

    empirical_accept = (counts_a == 2).float().mean()
    expected_accept = torch.minimum(p, q).sum()
    assert torch.allclose(empirical_accept, expected_accept, atol=0.025)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple Metal"
)
def test_fused_rejection_sampler_matches_vectorized_mps() -> None:
    """Serving dtypes take the fused path without changing seeded output."""
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils as rs

    torch.manual_seed(17)
    device = torch.device("mps")
    requests, steps, vocab = 8, 8, 1024
    rows = requests * (steps + 1)
    target = torch.randn(rows, vocab, device=device)
    draft = torch.randn(requests, steps, vocab, device=device)
    draft_ids = draft.argmax(dim=-1).to(torch.int32)
    draft_sampled = torch.zeros(rows, dtype=torch.int32, device=device)
    for req in range(requests):
        start = req * (steps + 1)
        draft_sampled[start + 1 : start + steps + 1] = draft_ids[req]
    cu = torch.arange(0, rows + 1, steps + 1, dtype=torch.int32, device=device)
    positions = torch.arange(100, 100 + rows, dtype=torch.int64, device=device)
    idx_mapping = torch.arange(requests, dtype=torch.int32, device=device)
    temperature = torch.ones(requests, device=device)
    seeds = torch.arange(42, 42 + requests, dtype=torch.int64, device=device)
    args = (
        target,
        draft,
        draft_sampled,
        cu,
        positions,
        idx_mapping,
        temperature,
        seeds,
        steps,
        vocab,
    )

    old = rs._MPS_REJECTION_FUSED
    try:
        rs._MPS_REJECTION_FUSED = True
        fused = rs._rejection_sample_mps(*args)
        rs._MPS_REJECTION_FUSED = False
        reference = rs._rejection_sample_mps(*args)
    finally:
        rs._MPS_REJECTION_FUSED = old
    assert torch.equal(fused[0], reference[0])
    assert torch.equal(fused[1], reference[1])
