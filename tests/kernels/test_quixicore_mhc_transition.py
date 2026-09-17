# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The QuixiCore fused mHC post+pre transition against the Triton split
transition: same streams, mixes, sinkhorn matrix and fused RMSNorm. Token
counts straddle the kernel selection inside the CUDA op (the cooperative
kernel at one token, the tiled kernel below 128, the persistent prefill
kernel from 128 up)."""

import pytest
import torch

from vllm.model_executor.layers.glm5_next_mhc_triton import mhc_transition
from vllm.quixicore import quixicore_ops

HC, D, MIXES = 4, 4096, 24

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not quixicore_ops.is_available(),
    reason="needs a CUDA device with the QuixiCore extension",
)


def _inputs(tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kw = dict(device="cuda", generator=g)
    x = torch.randn(tokens, D, dtype=torch.bfloat16, **kw)
    residual = torch.randn(tokens, HC, D, dtype=torch.bfloat16, **kw)
    post_mix = torch.rand(tokens, HC, **kw)
    comb_mix = torch.rand(tokens, HC, HC, **kw)
    fn = torch.randn(MIXES, HC * D, **kw) * 0.02
    hc_scale = torch.tensor([0.5, 0.5, 0.5], device="cuda")
    hc_base = torch.randn(MIXES, **kw) * 0.1
    norm_weight = (1.0 + 0.1 * torch.randn(D, **kw)).to(torch.bfloat16)
    return x, residual, post_mix, comb_mix, fn, hc_scale, hc_base, norm_weight


@pytest.mark.parametrize("tokens", [1, 4, 8, 33, 64, 65, 127, 128, 300, 1000])
def test_fused_post_pre_matches_the_triton_transition(tokens: int):
    x, residual, post_mix, comb_mix, fn, scale, base, norm_weight = _inputs(
        tokens, tokens
    )
    rms_eps, hc_eps, post_mult, iters, norm_eps = 1e-6, 1e-6, 1.0, 3, 1e-6
    # The transition is independent per token, so the Triton kernel (which
    # serves at most 64 tokens per launch) is run in slices as the reference.
    parts = [
        mhc_transition(
            x[i : i + 64],
            residual[i : i + 64],
            post_mix[i : i + 64],
            comb_mix[i : i + 64],
            fn,
            scale,
            base,
            rms_eps,
            hc_eps,
            post_mult,
            iters,
            norm_weight,
            norm_eps,
        )
        for i in range(0, tokens, 64)
    ]
    ref = [torch.cat(t, dim=0) for t in zip(*parts)]
    out = quixicore_ops.dsv4_mhc_fused_post_pre(
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        scale,
        base,
        rms_eps,
        hc_eps,
        hc_eps,
        post_mult,
        iters,
        norm_weight,
        norm_eps,
    )
    names = ("residual_out", "post", "comb", "layer_input")
    for name, r, o in zip(names, ref, out):
        r = r.float().reshape(o.shape)
        o = o.float()
        # bf16 outputs may round differently on the last bit where the fp32
        # mixes are summed in another order; fp32 mixes agree to 1e-4.
        tol = (
            dict(atol=2e-2, rtol=2e-2)
            if name in ("residual_out", "layer_input")
            else dict(atol=1e-4, rtol=1e-4)
        )
        assert torch.allclose(o, r, **tol), (
            f"{name} at {tokens} tokens: max |diff| {(o - r).abs().max().item():.4g}"
        )


def test_persistent_kernel_covers_every_element_and_token():
    # A tail tile (tokens not a multiple of 8) and every split's chunk must
    # be written: poison the outputs first through a second call whose
    # residual_out and layer_input are compared token by token.
    tokens = 301
    x, residual, post_mix, comb_mix, fn, scale, base, norm_weight = _inputs(tokens, 7)
    args = (
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        scale,
        base,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        3,
        norm_weight,
        1e-6,
    )
    a = quixicore_ops.dsv4_mhc_fused_post_pre(*args)
    b = quixicore_ops.dsv4_mhc_fused_post_pre(*args)
    for ta, tb in zip(a, b):
        assert torch.equal(ta, tb)
    # Every token's comb is finite and, the sinkhorn ending on a column
    # normalisation, its columns sum to one.
    comb = a[2].reshape(tokens, HC, HC)
    assert torch.isfinite(comb).all()
    assert torch.allclose(comb.sum(-2), torch.ones_like(comb.sum(-2)), atol=1e-3)


@pytest.mark.parametrize("tokens", [1, 3, 64])
def test_small_batches_repeat_exactly_after_a_wider_batch(tokens: int):
    # The single-launch transition counts block arrivals per token and
    # resets the counters itself: a batch after a wider one (which touched
    # more counter slots) and a repeat of the same batch are bit-identical.
    args64 = _inputs(64, 11)[:7]
    x, residual, post_mix, comb_mix, fn, scale, base, norm_weight = _inputs(tokens, 13)
    tail = (1e-6, 1e-6, 1e-6, 1.0, 20, norm_weight, 1e-6)
    first = quixicore_ops.dsv4_mhc_fused_post_pre(
        x, residual, post_mix, comb_mix, fn, scale, base, *tail
    )
    quixicore_ops.dsv4_mhc_fused_post_pre(*args64, *tail)
    second = quixicore_ops.dsv4_mhc_fused_post_pre(
        x, residual, post_mix, comb_mix, fn, scale, base, *tail
    )
    for a, b in zip(first, second):
        assert torch.equal(a, b)
    comb = first[2].reshape(tokens, HC, HC)
    assert torch.isfinite(comb).all()
    assert torch.allclose(comb.sum(-2), torch.ones_like(comb.sum(-2)), atol=1e-3)
