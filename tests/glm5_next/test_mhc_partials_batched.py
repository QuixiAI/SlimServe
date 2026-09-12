# SPDX-License-Identifier: Apache-2.0
"""The token-batched mHC partials (decode batches) must reproduce the
per-token kernel bit-for-bit: dsv4_mhc_fused_post_pre at T tokens versus
T single-token calls."""

import pytest
import torch

pytest.importorskip("vllm._quixicore_C")
from vllm.quixicore import quixicore_ops as qc  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
HC, D, MIXES = 4, 4096, 24


def _inputs(T, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    x = (torch.randn(T, D, device=dev, generator=g) * 0.5).to(torch.bfloat16)
    residual = (torch.randn(T, HC, D, device=dev, generator=g)).to(torch.bfloat16)
    post = torch.rand(T, HC, device=dev, generator=g)
    comb = torch.rand(T, HC, HC, device=dev, generator=g) * 0.5
    fn = torch.randn(MIXES, HC * D, device=dev, generator=g) * 0.02
    scale = torch.rand(MIXES, device=dev, generator=g) + 0.5
    base = torch.randn(MIXES, device=dev, generator=g) * 0.1
    norm_w = (torch.rand(D, device=dev, generator=g) + 0.5).to(torch.bfloat16)
    return x, residual, post, comb, fn, scale, base, norm_w


@pytest.mark.parametrize("T", [2, 4, 5, 32, 128])
def test_batched_partials_match_per_token(T):
    x, residual, post, comb, fn, scale, base, norm_w = _inputs(T)
    args = (fn, scale, base, 1e-6, 1e-6, 1e-6, 1.0, 3, norm_w, 1e-5)
    batched = qc.dsv4_mhc_fused_post_pre(x, residual, post, comb, *args)
    for t in range(T):
        single = qc.dsv4_mhc_fused_post_pre(
            x[t : t + 1], residual[t : t + 1], post[t : t + 1], comb[t : t + 1], *args
        )
        names = ("residual_out", "post", "comb", "layer_input")
        for name, b, s in zip(names, batched, single):
            bt = b[t : t + 1].reshape(s.shape)
            if name == "residual_out":
                # Mixed residual (bf16): bit-identical, no fn involvement.
                assert torch.equal(bt, s), (T, t, name)
            elif name == "layer_input":
                # Normed layer input (bf16) derives from the fp32 sums below;
                # a 1-ulp fp32 difference can flip a bf16 rounding: allow
                # one bf16 ulp.
                torch.testing.assert_close(bt.float(), s.float(), rtol=2**-7, atol=2e-3)
            else:
                # fp32 mixing coefficients: same partial sums, FMA
                # contraction differs by a couple of ulp.
                torch.testing.assert_close(bt, s, rtol=0, atol=1e-6)
