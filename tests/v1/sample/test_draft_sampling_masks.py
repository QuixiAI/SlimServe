# SPDX-License-Identifier: Apache-2.0
"""A drafter drawing under the request's top-k / top-p (draft_top_k_top_p): the
tempered, cut logits are the verifier's cut of the same rows, a row without a
cutoff passes through, and the Gumbel draw never lands outside the cut."""

import pytest
import torch

from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    apply_top_k_top_p_pytorch,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.speculator import mask_draft_logits

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)
DEVICE = torch.device("cuda")
VOCAB = 4096
MAX_REQS = 32


def _batch(rows: int, seed: int, dtype: torch.dtype):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    logits = (torch.randn(rows, VOCAB, device=DEVICE, generator=g) * 3).to(dtype)
    # The request states are wider than the batch, which sits at their top.
    top_k = torch.full((MAX_REQS,), VOCAB, dtype=torch.int32, device=DEVICE)
    top_p = torch.ones(MAX_REQS, dtype=torch.float32, device=DEVICE)
    temperature = torch.ones(MAX_REQS, dtype=torch.float32, device=DEVICE)
    idx = torch.arange(MAX_REQS - rows, MAX_REQS, dtype=torch.int32, device=DEVICE)
    return logits, top_k, top_p, temperature, idx


@pytest.mark.parametrize("native", [False, True])
def test_the_cut_draft_is_the_verifiers_cut(native):
    rows = 8
    # fp32 draws: no ties at the k-th value to resolve differently.
    logits, top_k, top_p, temperature, idx = _batch(rows, 0, torch.float32)
    top_k[idx[:6]] = 20
    top_p[idx[:6]] = 0.95
    temperature[idx[:6]] = 0.7
    top_k[idx[6]] = 5  # top-k only, at temperature 1
    if native:
        # The native cutoff serves batches whose every row has 1 <= k <= 32
        # (the fallback takes the others); row 7 is its widest cut.
        top_k[idx[7]] = 32
    # ... otherwise row 7 keeps no cutoff at all.

    out = mask_draft_logits(logits, idx, temperature, top_k, top_p, native)
    gather = idx.to(torch.int64)
    ref = logits / temperature[gather].unsqueeze(1)
    # What the verifier cuts the target's rows with: the native cutoff kernel
    # (exact against the torch reference) or the Triton op it falls back to.
    if native:
        ref = apply_top_k_top_p_pytorch(ref, top_k[gather], top_p[gather])
    else:
        ref = apply_top_k_top_p(ref, top_k[gather], top_p[gather])

    assert out.dtype == torch.float32 and out.shape == logits.shape
    kept = torch.isfinite(ref)
    assert torch.equal(torch.isfinite(out), kept)
    torch.testing.assert_close(out[kept], ref[kept], rtol=1e-6, atol=1e-6)
    assert (kept[:6].sum(dim=1) <= 20).all()
    assert int(kept[6].sum()) == 5
    if native:
        assert int(kept[7].sum()) == 32
    else:
        assert kept[7].all()


def test_the_draw_stays_inside_the_cut_the_verifier_sees():
    rows = 16
    logits, top_k, top_p, temperature, idx = _batch(rows, 1, torch.bfloat16)
    top_k[idx] = 20
    top_p[idx] = 0.95
    out = mask_draft_logits(logits, idx, temperature, top_k, top_p, native=True)
    seeds = torch.arange(100, 100 + MAX_REQS, dtype=torch.int64, device=DEVICE)
    pos = torch.arange(rows, dtype=torch.int64, device=DEVICE)
    draft_logits = torch.zeros(MAX_REQS, 1, VOCAB, dtype=torch.float32, device=DEVICE)
    col = torch.zeros((), dtype=torch.int64, device=DEVICE)
    for step in range(50):
        sampled = gumbel_sample(
            out,
            idx,
            temperature,
            seeds,
            pos + step,
            apply_temperature=False,
            output_processed_logits=draft_logits,
            output_processed_logits_col=col,
            is_drafting=True,
        )
        assert torch.isfinite(out.gather(1, sampled.view(-1, 1))).all()
    # What the verifier reads as the proposal is exactly the cut draft.
    assert torch.equal(draft_logits[idx.to(torch.int64), 0], out)
