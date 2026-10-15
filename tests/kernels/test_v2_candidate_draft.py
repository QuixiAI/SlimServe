# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The candidate draft sampler (v2_candidate_draft) against the full draft path
of the V2 speculator: mask_draft_logits (temperature, native top-k / top-p cut)
followed by gumbel_sample with the processed row written for the block
rejection sampler. On one rank the candidates are the top-K of the full
logits, which is what every rank's shard contributes under TP; the draw, the
sampled token and the processed row must match exactly."""

import pytest
import torch

from vllm.quixicore.ops import quixicore_ops

pytest.importorskip("vllm._quixicore_C")
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and quixicore_ops.has_v2_candidate_draft()),
    reason="needs CUDA and the QuixiCore v2_candidate_draft binding",
)

from vllm.v1.worker.gpu.sample import topk_sample  # noqa: E402
from vllm.v1.worker.gpu.sample.gumbel import DRAFT_NOISE_SALT, gumbel_sample  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.speculator import mask_draft_logits  # noqa: E402

DEV = "cuda"
V = 4096
MAX_REQS = 8
STEPS = 3


def _batch(seed: int, num_tokens: int, temps, top_ks, top_ps):
    g = torch.Generator(device=DEV).manual_seed(seed)
    logits = (torch.randn(num_tokens, V, device=DEV, generator=g) * 3).to(torch.bfloat16)
    idx_mapping = torch.arange(num_tokens, device=DEV, dtype=torch.int32)
    temperature = torch.zeros(MAX_REQS, device=DEV)
    temperature[:num_tokens] = torch.tensor(temps[:num_tokens], device=DEV)
    top_k = torch.full((MAX_REQS,), V, device=DEV, dtype=torch.int32)
    top_k[:num_tokens] = torch.tensor(top_ks[:num_tokens], device=DEV, dtype=torch.int32)
    top_p = torch.ones(MAX_REQS, device=DEV)
    top_p[:num_tokens] = torch.tensor(top_ps[:num_tokens], device=DEV)
    seeds = torch.arange(MAX_REQS, device=DEV, dtype=torch.int64) * 7919 + 13
    positions = torch.arange(num_tokens, device=DEV, dtype=torch.int64) * 31 + 5
    return logits, idx_mapping, temperature, top_k, top_p, seeds, positions


@pytest.mark.parametrize("use_fp64", [False, True])
@pytest.mark.parametrize("step", [0, 2])
@pytest.mark.parametrize(
    "num_tokens,temps,top_ks,top_ps",
    [
        (1, [1.0], [20], [0.95]),
        (4, [1.0, 0.7, 1.3, 0.0], [20, 5, 32, 1], [0.95, 1.0, 0.6, 1.0]),
        (8, [1.0] * 8, [20, 20, 3, 32, 8, 20, 1, 2], [0.95, 0.5, 1.0, 0.9, 0.8, 0.99, 1.0, 0.7]),
    ],
)
def test_candidate_draft_matches_full_path(num_tokens, temps, top_ks, top_ps, step, use_fp64):
    logits, idx_mapping, temperature, top_k, top_p, seeds, positions = _batch(
        11 + num_tokens + step, num_tokens, temps, top_ks, top_ps
    )
    do_top_p = bool((top_p[:num_tokens] != 1.0).any())
    draft_step = torch.tensor(step, device=DEV, dtype=torch.int64)
    # The full path: temper + native cut over the vocabulary, then the Gumbel draw.
    ref_rows = torch.full((MAX_REQS, STEPS, V), -float("inf"), device=DEV)
    masked = mask_draft_logits(
        logits.clone(), idx_mapping, temperature, top_k, top_p if do_top_p else None, native=True
    )
    ref = gumbel_sample(
        masked, idx_mapping, temperature, seeds, positions + 1, apply_temperature=False,
        output_processed_logits=ref_rows, output_processed_logits_col=draft_step,
        use_fp64=use_fp64, is_drafting=True,
    )
    # The candidate path: the top-K of the (one) shard, untempered, then the kernel.
    k = topk_sample.MAX_TOP_K
    vals, ids = logits.float().topk(k, dim=-1)
    rows = torch.full((MAX_REQS, STEPS, V), -float("inf"), device=DEV)
    out = quixicore_ops.v2_candidate_draft(
        vals.contiguous(), ids.to(torch.int32).contiguous(), top_k, top_p if do_top_p else None,
        idx_mapping, seeds, positions + (1 + DRAFT_NOISE_SALT), temperature, V, rows, draft_step, use_fp64,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, ref), (out, ref)
    assert torch.equal(rows, ref_rows)
    # The drawn token is one the processed row keeps.
    kept = rows[idx_mapping.long(), step, :]
    assert bool(torch.isfinite(kept.gather(1, out.view(-1, 1))).all())


def test_candidate_draft_rejects_bad_shapes():
    logits, idx_mapping, temperature, top_k, top_p, seeds, positions = _batch(3, 2, [1.0, 1.0], [8, 8], [1.0, 1.0])
    vals, ids = logits.float().topk(8, dim=-1)
    with pytest.raises(RuntimeError):
        quixicore_ops.v2_candidate_draft(
            vals.contiguous(), ids.contiguous(), top_k, None, idx_mapping, seeds, positions, temperature, V,
            None, None, False,
        )
    with pytest.raises(RuntimeError):
        quixicore_ops.v2_candidate_draft(
            torch.zeros(2, 200, device=DEV), torch.zeros(2, 200, device=DEV, dtype=torch.int32), top_k, None,
            idx_mapping, seeds, positions, temperature, V, None, None, False,
        )
