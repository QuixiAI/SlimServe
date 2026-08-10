# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression: rejection_sample must never emit an out-of-vocab token id.

A NaN target-logits row used to poison the argmax reduction
(`argmax_combine` in csrc/quixicore/serving/v2_sample_kernels.cuh let NaN
replace the running best), emitting token id == vocab_size. That value then
propagated through last_sampled_tokens into input_ids and crashed the DSV4
hash router with a wild weight-row read. Covers both the native and Triton
paths for greedy and stochastic sampling with degenerate logits.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample

VOCAB = 129280  # DeepSeek V4 vocab; also exercises a partial final block.
N_SPEC = 5
NUM_LOGITS = N_SPEC + 1
MAX_REQS = 4


def _run(target_logits, draft_logits, temp):
    device = target_logits.device
    sampled, num_sampled = rejection_sample(
        target_logits,
        draft_logits,
        torch.tensor([11, 22, 33, 44, 55, 66], dtype=torch.int64, device=device),
        torch.tensor([0, NUM_LOGITS], dtype=torch.int32, device=device),
        torch.arange(1000, 1000 + NUM_LOGITS, dtype=torch.int64, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(NUM_LOGITS, dtype=torch.int32, device=device),
        torch.arange(NUM_LOGITS, dtype=torch.int32, device=device),
        torch.full((MAX_REQS,), temp, dtype=torch.float32, device=device),
        torch.zeros(MAX_REQS, dtype=torch.int64, device=device),
        N_SPEC,
    )
    n = int(num_sampled[0])
    assert n >= 1
    return sampled[0, :n].tolist()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("temp", [0.0, 0.9])
@pytest.mark.parametrize(
    "target_kind", ["normal", "nan_row0", "all_nan", "neg_inf_row0"]
)
@pytest.mark.parametrize("draft_kind", [None, "normal", "nan", "neg_inf", "huge"])
def test_rejection_sample_never_out_of_vocab(temp, target_kind, draft_kind):
    torch.manual_seed(0)
    device = "cuda"
    target = torch.randn(NUM_LOGITS, VOCAB, device=device) * 4.0
    if target_kind == "nan_row0":
        target[0].fill_(float("nan"))
    elif target_kind == "all_nan":
        target.fill_(float("nan"))
    elif target_kind == "neg_inf_row0":
        target[0].fill_(float("-inf"))

    draft = None
    if draft_kind is not None:
        fill = {
            "normal": None,
            "nan": float("nan"),
            "neg_inf": float("-inf"),
            "huge": 50.0,
        }[draft_kind]
        if fill is None:
            draft = torch.randn(MAX_REQS, N_SPEC, VOCAB, device=device)
        else:
            draft = torch.full((MAX_REQS, N_SPEC, VOCAB), fill, device=device)

    sampled = _run(target, draft, temp)
    for token in sampled:
        assert 0 <= token < VOCAB, (
            f"out-of-vocab token {token} for target={target_kind} "
            f"draft={draft_kind} temp={temp}: {sampled}"
        )
