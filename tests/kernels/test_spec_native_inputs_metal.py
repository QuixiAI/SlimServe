# SPDX-License-Identifier: Apache-2.0
"""Torch-native (Metal) replacements for the autoregressive speculator's
Triton input-prep kernels, checked against a literal per-request Python
transcription of the kernels' semantics."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal only"
)
DEV = "mps"


def _buffers(max_num_reqs, max_num_tokens):
    from vllm.v1.worker.gpu.input_batch import InputBuffers

    b = InputBuffers(max_num_reqs, max_num_tokens, torch.device(DEV))
    # Poison so untouched pads are visible.
    b.input_ids.fill_(-7)
    b.positions.fill_(-7)
    b.query_start_loc.fill_(-7)
    b.seq_lens.fill_(-7)
    return b


def _ref_prefill(
    qsl,
    seq_lens,
    idx_mapping,
    num_sampled,
    num_rejected,
    last_sampled,
    next_prefill,
    target_ids,
    target_pos,
    max_num_reqs,
):
    """Literal transcription of _prepare_prefill_inputs_kernel."""
    num_reqs = len(seq_lens)
    draft_ids = {}
    draft_pos = {}
    last_idx = [0] * max_num_reqs
    for r in range(num_reqs):
        qs, qe = qsl[r], qsl[r + 1]
        qlen = qe - qs - num_rejected[r]
        state = idx_mapping[r]
        nxt = last_sampled[state] if num_sampled[r] > 0 else next_prefill[state]
        for i in range(1, qlen):
            draft_ids[qs + i - 1] = target_ids[qs + i]
        last = qs + qlen - 1
        last_idx[r] = last
        draft_ids[last] = nxt
        for i in range(qlen):
            draft_pos[qs + i] = target_pos[qs + i]
    d_qsl = list(qsl[: num_reqs + 1]) + [qsl[num_reqs]] * (max_num_reqs - num_reqs)
    d_seq = list(seq_lens) + [0] * (max_num_reqs - num_reqs)
    return draft_ids, draft_pos, d_qsl, d_seq, last_idx


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_prepare_prefill_inputs_native(seed):
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        _prepare_prefill_inputs_native,
    )

    g = torch.Generator().manual_seed(seed)
    max_num_reqs, max_num_tokens = 8, 64
    num_reqs = int(torch.randint(1, 6, (1,), generator=g))
    qlens = torch.randint(1, 6, (num_reqs,), generator=g).tolist()
    qsl = [0]
    for q in qlens:
        qsl.append(qsl[-1] + q)
    num_tokens = qsl[-1]
    seq_lens = [q + int(torch.randint(0, 20, (1,), generator=g)) for q in qlens]
    idx_mapping = torch.randperm(max_num_reqs, generator=g)[:num_reqs].tolist()
    num_rejected = [int(torch.randint(0, q, (1,), generator=g)) for q in qlens]
    num_sampled = [int(torch.randint(0, 2, (1,), generator=g)) for _ in qlens]
    # the runner keeps last_sampled_tokens / next_prefill_tokens as
    # [max_num_reqs, 1]; the kernel reads them flat
    last_sampled = torch.randint(0, 1000, (max_num_reqs, 1), generator=g)
    next_prefill = torch.randint(1000, 2000, (max_num_reqs, 1), generator=g)
    target_ids = torch.randint(2000, 3000, (num_tokens,), generator=g)
    target_pos = torch.randint(0, 500, (num_tokens,), generator=g)

    ref_ids, ref_pos, ref_qsl, ref_seq, ref_last = _ref_prefill(
        qsl,
        seq_lens,
        idx_mapping,
        num_sampled,
        num_rejected,
        last_sampled.reshape(-1).tolist(),
        next_prefill.reshape(-1).tolist(),
        target_ids.tolist(),
        target_pos.tolist(),
        max_num_reqs,
    )

    bufs = _buffers(max_num_reqs, max_num_tokens)
    batch = SimpleNamespace(
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=DEV),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=DEV),
        idx_mapping=torch.tensor(idx_mapping, dtype=torch.int32, device=DEV),
        input_ids=target_ids.to(torch.int32).to(DEV),
        positions=target_pos.to(DEV),
    )
    last_token_indices = torch.full((max_num_reqs,), -7, dtype=torch.int64, device=DEV)
    step = torch.tensor(5, dtype=torch.int64, device=DEV)
    _prepare_prefill_inputs_native(
        last_token_indices,
        step,
        bufs,
        batch,
        torch.tensor(num_sampled, dtype=torch.int32, device=DEV),
        torch.tensor(num_rejected, dtype=torch.int32, device=DEV),
        last_sampled.to(DEV),
        next_prefill.to(DEV),
        max_num_reqs,
    )
    ids = bufs.input_ids.cpu().tolist()
    pos = bufs.positions.cpu().tolist()
    for i, v in ref_ids.items():
        assert ids[i] == v, (i, ids[i], v)
    for i, v in ref_pos.items():
        assert pos[i] == v, (i, pos[i], v)
    assert bufs.query_start_loc.cpu().tolist() == ref_qsl
    assert bufs.seq_lens.cpu().tolist() == ref_seq
    assert last_token_indices.cpu().tolist() == ref_last
    assert step.item() == 0


def test_prepare_decode_and_update_native():
    from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
        _prepare_decode_inputs_native,
        _update_draft_inputs_native,
    )

    max_num_reqs, num_reqs, max_model_len = 8, 3, 100
    bufs = _buffers(max_num_reqs, 32)
    bufs.positions[:num_reqs] = torch.tensor([10, 99, 50], device=DEV)
    draft0 = torch.tensor([[7, 0], [8, 0], [9, 0]], dtype=torch.int64, device=DEV)
    target_seq = torch.tensor([12, 100, 60], dtype=torch.int32, device=DEV)
    rejected = torch.tensor([1, 0, 2], dtype=torch.int32, device=DEV)
    _prepare_decode_inputs_native(
        draft0[:, 0], target_seq, rejected, bufs, max_model_len, max_num_reqs, True
    )
    assert bufs.input_ids[:num_reqs].cpu().tolist() == [7, 8, 9]
    assert bufs.positions[:num_reqs].cpu().tolist() == [11, 99, 51]
    assert bufs.seq_lens.cpu().tolist() == [12, 100, 59, 0, 0, 0, 0, 0]
    assert bufs.query_start_loc.cpu().tolist() == [0, 1, 2, 3, 3, 3, 3, 3, 3]

    hidden = torch.randn(num_reqs + 2, 16, device=DEV)
    next_hidden = torch.zeros(max_num_reqs, 16, device=DEV)
    out_tokens = torch.zeros(max_num_reqs, 3, dtype=torch.int64, device=DEV)
    step = torch.tensor(1, dtype=torch.int64, device=DEV)
    sampled = torch.tensor([21, 22, 23], dtype=torch.int64, device=DEV)
    _update_draft_inputs_native(
        sampled,
        step,
        hidden,
        out_tokens,
        next_hidden,
        bufs,
        num_reqs,
        max_model_len,
        True,
    )
    assert out_tokens[:num_reqs].cpu().tolist() == [[0, 21, 0], [0, 22, 0], [0, 23, 0]]
    assert bufs.input_ids[:num_reqs].cpu().tolist() == [21, 22, 23]
    assert torch.equal(next_hidden[:num_reqs], hidden[:num_reqs])
    assert bufs.positions[:num_reqs].cpu().tolist() == [12, 99, 52]
    assert bufs.seq_lens[:num_reqs].cpu().tolist() == [13, 100, 60]
