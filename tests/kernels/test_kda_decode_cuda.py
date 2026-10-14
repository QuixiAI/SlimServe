# SPDX-License-Identifier: Apache-2.0
"""CUDA KDA decode recurrence vs the vendored Triton packed-decode kernel
(GLM-5.3 geometry: K = V = 128, 16 heads per TP4 rank)."""

import pytest
import torch

pytest.importorskip("vllm._quixicore_C")
from vllm.quixicore import quixicore_ops as qc  # noqa: E402
from vllm.models.kimi_k3.amd.ops.third_party.kda.fused_recurrent import (  # noqa: E402
    fused_recurrent_kda_packed_decode,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
H, K, V = 16, 128, 128


def _inputs(N, slots=256, seed=0, extra_cols=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    state = torch.randn(slots, H, V, K, device="cuda", generator=g) * 0.1
    mixed = torch.randn(N, 2 * H * K + H * V + extra_cols, device="cuda", generator=g).to(torch.bfloat16)
    raw_g = torch.randn(1, N, H, K, device="cuda", generator=g).to(torch.bfloat16)
    raw_beta = torch.randn(1, N, H, device="cuda", generator=g).to(torch.bfloat16)
    A_log = torch.randn(H, device="cuda", generator=g) * 0.1
    dt_bias = torch.randn(H * K, device="cuda", generator=g) * 0.1
    idx = torch.randperm(slots, device="cuda", generator=g)[:N].to(torch.int32)
    return state, mixed, raw_g, raw_beta, A_log, dt_bias, idx


@pytest.mark.parametrize("N", [1, 7, 32, 128])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_cuda_matches_triton(N, lower_bound):
    state, mixed, raw_g, raw_beta, A_log, dt_bias, idx = _inputs(N)
    if N >= 7:
        idx[2] = 0  # a row without state writes zeros and leaves the state alone
    s_ref = state.clone(); s_new = state.clone()
    o_ref, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv=mixed, raw_g=raw_g, raw_beta=raw_beta, A_log=A_log, dt_bias=dt_bias,
        lower_bound=lower_bound, initial_state=s_ref, state_indices=idx)
    o_new = qc.kda_decode(mixed, raw_g, raw_beta, A_log, dt_bias, s_new, idx, K ** -0.5, lower_bound)
    torch.testing.assert_close(o_new.float(), o_ref.float(), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(s_new, s_ref, rtol=1e-4, atol=1e-4)

