# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic oracle checks for the fused Qwen3.5 GDN Metal kernels."""

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    _causal_conv1d_native,
    _gdn_recurrent_scan_native,
    _gdn_spec_state_step_native,
)
from vllm.quixicore.ops import quixicore_ops

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple Metal"
)

HK = 16
HV = 48
D = 128
CONV_DIM = 2 * HK * D + HV * D
WIDTH = 4


def _inputs(slots: int, conv_cols: int):
    device = torch.device("mps")
    conv_state = (torch.randn(slots, conv_cols, CONV_DIM, device=device) * 0.2).to(
        torch.bfloat16
    )
    ssm_state = torch.randn(slots, HV, D, D, device=device) * 0.05
    weight = torch.randn(CONV_DIM, WIDTH, device=device) * 0.2
    bias = torch.randn(CONV_DIM, device=device) * 0.05
    a_log = torch.log(torch.rand(HV, device=device) * 8 + 1)
    dt_bias = torch.randn(HV, device=device) * 0.3
    return conv_state, ssm_state, weight, bias, a_log, dt_bias


def _gates(a: torch.Tensor, b: torch.Tensor, a_log: torch.Tensor, dt_bias):
    g = -a_log.exp() * torch.nn.functional.softplus(
        a.float() + dt_bias, beta=1.0, threshold=20.0
    )
    return g, torch.sigmoid(b.float())


def test_fused_gdn_decode_matches_torch_oracle() -> None:
    torch.manual_seed(10)
    device = torch.device("mps")
    n = 2
    conv_state, ssm_state, weight, bias, a_log, dt_bias = _inputs(5, 10)
    qkvz = torch.randn(n, CONV_DIM + HV * D, device=device).to(torch.bfloat16)
    ba = torch.randn(n, 2 * HV, device=device).to(torch.bfloat16)
    x = qkvz[:, :CONV_DIM]  # non-contiguous row stride is part of the contract
    b, a = ba.chunk(2, dim=-1)
    slots = torch.tensor([1, 0], dtype=torch.int32, device=device)
    scale = D**-0.5

    conv_ref = conv_state.clone()
    ssm_ref = ssm_state.clone()
    conv_out, conv_final = _causal_conv1d_native(
        x[:1].float().unsqueeze(-1),
        conv_ref[1:2].transpose(-1, -2)[:, :, : WIDTH - 1].float(),
        weight,
        bias,
        "silu",
    )
    conv_ref[1, : WIDTH - 1] = conv_final[0].to(conv_ref.dtype).transpose(0, 1)
    tokens = conv_out.transpose(1, 2)
    q, k, v = torch.split(tokens, [HK * D, HK * D, HV * D], dim=-1)
    q = q.view(1, 1, HK, D)
    k = k.view(1, 1, HK, D)
    v = v.view(1, 1, HV, D)
    g, beta = _gates(a[:1], b[:1], a_log, dt_bias)
    out_ref, state_final = _gdn_recurrent_scan_native(
        q,
        k,
        v,
        g.unsqueeze(1),
        beta.unsqueeze(1),
        scale,
        ssm_ref[1:2],
        tiled_gqa=True,
    )
    ssm_ref[1] = state_final[0]

    conv_got = conv_state.clone()
    ssm_got = ssm_state.clone()
    out = torch.empty(n, HV, D, dtype=torch.bfloat16, device=device)
    token_map = torch.arange(n, dtype=torch.int32, device=device)
    quixicore_ops.qwen_gdn_step(
        x,
        a,
        b,
        conv_got.transpose(-1, -2),
        ssm_got,
        weight,
        bias,
        a_log,
        dt_bias,
        token_map,
        slots,
        slots,
        slots.view(n, 1),
        None,
        out,
        n,
        1,
        HK,
        True,
        True,
        scale,
    )
    torch.mps.synchronize()

    assert torch.allclose(out[0].float(), out_ref[0, 0], atol=2e-4, rtol=0)
    assert out[1].count_nonzero() == 0
    assert torch.equal(conv_got[1], conv_ref[1])
    assert torch.allclose(ssm_got[1], ssm_ref[1], atol=5e-7, rtol=0)
    assert torch.equal(conv_got[0], conv_state[0])
    assert torch.equal(ssm_got[0], ssm_state[0])


def test_fused_gdn_verify_matches_rollback_oracle() -> None:
    torch.manual_seed(11)
    device = torch.device("mps")
    n, seq_len = 2, 8
    conv_state, ssm_state, weight, bias, a_log, dt_bias = _inputs(20, 10)
    x = torch.randn(n, seq_len, CONV_DIM, device=device).to(torch.bfloat16)
    ba = torch.randn(n * seq_len, 2 * HV, device=device).to(torch.bfloat16)
    b, a = ba.chunk(2, dim=-1)
    slot_rows = torch.arange(1, n * seq_len + 1, device=device).view(n, seq_len)
    slot_rows[0, 6] = 0  # NULL per-position store, not a resume slot
    accepted = torch.tensor([1, 4], device=device)
    g, beta = _gates(a, b, a_log, dt_bias)
    scale = D**-0.5

    conv_ref = conv_state.clone()
    ssm_ref = ssm_state.clone()
    out_ref = _gdn_spec_state_step_native(
        x,
        g.view(n, seq_len, HV),
        beta.view(n, seq_len, HV),
        conv_ref.transpose(-1, -2),
        ssm_ref,
        slot_rows,
        accepted,
        weight,
        bias,
        "silu",
        HK,
        D,
        D,
        scale,
        True,
    )

    conv_got = conv_state.clone()
    ssm_got = ssm_state.clone()
    out = torch.empty(n * seq_len, HV, D, dtype=torch.bfloat16, device=device)
    token_map = torch.arange(n * seq_len, dtype=torch.int32, device=device)
    resume = slot_rows.gather(1, (accepted - 1).view(-1, 1)).squeeze(1)
    quixicore_ops.qwen_gdn_step(
        x.view(n * seq_len, CONV_DIM),
        a,
        b,
        conv_got.transpose(-1, -2),
        ssm_got,
        weight,
        bias,
        a_log,
        dt_bias,
        token_map,
        slot_rows[:, 0].to(torch.int32),
        resume.to(torch.int32),
        slot_rows.to(torch.int32),
        accepted.to(torch.int32),
        out,
        n,
        seq_len,
        HK,
        True,
        True,
        scale,
    )
    torch.mps.synchronize()

    assert torch.allclose(out.float(), out_ref.view_as(out).float(), atol=2e-4, rtol=0)
    assert torch.equal(conv_got, conv_ref)
    assert torch.allclose(ssm_got, ssm_ref, atol=5e-7, rtol=0)


def test_fused_gdn_gated_norm_matches_torch() -> None:
    torch.manual_seed(12)
    device = torch.device("mps")
    x = torch.randn(8, HV, D, device=device).to(torch.bfloat16)
    z_storage = torch.randn(8, 2 * HV * D, device=device).to(torch.bfloat16)
    z = z_storage[:, : HV * D].view(8, HV, D)
    weight = torch.randn(D, device=device)
    eps = 1e-6
    got = quixicore_ops.qwen_gdn_gated_norm(x, z, weight, eps)
    ref = RMSNormGated.forward_static(
        x.reshape(-1, D),
        z.reshape(-1, D),
        weight,
        eps,
        torch.bfloat16,
        None,
        True,
        "silu",
    ).view_as(x)
    error = (got.float() - ref.float()).abs().max()
    assert float(error / ref.float().abs().max()) < 1e-2
