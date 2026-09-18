# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused KDA decode chain (kda_chain_decode: conv + gate GEMVs + recurrence
+ gated RMS norm in two launches) against the Triton chain the KDA layer runs
otherwise: kda_gate_pair, causal_conv1d_update, fused_recurrent_kda,
FusedRMSNormGated. Speculative rows (accepted-count conv roll, per-row state
columns) and plain decode rows, bf16 and fp32 conv states, every V split."""

import pytest
import torch

qc = pytest.importorskip("vllm._quixicore_C")
from einops import rearrange  # noqa: E402

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (  # noqa: E402
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.kda_gate_projection import (  # noqa: E402
    kda_gate_pair,
)
from vllm.models.kimi_k3.amd.ops.third_party.kda.fused_recurrent import (  # noqa: E402
    fused_recurrent_kda,
)
from vllm.third_party.flash_linear_attention.ops.kda import (  # noqa: E402
    rms_norm_gated,
)

H, D = 16, 128
LB = -5.0
EPS = 1e-5


def _inputs(R, T, state_len, conv_dtype, seed, pad=48, ragged=False):
    torch.manual_seed(seed)
    dev = "cuda"
    lens = torch.full((R,), T, dtype=torch.int64)
    if ragged:
        lens[1::2] = torch.randint(1, T + 1, (len(lens[1::2]),))
    N = int(lens.sum()) if T > 1 else R
    dim = 3 * H * D
    # Row-padded projection so the views carry a row stride wider than their width.
    proj = torch.randn(N, dim + 2 * D + H + pad, device=dev, dtype=torch.bfloat16)
    mixed = proj[:, :dim]
    f_a = proj[:, dim : dim + D]
    g_a = proj[:, dim + D : dim + 2 * D]
    beta = proj[:, dim + 2 * D : dim + 2 * D + H].unsqueeze(0)
    f_w = torch.randn(H * D, D, device=dev, dtype=torch.bfloat16) * 0.05
    g_w = torch.randn(H * D, D, device=dev, dtype=torch.bfloat16) * 0.05
    A_log = torch.randn(H, device=dev) * 0.1
    dt_bias = torch.randn(H * D, device=dev) * 0.1
    slots = R * (T + 1) + 4
    conv_sd = (torch.randn(slots, state_len, dim, device=dev) * 0.5).to(conv_dtype)
    conv_w = torch.randn(dim, 4, device=dev) * 0.3
    ssm = torch.randn(slots, H, D, D, device=dev) * 0.1
    norm_w = (1.0 + 0.1 * torch.randn(D, device=dev)).to(torch.bfloat16)
    if T == 1:
        sidx = (1 + torch.arange(N, device=dev)).to(torch.int32)
        cu = None
        acc = None
        conv_idx = sidx
    else:
        sidx = (1 + torch.arange(R, device=dev)[:, None] * T + torch.arange(T, device=dev)[None, :]).to(torch.int32).contiguous()
        cu = torch.zeros(R + 1, dtype=torch.int32)
        cu[1:] = lens.cumsum(0).to(torch.int32)
        cu = cu.to(dev)
        acc = torch.randint(1, T + 1, (R,), device=dev, dtype=torch.int32)
        conv_idx = sidx[:, 0]
    return dict(mixed=mixed, f_a=f_a, g_a=g_a, beta=beta, f_w=f_w, g_w=g_w, A_log=A_log, dt_bias=dt_bias,
                conv_sd=conv_sd, conv_w=conv_w, ssm=ssm, norm_w=norm_w, sidx=sidx, cu=cu, acc=acc,
                conv_idx=conv_idx, N=N, T=T)


def _reference(i):
    conv_state = i["conv_sd"].clone().transpose(-1, -2)
    ssm = i["ssm"].clone()
    g1, g2 = kda_gate_pair(i["f_a"], i["g_a"], i["f_w"], i["g_w"])
    conv_out = torch.empty_like(i["mixed"])
    if i["T"] == 1:
        causal_conv1d_update(i["mixed"], conv_state, i["conv_w"], None, activation="silu",
                             conv_state_indices=i["conv_idx"], validate_data=True, out=conv_out)
    else:
        causal_conv1d_update(i["mixed"], conv_state, i["conv_w"], None, activation="silu",
                             conv_state_indices=i["conv_idx"], num_accepted_tokens=i["acc"],
                             query_start_loc=i["cu"], max_query_len=i["T"], validate_data=False, out=conv_out)
    q, k, v = (rearrange(x, "n (h d) -> 1 n h d", d=D) for x in conv_out.split(H * D, dim=-1))
    cu = i["cu"] if i["cu"] is not None else torch.arange(0, i["N"] + 1, device="cuda", dtype=torch.int32)
    out, _ = fused_recurrent_kda(q=q, k=k, v=v, raw_g=rearrange(g1, "n (h d) -> 1 n h d", d=D), raw_beta=i["beta"],
                                 A_log=i["A_log"], dt_bias=i["dt_bias"], lower_bound=LB, initial_state=ssm,
                                 cu_seqlens=cu, ssm_state_indices=i["sidx"], num_accepted_tokens=i["acc"])
    y = rms_norm_gated(out, rearrange(g2, "n (h d) -> n h d", d=D), i["norm_w"], None,
                       activation="sigmoid", eps=EPS)
    return y, conv_state.transpose(-1, -2), ssm


def _fused(i, split):
    conv_state = i["conv_sd"].clone().transpose(-1, -2)
    ssm = i["ssm"].clone()
    out = torch.empty(1, i["N"], H, D, device="cuda", dtype=torch.bfloat16)
    qc.kda_chain_decode(i["mixed"], conv_state, i["conv_w"], i["conv_idx"], i["f_a"], i["g_a"], i["f_w"], i["g_w"],
                        i["beta"], i["A_log"], i["dt_bias"], ssm, i["sidx"], i["cu"], i["acc"], i["norm_w"], out,
                        D**-0.5, LB, True, EPS, split)
    return out, conv_state.transpose(-1, -2), ssm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("conv_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("split", [1, 2, 4])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("R,T,state_len", [(1, 1, 3), (16, 1, 3), (3, 1, 6), (1, 4, 6), (8, 4, 6), (5, 3, 5), (4, 8, 10), (16, 4, 6)])
def test_chain_matches_triton(R, T, state_len, split, conv_dtype, ragged):
    if ragged and (T == 1 or R < 2):
        pytest.skip("ragged needs several multi-row requests")
    i = _inputs(R, T, state_len, conv_dtype, seed=R * 100 + T * 10 + state_len, ragged=ragged)
    y_ref, conv_ref, ssm_ref = _reference(i)
    y, conv, ssm = _fused(i, split)
    torch.cuda.synchronize()
    torch.testing.assert_close(conv, conv_ref, rtol=0, atol=0)
    # The gate GEMV's fp32 summation order differs from Triton's tree sum, so a
    # g1 value on a bf16 rounding boundary can flip one ulp (0.4 %) and scale
    # its state column by that much; everything else is bit-for-bit.
    torch.testing.assert_close(ssm, ssm_ref, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(y.float(), y_ref.float(), rtol=2e-2, atol=2e-2)
