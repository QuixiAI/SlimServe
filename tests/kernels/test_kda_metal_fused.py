# SPDX-License-Identifier: Apache-2.0
"""Parity of the Metal ``kda_step`` kernels against the torch-native KDA
reference (``kda_mps_fallback.py``): decode rows, varlen prefill, null slots,
and state-pool persistence across two steps."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal only"
)

H, DK, DV, KS = 4, 128, 128, 4


def _qc():
    from vllm.quixicore.ops import quixicore_ops

    if not quixicore_ops.has("kda_step"):
        pytest.skip("kda_step not built")
    return quixicore_ops


def _reference(mixed, g1, beta, conv_w, conv_state, ssm, cu, slots, A_log,
               dt_bias, lb, norm_w, z, eps, has_init):
    """Sequential torch reference in fp32 on the same (cloned) pools."""
    from vllm.model_executor.layers.mamba.gdn import kda_mps_fallback as F

    T = mixed.shape[0]
    out = torch.zeros(T, H * DV, dtype=mixed.dtype, device=mixed.device)
    conv_state = conv_state.clone()
    ssm = ssm.clone()
    for r in range(len(slots)):
        s = int(slots[r])
        a, b = int(cu[r]), int(cu[r + 1])
        if s <= 0:
            continue
        if not has_init[r]:
            conv_state[s] = 0
            ssm[s] = 0
        for t in range(a, b):
            x = mixed[t : t + 1]
            co = F.kda_conv_update_native(
                x, conv_state, conv_w.to(mixed.dtype), None, "silu",
                torch.tensor([s], device=mixed.device),
            )
            o = F.kda_recurrent_decode_native(
                co, g1[t : t + 1].view(1, 1, H, DK), beta[t : t + 1].view(1, 1, H),
                A_log, dt_bias, lb, ssm, torch.tensor([s], device=mixed.device),
            )  # [1, 1, H, DV]
            on = F.gated_rmsnorm_sigmoid_native(
                o[0, 0], z[t].view(H, DV), norm_w, eps
            )
            out[t] = on.reshape(-1)
    return out, conv_state, ssm


def _make(T, slots_n, seed=0, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    dev = "mps"
    mixed = (torch.randn(T, 3 * H * DK, device=dev) * 0.5).to(dtype)
    g1 = (torch.randn(T, H * DK, device=dev) * 0.5).to(dtype)
    beta = torch.randn(T, H, device=dev).to(dtype)
    conv_w = torch.randn(3 * H * DK, KS, device=dev) * 0.3
    conv_state = (torch.randn(slots_n, 3 * H * DK, KS - 1, device=dev) * 0.5).to(dtype)
    ssm = torch.randn(slots_n, H, DV, DK, device=dev) * 0.05
    A_log = torch.randn(H, device=dev) * 0.3
    dt_bias = torch.randn(H * DK, device=dev) * 0.2
    norm_w = (1.0 + 0.1 * torch.randn(DV, device=dev)).to(dtype)
    z = torch.randn(T, H * DV, device=dev).to(dtype)
    return mixed, g1, beta, conv_w, conv_state, ssm, A_log, dt_bias, norm_w, z


def _run(cu, slots, has_init, T, slots_n, lb=-5.0, dtype=torch.bfloat16):
    qc = _qc()
    mixed, g1, beta, conv_w, conv_state, ssm, A_log, dt_bias, norm_w, z = _make(
        T, slots_n, dtype=dtype
    )
    eps = 1e-5
    # lower_bound 0.0 selects the softplus gate form in the kernel; the torch
    # reference expresses that form as lower_bound=None.
    ref, ref_conv, ref_ssm = _reference(
        mixed, g1, beta, conv_w, conv_state, ssm, cu, slots, A_log, dt_bias,
        None if lb == 0.0 else lb, norm_w, z, eps, has_init,
    )
    conv_k = conv_state.clone()
    ssm_k = ssm.clone()
    for r, s in enumerate(slots):
        if s > 0 and not has_init[r]:
            conv_k[s] = 0
            ssm_k[s] = 0
    cu_t = torch.tensor(cu, dtype=torch.int32, device="mps")
    sl_t = torch.tensor(slots, dtype=torch.int32, device="mps")
    out = qc.kda_step(
        mixed, g1, beta, conv_w.contiguous(), conv_k, ssm_k, cu_t, sl_t,
        A_log.contiguous(), dt_bias.contiguous(), lb, True, norm_w, z, eps,
        DK ** -0.5, 1e-6,
    )
    torch.mps.synchronize()
    return out, ref, conv_k, ref_conv, ssm_k, ref_ssm


def _check(out, ref, conv_k, ref_conv, ssm_k, ref_ssm, tol=3e-2):
    assert torch.isfinite(out.float()).all()
    err = (out.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item() + 1e-6
    assert err / scale < tol, f"output max abs err {err} (scale {scale})"
    # The MPS torch reference itself carries ~1e-3 error against float64
    # (measured 2026-09-11: kernel 4e-8, torch path 1.6e-3); the kernel is
    # held to float64 in test_decode_rows_float64 and to the torch path here.
    assert torch.allclose(ssm_k, ref_ssm, atol=5e-3, rtol=1e-2), (
        (ssm_k - ref_ssm).abs().max()
    )
    assert torch.allclose(conv_k.float(), ref_conv.float(), atol=1e-2, rtol=1e-2)


def test_decode_rows():
    B = 5
    cu = list(range(B + 1))
    slots = [3, 1, 6, 2, 4]
    _check(*_run(cu, slots, [True] * B, B, 8))


def test_decode_null_slot_rows():
    B = 4
    cu = list(range(B + 1))
    slots = [2, 0, 5, -1]
    out, ref, conv_k, ref_conv, ssm_k, ref_ssm = _run(cu, slots, [True] * B, B, 8)
    _check(out, ref, conv_k, ref_conv, ssm_k, ref_ssm)
    assert out[1].abs().sum() == 0 and out[3].abs().sum() == 0


def test_varlen_prefill_with_fresh_and_resumed_rows():
    cu = [0, 7, 8, 20]
    slots = [2, 5, 3]
    has_init = [False, True, False]
    _check(*_run(cu, slots, has_init, 20, 8))


def test_varlen_prefill_multi_chunk_rows():
    """Requests longer than the prepare kernel's 32-token grid chunk (chunk
    seams at 32/64/96: history from the raw rows, pool written by the last
    chunk) next to a 1-token row and a null slot."""
    cu = [0, 70, 71, 150, 183]
    slots = [2, 5, 0, 3]
    has_init = [False, True, True, True]
    _check(*_run(cu, slots, has_init, 183, 8))


def test_softplus_gate_form():
    B = 3
    _check(*_run(list(range(B + 1)), [1, 2, 3], [True] * B, B, 4, lb=0.0))


def test_two_steps_persist_state():
    qc = _qc()
    mixed, g1, beta, conv_w, conv_state, ssm, A_log, dt_bias, norm_w, z = _make(2, 4)
    cu = [0, 1, 2]
    slots = [1, 2]
    eps = 1e-5
    # reference over both steps sequentially
    ref, ref_conv, ref_ssm = _reference(
        mixed, g1, beta, conv_w, conv_state, ssm, cu, slots, A_log, dt_bias,
        -5.0, norm_w, z, eps, [True, True],
    )
    # kernel: step token 0 and token 1 as separate launches on the same pools
    conv_k, ssm_k = conv_state.clone(), ssm.clone()
    outs = []
    for t in range(2):
        cu_t = torch.tensor([0, 1], dtype=torch.int32, device="mps")
        sl_t = torch.tensor([slots[t]], dtype=torch.int32, device="mps")
        outs.append(
            qc.kda_step(
                mixed[t : t + 1], g1[t : t + 1], beta[t : t + 1], conv_w.contiguous(),
                conv_k, ssm_k, cu_t, sl_t, A_log.contiguous(), dt_bias.contiguous(),
                -5.0, True, norm_w, z[t : t + 1], eps, DK ** -0.5, 1e-6,
            ).clone()
        )
    torch.mps.synchronize()
    _check(torch.cat(outs), ref, conv_k, ref_conv, ssm_k, ref_ssm)


def test_decode_rows_float64():
    """The kernel's state update against a float64 CPU recurrence."""
    qc = _qc()
    B = 5
    slots = [3, 1, 6, 2, 4]
    mixed, g1, beta, conv_w, conv_state, ssm, A_log, dt_bias, norm_w, z = _make(B, 8)
    c = lambda t: t.cpu().double()  # noqa: E731
    cs, S = c(conv_state), c(ssm)
    for r in range(B):
        s = slots[r]
        x = c(mixed[r])
        win = torch.cat([cs[s], x[:, None]], -1)
        co = torch.nn.functional.silu((win * c(conv_w)).sum(-1))
        co = co.to(torch.bfloat16).double()
        q, k, v = co.split(H * DK)
        q = q.view(H, DK)
        k = k.view(H, DK)
        v = v.view(H, DV)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * DK**-0.5
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        g = c(g1[r]).view(H, DK) + c(dt_bias).view(H, DK)
        a = c(A_log).exp().view(H, 1)
        dec = torch.exp(-5.0 * torch.sigmoid(a * g))
        b = torch.sigmoid(c(beta[r]))
        St = S[s] * dec[:, None, :]
        hk = (St * k[:, None, :]).sum(-1)
        d = (v - hk) * b[:, None]
        S[s] = St + d[:, :, None] * k[:, None, :]
    cu_t = torch.tensor(list(range(B + 1)), dtype=torch.int32, device="mps")
    sl_t = torch.tensor(slots, dtype=torch.int32, device="mps")
    ssm_k = ssm.clone()
    qc.kda_step(
        mixed, g1, beta, conv_w.contiguous(), conv_state.clone(), ssm_k, cu_t,
        sl_t, A_log.contiguous(), dt_bias.contiguous(), -5.0, True, norm_w, z,
        1e-5, DK**-0.5, 1e-6,
    )
    torch.mps.synchronize()
    err = (c(ssm_k) - S).abs().max().item()
    assert err < 1e-6, err
