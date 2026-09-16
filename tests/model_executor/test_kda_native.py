# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-native KDA (Kimi Delta Attention) core: reference checks.

The Triton reference (vllm/models/kimi_k3/amd/ops/third_party/kda/
fused_recurrent.py) cannot run here (no CUDA); the oracle is an
independently written float64 recurrence on CPU. The native functions run on
CPU and, when available, MPS.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.gdn import kimi_gdn_linear_attn as kda_layer
from vllm.model_executor.layers.mamba.gdn.kda_mps_fallback import (
    KdaVarlenPlan,
    gated_rmsnorm_sigmoid_native,
    kda_conv_prefill_native,
    kda_conv_spec_update_native,
    kda_conv_update_native,
    kda_gate_native,
    kda_recurrent_decode_native,
    kda_recurrent_prefill_native,
    kda_recurrent_spec_native,
)

DEVICES = ["cpu"]
if torch.backends.mps.is_available():
    DEVICES.append("mps")

LOWER_BOUND = -5.0  # GLM-5.3-Flash gate_lower_bound
WIDTH = 4


# --------------------------------------------------------------------------
# float64 reference (independent formulation: explicit per-head loops)
# --------------------------------------------------------------------------


def ref_gate(raw_g, A_log, dt_bias, lower_bound):
    """[T, H, K] float64 log-decay per key channel."""
    H, K = raw_g.shape[-2], raw_g.shape[-1]
    x = raw_g.double()
    if dt_bias is not None:
        x = x + dt_bias.double().view(H, K)
    a = A_log.double().exp()[:, None]
    if lower_bound is not None:
        return lower_bound * torch.sigmoid(a * x)
    return -a * F.softplus(x, beta=1.0, threshold=20.0)


def ref_recurrence(q, k, v, gate, beta, S0):
    """Single sequence. q/k [T, H, K], v [T, H, V], gate [T, H, K] (log),
    beta [T, H], S0 [H, V, K]; all float64 on CPU.
    Returns (o [T, H, V], states [T, H, V, K] after every step)."""
    T, H, K = q.shape
    V = v.shape[-1]
    scale = K**-0.5
    o = torch.zeros(T, H, V, dtype=torch.float64)
    states = torch.zeros(T, H, V, K, dtype=torch.float64)
    for h in range(H):
        S = S0[h].clone()
        for t in range(T):
            qt = q[t, h] / torch.sqrt((q[t, h] ** 2).sum() + 1e-6) * scale
            kt = k[t, h] / torch.sqrt((k[t, h] ** 2).sum() + 1e-6)
            S = S * torch.exp(gate[t, h])[None, :]  # decay along the key axis
            hk = S @ kt
            d = (v[t, h] - hk) * beta[t, h]
            S = S + torch.outer(d, kt)
            o[t, h] = S @ qt
            states[t, h] = S
    return o, states


def ref_conv(x, prev, w, bias):
    """x [T, dim], prev [dim, width-1], w [dim, width] -> [T, dim] float64,
    SiLU; plus the trailing width-1 inputs (the next state)."""
    T, dim = x.shape
    width = w.shape[1]
    padded = torch.cat([prev.double(), x.double().T], dim=1)  # [dim, W-1+T]
    out = torch.zeros(dim, T, dtype=torch.float64)
    for t in range(T):
        acc = torch.zeros(dim, dtype=torch.float64)
        for j in range(width):
            acc += w.double()[:, j] * padded[:, t + j]
        if bias is not None:
            acc += bias.double()
        out[:, t] = F.silu(acc)
    return out.T, padded[:, -(width - 1) :]


def _rand(*shape, device, scale=1.0, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return (torch.randn(*shape) * scale).to(device)


def _params(H, K, device, seed):
    torch.manual_seed(seed)
    A_log = torch.randn(H) * 0.5
    dt_bias = torch.randn(H * K) * 0.5
    return A_log.to(device), dt_bias.to(device)


def _pack(q, k, v):
    """[T, H, K] x3 -> packed [T, 3*H*K] rows (post-conv layout)."""
    T = q.shape[0]
    return torch.cat([q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)], -1)


def _cpu64(x):
    return x.detach().to("cpu").double()


# --------------------------------------------------------------------------
# (b) gate formula, both forms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("lower_bound", [LOWER_BOUND, None])
@pytest.mark.parametrize("with_bias", [True, False])
def test_gate_both_forms(device, lower_bound, with_bias):
    H, K, T = 3, 16, 7
    A_log, dt_bias = _params(H, K, device, 1)
    torch.manual_seed(2)
    raw_g = (torch.randn(T, H, K) * 12).to(device)  # crosses softplus x > 20
    assert bool((raw_g.cpu() > 20).any())
    bias = dt_bias if with_bias else None
    got = kda_gate_native(raw_g, A_log, bias, lower_bound)
    assert got.dtype == torch.float32 and got.shape == raw_g.shape
    ref = ref_gate(
        raw_g.cpu(), A_log.cpu(), bias.cpu() if bias is not None else None, lower_bound
    )
    torch.testing.assert_close(_cpu64(got), ref, rtol=1e-5, atol=1e-5)
    if lower_bound is not None:
        assert bool((got.cpu() <= 0).all()) and bool((got.cpu() >= lower_bound).all())


@pytest.mark.parametrize("device", DEVICES)
def test_gate_bf16_input_promotes_to_fp32(device):
    H, K = 2, 8
    A_log, dt_bias = _params(H, K, device, 3)
    raw_g = _rand(5, H, K, device=device, seed=4).to(torch.bfloat16)
    got = kda_gate_native(raw_g, A_log, dt_bias, LOWER_BOUND)
    ref = ref_gate(raw_g.cpu(), A_log.cpu(), dt_bias.cpu(), LOWER_BOUND)
    assert got.dtype == torch.float32
    torch.testing.assert_close(_cpu64(got), ref, rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# (a) decode
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("H,D", [(2, 8), (4, 16)])
@pytest.mark.parametrize("lower_bound", [LOWER_BOUND, None])
def test_decode_matches_reference(device, H, D, lower_bound):
    K = V = D
    B, slots = 4, 6
    A_log, dt_bias = _params(H, K, device, 10)
    q = _rand(B, H, K, device=device, seed=11)
    k = _rand(B, H, K, device=device, seed=12)
    v = _rand(B, H, V, device=device, seed=13)
    raw_g = _rand(1, B, H, K, device=device, seed=14)
    raw_beta = _rand(1, B, H, device=device, seed=15)
    pool = _rand(slots, H, V, K, device=device, scale=0.1, seed=16)
    pool_before = pool.clone()
    # row 2 is a NULL/padded row (slot 0): zero output, pool untouched.
    idx = torch.tensor([3, 1, 0, 5], dtype=torch.int32, device=device)

    out = kda_recurrent_decode_native(
        _pack(q, k, v), raw_g, raw_beta, A_log, dt_bias, lower_bound, pool, idx
    )
    assert out.shape == (1, B, H, V) and out.dtype == q.dtype

    gate = ref_gate(raw_g[0].cpu(), A_log.cpu(), dt_bias.cpu(), lower_bound)
    beta = torch.sigmoid(_cpu64(raw_beta[0]))
    for b in range(B):
        s = int(idx[b])
        if s <= 0:
            assert torch.equal(out[0, b].cpu(), torch.zeros(H, V))
            continue
        o_ref, st = ref_recurrence(
            _cpu64(q[b : b + 1]),
            _cpu64(k[b : b + 1]),
            _cpu64(v[b : b + 1]),
            gate[b : b + 1],
            beta[b : b + 1],
            _cpu64(pool_before[s]),
        )
        torch.testing.assert_close(_cpu64(out[0, b]), o_ref[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(_cpu64(pool[s]), st[0], rtol=1e-4, atol=1e-4)
    # Untouched slots (incl. the null block) are byte-identical.
    for s in (0, 2, 4):
        assert torch.equal(pool[s].cpu(), pool_before[s].cpu())


# --------------------------------------------------------------------------
# (a) prefill (varlen, with and without initial state)
# --------------------------------------------------------------------------


def _prefill_case(device, H, D, seed, lens, slots_list, has_init_list, pool_slots=8):
    K = V = D
    T = sum(lens)
    A_log, dt_bias = _params(H, K, device, seed)
    q = _rand(1, T, H, K, device=device, seed=seed + 1)
    k = _rand(1, T, H, K, device=device, seed=seed + 2)
    v = _rand(1, T, H, V, device=device, seed=seed + 3)
    raw_g = _rand(1, T, H, K, device=device, seed=seed + 4)
    raw_beta = _rand(1, T, H, device=device, seed=seed + 5)
    pool = _rand(pool_slots, H, V, K, device=device, scale=0.1, seed=seed + 6)
    cu = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=device
    )
    slots = torch.tensor(slots_list, dtype=torch.int32, device=device)
    has_init = torch.tensor(has_init_list, dtype=torch.bool, device=device)
    return A_log, dt_bias, q, k, v, raw_g, raw_beta, pool, cu, slots, has_init


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("H,D", [(2, 8), (3, 16)])
@pytest.mark.parametrize("lower_bound", [LOWER_BOUND, None])
def test_prefill_varlen_matches_reference(device, H, D, lower_bound):
    # Mixed batch: decodes-as-length-1 rows, a zero-length (padded) row, a
    # NULL slot, rows with and without initial state, duplicate lengths
    # (exercise the length-group batching).
    lens = [5, 1, 9, 1, 0, 3, 3]
    slots_list = [2, 3, 4, 0, 5, 6, 7]
    has_init_list = [True, True, False, True, True, False, True]
    (A_log, dt_bias, q, k, v, raw_g, raw_beta, pool, cu, slots, has_init) = (
        _prefill_case(device, H, D, 20, lens, slots_list, has_init_list)
    )
    pool_before = pool.clone()
    plan = KdaVarlenPlan.build(cu, slots, has_init)
    assert sorted(plan.groups) == [1, 3, 5, 9]
    assert plan.groups[1] == [1, 3] and plan.groups[3] == [5, 6]

    out = kda_recurrent_prefill_native(
        q, k, v, raw_g, raw_beta, A_log, dt_bias, lower_bound, pool, plan
    )
    assert out.shape == v.shape and out.dtype == v.dtype

    gate = ref_gate(raw_g[0].cpu(), A_log.cpu(), dt_bias.cpu(), lower_bound)
    beta = torch.sigmoid(_cpu64(raw_beta[0]))
    starts = [0] + list(torch.tensor(lens).cumsum(0).tolist())
    for i, L in enumerate(lens):
        if L == 0:
            continue
        s, e = starts[i], starts[i + 1]
        slot = slots_list[i]
        S0 = (
            _cpu64(pool_before[slot])
            if (has_init_list[i] and slot > 0)
            else torch.zeros(H, D, D, dtype=torch.float64)
        )
        o_ref, st = ref_recurrence(
            _cpu64(q[0, s:e]),
            _cpu64(k[0, s:e]),
            _cpu64(v[0, s:e]),
            gate[s:e],
            beta[s:e],
            S0,
        )
        torch.testing.assert_close(_cpu64(out[0, s:e]), o_ref, rtol=1e-4, atol=1e-4)
        if slot > 0:
            torch.testing.assert_close(_cpu64(pool[slot]), st[-1], rtol=1e-4, atol=1e-4)
    # NULL slot and the untouched slots stay byte-identical (slot 5 belongs
    # to the zero-length row: nothing computed, nothing written).
    for s in (0, 1, 5):
        assert torch.equal(pool[s].cpu(), pool_before[s].cpu())


# --------------------------------------------------------------------------
# (d) prefill == token-by-token decode
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("H,D,T", [(2, 8, 1), (3, 16, 9), (4, 8, 6)])
def test_prefill_equals_stepwise_decode(device, H, D, T):
    (A_log, dt_bias, q, k, v, raw_g, raw_beta, pool, cu, slots, has_init) = (
        _prefill_case(device, H, D, 30, [T], [2], [True], pool_slots=4)
    )
    pool_dec = pool.clone()
    plan = KdaVarlenPlan.build(cu, slots, has_init)
    out_pre = kda_recurrent_prefill_native(
        q, k, v, raw_g, raw_beta, A_log, dt_bias, LOWER_BOUND, pool, plan
    )
    outs = []
    idx = torch.tensor([2], dtype=torch.int32, device=device)
    for t in range(T):
        outs.append(
            kda_recurrent_decode_native(
                _pack(q[0, t : t + 1], k[0, t : t + 1], v[0, t : t + 1]),
                raw_g[:, t : t + 1],
                raw_beta[:, t : t + 1],
                A_log,
                dt_bias,
                LOWER_BOUND,
                pool_dec,
                idx,
            )
        )
    out_dec = torch.cat(outs, dim=1)
    torch.testing.assert_close(out_pre.cpu(), out_dec.cpu(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(pool[2].cpu(), pool_dec[2].cpu(), rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# (a) spec verify with num_accepted rollback
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("H,D", [(2, 8), (4, 16)])
def test_spec_verify_matches_reference(device, H, D):
    """Rows resume from slot[num_accepted-1]; every position t stores the
    running state to slot[t]; a NULL resume row is skipped; short rows
    (len < num_spec+1) and zero-length rows are honored."""
    K = V = D
    num_spec = 3
    max_len = num_spec + 1
    lens = [4, 2, 4, 0, 4]
    T = sum(lens)
    A_log, dt_bias = _params(H, K, device, 40)
    q = _rand(1, T, H, K, device=device, seed=41)
    k = _rand(1, T, H, K, device=device, seed=42)
    v = _rand(1, T, H, V, device=device, seed=43)
    raw_g = _rand(1, T, H, K, device=device, seed=44)
    raw_beta = _rand(1, T, H, device=device, seed=45)
    pool = _rand(24, H, V, K, device=device, scale=0.1, seed=46)
    pool_before = pool.clone()
    rows_list = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 0, 10, 11],  # NULL store slot at position 1: skipped
        [12, 13, 14, 15],
        [0, 16, 17, 18],  # num_accepted=1 -> resume slot 0 (NULL): skipped
    ]
    acc_list = [2, 1, 3, 1, 1]
    rows = torch.tensor(rows_list, dtype=torch.int32, device=device)
    acc = torch.tensor(acc_list, dtype=torch.int32, device=device)
    cu = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=device
    )

    out = kda_recurrent_spec_native(
        q,
        k,
        v,
        raw_g,
        raw_beta,
        A_log,
        dt_bias,
        LOWER_BOUND,
        pool,
        rows,
        acc,
        cu,
        max_len,
    )
    assert out.shape == (1, T, H, V)

    gate = ref_gate(raw_g[0].cpu(), A_log.cpu(), dt_bias.cpu(), LOWER_BOUND)
    beta = torch.sigmoid(_cpu64(raw_beta[0]))
    starts = [0] + list(torch.tensor(lens).cumsum(0).tolist())
    touched = set()
    for i, L in enumerate(lens):
        s, e = starts[i], starts[i + 1]
        resume = rows_list[i][acc_list[i] - 1]
        if L == 0 or resume <= 0:
            assert torch.equal(out[0, s:e].cpu(), torch.zeros(L, H, V))
            continue
        o_ref, st = ref_recurrence(
            _cpu64(q[0, s:e]),
            _cpu64(k[0, s:e]),
            _cpu64(v[0, s:e]),
            gate[s:e],
            beta[s:e],
            _cpu64(pool_before[resume]),
        )
        torch.testing.assert_close(_cpu64(out[0, s:e]), o_ref, rtol=1e-4, atol=1e-4)
        for t in range(L):
            slot = rows_list[i][t]
            if slot > 0:
                touched.add(slot)
                torch.testing.assert_close(
                    _cpu64(pool[slot]), st[t], rtol=1e-4, atol=1e-4
                )
    for s in range(pool.shape[0]):
        if s not in touched:
            assert torch.equal(pool[s].cpu(), pool_before[s].cpu()), s


@pytest.mark.parametrize("device", DEVICES)
def test_spec_rollback_equals_decode_over_accepted_prefix(device):
    """Two verify steps: step 1 verifies 4 draft positions from a fresh
    state, 2 are accepted; step 2 resumes from slot[num_accepted-1]. The
    result must equal decoding the accepted prefix + step-2 tokens one token
    at a time (the fused_recurrent_kda rollback contract)."""
    H, D = 3, 8
    K = V = D
    max_len = 4
    A_log, dt_bias = _params(H, K, device, 50)
    tok = {
        name: _rand(1, 8, H, K if name != "v" else V, device=device, seed=51 + j)
        for j, name in enumerate(("q", "k", "v", "g"))
    }
    raw_beta = _rand(1, 8, H, device=device, seed=55)
    pool = torch.zeros(12, H, V, K, device=device)
    pool_dec = pool.clone()
    cu = torch.tensor([0, 4], dtype=torch.int32, device=device)
    # step 1: tokens 0..3, resume from slot 1 (fresh zeros), stores to 1..4
    rows1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=device)
    acc1 = torch.tensor([1], dtype=torch.int32, device=device)
    kda_recurrent_spec_native(
        tok["q"][:, :4],
        tok["k"][:, :4],
        tok["v"][:, :4],
        tok["g"][:, :4],
        raw_beta[:, :4],
        A_log,
        dt_bias,
        LOWER_BOUND,
        pool,
        rows1,
        acc1,
        cu,
        max_len,
    )
    # Oracle: decode tokens 0, 1 (the accepted prefix) one at a time in
    # slot 9; the step-1 checkpoint at slot[1] == 2 must equal that state.
    idx = torch.tensor([9], dtype=torch.int32, device=device)

    def decode_one(t):
        return kda_recurrent_decode_native(
            _pack(
                tok["q"][0, t : t + 1],
                tok["k"][0, t : t + 1],
                tok["v"][0, t : t + 1],
            ),
            tok["g"][:, t : t + 1],
            raw_beta[:, t : t + 1],
            A_log,
            dt_bias,
            LOWER_BOUND,
            pool_dec,
            idx,
        )

    for t in (0, 1):
        decode_one(t)
    torch.testing.assert_close(pool[2].cpu(), pool_dec[9].cpu(), rtol=1e-5, atol=1e-5)
    # 2 accepted: the next step presents the SAME block slots with
    # num_accepted=2, so the row resumes from slot[1] == 2 and re-stores
    # positions 0..3 into slots 1..4. Step 2 verifies tokens 4..7.
    acc2 = torch.tensor([2], dtype=torch.int32, device=device)
    out2 = kda_recurrent_spec_native(
        tok["q"][:, 4:],
        tok["k"][:, 4:],
        tok["v"][:, 4:],
        tok["g"][:, 4:],
        raw_beta[:, 4:],
        A_log,
        dt_bias,
        LOWER_BOUND,
        pool,
        rows1,
        acc2,
        cu,
        max_len,
    )
    dec = [decode_one(t) for t in (4, 5, 6, 7)]
    torch.testing.assert_close(
        out2.cpu(), torch.cat(dec, 1).cpu(), rtol=1e-5, atol=1e-5
    )
    # Final per-position store of step 2 (slot 4) == decode state.
    torch.testing.assert_close(pool[4].cpu(), pool_dec[9].cpu(), rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------
# conv: decode ring, varlen prefill, spec rollback
# --------------------------------------------------------------------------


def _conv_params(dim, device, seed):
    torch.manual_seed(seed)
    w = (torch.randn(dim, WIDTH) * 0.3).to(device)
    bias = (torch.randn(dim) * 0.1).to(device)
    return w, bias


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_conv_decode_matches_reference(device, state_dtype):
    dim, B, L = 12, 3, WIDTH - 1 + 2  # spec-configured pool (wider rows)
    w, bias = _conv_params(dim, device, 60)
    pool = _rand(6, dim, L, device=device, seed=61).to(state_dtype)
    pool_before = pool.clone()
    x = _rand(B, dim, device=device, seed=62)
    idx = torch.tensor([4, 0, 2], dtype=torch.int32, device=device)
    out = kda_conv_update_native(x, pool, w, bias, "silu", idx)
    assert out.shape == x.shape and out.dtype == x.dtype
    for b, s in enumerate(idx.tolist()):
        if s <= 0:
            assert torch.equal(out[b].cpu(), x[b].cpu())
            continue
        o_ref, st_ref = ref_conv(
            x[b : b + 1].cpu(),
            pool_before[s, :, : WIDTH - 1].cpu(),
            w.cpu(),
            bias.cpu(),
        )
        torch.testing.assert_close(_cpu64(out[b]), o_ref[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            _cpu64(pool[s, :, : WIDTH - 1]),
            st_ref.to(state_dtype).double(),
            rtol=0,
            atol=0,
        )
        # Spec-tail columns of the ring are not the non-spec ring: untouched.
        assert torch.equal(
            pool[s, :, WIDTH - 1 :].cpu(), pool_before[s, :, WIDTH - 1 :].cpu()
        )
    for s in (0, 1, 3, 5):
        assert torch.equal(pool[s].cpu(), pool_before[s].cpu())


@pytest.mark.parametrize("device", DEVICES)
def test_conv_prefill_varlen_matches_reference(device):
    dim = 10
    lens = [5, 1, 1, 0, 3, 6]
    slots_list = [2, 3, 0, 4, 5, 6]
    has_init_list = [True, False, True, True, True, False]
    T = sum(lens)
    w, bias = _conv_params(dim, device, 70)
    pool = _rand(8, dim, WIDTH - 1, device=device, seed=71)
    pool_before = pool.clone()
    x = _rand(T, dim, device=device, seed=72)
    cu = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=device
    )
    slots = torch.tensor(slots_list, dtype=torch.int32, device=device)
    has_init = torch.tensor(has_init_list, dtype=torch.bool, device=device)
    plan = KdaVarlenPlan.build(cu, slots, has_init)
    out = kda_conv_prefill_native(x, pool, w, bias, "silu", plan)
    starts = [0] + list(torch.tensor(lens).cumsum(0).tolist())
    for i, L in enumerate(lens):
        if L == 0:
            continue
        s, e = starts[i], starts[i + 1]
        slot = slots_list[i]
        prev = (
            pool_before[slot].cpu()
            if has_init_list[i] and slot > 0
            else torch.zeros(dim, WIDTH - 1)
        )
        o_ref, st_ref = ref_conv(x[s:e].cpu(), prev, w.cpu(), bias.cpu())
        torch.testing.assert_close(_cpu64(out[s:e]), o_ref, rtol=1e-4, atol=1e-4)
        if slot > 0:
            torch.testing.assert_close(_cpu64(pool[slot]), st_ref, rtol=1e-6, atol=1e-6)
    for s in (0, 1, 4, 7):
        assert torch.equal(pool[s].cpu(), pool_before[s].cpu())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_conv_spec_rollback_matches_reference(device, state_dtype):
    """Row i reads its window at offset num_accepted-1 and rewrites the row
    from column 0 as [window[1:], x_0..x_{s-1}], keeping the tail."""
    dim, num_spec = 8, 3
    max_len = num_spec + 1
    L = WIDTH - 1 + num_spec
    lens = [4, 2, 4, 0]
    T = sum(lens)
    w, bias = _conv_params(dim, device, 80)
    pool = _rand(6, dim, L, device=device, seed=81).to(state_dtype)
    pool_before = pool.clone()
    x = _rand(T, dim, device=device, seed=82)
    idx = torch.tensor([1, 2, 0, 3], dtype=torch.int32, device=device)
    acc = torch.tensor([3, 1, 2, 4], dtype=torch.int32, device=device)
    cu = torch.tensor(
        [0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=device
    )
    out = kda_conv_spec_update_native(x, pool, w, bias, "silu", idx, acc, cu, max_len)
    assert out.shape == x.shape and out.dtype == x.dtype
    starts = [0] + list(torch.tensor(lens).cumsum(0).tolist())
    for i, Ln in enumerate(lens):
        s, e = starts[i], starts[i + 1]
        slot = int(idx[i])
        if Ln == 0 or slot <= 0:
            if slot > 0:
                assert torch.equal(pool[slot].cpu(), pool_before[slot].cpu())
            continue
        off = int(acc[i]) - 1
        row = pool_before[slot].cpu()
        window = row[:, off : off + WIDTH - 1]
        o_ref, _ = ref_conv(x[s:e].cpu(), window, w.cpu(), bias.cpu())
        torch.testing.assert_close(_cpu64(out[s:e]), o_ref, rtol=1e-4, atol=1e-4)
        expect = row.clone()
        new_cols = torch.cat([window[:, 1:], x[s:e].cpu().T.to(state_dtype)], 1)
        expect[:, : WIDTH - 2 + Ln] = new_cols
        assert torch.equal(pool[slot].cpu(), expect), i
    for s in (0, 4, 5):
        assert torch.equal(pool[s].cpu(), pool_before[s].cpu())


# --------------------------------------------------------------------------
# output gate norm
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gated_rmsnorm_sigmoid(device, dtype):
    H, D, T = 3, 16, 5
    torch.manual_seed(90)
    x = torch.randn(1, T, H, D).to(device).to(dtype)
    g = torch.randn(T, H, D).to(device).to(dtype)
    weight = (1 + 0.1 * torch.randn(D)).to(device)
    eps = 1e-5
    got = gated_rmsnorm_sigmoid_native(x, g, weight, eps)
    xf = _cpu64(x)
    ref = (
        xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.cpu().double()
    )
    ref = ref * torch.sigmoid(_cpu64(g))
    assert got.dtype == dtype
    tol = 1e-5 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(_cpu64(got), ref, rtol=tol, atol=tol)


# --------------------------------------------------------------------------
# (c) determinism on MPS
# --------------------------------------------------------------------------


@pytest.mark.skipif("mps" not in DEVICES, reason="requires Apple Metal (MPS)")
def test_mps_determinism_x2():
    device = "mps"
    H, D = 4, 16
    lens = [7, 1, 3, 1]
    (A_log, dt_bias, q, k, v, raw_g, raw_beta, pool, cu, slots, has_init) = (
        _prefill_case(
            device,
            H,
            D,
            100,
            lens,
            [1, 2, 3, 4],
            [True, False, True, True],
            pool_slots=12,
        )
    )
    runs = []
    for _ in range(2):
        p = pool.clone()
        plan = KdaVarlenPlan.build(cu, slots, has_init)
        o = kda_recurrent_prefill_native(
            q, k, v, raw_g, raw_beta, A_log, dt_bias, LOWER_BOUND, p, plan
        )
        rows = torch.tensor(
            [[1, 5, 6, 7], [2, 8, 9, 10]], dtype=torch.int32, device=device
        )
        acc = torch.tensor([1, 1], dtype=torch.int32, device=device)
        cu_s = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)
        os_ = kda_recurrent_spec_native(
            q[:, :8],
            k[:, :8],
            v[:, :8],
            raw_g[:, :8],
            raw_beta[:, :8],
            A_log,
            dt_bias,
            LOWER_BOUND,
            p,
            rows,
            acc,
            cu_s,
            4,
        )
        od = kda_recurrent_decode_native(
            _pack(q[0, :2], k[0, :2], v[0, :2]),
            raw_g[:, :2],
            raw_beta[:, :2],
            A_log,
            dt_bias,
            LOWER_BOUND,
            p,
            torch.tensor([3, 4], dtype=torch.int32, device=device),
        )
        runs.append((o.cpu(), os_.cpu(), od.cpu(), p.cpu()))
    for a, b in zip(runs[0], runs[1]):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------
# platform dispatch switch
# --------------------------------------------------------------------------


def test_native_path_switch(monkeypatch):
    plat = kda_layer.current_platform
    monkeypatch.setattr(plat, "is_cuda", lambda: False)
    monkeypatch.setattr(plat, "is_rocm", lambda: False)
    monkeypatch.delenv("VLLM_METAL_KDA", raising=False)
    assert kda_layer._kda_native_path_default() is True
    monkeypatch.setenv("VLLM_METAL_KDA", "0")
    assert kda_layer._kda_native_path_default() is False
    monkeypatch.setattr(plat, "is_cuda", lambda: True)
    monkeypatch.delenv("VLLM_METAL_KDA", raising=False)
    assert kda_layer._kda_native_path_default() is False
    monkeypatch.setenv("VLLM_METAL_KDA", "native")
    assert kda_layer._kda_native_path_default() is True


# --------------------------------------------------------------------------
# layer wiring: KimiGatedDeltaNetAttention._forward_native end to end
# --------------------------------------------------------------------------


def _ref_row(x_row, conv_prev, S0, raw_g_row, raw_beta_row, g2_row, P):
    """One sequence through conv -> KDA recurrence -> gated norm (float64).
    x_row [s, 3*H*D] pre-conv; conv_prev [dim, width-1]; S0 [H, D, D]."""
    H, D = P["H"], P["D"]
    y, _ = ref_conv(x_row, conv_prev, P["w"], P["bias"])
    q, k, v = (t.reshape(-1, H, D) for t in y.split(H * D, dim=-1))
    gate = ref_gate(raw_g_row, P["A_log"], P["dt_bias"], LOWER_BOUND)
    beta = torch.sigmoid(raw_beta_row.double())
    o, _ = ref_recurrence(q, k, v, gate, beta, S0)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + P["eps"]) * P["norm_w"]
    return o * torch.sigmoid(g2_row.double())


def _fake_layer(device, H, D, width, num_spec, slots):
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    dim = 3 * H * D
    torch.manual_seed(200)
    w = torch.randn(dim, width) * 0.3
    A_log = torch.randn(H) * 0.5
    dt_bias = torch.randn(H * D) * 0.5
    norm_w = 1 + 0.1 * torch.randn(D)
    L = width - 1 + num_spec
    conv_pool = torch.randn(slots, dim, L) * 0.5
    ssm_pool = torch.randn(slots, H, D, D) * 0.1
    conv_kv = conv_pool if is_conv_state_dim_first() else conv_pool.transpose(1, 2)
    eps = 1e-5
    layer = SimpleNamespace(
        kv_cache=(conv_kv.clone().to(device), ssm_pool.clone().to(device)),
        conv1d=SimpleNamespace(weight=w.view(dim, 1, width).to(device), bias=None),
        local_num_heads=H,
        head_dim=D,
        local_projection_size=H * D,
        prefix="model.layers.0.self_attn",
        A_log=A_log.to(device),
        dt_bias=dt_bias.to(device),
        gate_lower_bound=LOWER_BOUND,
        o_norm=SimpleNamespace(
            forward_native=lambda x, g: gated_rmsnorm_sigmoid_native(
                x, g, norm_w.to(device), eps
            )
        ),
    )
    P = dict(
        H=H, D=D, w=w, bias=None, A_log=A_log, dt_bias=dt_bias, norm_w=norm_w, eps=eps
    )
    return layer, P, conv_pool, ssm_pool


def _conv_rows(layer):
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    c = layer.kv_cache[0]
    return c if is_conv_state_dim_first() else c.transpose(1, 2)


@pytest.mark.parametrize("device", DEVICES)
def test_layer_forward_native_mixed_spec_prefill(device):
    """Spec rows (verify with rollback) + prefill rows (incl. a length-1
    decode reclassified as prefill) in one batch, tokens interleaved, two
    padded rows, through the layer's native body."""
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
        KimiGatedDeltaNetAttention,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    H, D, width, num_spec = 2, 8, WIDTH, 2
    layer, P, conv_before, ssm_before = _fake_layer(device, H, D, width, num_spec, 12)
    dim = 3 * H * D
    # Batch token order: spec0 (3) | prefill A (5) | spec1 (2) | prefill B (1)
    spec_tok = [0, 1, 2, 8, 9]
    ns_tok = [3, 4, 5, 6, 7, 10]
    T = 11
    pad = 2
    torch.manual_seed(201)
    mixed = torch.randn(T + pad, dim).to(device)
    g1 = torch.randn(1, T + pad, H, D).to(device)
    beta = torch.randn(1, T + pad, H).to(device)
    g2 = torch.randn(T + pad, H, D).to(device)
    core = torch.full((1, T + pad, H, D), float("nan"), device=device)

    spec_rows = [[1, 2, 3], [4, 5, 6]]
    spec_acc = [2, 1]
    ns_slots = [7, 8]
    ns_has_init = [True, False]
    m = GDNAttentionMetadata(
        num_prefills=2,
        num_prefill_tokens=6,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=5,
        num_actual_tokens=T,
        has_initial_state=torch.tensor(ns_has_init, device=device),
        spec_query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32, device=device),
        non_spec_query_start_loc=torch.tensor(
            [0, 5, 6], dtype=torch.int32, device=device
        ),
        spec_state_indices_tensor=torch.tensor(
            spec_rows, dtype=torch.int32, device=device
        ),
        non_spec_state_indices_tensor=torch.tensor(
            ns_slots, dtype=torch.int32, device=device
        ),
        spec_sequence_masks=torch.tensor([True, False, True, False], device=device),
        spec_token_indx=torch.tensor(spec_tok, dtype=torch.long, device=device),
        non_spec_token_indx=torch.tensor(ns_tok, dtype=torch.long, device=device),
        num_accepted_tokens=torch.tensor(spec_acc, dtype=torch.int32, device=device),
    )
    KimiGatedDeltaNetAttention._forward_native(
        layer, mixed, g1, g2, beta, core, {layer.prefix: m}
    )
    assert torch.isfinite(core).all()
    assert torch.equal(core[0, T:].cpu(), torch.zeros(pad, H, D))

    mixed_c, g1_c, beta_c, g2_c = (t.cpu() for t in (mixed, g1[0], beta[0], g2))
    conv_after = _conv_rows(layer).cpu()
    ssm_after = layer.kv_cache[1].cpu()
    # Spec rows: window at offset acc-1, resume from rows[acc-1].
    spec_segments = [(0, [0, 1, 2]), (1, [8, 9])]
    for i, toks in spec_segments:
        off = spec_acc[i] - 1
        slot0 = spec_rows[i][0]
        prev = conv_before[slot0][:, off : off + width - 1]
        S0 = ssm_before[spec_rows[i][off]].double()
        ref = _ref_row(mixed_c[toks], prev, S0, g1_c[toks], beta_c[toks], g2_c[toks], P)
        torch.testing.assert_close(
            core[0, toks].cpu().double(), ref, rtol=1e-4, atol=1e-4
        )
        # conv row rewritten from col 0: [window[1:], x...]
        new_cols = torch.cat([prev[:, 1:], mixed_c[toks].T], 1)
        assert torch.allclose(conv_after[slot0][:, : new_cols.shape[1]], new_cols)
    # Non-spec rows: prefill A (5 tokens, has init), B (1 token, no init).
    for i, toks in ((0, [3, 4, 5, 6, 7]), (1, [10])):
        slot = ns_slots[i]
        prev = (
            conv_before[slot][:, : width - 1]
            if ns_has_init[i]
            else torch.zeros(dim, width - 1)
        )
        S0 = (
            ssm_before[slot].double()
            if ns_has_init[i]
            else torch.zeros(H, D, D).double()
        )
        ref = _ref_row(mixed_c[toks], prev, S0, g1_c[toks], beta_c[toks], g2_c[toks], P)
        torch.testing.assert_close(
            core[0, toks].cpu().double(), ref, rtol=1e-4, atol=1e-4
        )
        assert torch.allclose(
            conv_after[slot][:, : width - 1],
            torch.cat([prev, mixed_c[toks].T], 1)[:, -(width - 1) :],
        )
    # Untouched slots (null block, 9..11) are byte-identical.
    for s in (0, 9, 10, 11):
        assert torch.equal(ssm_after[s], ssm_before[s])
        assert torch.equal(conv_after[s], conv_before[s])


@pytest.mark.parametrize("device", DEVICES)
def test_layer_forward_native_pure_decode(device):
    from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
        KimiGatedDeltaNetAttention,
    )
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    H, D, width = 3, 8, WIDTH
    layer, P, conv_before, ssm_before = _fake_layer(device, H, D, width, 0, 6)
    dim = 3 * H * D
    B = 3
    torch.manual_seed(202)
    mixed = torch.randn(B, dim).to(device)
    g1 = torch.randn(1, B, H, D).to(device)
    beta = torch.randn(1, B, H).to(device)
    g2 = torch.randn(B, H, D).to(device)
    core = torch.full((1, B, H, D), float("nan"), device=device)
    slots = [2, 0, 5]  # middle row is a NULL/padded row
    m = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=B,
        num_decode_tokens=B,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=B,
        non_spec_query_start_loc=torch.arange(B + 1, dtype=torch.int32, device=device),
        non_spec_state_indices_tensor=torch.tensor(
            slots, dtype=torch.int32, device=device
        ),
    )
    KimiGatedDeltaNetAttention._forward_native(
        layer, mixed, g1, g2, beta, core, {layer.prefix: m}
    )
    assert torch.isfinite(core).all()
    mixed_c, g1_c, beta_c, g2_c = (t.cpu() for t in (mixed, g1[0], beta[0], g2))
    for b, slot in enumerate(slots):
        if slot <= 0:
            # Null row: the recurrence output is zero, so only the norm's
            # rmsnorm(0) * sigmoid(g2) == 0 remains.
            assert torch.equal(core[0, b].cpu(), torch.zeros(H, D))
            continue
        ref = _ref_row(
            mixed_c[b : b + 1],
            conv_before[slot][:, : width - 1],
            ssm_before[slot].double(),
            g1_c[b : b + 1],
            beta_c[b : b + 1],
            g2_c[b : b + 1],
            P,
        )
        torch.testing.assert_close(
            core[0, b : b + 1].cpu().double(), ref, rtol=1e-4, atol=1e-4
        )
    assert torch.equal(layer.kv_cache[1][0].cpu(), ssm_before[0])
    assert torch.equal(_conv_rows(layer)[0].cpu(), conv_before[0])
