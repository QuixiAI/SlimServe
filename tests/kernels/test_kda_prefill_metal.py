# SPDX-License-Identifier: Apache-2.0
"""Metal KDA prefill recurrence (kda_recur over prepared rows, via
kda_recurrent_prefill_metal) against the torch per-token reference
(kda_recurrent_prefill_native): varlen batches with and without initial
state, state writeback, and zero output / untouched pool on null slots."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal only"
)
DEV = "mps"


def _available():
    from vllm.quixicore import quixicore_ops

    return quixicore_ops.is_available() and hasattr(quixicore_ops, "kda_recur_prefill")


@pytest.mark.parametrize("lens", [[7], [1, 5, 3], [64, 1, 130], [300]])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_kda_prefill_metal_matches_native(lens, dtype):
    if not _available():
        pytest.skip("kda_recur_prefill not built")
    from vllm.model_executor.layers.mamba.gdn.kda_mps_fallback import (
        KdaVarlenPlan,
        kda_recurrent_prefill_metal,
        kda_recurrent_prefill_native,
    )

    H, K, V = 4, 128, 128
    T = sum(lens)
    g = torch.Generator().manual_seed(T + len(lens))
    q = (torch.randn(1, T, H, K, generator=g) * 0.5).to(dtype)
    k = (torch.randn(1, T, H, K, generator=g) * 0.5).to(dtype)
    v = (torch.randn(1, T, H, V, generator=g) * 0.5).to(dtype)
    raw_g = (torch.randn(1, T, H, K, generator=g) * 0.5).to(dtype)
    raw_beta = torch.randn(1, T, H, generator=g).to(dtype)
    A_log = torch.randn(H, generator=g) * 0.1
    dt_bias = torch.randn(H * K, generator=g) * 0.1
    # Real rows take slots 1..n. In multi-row batches row 0 is the null
    # slot (0): the kernel emits zeros there and leaves the pool alone (the
    # decode kda_step contract; the torch reference computes a throwaway
    # output for it), and row 1 carries no initial state.
    n = len(lens)
    slots = list(range(1, n + 1))
    has_init = [True] * n
    if n > 1:
        slots[0] = 0
        has_init[1] = False
    starts = [0]
    for L in lens:
        starts.append(starts[-1] + L)
    cu = torch.tensor(starts, dtype=torch.int32)
    idx = torch.tensor(slots, dtype=torch.int32)
    hi = torch.tensor(has_init)
    pool = torch.randn(n + 1, H, V, K, generator=g) * 0.3

    args = (A_log.to(DEV), dt_bias.to(DEV), -5.0)
    plan = KdaVarlenPlan.build(cu, idx, hi)
    pool_ref = pool.clone().to(DEV)
    ref = kda_recurrent_prefill_native(
        q.to(DEV),
        k.to(DEV),
        v.to(DEV),
        raw_g.to(DEV),
        raw_beta.to(DEV),
        *args,
        pool_ref,
        plan,
    )
    pool_got = pool.clone().to(DEV)
    got = kda_recurrent_prefill_metal(
        q.to(DEV),
        k.to(DEV),
        v.to(DEV),
        raw_g.to(DEV),
        raw_beta.to(DEV),
        *args,
        pool_got,
        plan,
        cu.to(DEV),
        idx.to(DEV),
    )
    torch.mps.synchronize()
    tol = 3e-2 if dtype == torch.bfloat16 else 1e-2
    assert got.shape == ref.shape
    for i in range(n):
        a, b = starts[i], starts[i + 1]
        if slots[i] <= 0:
            assert not got[0, a:b].float().any(), (lens, dtype, i)
            continue
        err = (got[0, a:b].float() - ref[0, a:b].float()).abs().max().item()
        scale = ref[0, a:b].float().abs().max().item() + 1e-6
        assert err / scale <= tol, (err, scale, lens, dtype, i)
    # State writeback: identical slots touched, values close (relative).
    s_err = (pool_got - pool_ref).abs().max().item()
    s_scale = pool_ref.abs().max().item() + 1e-6
    assert s_err / s_scale <= 2e-2, (s_err, s_scale)
    # The null-slot row 0 of the pool is untouched by both.
    assert torch.equal(pool_got[0], pool.to(DEV)[0])
