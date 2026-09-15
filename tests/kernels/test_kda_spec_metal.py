# SPDX-License-Identifier: Apache-2.0
"""Metal KDA speculative-verify step (kda_step with slot_table +
num_accepted: conv rewind mode + checkpointing per-channel delta rule +
gated norm) against the torch-native spec chain (kda_conv_spec_update_native
-> kda_recurrent_spec_native -> gated rmsnorm): outputs, conv-state rows and
every checkpointed SSM slot."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal only"
)
DEV = "mps"


def _qc():
    from vllm.quixicore.ops import quixicore_ops

    if not quixicore_ops.has_kernel("kda_recur_spec_d128"):
        pytest.skip("kda spec kernels not built")
    return quixicore_ops


@pytest.mark.parametrize(
    "lens,accepted",
    [([2], [1]), ([2, 2, 2], [2, 1, 2]), ([1, 3, 2], [1, 2, 3]), ([3, 3], [3, 1])],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_kda_spec_step_matches_torch(lens, accepted, dtype):
    from vllm.model_executor.layers.mamba.gdn.kda_mps_fallback import (
        gated_rmsnorm_sigmoid_native,
        kda_conv_spec_update_native,
        kda_recurrent_spec_native,
    )

    qc = _qc()
    torch.manual_seed(0)
    H, D, KS = 4, 128, 4
    R = len(lens)
    max_len = max(lens)
    T = sum(lens)
    num_spec = 3  # slots per request (K + 1)
    C = 3 * H * D
    L = KS - 1 + num_spec  # conv columns: kernel_size-1 + num_spec
    slots = 1 + R * num_spec  # slot 0 = null block
    conv_pool = (torch.randn(slots, C, L) * 0.5).to(dtype).to(DEV)
    ssm_pool = (torch.randn(slots, H, D, D) * 0.1).float().to(DEV)
    table = torch.zeros(R, num_spec, dtype=torch.int32)
    for r in range(R):
        table[r] = torch.arange(1 + r * num_spec, 1 + (r + 1) * num_spec)
    table = table.to(DEV)
    num_accepted = torch.tensor(accepted, dtype=torch.int32, device=DEV)
    cu = torch.tensor(
        [0] + [int(v) for v in torch.tensor(lens).cumsum(0)], dtype=torch.int32
    ).to(DEV)
    mixed_qkv = (torch.randn(T, C) * 0.7).to(dtype).to(DEV)
    g1 = (torch.randn(T, H * D) * 0.5).to(dtype).to(DEV)
    beta = torch.randn(T, H).to(dtype).to(DEV)
    g2 = torch.randn(T, H * D).to(dtype).to(DEV)
    conv_w = (torch.randn(C, KS) * 0.3).float().to(DEV)
    A_log = torch.randn(H).float().to(DEV)
    dt_bias = (torch.randn(H * D) * 0.2).float().to(DEV)
    norm_w = (torch.rand(D) + 0.5).to(dtype).to(DEV)
    lb, eps, l2_eps = -5.0, 1e-5, 1e-6
    scale = D**-0.5

    # ---- torch reference (its own copies of the pools)
    conv_ref = conv_pool.clone()
    ssm_ref = ssm_pool.clone()
    conv_out = kda_conv_spec_update_native(
        mixed_qkv, conv_ref, conv_w.to(dtype), None, "silu",
        table[:, 0], num_accepted, cu, max_len,
    )
    q, k, v = (x.reshape(1, T, H, D) for x in conv_out.split(H * D, dim=-1))
    o = kda_recurrent_spec_native(
        q, k, v, g1.view(1, T, H, D), beta.view(1, T, H), A_log, dt_bias, lb,
        ssm_ref, table, num_accepted, cu, max_len, scale=scale,
    )
    ref = gated_rmsnorm_sigmoid_native(
        o.view(T, H, D), g2.view(T, H, D), norm_w, eps
    ).reshape(T, H * D)

    # ---- fused kernel
    got = qc.kda_step(
        mixed_qkv, g1, beta, conv_w, conv_pool, ssm_pool, cu,
        table[:, 0].contiguous(), A_log, dt_bias, lb, True, norm_w, g2, eps,
        scale, l2_eps, slot_table=table, num_accepted=num_accepted,
    ).clone()
    torch.mps.synchronize()

    tol = 3e-2 if dtype == torch.bfloat16 else 1e-2
    err = (got.float() - ref.float()).abs().max().item()
    assert err <= tol, err
    # conv state: the rewritten columns [0, KS-2+len) of every request row
    for r in range(R):
        s = int(table[r, 0])
        n = KS - 2 + lens[r]
        a = conv_pool[s, :, :n].float()
        b = conv_ref[s, :, :n].float()
        assert torch.allclose(a, b, atol=1e-3, rtol=1e-2), (r, (a - b).abs().max())
    # ssm checkpoints: slots [r, t] for t < len (relative to the slot's
    # magnitude: the conv output is rounded to the activation dtype on both
    # sides, and three delta-rule steps compound that rounding)
    stol = 2e-2 if dtype == torch.bfloat16 else 5e-3
    for r in range(R):
        for t in range(lens[r]):
            s = int(table[r, t])
            d = (ssm_pool[s] - ssm_ref[s]).abs().max().item()
            d /= max(ssm_ref[s].abs().max().item(), 1e-6)
            assert d <= stol, (r, t, d)
    # untouched slots (null block and unused checkpoints) unchanged
    assert torch.equal(ssm_pool[0], ssm_ref[0])
