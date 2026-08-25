# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metal SwiGLU parity for the native default and guarded OAI parameters."""

import torch

try:
    import pytest

    pytestmark = pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="requires Apple Metal (MPS)",
    )
except ModuleNotFoundError:
    pass


def main() -> None:
    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation,
        _metal_swiglu,
        apply_moe_activation,
    )

    torch.manual_seed(17)
    for dtype in (torch.float16, torch.bfloat16):
        x = (torch.randn(19, 128, dtype=dtype, device="mps") * 3).contiguous()

        gate = torch.clamp(x[:, :64], max=7.0)
        up = torch.clamp(x[:, 64:], min=-7.0, max=7.0)
        ref = gate * torch.sigmoid(gate) * up
        got = torch.empty_like(ref)
        assert _metal_swiglu(got, x, 7.0, oai_form=True, alpha=1.0, beta=0.0)
        assert torch.equal(got.cpu(), ref.cpu())

        ref = gate * torch.sigmoid(1.702 * gate) * (up + 1.0)
        got = torch.empty_like(ref)
        apply_moe_activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            got,
            x,
            clamp_limit=7.0,
            alpha=1.702,
            beta=1.0,
        )
        assert torch.equal(got.cpu(), ref.cpu())
        probe = torch.empty_like(ref)
        assert not _metal_swiglu(probe, x, 7.0, oai_form=True, alpha=1.702, beta=1.0)

    print("Metal SwiGLU default-native and nondefault-eager parity passed")


def test_metal_swiglu() -> None:
    main()


if __name__ == "__main__":
    main()
