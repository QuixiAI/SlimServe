# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF dequantization dispatch.

`ggml_get_to_cuda` returning nullptr for a type the caller then invokes is a
segfault, not an error, so both the coverage and the guard are worth pinning.
"""

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf import ops
from vllm.model_executor.layers.quantization.gguf.utils import DEQUANT_TYPES

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="dequant kernels are device code"
)

MXFP4 = 39
# OCP E2M1, the true values. The kernels carry 2x these with the factor folded
# into the scale; dequantization must undo that.
E2M1 = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


def _reference(raw: np.ndarray, cols: int) -> np.ndarray:
    """Decode MXFP4 straight from the spec: e8m0 scale, then low/high nibbles."""
    rows = raw.shape[0]
    view = raw.reshape(rows, cols // 32, 17)
    scale = np.exp2(view[:, :, 0].astype(np.int32) - 127).astype(np.float32)
    qs = view[:, :, 1:]
    values = np.concatenate([E2M1[qs & 0x0F], E2M1[qs >> 4]], axis=2)
    return (values * scale[:, :, None]).reshape(rows, cols)


def test_mxfp4_dequantize_matches_the_spec_exactly():
    rows, cols = 4, 256
    rng = np.random.default_rng(0)
    raw = np.empty((rows, cols // 32, 17), dtype=np.uint8)
    # Exponents around 127 keep the reference in a range float32 holds exactly.
    raw[:, :, 0] = rng.integers(120, 135, size=raw.shape[:2], dtype=np.uint8)
    raw[:, :, 1:] = rng.integers(0, 256, size=(rows, cols // 32, 16), dtype=np.uint8)
    raw = raw.reshape(rows, -1)

    got = ops.ggml_dequantize(
        torch.from_numpy(raw).cuda().contiguous(), MXFP4, rows, cols, torch.float32
    )
    want = torch.from_numpy(_reference(raw, cols)).cuda()
    # Every value is a table entry times a power of two, so this is exact.
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_every_advertised_dequant_type_has_a_kernel():
    """DEQUANT_TYPES is what the Python layer promises it can dequantize."""
    for weight_type in sorted(DEQUANT_TYPES, key=int):
        rows, cols = 1, 256
        raw = torch.zeros(rows, 4096, dtype=torch.uint8).cuda()
        # A missing kernel raises; a present one returns zeros for zero input.
        ops.ggml_dequantize(raw, int(weight_type), rows, cols, torch.float32)


def test_unknown_type_raises_instead_of_segfaulting():
    with pytest.raises(Exception, match="no dequant kernel"):
        ops.ggml_dequantize(
            torch.zeros(64, dtype=torch.uint8).cuda(), 31, 1, 32, torch.float32
        )
