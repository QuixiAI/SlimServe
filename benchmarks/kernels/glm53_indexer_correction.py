# SPDX-License-Identifier: Apache-2.0
"""Isolated indexer cancellation correction; not installed into serving."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

EXPONENT = -12
OPTIONS = dict(num_warps=1, num_stages=1, enable_fp_fusion=False)


def check_layout(packed, gamma, bias, output, selected, rows):
    import torch

    tensors = (packed, gamma, bias, output, selected)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("tensor arguments required")
    if any(t.device != packed.device for t in tensors):
        raise ValueError("one device required")
    if (
        any(t.dtype != torch.bfloat16 for t in tensors[:-1])
        or selected.dtype != torch.uint8
    ):
        raise ValueError("BF16 values and uint8 selection required")
    expected = (
        ((rows, 2336), (2336, 1)),
        ((128,), (1,)),
        ((128,), (1,)),
        ((rows, 128), (256, 1)),
        ((rows, 128), (128, 1)),
    )
    if any(
        (tuple(t.shape), t.stride()) != layout for t, layout in zip(tensors, expected)
    ):
        raise ValueError("packed/indexer/selection layout differs")


@triton.jit
def correct_indexer(Packed, Gamma, Bias, Output, Selected, N):
    row = tl.program_id(0)
    col = tl.arange(0, 128)
    valid = row < N
    y = tl.load(Output + row * 256 + col, valid, other=0).to(tl.float32)
    b = tl.load(Bias + col).to(tl.float32)
    scale = tl.abs(y - b) + tl.abs(b)
    selected = valid & (scale > 0) & (tl.abs(y) <= 0.000244140625 * scale)
    # Explicit diagnostic coverage; its bandwidth is included in probe timing.
    tl.store(Selected + row * 128 + col, selected.to(tl.uint8), valid)
    if tl.sum(selected.to(tl.int32), 0) > 0:
        x = tl.load(Packed + row * 2336 + 2048 + col, valid, other=0).to(tl.float64)
        mean = tl.sum(x, 0) / 128.0
        centered = x - mean
        variance = tl.sum(centered * centered, 0) / 128.0
        inverse = 1.0 / libdevice.sqrt(variance + tl.full((), 1e-6, tl.float64))
        gamma = tl.load(Gamma + col).to(tl.float64)
        value = (centered * inverse) * gamma + b.to(tl.float64)
        # Keep FP64 through affine; only the final result converts to BF16.
        tl.store(Output + row * 256 + col, value.to(tl.bfloat16), selected)


class IndexerOnlyCorrection:
    """Original combo ABI plus a precompiled correction on the same stream."""

    def __init__(self, combo, correction, selected):
        if not callable(combo) or not callable(correction):
            raise ValueError("two precompiled launchers required")
        self.combo, self.correction, self.selected = combo, correction, selected
        self.combo_calls = self.correction_calls = 0

    def __call__(self, *args, stream):
        if len(args) != 11:
            raise ValueError("original combo ABI requires eleven arguments")
        rows = args[8]
        if type(rows) is not int or rows <= 0 or args[9:] != (rows, rows):
            raise ValueError("matching positive row counts required")
        check_layout(args[0], args[3], args[4], args[7], self.selected, rows)
        result = self.combo(*args, stream=stream)
        self.combo_calls += 1
        self.correction(
            args[0], args[3], args[4], args[7], self.selected, rows, stream=stream
        )
        self.correction_calls += 1
        return result
