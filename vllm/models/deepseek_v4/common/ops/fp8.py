# SPDX-License-Identifier: Apache-2.0
"""Bit-level FP8 helpers for platforms without native OCP E4M3 conversion."""

from vllm.triton_utils import tl, tldevice, triton


@triton.jit
def e4m3fn_encode_software(value):
    """Return OCP E4M3FN bytes with round-to-nearest-even and saturation."""
    sign = tl.where(value < 0.0, 0x80, 0).to(tl.int32)
    absolute = tl.abs(value)

    # Keep the exponent path finite for zero/subnormal lanes. Triton evaluates
    # vector expressions eagerly even when the final tl.where selects another
    # branch.
    exponent = tl.floor(tl.log2(tl.maximum(absolute, 0.0009765625))).to(tl.int32)
    significand = absolute * tl.exp2(-exponent.to(tl.float32))
    mantissa = tldevice.rint((significand - 1.0) * 8.0).to(tl.int32)
    exponent_field = exponent + 7
    carry = mantissa >= 8
    mantissa = tl.where(carry, 0, mantissa)
    exponent_field += carry.to(tl.int32)
    normal = (exponent_field << 3) | mantissa

    subnormal_mantissa = tldevice.rint(absolute * 512.0).to(tl.int32)
    subnormal = tl.where(subnormal_mantissa >= 8, 0x08, subnormal_mantissa)
    code = tl.where(exponent < -6, subnormal, normal)
    code = tl.where(absolute < 0.0009765625, subnormal_mantissa, code)

    saturated = (absolute >= 448.0) | (exponent_field > 15) | (
        (exponent_field == 15) & (mantissa > 6)
    )
    code = tl.where(saturated, 0x7E, code)
    return (sign | code).to(tl.uint8)


@triton.jit
def e4m3fn_decode_software(code):
    # e4b15 and E4M3FN share their bit fields and differ only by an exponent
    # bias of eight. The bitcast is supported on Ampere; scaling by 2^8 then
    # recovers the exact OCP E4M3FN value represented by the byte.
    return code.to(tl.float8e4b15, bitcast=True).to(tl.float32) * 256.0
