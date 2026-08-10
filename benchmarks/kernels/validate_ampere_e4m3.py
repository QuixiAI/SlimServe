# SPDX-License-Identifier: Apache-2.0
"""Validate Ampere's software E4M3FN byte encoder."""

import torch

from vllm.models.deepseek_v4.common.ops.fp8 import (
    e4m3fn_decode_software,
    e4m3fn_encode_software,
)
from vllm.triton_utils import tl, triton


@triton.jit
def _encode_kernel(
    source, unscaled, corrected, decoded, count: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    values = tl.load(source + offsets, mask=mask)
    old = values.to(tl.float8e4b15).to(tl.uint8, bitcast=True)
    fixed = e4m3fn_encode_software(values)
    reconstructed = e4m3fn_decode_software(fixed)
    tl.store(unscaled + offsets, old, mask=mask)
    tl.store(corrected + offsets, fixed, mask=mask)
    tl.store(decoded + offsets, reconstructed, mask=mask)


def main() -> None:
    torch.manual_seed(7)
    values = torch.cat(
        (
            torch.linspace(-448.0, 448.0, 8193),
            torch.randn(8192) * 100.0,
            torch.tensor(
                [
                    -448.0,
                    -240.0,
                    -1.0,
                    -0.001953125,
                    0.0,
                    0.001953125,
                    1.0,
                    240.0,
                    448.0,
                ]
            ),
        )
    ).float()
    source = values.cuda()
    old = torch.empty_like(source, dtype=torch.uint8)
    corrected = torch.empty_like(old)
    decoded = torch.empty_like(source)
    block = 256
    _encode_kernel[(triton.cdiv(source.numel(), block),)](
        source, old, corrected, decoded, source.numel(), BLOCK=block
    )

    reference = values.to(torch.float8_e4m3fn).view(torch.uint8)
    old_matches = int((old.cpu() == reference).sum())
    corrected_matches = int((corrected.cpu() == reference).sum())
    decoded_reference = values.to(torch.float8_e4m3fn).float()
    decoded_matches = int((decoded.cpu() == decoded_reference).sum())
    print(
        {
            "values": values.numel(),
            "old_matches": old_matches,
            "corrected_matches": corrected_matches,
            "decoded_matches": decoded_matches,
        }
    )
    if corrected_matches != values.numel() or decoded_matches != values.numel():
        bad = (corrected.cpu() != reference).nonzero().flatten()[:16]
        raise AssertionError(
            torch.stack(
                (values[bad], corrected.cpu()[bad].float(), reference[bad].float()),
                dim=1,
            )
        )


if __name__ == "__main__":
    main()
