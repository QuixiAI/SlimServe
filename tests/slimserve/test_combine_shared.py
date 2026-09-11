# SPDX-License-Identifier: Apache-2.0
"""Unsupported native combine inputs must leave the runner's add available."""

import pytest
import torch

from vllm.model_executor.layers.fused_moe import combine_shared


class SharedOutput:
    def __init__(self, value):
        self.value = value
        self.consumed = False

    def peek_output(self):
        return self.value

    @property
    def output(self):
        self.consumed = True
        return self.value


@pytest.fixture(autouse=True)
def clear_publication():
    combine_shared.retire()
    yield
    combine_shared.retire()


def test_supported_combine_consumes_shared_output_once():
    ids = torch.zeros(2, 8, dtype=torch.int32)
    output = torch.empty(2, 16, dtype=torch.bfloat16)
    shared = SharedOutput(torch.ones_like(output))
    combine_shared.publish(ids, shared)
    assert combine_shared.consume(ids, output) is shared.value
    assert shared.consumed
    assert combine_shared.consume(ids, output) is None
    assert combine_shared.retire()


@pytest.mark.parametrize(
    "case",
    [
        "fp16",
        "fp32",
        "width",
        "shared_offset",
        "output_offset",
        "strided",
        "wrong_ids",
        "shape",
        "device",
    ],
)
def test_unsupported_combine_keeps_shared_output_for_fallback(case):
    ids = torch.zeros(2, 8, dtype=torch.int32)
    output = torch.empty(2, 16, dtype=torch.bfloat16)
    value = torch.ones_like(output)
    if case in ("fp16", "fp32"):
        dtype = torch.float16 if case == "fp16" else torch.float32
        output, value = output.to(dtype), value.to(dtype)
    elif case == "width":
        output = torch.empty(2, 7, dtype=torch.bfloat16)
        value = torch.ones_like(output)
    elif case == "shared_offset":
        value = torch.ones(33, dtype=torch.bfloat16)[1:].view(2, 16)
    elif case == "output_offset":
        output = torch.empty(33, dtype=torch.bfloat16)[1:].view(2, 16)
    elif case == "strided":
        value = torch.ones(2, 32, dtype=torch.bfloat16)[:, ::2]
    elif case == "shape":
        value = torch.ones(1, 16, dtype=torch.bfloat16)
    elif case == "device":
        value = torch.empty_like(output, device="meta")
    shared = SharedOutput(value)
    combine_shared.publish(ids, shared)
    assert (
        combine_shared.consume(ids.clone() if case == "wrong_ids" else ids, output)
        is None
    )
    assert not shared.consumed
    assert not combine_shared.retire()
    assert shared.output is value
