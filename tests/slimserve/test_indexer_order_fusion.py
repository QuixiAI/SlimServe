# SPDX-License-Identifier: Apache-2.0
from itertools import islice

import pytest
import torch

from benchmarks.kernels.benchmark_glm53_indexer_order_fusion import (
    make_call,
    timing_cases,
    verify_assertion_only_changes,
)


@pytest.mark.parametrize("label", ["post-sort", "fused"])
def test_timing_callback_only_sorts_control(label):
    values = [torch.zeros(3, 600), torch.zeros(3), torch.zeros(3)]
    output = torch.empty(3, 512, dtype=torch.int32)
    calls = []

    def operation(*args):
        assert all(a is b for a, b in zip(args[:3], values))
        assert args[3] is output and args[4:] == (3, 600, 1, 512)
        output.fill_(42)
        calls.append("native")

    def order(value):
        assert value is output and (output == 42).all()
        calls.append("order")

    callback = make_call(operation, label, *values, output, order)
    callback()
    assert calls == (["native", "order"] if label == "post-sort" else ["native"])


def test_prefill_cases_copy_actual_rows_without_changing_ranges():
    captured = dict(
        logits=torch.arange(7616 * 2).reshape(7616, 2).float(),
        starts=torch.zeros(7616, dtype=torch.int32),
        ends=torch.arange(7616, dtype=torch.int32),
    )
    cases = list(islice(timing_cases(captured), 2))
    assert [name for name, _ in cases] == ["actual-prefill-7616", "actual-prefill-583"]
    for (_, values), begin in zip(cases, (0, 7033)):
        for key in captured:
            assert torch.equal(values[key], captured[key][begin:])
            assert values[key].is_contiguous()
            assert values[key].data_ptr() != captured[key][begin:].data_ptr()


def test_synthetic_decode_uses_registered_physical_width_and_exact_visible_range():
    captured = dict(
        logits=torch.zeros(7616, 1), starts=torch.zeros(7616), ends=torch.zeros(7616)
    )
    first = list(islice(timing_cases(captured), 2, 6))
    second = list(islice(timing_cases(captured), 2, 6))
    for (name, values), (repeat_name, repeat_values), visible in zip(
        first, second, (250, 8192, 32768, 262144)
    ):
        assert name == repeat_name == f"synthetic-decode-1-visible-{visible}"
        assert values["logits"].shape == (1, 262144)
        assert values["starts"].dtype == values["ends"].dtype == torch.int32
        assert (values["starts"] == 0).all() and (values["ends"] == visible).all()
        assert (values["logits"][:, visible:] == 0).all()
        assert torch.isfinite(values["logits"]).all()
        assert all(torch.equal(values[k], repeat_values[k]) for k in values)


@pytest.mark.parametrize("extra_change", [None, "math", "schedule", "branch"])
def test_binary_exception_rejects_any_selection_or_control_change(extra_change):
    before = {
        0xE0: "@!P0 BRA P1, 0x200\ncontrol",
        0xF0: "metadata f0\ncontrol",
        0x100: "metadata100\ncontrol",
        0x110: "metadata110\ncontrol",
        0x130: "metadata130\ncontrol",
        0x1E0: "CALL.ABS.NOINC R2\ncontrol",
        0x200: "real selection instruction\ncontrol",
    }
    after = dict(before)
    for pc in (0xF0, 0x100, 0x110, 0x130):
        after[pc] = "relocated " + after[pc]
    if extra_change == "math":
        after[0x200] = "changed math\ncontrol"
    elif extra_change == "schedule":
        after[0x100] += "changed control"
    elif extra_change == "branch":
        after[0xE0] = "changed branch\ncontrol"
    if extra_change is None:
        assert len(verify_assertion_only_changes(before, after)) == 4
    else:
        with pytest.raises(AssertionError):
            verify_assertion_only_changes(before, after)
