# SPDX-License-Identifier: Apache-2.0
from collections import Counter

import pytest

from benchmarks.kernels.compare_cuda_sass import body_sha, compare, function_bodies


def fixture(control="0x000e220008000a00", spaces=True):
    word = f"/* {control} */" if spaces else f"/*{control}*/"
    return [
        "arch = sm_120f\n",
        "Function : selector\n",
        " /*00f0*/ LDCU.64 UR4, c[0x4][0x150] ; /* 0x01002a00ff0477ac */\n",
        "   " + word + "\n",
    ]


@pytest.mark.parametrize("spaces", [True, False])
def test_parser_keeps_separate_scheduling_words(spaces):
    name, body = list(function_bodies(fixture(spaces=spaces)))[0]
    assert name == "arch = sm_120f selector"
    assert "0x01002a00ff0477ac" in body[0xF0]
    assert "0x000e220008000a00" in body[0xF0]
    assert len(body[0xF0].splitlines()) == 2


def test_scheduling_only_change_is_not_identical():
    name, first = list(function_bodies(fixture()))[0]
    _, second = list(function_bodies(fixture(control="0x000f220008000a00")))[0]
    result = compare(
        {name: Counter({body_sha(first): 1})}, {name: Counter({body_sha(second): 1})}
    )
    assert result["identical_common_functions"] == 0
    assert result["changed_common_functions"] == [name]


@pytest.mark.parametrize("tail", [[], ["Function : next\n"], ["/* 0x1 */\n"]])
def test_missing_or_malformed_control_word_fails(tail):
    with pytest.raises(AssertionError):
        list(function_bodies(fixture()[:-1] + tail))


def test_duplicate_function_copies_are_retained():
    bodies = list(function_bodies(fixture() + fixture()))
    assert len(bodies) == 2 and bodies[0] == bodies[1]
    name, body = bodies[0]
    result = compare(
        {name: Counter({body_sha(body): 2})}, {name: Counter({body_sha(body): 1})}
    )
    assert result["identical_common_functions"] == 1
    assert result["changed_common_functions"] == [name]


@pytest.mark.parametrize("added", ["original", "different-codegen"])
def test_added_common_function_copy_is_reported(added):
    before = {"selector": Counter({"original": 1})}
    candidate = {"selector": Counter({"original": 1})}
    candidate["selector"][added] += 1
    result = compare(before, candidate)
    assert result["changed_common_functions"] == ["selector"]
    assert result["identical_common_functions"] == 1
    assert result["before_functions"] == 1
    assert result["candidate_functions"] == 2
