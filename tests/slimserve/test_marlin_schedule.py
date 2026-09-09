# SPDX-License-Identifier: Apache-2.0
import argparse

import pytest

from benchmarks.kernels.benchmark_glm53_marlin_schedule import footprint, parse_config


def test_only_existing_generated_launch_shapes():
    assert parse_config("auto") == (-1, -1, -1)
    assert parse_config("64,128,3") == (64, 128, 3)
    for invalid in ("64,64,1", "128,128,0", "64,128,5", "bad", "1,2"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_config(invalid)


def test_unique_weight_footprint_groups_by_layer_not_route():
    cases = [
        {"layer": 3, "expert_ids": [[0, 1], [1, 2]]},
        {"layer": 3, "expert_ids": [[2, 3]]},
        {"layer": 4, "expert_ids": [[0, 1]]},
    ]
    assert footprint(cases, "gate_up") == 6 * 1024 * 4096 * 9 // 16
    assert footprint(cases, "down") == 6 * 4096 * 512 * 9 // 16
