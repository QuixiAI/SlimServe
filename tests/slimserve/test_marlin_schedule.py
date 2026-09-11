# SPDX-License-Identifier: Apache-2.0
import argparse

import pytest
import torch

from benchmarks.kernels.benchmark_glm53_marlin_schedule import footprint, parse_config
from benchmarks.kernels.check_glm53_marlin_schedule import decode_fragments


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
    assert footprint(cases, "moe") == 6 * 4096 * 1536 * 9 // 16
    with pytest.raises(ValueError, match="unknown phase"):
        footprint(cases, "not-a-phase")


def test_independent_planar_nvfp4_decode():
    # Low nibble is the earlier K element; sign is bit 3, groups are 16.
    packed = torch.tensor(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8
    )
    scales = torch.tensor([[2.0, 0.5]], dtype=torch.float8_e4m3fn)
    values = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        decode_fragments(packed, scales), torch.cat([values * 2, values / 2])[None]
    )
    with pytest.raises(ValueError, match="group-16"):
        decode_fragments(packed, scales[:, :1])
