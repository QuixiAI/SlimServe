# SPDX-License-Identifier: Apache-2.0
import copy
import json
import math

import pytest

from benchmarks.analyze_glm53_kda_choices import (
    KERNELS,
    compare_choices,
    disk_choice,
    read_pinned,
    scan_choices,
    within,
)
from benchmarks.analyze_glm53_reduction_receipts import sha


def config(warps=4):
    return dict(
        kwargs={"BS": 32}, num_warps=warps, num_stages=3, num_ctas=1, pre_hook=None
    )


def document():
    return dict(
        key=[16, 128, 64, True, "torch.bfloat16"],
        configs_timings=[
            [config(8), [1.0, 0.9, 1.1]],
            [config(2), [1.0, 0.8, 1.2]],
        ],
    )


def test_matches_lexicographic_selection_and_first_exact_tie():
    data = document()
    assert disk_choice(data)["selected"] == config(2)
    data["configs_timings"][1][1] = [1.0, 0.9, 1.1]
    assert disk_choice(data)["selected"] == config(8)
    data["configs_timings"][0][1] = [math.inf] * 3
    assert disk_choice(data)["selected"] == config(2)


@pytest.mark.parametrize("value", [math.nan, -1, -math.inf, True, "1.0"])
def test_invalid_timing_rejected(value):
    data = document()
    data["configs_timings"][0][1][0] = value
    with pytest.raises(ValueError, match="timing value"):
        disk_choice(data)


def test_incomplete_duplicate_and_all_failed_matrices_rejected():
    for data in (
        dict(key=[], configs_timings=[]),
        dict(key=[16], configs_timings=[]),
        dict(key=[16], configs_timings=[[config(), [math.inf] * 3]]),
        dict(key=[16], configs_timings=[[config(), [1, 2, 3]]] * 2),
    ):
        with pytest.raises(ValueError):
            disk_choice(data)


def make_cache(root, private=True):
    frozen = {}
    for rank in range(4) if private else [None]:
        for name in sorted(KERNELS):
            path = root / str(rank) / name if private else root / name
            path.mkdir(parents=True)
            path = path / (name + ".autotune.json")
            path.write_text(json.dumps(document()))
            frozen[str(path.relative_to(root))] = sha(path)
    return frozen


def test_complete_rank_private_cache_is_hash_checked(tmp_path):
    frozen = make_cache(tmp_path)
    rows = scan_choices(tmp_path, rank_private=True, frozen_files=frozen)
    assert len(rows) == 24
    path = tmp_path / next(iter(frozen))
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed tuning"):
        scan_choices(tmp_path, rank_private=True, frozen_files=frozen)


def test_extra_missing_or_ambiguous_cache_entries_rejected(tmp_path):
    frozen = make_cache(tmp_path)
    path = tmp_path / next(iter(frozen))
    path.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        scan_choices(tmp_path, rank_private=True)
    path.write_text(json.dumps(document()))
    duplicate = path.parent.parent / "another-key" / path.name
    duplicate.parent.mkdir()
    duplicate.write_text(path.read_text())
    with pytest.raises(ValueError, match="ambiguous"):
        scan_choices(tmp_path, rank_private=True)


def test_shared_cache_not_mislabelled_as_rank_receipts(tmp_path):
    make_cache(tmp_path, private=False)
    rows = scan_choices(tmp_path, rank_private=False)
    assert len(rows) == 6
    assert {key[0] for key in rows} == {"shared"}
    with pytest.raises(ValueError, match="incomplete"):
        scan_choices(tmp_path, rank_private=True)


def test_path_escape_and_wrong_receipt_rejected(tmp_path):
    for path in ("../other", "/tmp/other"):
        with pytest.raises(ValueError, match="unbounded"):
            within(tmp_path, path)
    link = tmp_path / "escape"
    link.symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        within(tmp_path, "escape/outside")
    path = tmp_path / "receipt.json"
    path.write_text("{}")
    assert read_pinned(path, sha(path)) == {}
    with pytest.raises(ValueError, match="receipt changed"):
        read_pinned(path, "0" * 64)


def test_compare_requires_identical_tuning_domain_not_just_kernel_name():
    row = dict(disk_key="key", **disk_choice(document()))
    control = {("0", "gate"): copy.deepcopy(row)}
    fresh = {("shared", "gate"): copy.deepcopy(row)}
    assert not compare_choices(control, fresh)[0]["changed"]
    fresh["shared", "gate"]["selected"] = config(8)
    assert compare_choices(control, fresh)[0]["changed"]
    for field in ("disk_key", "key", "candidates"):
        changed = copy.deepcopy(fresh)
        changed["shared", "gate"][field] = "different"
        with pytest.raises(ValueError, match="unmatched"):
            compare_choices(control, changed)
