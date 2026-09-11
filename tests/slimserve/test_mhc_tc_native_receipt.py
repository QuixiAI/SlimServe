# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from benchmarks.kernels.check_glm53_mhc_tc_native import qualified_kernel_body


@pytest.mark.parametrize("mutation", [None, "arithmetic", "layout"])
def test_native_kernel_body_is_unchanged_except_namespace(mutation):
    root = Path(__file__).resolve().parents[2]
    probe = (root / "benchmarks/kernels/mhc_prefill_tc_probe.cu").read_text()
    native = (root / "csrc/quixicore/serving/glm53_mhc_prefill_tc.cuh").read_text()
    anchors = {
        "arithmetic": ("sum += v.x * v.x;", "sum += v.x;"),
        "layout": ("ROW = 520", "ROW = 512"),
    }
    if mutation is not None:
        original, replacement = anchors[mutation]
        mutated = native.replace(original, replacement)
        assert mutated != native, f"mutation anchor {original!r} no longer exists"
        native = mutated
    if mutation is None:
        qualified_kernel_body(probe, native)
    else:
        with pytest.raises(ValueError, match="kernel body changed"):
            qualified_kernel_body(probe, native)


def test_missing_census_source_fails_before_hash_or_gpu_probe(tmp_path, monkeypatch):
    from benchmarks.kernels import check_glm53_mhc_tc_native as check

    missing = tmp_path / "missing.cuh"
    monkeypatch.setattr(
        check.sys,
        "argv",
        [
            "check",
            "--census",
            str(tmp_path / "census.json"),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    monkeypatch.setattr(
        check,
        "qualified_census",
        lambda path: (
            {"source_sha256": {str(missing): "not-a-digest"}},
            {},
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("missing sources must fail before hashing or GPU work")

    monkeypatch.setattr(check, "sha", forbidden)
    monkeypatch.setattr(check.subprocess, "check_output", forbidden)
    with pytest.raises(ValueError, match="qualified source missing"):
        check.main()
