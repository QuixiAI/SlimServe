# SPDX-License-Identifier: Apache-2.0
import csv

import pytest

from benchmarks.analyze_marlin_counters import METRICS, analyze


def data(tmp_path, missing=False):
    names = [
        name for name in METRICS if not missing or name != "dram__bytes_op_read.sum"
    ]
    fields = ["ID", "Kernel Name", *names]
    path = tmp_path / "raw.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({name: METRICS[name] for name in names})
        for i in range(2):
            writer.writerow(
                {
                    "ID": i,
                    "Kernel Name": "void Marlin<0>()",
                    **{name: 100 for name in names},
                }
            )
    cases = {
        "status": "complete",
        "cases": [{"label": "test", "batch": 1, "layer": 3, "unique_experts": 8}],
    }
    return path, cases


def test_exact_pairing_and_byte_units(tmp_path):
    path, cases = data(tmp_path)
    result = analyze(path, cases)
    assert [r["phase"] for r in result["rows"]] == ["gate_up", "down"]
    row = result["rows"][0]
    assert row["unique_weight_footprint_bytes"] == 8 * 1024 * 4096 * 9 / 16
    assert row["profiled_dram_read_GB_per_s"] == 0.001


def test_missing_counter_is_not_zero(tmp_path):
    path, cases = data(tmp_path, missing=True)
    with pytest.raises(ValueError, match="missing metrics"):
        analyze(path, cases)


def test_scaled_bytes_are_decimal_not_binary(tmp_path):
    path, cases = data(tmp_path)
    lines = path.read_text().splitlines()
    lines[1] = lines[1].replace("byte", "Mbyte")
    path.write_text("\n".join(lines) + "\n")
    row = analyze(path, cases)["rows"][0]
    assert row["dram_read_bytes"] == 100_000_000
    assert row["dram_write_bytes"] == 100_000_000
    assert row["profiled_dram_read_GB_per_s"] == 1000


def test_incomplete_case_count_rejected(tmp_path):
    path, cases = data(tmp_path)
    cases["cases"] *= 2
    with pytest.raises(ValueError, match="unexpected number"):
        analyze(path, cases)
