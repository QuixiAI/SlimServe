# SPDX-License-Identifier: Apache-2.0
import copy
import json

import pytest

from benchmarks.kernels.check_glm53_attention_norms import (
    LAYOUTS,
    checked_debug_source,
    flat_config,
    layernorm_oracle,
    load_preserving_provenance,
    mismatch_examples,
    packed_inputs,
    rank_cache,
    source_info,
)
from benchmarks.kernels.check_glm53_cached_rmsnorm import compare, oracle


def source(body, hint=2048):
    return (
        f"@heuristics.reduction(size_hints={{'x': 8192, 'r0_': {hint}}})\n"
        f"def triton_norm(x):\n{body}\n"
    )


def test_logical_extent_not_rounded_hint():
    info = source_info(source("    r0_numel = 1536\n    x = x.to(tl.float32)"))
    assert info["widths"] == [1536]
    assert info["size_hints"]["r0_"] == 2048
    assert info["bf16_intermediate_casts"] == 0
    assert info["args"] == ["x"]


def test_combo_reports_each_logical_extent():
    info = source_info(
        source(
            "    if pid < n0:\n        r0_numel = 512\n"
            "    elif pid < n1:\n        r0_numel = 1536\n"
            "    else:\n        r0_numel = 128"
        )
    )
    assert info["widths"] == [512, 1536, 128]


def test_fingerprint_ignores_metadata_not_rounding():
    a = source_info(source("    r0_numel = 1536\n    x = x.to(tl.float32)"))
    b = source_info(source("    r0_numel = 1536\n    x = x.to(tl.float32)", 4096))
    c = source_info(source("    r0_numel = 1536\n    x = x.to(tl.bfloat16)"))
    assert a["body_sha256"] == b["body_sha256"]
    assert a["body_sha256"] != c["body_sha256"]
    assert c["bf16_intermediate_casts"] == 1


@pytest.mark.parametrize(
    "text", ["", "def triton_norm(x):\n    pass", source("    pass") * 2]
)
def test_ambiguous_sources_rejected(text):
    with pytest.raises(ValueError):
        source_info(text)


def test_actual_layouts_include_indexer_output_gap():
    assert LAYOUTS == {512: (1536, 512), 1536: (0, 1536), 128: (2048, 256)}
    assert flat_config(dict(kwargs=dict(XBLOCK=2), num_warps=1, num_stages=1)) == dict(
        XBLOCK=2, num_warps=1, num_stages=1
    )


def test_copy_preserves_code_provenance_but_not_cache_filename(tmp_path):
    import inspect

    original = tmp_path / "original.py"
    copied = tmp_path / "private.py"
    source = "def triton_norm():\n    return __file__\n"
    original.write_text(source)
    copied.write_text(source)
    function = load_preserving_provenance(
        copied, original, "triton_norm", "norm_provenance_test"
    )
    assert function() == str(copied)
    assert function.__code__.co_filename == str(original)
    assert inspect.getsourcefile(function) == str(original)
    assert inspect.getsource(function) == source
    assert original.read_text() == copied.read_text() == source


def test_provenance_rejects_original_import_and_changed_copy(tmp_path):
    original = tmp_path / "original.py"
    original.write_text("def triton_norm():\n    return 1\n")
    with pytest.raises(ValueError, match="private copy"):
        load_preserving_provenance(original, original, "triton_norm", "unused")
    copied = tmp_path / "private.py"
    copied.write_text("def triton_norm():\n    return 2\n")
    with pytest.raises(ValueError, match="copy differs"):
        load_preserving_provenance(copied, original, "triton_norm", "unused")


def test_first_writer_alias_preserves_function_but_not_rank_metadata(tmp_path):
    original, copied, debug = [
        tmp_path / n for n in ("original.py", "private.py", "writer.py")
    ]
    original.write_text("rank = 0\ndef triton_norm():\n    return rank, __file__\n")
    copied.write_bytes(original.read_bytes())
    debug.write_text(original.read_text().replace("rank = 0", "rank = 1"))
    function = load_preserving_provenance(
        copied, original, "triton_norm", "norm_alias_test", debug_source=debug
    )
    assert function() == (0, str(copied))
    assert function.__code__.co_filename == str(debug)
    debug.write_text("\n" + debug.read_text())
    with pytest.raises(ValueError, match="function/line differs"):
        load_preserving_provenance(
            copied, original, "triton_norm", "unused", debug_source=debug
        )
    debug.write_text(original.read_text().replace("return rank", "return 2"))
    with pytest.raises(ValueError, match="function/line differs"):
        load_preserving_provenance(
            copied, original, "triton_norm", "unused", debug_source=debug
        )


def test_debug_path_requires_exact_bounded_record(tmp_path):
    ptx = tmp_path / "kernel.ptx"
    writer = tmp_path / "writer.py"
    ptx.write_text(f'\t.file\t1 "{writer}"\n\t.file\t2 "/helper.py"\n')
    assert checked_debug_source(ptx, tmp_path) == writer
    for text in (
        '\t.file 1 "/elsewhere.py"',
        '\t.file 1 "relative.py"',
        "",
        ptx.read_text() * 2,
    ):
        ptx.write_text(text)
        with pytest.raises(ValueError):
            checked_debug_source(ptx, tmp_path)


def test_same_key_isolated_in_real_rank_cache_managers(tmp_path):
    import triton
    from triton.runtime.cache import get_cache_manager

    original = triton.knobs.cache.dir
    for rank in range(4):
        with triton.knobs.cache.scope():
            triton.knobs.cache.dir = str(rank_cache(tmp_path, rank))
            manager = get_cache_manager("ab" * 32)
            manager.put(f"debug-image-{rank}", "kernel.cubin")
    assert triton.knobs.cache.dir == original
    for rank in range(4):
        with triton.knobs.cache.scope():
            triton.knobs.cache.dir = str(rank_cache(tmp_path, rank))
            manager = get_cache_manager("ab" * 32)
            from pathlib import Path

            assert (
                Path(manager.get_file("kernel.cubin")).read_text()
                == f"debug-image-{rank}"
            )
    for bad in (-1, 4, True, "../elsewhere"):
        with pytest.raises(ValueError):
            rank_cache(tmp_path, bad)


def test_packed_inputs_replay_seed_and_stride():
    import torch

    x = packed_inputs(3, 530901, 1.0)
    assert x.shape == (3, 2336) and x.dtype == torch.bfloat16
    assert torch.equal(x, packed_inputs(3, 530901, 1.0))
    assert not torch.equal(x, packed_inputs(3, 531001, 1.0))
    assert x[:, 1536:2048].stride() == (2336, 1)


def test_fp64_reference_uses_single_final_round_and_centered_variance():
    import torch

    x = packed_inputs(3, 530901, 1.0)[:, :128]
    w = torch.linspace(0.25, 1.75, 128).bfloat16()
    b = torch.linspace(-0.125, 0.125, 128).bfloat16()
    centered = x.double() - x.double().mean(-1, keepdim=True)
    expected = (
        centered
        * torch.rsqrt(centered.square().mean(-1, keepdim=True) + 1e-6)
        * w.double()
        + b.double()
    ).bfloat16()
    assert torch.equal(layernorm_oracle(x, w, b), expected)
    normalized = x.double() * torch.rsqrt(
        x.double().square().mean(-1, keepdim=True) + 1e-5
    )
    staged = (normalized.bfloat16().double() * w.double()).bfloat16()
    actual = oracle(x, w)
    assert compare(actual, staged)["bit_mismatches"] > 0
    assert torch.equal(actual, (normalized * w.double()).bfloat16())


def test_mismatch_examples_preserve_signed_near_zero_values():
    import torch

    reference = torch.tensor([[0.0, 1e-6, -1e-6]], dtype=torch.bfloat16)
    actual = reference.clone()
    actual.view(torch.int16)[0, 1] += 3
    actual.view(torch.int16)[0, 2] += 4
    evidence = mismatch_examples(actual, reference)
    assert evidence["count"] == 2
    assert [r["bf16_ulp"] for r in evidence["worst"]] == [4, 3]
    assert evidence["worst"][0]["actual"] < 0
    assert evidence["worst"][1]["reference"] == float(reference[0, 1])
    assert mismatch_examples(reference, reference) == dict(count=0, worst=[])


@pytest.fixture
def audit_fixture(monkeypatch, tmp_path):
    from benchmarks.kernels import check_glm53_attention_norms as probe

    monkeypatch.setattr(probe, "verify", lambda _: None)
    monkeypatch.setattr(probe, "sha", lambda _: "digest")
    monkeypatch.setattr(probe, "ROWS", (1,))
    monkeypatch.setattr(probe, "SEEDS", (7,))
    monkeypatch.setattr(probe, "MAGNITUDES", (1.0,))
    records = [
        dict(
            rank=rank,
            arm=arm,
            source="norm.py",
            source_sha256="digest",
            selected=dict(hash="key"),
            cubin_sha256="digest",
            info=dict(kernel="triton_norm"),
        )
        for rank in range(4)
        for arm in ("combo", "split", "split", "split")
    ]
    manifest = dict(records=records, sources={}, original_files={}, weight_sha256={})
    metrics = [dict(elements=w, max_bf16_ulp=1, bit_mismatches=0) for w in LAYOUTS]
    checks = [
        dict(
            rank=rank,
            rows=1,
            seed=7,
            magnitude=1.0,
            passed=True,
            repeat_graph_guards_mutation_pass=True,
            oracle={a: copy.deepcopy([metrics, metrics]) for a in ("combo", "split")},
            outputs={a: [["output"] * 3] * 2 for a in ("combo", "split")},
            combo_vs_split=copy.deepcopy([metrics, metrics]),
        )
        for rank in range(4)
    ]
    summary = dict(
        status="complete",
        manifest_sha256="digest",
        weights={},
        checks=checks,
        binaries=[
            {
                k: r[k]
                for k in ("rank", "arm", "source_sha256", "selected", "cubin_sha256")
            }
            for r in records
        ],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    def run():
        (tmp_path / "summary.json").write_text(json.dumps(summary))
        probe.audit(path, tmp_path)
        return json.loads((tmp_path / "analysis.json").read_text())

    return summary, run


def test_audit_accepts_complete_receipts(audit_fixture):
    _, run = audit_fixture
    report = run()
    assert report["status"] == "complete" and report["numerical_pass"]
    assert report["pairs"] == 4


def test_audit_closes_failed_numerical_probe_without_promoting(audit_fixture):
    summary, run = audit_fixture
    summary["checks"][0]["oracle"]["split"][0][2]["max_bf16_ulp"] = 2
    summary["checks"][0]["passed"] = False
    summary["status"] = "failed"
    report = run()
    assert report["status"] == "complete" and not report["numerical_pass"]
    assert len(report["failed_cases"]) == 1


@pytest.mark.parametrize("corruption", ["count", "verdict", "extent", "hash"])
def test_audit_rejects_inconsistent_receipts(audit_fixture, corruption):
    summary, run = audit_fixture
    row = summary["checks"][0]
    if corruption == "count":
        summary["checks"].pop()
    elif corruption == "verdict":
        row["passed"] = False
    elif corruption == "extent":
        row["oracle"]["combo"][0][0]["elements"] = 2048
    else:
        row["combo_vs_split"][0][0]["bit_mismatches"] = 1
    with pytest.raises(ValueError):
        run()
