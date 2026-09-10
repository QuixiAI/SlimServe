# SPDX-License-Identifier: Apache-2.0
import base64
import copy
import json
from types import SimpleNamespace

import pytest

from benchmarks.analyze_glm53_reduction_receipts import (
    audit_receipt,
    binary_cache_directory,
    declared_symbols,
    reduction_metadata,
)
from slimserve.reduction_receipts import graph_snapshot


@pytest.fixture
def receipt(tmp_path):
    name = "triton_red_fused_rms_norm_0"
    graph = tmp_path / "graph.py"
    graph.write_text(
        f"{name} = async_compile.triton('kernel', 'source')\n"
        f"def call(args):\n    {name}.run(*args)\n"
    )
    source = tmp_path / "inductor/aa/kernel.py"
    source.parent.mkdir(parents=True)
    metadata = dict(
        kernel_name=name,
        num_reduction=1,
        deterministic=True,
        batch_invariant=False,
        are_deterministic_algorithms_enabled=False,
    )
    source.write_text(
        f"@triton_heuristics.reduction(inductor_meta={metadata!r})\n"
        f"def {name}(): pass\n"
    )
    digest = "aa" * 32
    key = base64.b32encode(bytes.fromhex(digest)).decode().rstrip("=")
    config = dict(
        kwargs={"XBLOCK": 1, "R0_BLOCK": 1024},
        num_warps=8,
        num_stages=1,
        num_ctas=1,
        maxnreg=None,
    )
    folder = tmp_path / "triton" / key
    folder.mkdir(parents=True)
    (folder / (name + ".cubin")).write_bytes(b"synthetic compiled bytes")
    (folder / (name + ".json")).write_text(
        json.dumps(
            {
                **{k: v for k, v in config.items() if k != "kwargs"},
                "hash": digest,
                "target": {"backend": "cuda", "arch": 120},
            }
        )
    )
    tuner = SimpleNamespace(
        filename=str(source),
        inductor_meta=metadata,
        deterministic_mode=True,
        _could_rblock_scale=False,
        launchers=[SimpleNamespace(cache_hash=key, config=SimpleNamespace(**config))],
    )
    module = SimpleNamespace(__file__=str(graph), call=lambda: None, **{name: tuner})
    snap = graph_snapshot([module], tmp_path)
    document = dict(
        status="complete",
        rank=0,
        cache_root=str(tmp_path),
        source_sha256="recorder",
        heuristic_source_sha256="compiler",
        compiler_options={"deterministic": True},
        snapshots={phase: copy.deepcopy(snap) for phase in ("before", "after")},
    )
    return document, dict(
        cache_root=tmp_path,
        rank=0,
        source_sha256="recorder",
        heuristic_sha256="compiler",
    )


def test_audit_uses_source_inventory_and_cubin_bytes(receipt):
    document, args = receipt
    rows = audit_receipt(document, **args)
    assert len(rows) == 1 and rows[0]["rmsnorm"]
    assert len(rows[0]["cubin_sha256"]) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        "rank",
        "compiler",
        "capture",
        "source",
        "coverage",
        "config",
        "binary",
        "target",
        "mode",
        "classification",
        "duplicate",
        "phase",
    ],
)
def test_audit_fails_closed(receipt, mutation):
    document, args = receipt
    snap = document["snapshots"]["after"]
    row = snap["bindings"][0]
    if mutation == "rank":
        document["rank"] = 1
    elif mutation == "compiler":
        document["heuristic_source_sha256"] = "other"
    elif mutation == "capture":
        document["status"] = "failed"
    elif mutation == "source":
        row["sha256"] = "other"
    elif mutation == "coverage":
        snap["bindings"] = []
    elif mutation == "config":
        row["selected"][0]["config"]["num_warps"] = 4
    elif mutation in ("binary", "target"):
        folder = args["cache_root"] / "triton" / row["selected"][0]["hash"]
        path = folder / (row["kernel"] + ".json")
        data = json.loads(path.read_text())
        if mutation == "binary":
            data["hash"] = "bb" * 32
        else:
            data["target"]["arch"] = 80
        path.write_text(json.dumps(data))
    elif mutation == "mode":
        row["runtime_deterministic"] = False
    elif mutation == "classification":
        row["reduction"] = False
    elif mutation == "duplicate":
        snap["bindings"].append(copy.deepcopy(row))
    else:
        del document["snapshots"]["before"]
    with pytest.raises(ValueError):
        audit_receipt(document, **args)


def test_inventory_includes_assignments_and_calls_not_comments():
    assert declared_symbols(
        "# triton_fake\ntriton_a = source()\ndef call(args):\n    triton_b.run(args)\n"
    ) == {"triton_a", "triton_b"}


def test_metadata_does_not_execute_generated_source():
    assert reduction_metadata(
        "@decorator(inductor_meta={'deterministic': True, "
        "'autotune_hints': forbidden()})\ndef kernel(): pass"
    ) == {"deterministic": True}
    with pytest.raises(ValueError):
        reduction_metadata(
            "@decorator(inductor_meta={'deterministic': forbidden()})\n"
            "def kernel(): pass"
        )


def test_vllm_redirected_cache_is_resolved_from_actual_source(tmp_path):
    namespace = tmp_path / "torch_compile_cache/torch_aot_compile/key"
    source = namespace / "inductor_cache/aa/kernel.py"
    assert binary_cache_directory(source, tmp_path) == namespace / "triton_cache"
    with pytest.raises(ValueError, match="unrecognized"):
        binary_cache_directory(tmp_path / "unknown/kernel.py", tmp_path)
    with pytest.raises(ValueError, match="outside"):
        binary_cache_directory(tmp_path.parent / "inductor/aa/kernel.py", tmp_path)
