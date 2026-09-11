#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Qualify Inductor's deterministic runtime policy on exact GLM53 sources.

This changes only decorator metadata in an isolated probe, not generated source
bytes or serving code. It exercises normal autotuner.run, forbids benchmarking,
and requires an already-qualified binary. It does NOT test frontend codegen,
fresh full-model compilation, model quality or serving performance.
"""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from benchmarks.kernels.check_glm53_cached_rmsnorm import (
    compare,
    make_inputs,
    oracle,
    run_configuration,
    sha,
    source_layout,
    tensor_sha,
)
from benchmarks.kernels.check_glm53_rmsnorm_intervention import load_source
from slimserve.rmsnorm_diagnostic import AUDIT_SHA, single_launcher_receipt

ROWS = (1, 16, 640, 7616)
WEIGHT = "model.language_model.layers.22.post_attention_layernorm.weight"


def policy_metadata(metadata):
    """Only the supported deterministic option changes; never mutate callers."""
    if metadata.get("deterministic") is not False:
        raise ValueError("requires recorded nondeterministic source metadata")
    return {**copy.deepcopy(metadata), "deterministic": True}


def forbid_benchmark(*args, **kwargs):
    raise ValueError("deterministic reduction attempted GPU benchmarking")


def recorded_choice(config, saved):
    actual = dict(
        **config.kwargs, num_warps=config.num_warps, num_stages=config.num_stages
    )
    matches = [
        row
        for row in saved
        if actual
        == {key: row[key] for key in ("XBLOCK", "R0_BLOCK", "num_warps", "num_stages")}
    ]
    if len(matches) != 1:
        raise ValueError(f"policy selected an unqualified configuration: {actual}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-audit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sha(args.audit) != AUDIT_SHA:
        parser.error("requires new output and the exact cache-comparison audit")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        parser.error("GPUs are already in use")
    args.output.mkdir(parents=True)
    os.environ["TRITON_CACHE_DIR"] = str(args.output.resolve() / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(args.output.resolve() / "inductor")
    sys.dont_write_bytecode = True
    manifest = json.loads(args.manifest.read_text())
    original = Path(manifest["original_namespace"])
    snapshot = manifest["original_files"]
    root = Path(__file__).resolve().parents[2]
    if sha(args.native_audit) != (
        "b8a3f1223798174e1f1a4ca691b60e2b2b75357f4f7fa027cd0e073b343fe415"
    ):
        raise ValueError("requires the completed graph-control native receipt")
    native = json.loads(args.native_audit.read_text())["native_sha256"]
    source_receipts = {
        name: sha(root / name)
        for name in (
            "benchmarks/kernels/check_glm53_deterministic_reductions.py",
            "benchmarks/kernels/check_glm53_cached_rmsnorm.py",
            "benchmarks/kernels/check_glm53_rmsnorm_intervention.py",
            "slimserve/rmsnorm_diagnostic.py",
            "slimserve/glm53_ordering.py",
        )
    }

    def verify_original():
        if {
            str(p.relative_to(original)) for p in original.rglob("*") if p.is_file()
        } != set(snapshot) or any(
            sha(original / p) != digest for p, digest in snapshot.items()
        ):
            raise ValueError("original cache changed")
        for name, digest in (native | source_receipts).items():
            if sha(root / name) != digest:
                raise ValueError(f"native/helper source changed: {name}")

    summary = dict(
        status="running",
        source_sha256=sha(__file__),
        audit_sha256=sha(args.audit),
        manifest_sha256=sha(args.manifest),
        native_audit_sha256=sha(args.native_audit),
        native_sha256=native,
        implementation_sha256=source_receipts,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        rows=ROWS,
        weight=WEIGHT,
        sources=[],
        checks=[],
        limitation=__doc__,
    )
    output = args.output / "summary.json"

    def save():
        output.write_text(json.dumps(summary, indent=2) + "\n")

    save()
    try:
        verify_original()
        import torch
        import triton
        from safetensors import safe_open
        from torch._inductor.runtime import triton_heuristics as heuristics

        summary.update(
            torch=torch.__version__,
            triton=triton.__version__,
            cuda=torch.version.cuda,
            heuristic_source_sha256=sha(heuristics.__file__),
        )
        if torch.cuda.device_count() != 4 or any(
            torch.cuda.get_device_capability(i) != (12, 0) for i in range(4)
        ):
            raise ValueError("requires four rank-matched SM120 devices")
        index = args.model / "model.safetensors.index.json"
        weight_map = json.loads(index.read_text())["weight_map"]
        with safe_open(
            args.model / weight_map[WEIGHT], framework="pt", device="cpu"
        ) as f:
            weight = f.get_tensor(WEIGHT)
        if weight.dtype != torch.bfloat16 or weight.shape != (4096,):
            raise ValueError("expected checkpoint BF16 H4096 norm vector")
        summary.update(weight_sha256=tensor_sha(weight), weight_index_sha256=sha(index))
        audit = json.loads(args.audit.read_text())
        if len(audit["reduction_width_changes"]) != 4:
            raise ValueError("four source-bound changes required")
        for entry in audit["reduction_width_changes"]:
            (source,) = entry["sources"]
            (rank_text,) = source["device_indices"]
            rank = int(rank_text)
            (kernel,) = source["kernels"]
            source_path = original / "inductor_cache" / source["relative"]
            if sha(source_path) != source["source_sha256"]:
                raise ValueError("source no longer matches audit")
            folder = args.output / f"rank-{rank}"
            folder.mkdir()
            copied = folder / source_path.name
            shutil.copyfile(source_path, copied)
            original_reduction = heuristics.reduction
            calls = []

            def deterministic_reduction(
                *positional,
                calls=calls,
                original_reduction=original_reduction,
                **keywords,
            ):
                metadata = keywords["inductor_meta"]
                keywords["inductor_meta"] = policy_metadata(metadata)
                calls.append(dict(before=metadata["deterministic"], after=True))
                return original_reduction(*positional, **keywords)

            with torch.cuda.device(rank):
                # Scoped import-only metadata substitution in this isolated process.
                heuristics.reduction = deterministic_reduction
                try:
                    template = load_source(copied, kernel, f"fixed_norm_{rank}")
                finally:
                    heuristics.reduction = original_reduction
                if calls != [dict(before=False, after=True)]:
                    raise ValueError("unexpected source decorator coverage")
                if not template.deterministic_mode or len(template.configs) != 1:
                    raise ValueError("policy did not select exactly one configuration")
                if template._could_rblock_scale or template._should_coordesc_tune:
                    raise ValueError("reduction retuning remains enabled")
                saved = recorded_choice(template.configs[0], entry["configs"])
                template.bench = forbid_benchmark
                template.benchmark_all_configs = forbid_benchmark
                layout = source_layout(copied.read_text(), kernel)

                def launch(*call_args, stream, template=template):
                    template.run(*call_args, stream=stream)

                record = dict(
                    rank=rank,
                    source_sha256=sha(copied),
                    layout=layout,
                    expected_hash=saved["triton_cache_hash"],
                    deterministic=True,
                    candidate_configs=1,
                    dynamic_rblock=False,
                    coordinate_descent=False,
                )
                summary["sources"].append(record)
                for rows in ROWS:
                    x = make_inputs(rows, 530901, 1.0)
                    changed = make_inputs(rows, 531001, 1.0)
                    actual = run_configuration(launch, layout, x, changed, weight)
                    (launcher,) = template.launchers
                    receipt = single_launcher_receipt(launcher)
                    if receipt["hash"] != saved["triton_cache_hash"]:
                        raise ValueError("automatic policy emitted a different binary")
                    record["selected"] = receipt
                    metrics = [
                        compare(actual[i], oracle(data, weight))
                        for i, data in ((0, x), (2, changed))
                    ]
                    if any(m["max_bf16_ulp"] > 1 for m in metrics):
                        raise ValueError("predeclared one-BF16-ULP FP64 gate failed")
                    check = dict(
                        rank=rank,
                        rows=rows,
                        oracle=metrics,
                        input_sha256=tensor_sha(x),
                        changed_sha256=tensor_sha(changed),
                        output_sha256=[tensor_sha(t) for t in actual],
                        repeat_graph_guards_mutation_pass=True,
                        selected=receipt,
                    )
                    summary["checks"].append(check)
                    save()
                    print(
                        json.dumps(
                            dict(
                                rank=rank,
                                rows=rows,
                                status="pass",
                                config=receipt["config"],
                            )
                        ),
                        flush=True,
                    )
        verify_original()
        summary.update(status="complete", original_cache_files_unchanged=len(snapshot))
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
