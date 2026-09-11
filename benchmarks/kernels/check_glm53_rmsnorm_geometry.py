# SPDX-License-Identifier: Apache-2.0
"""Source-exact H4096 geometry qualification, not serving or performance evidence."""

import argparse
import base64
import json
import math
import os
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path

from benchmarks.kernels.check_glm53_attention_norms import (
    MODEL,
    load_checked,
    load_preserving_provenance,
    rank_cache,
    require,
    verify,
)
from benchmarks.kernels.check_glm53_cached_rmsnorm import (
    ROWS,
    SEEDS,
    SITES,
    compare,
    make_inputs,
    oracle,
    run_configuration,
    tensor_sha,
)
from benchmarks.kernels.glm53_rmsnorm_geometry import CONTROL, GEOMETRY, binary_sha
from slimserve.rmsnorm_diagnostic import sha, single_launcher_receipt

DISCOVERY_SHA = "d98fffd11b5f0f60880a3754ea6641201435aa9779bcfeb1bdb4ff95bd0d3a5f"
SCHEMA = "glm53-rmsnorm-geometry-probe-v1"
ARMS = ("control", "geometry")


def write_new(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(document, indent=2) + "\n")


def load_weights():
    import torch
    from safetensors import safe_open

    index = json.loads((MODEL / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    weights = []
    for layer, norm, _ in SITES:
        key = f"model.language_model.layers.{layer}.{norm}.weight"
        with safe_open(MODEL / index[key], framework="pt", device="cpu") as stream:
            weight = stream.get_tensor(key)
        require(
            weight.dtype == torch.bfloat16 and weight.shape == (4096,),
            "BF16 H4096 weight required",
        )
        weights.append(weight)
    return weights


def case_keys(targets):
    return list(product(range(len(targets)), ROWS, SEEDS, range(len(SITES))))


def validate_manifest(manifest):
    require(manifest["schema"] == SCHEMA, "qualified probe manifest required")
    require(
        manifest["rows"] == list(ROWS)
        and manifest["seeds"] == list(SEEDS)
        and manifest["sites"] == [list(site) for site in SITES]
        and manifest["changed_seed_offset"] == 100
        and manifest["max_bf16_ulp"] == 1,
        "fixed matrix or numerical contract changed",
    )
    require(
        len(manifest["targets"]) == 13 and manifest["expected_pairs"] == 312,
        "thirteen-source matrix required",
    )
    require(
        [t["rank"] for t in manifest["targets"]]
        == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 4,
        "rank order changed",
    )
    require(set(manifest["outputs"]) == {"a", "b"}, "two prescribed processes required")
    a, b = (Path(manifest["outputs"][key]) for key in ("a", "b"))
    require(
        a.is_absolute() and b.is_absolute() and a != b and a.parent == b.parent,
        "independent sibling outputs required",
    )
    for target in manifest["targets"]:
        require(
            {k: target["control"][k] for k in CONTROL} == CONTROL
            and target["geometry_config"] == GEOMETRY,
            "only the recorded geometry change is allowed",
        )


def prepare(discovery_path, manifest_path, series_root):
    require(
        not manifest_path.exists() and not series_root.exists(),
        "new manifest and series required",
    )
    discovery = load_checked(discovery_path, DISCOVERY_SHA)
    verify(discovery)
    weights = load_weights()
    sources = dict(discovery["sources"])
    for path in (Path(__file__), discovery_path):
        sources[str(path.resolve())] = sha(path)
    import torch._inductor.codecache as codecache
    import torch._inductor.runtime.static_triton_launcher as static_launcher
    import triton.language.standard as standard

    for module in (codecache, static_launcher, standard):
        sources[module.__file__] = sha(module.__file__)
    manifest = dict(
        schema=SCHEMA,
        discovery=str(discovery_path.resolve()),
        discovery_sha256=DISCOVERY_SHA,
        sources=sources,
        targets=discovery["targets"],
        original_namespace=discovery["original_namespace"],
        original_files=discovery["original_files"],
        rows=list(ROWS),
        seeds=list(SEEDS),
        sites=[list(site) for site in SITES],
        changed_seed_offset=100,
        max_bf16_ulp=1,
        expected_pairs=312,
        weight_sha256=[tensor_sha(w) for w in weights],
        outputs={arm: str(series_root.resolve() / arm) for arm in ("a", "b")},
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    )
    validate_manifest(manifest)
    verify(manifest)
    write_new(manifest_path, manifest)
    print(
        json.dumps(
            dict(manifest=str(manifest_path), sha256=sha(manifest_path), pairs=312)
        ),
        flush=True,
    )


def read_manifest(path):
    manifest = json.loads(path.read_text())
    validate_manifest(manifest)
    discovery = load_checked(Path(manifest["discovery"]), DISCOVERY_SHA)
    require(
        manifest["targets"] == discovery["targets"], "source target inventory changed"
    )
    verify(manifest)
    return manifest


def binary_files(output, target, selected):
    root = rank_cache(output, target["rank"]) / selected["hash"]
    # Triton cache keys are base32 encodings of the actual compiler metadata hash.
    require(
        len(selected["hash"]) == 52
        and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567" for c in selected["hash"]),
        "invalid cache key",
    )
    cubin, metadata_path = (
        root / (target["kernel"] + suffix) for suffix in (".cubin", ".json")
    )
    metadata = json.loads(metadata_path.read_text())
    encoded = base64.b32encode(bytes.fromhex(metadata["hash"])).decode().rstrip("=")
    require(
        encoded == selected["hash"]
        and metadata["name"] == target["kernel"]
        and metadata["num_warps"] == selected["config"]["num_warps"]
        and metadata["num_stages"] == selected["config"]["num_stages"],
        "binary metadata/config/key mismatch",
    )
    return dict(cubin_sha256=sha(cubin), metadata_sha256=sha(metadata_path))


def prior_a(manifest, manifest_path):
    output = Path(manifest["outputs"]["a"])
    path = output / "analysis.json"
    report = json.loads(path.read_text())
    require(
        report["status"] == "complete"
        and report["numerical_pass"] is True
        and report["summary_sha256"] == sha(output / "summary.json")
        and report["manifest_sha256"] == sha(manifest_path),
        "process A and its audit must pass before B",
    )
    summary = json.loads((output / "summary.json").read_text())
    require(
        analyze_summary(summary, manifest)["numerical_pass"],
        "process A numerical gate failed",
    )
    verify_binaries(manifest, output, summary)
    return dict(analysis_sha256=sha(path), summary_sha256=report["summary_sha256"])


def compile_recorded(template, config):
    """Observe the binary before make_launcher consumes static CUDA raw bytes."""
    import triton

    compiled = template._precompile_config(
        triton.Config(
            {k: config[k] for k in ("XBLOCK", "R0_BLOCK")},
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
    )
    in_memory_sha = binary_sha(compiled)
    return compiled.make_launcher(), in_memory_sha


def run(manifest_path, arm):
    manifest = read_manifest(manifest_path)
    output = Path(manifest["outputs"][arm])
    require(not output.exists(), "new output required; preserve every attempt")
    previous = prior_a(manifest, manifest_path) if arm == "b" else None
    require(
        not subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "GPU workload already active",
    )
    output.mkdir(parents=True)
    os.environ["TRITON_CACHE_DIR"] = str(output / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(output / "inductor")
    sys.dont_write_bytecode = True
    summary = dict(
        status="running",
        scope=__doc__,
        arm=arm,
        prior_a=previous,
        manifest_sha256=sha(manifest_path),
        git_commit=manifest["git_commit"],
        checks=[],
        binaries=[],
        pid=os.getpid(),
    )

    def save():
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    save()
    try:
        import torch
        import triton

        require(
            torch.cuda.device_count() == 4
            and all(torch.cuda.get_device_capability(i) == (12, 0) for i in range(4)),
            "four SM120 devices required",
        )
        summary.update(
            torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda
        )
        weights = load_weights()
        summary["weights"] = [tensor_sha(w) for w in weights]
        require(
            summary["weights"] == manifest["weight_sha256"],
            "checkpoint weights changed",
        )
        launches = {}
        # Finish all source/config/binary checks before numerical launches.
        for index, target in enumerate(manifest["targets"]):
            rank = target["rank"]
            with torch.cuda.device(rank), triton.knobs.cache.scope():
                triton.knobs.cache.dir = str(rank_cache(output, rank))
                copied = output / f"source-{index}" / Path(target["source"]).name
                copied.parent.mkdir()
                shutil.copyfile(target["source"], copied)
                template = load_preserving_provenance(
                    copied,
                    Path(target["source"]),
                    target["kernel"],
                    f"glm53_geometry_{index}",
                    debug_source=Path(target["debug_source"]),
                )
                launches[index] = []
                for label, config in (("control", CONTROL), ("geometry", GEOMETRY)):
                    launcher, in_memory_sha = compile_recorded(template, config)
                    selected = single_launcher_receipt(launcher)
                    require(selected["config"] == config, "compiled geometry changed")
                    files = binary_files(output, target, selected)
                    require(
                        files["cubin_sha256"] == in_memory_sha,
                        "in-memory/disk binary mismatch",
                    )
                    if label == "control":
                        require(
                            selected["hash"] == target["control"]["triton_cache_hash"]
                            and files["cubin_sha256"]
                            == target["control"]["cubin_sha256"],
                            "control binary differs from original serving image",
                        )
                    summary["binaries"].append(
                        dict(
                            source_index=index,
                            rank=rank,
                            arm=label,
                            source_sha256=target["source_sha256"],
                            selected=selected,
                            **files,
                        )
                    )
                    launches[index].append(launcher)
                    save()
        require(
            len(summary["binaries"]) == 26,
            "complete binary set required before numerics",
        )
        print(json.dumps(dict(binaries_verified=26)), flush=True)
        for index, rows, seed, site in case_keys(manifest["targets"]):
            target = manifest["targets"][index]
            _, _, magnitude = SITES[site]
            x, changed = (
                make_inputs(rows, value, magnitude) for value in (seed, seed + 100)
            )
            references = [oracle(data, weights[site]) for data in (x, changed)]
            with torch.cuda.device(target["rank"]):
                outputs = [
                    run_configuration(
                        launcher, target["layout"], x, changed, weights[site]
                    )
                    for launcher in launches[index]
                ]
            row = dict(
                source_index=index,
                rank=target["rank"],
                rows=rows,
                seed=seed,
                site=site,
                input_sha256=tensor_sha(x),
                changed_sha256=tensor_sha(changed),
                oracle_sha256=[tensor_sha(r) for r in references],
                repeat_graph_guards_mutation_pass=True,
                outputs={
                    label: [tensor_sha(outputs[i][phase]) for phase in (0, 2)]
                    for i, label in enumerate(ARMS)
                },
                oracle={
                    label: [
                        compare(outputs[i][phase], reference)
                        for phase, reference in zip((0, 2), references)
                    ]
                    for i, label in enumerate(ARMS)
                },
                cross_config=[
                    compare(outputs[0][phase], outputs[1][phase]) for phase in (0, 2)
                ],
            )
            row["passed"] = all(
                m["max_bf16_ulp"] <= 1
                for metrics in row["oracle"].values()
                for m in metrics
            )
            summary["checks"].append(row)
            save()
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "source_index",
                            "rank",
                            "rows",
                            "seed",
                            "site",
                            "passed",
                        )
                    }
                ),
                flush=True,
            )
        summary["status"] = (
            "complete" if all(r["passed"] for r in summary["checks"]) else "failed"
        )
    except BaseException as error:
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            verify(manifest)
            summary["frozen_sources_verified"] = True
        except BaseException as error:
            summary.update(
                status="failed",
                frozen_sources_verified=False,
                freeze_error=f"{type(error).__name__}: {error}",
            )
        save()
    return summary["status"] == "complete"


def check_digest(value):
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "SHA256 receipt required",
    )


def check_metric(metric, rows):
    require(metric["elements"] == rows * 4096, "wrong metric extent")
    for key in (
        "bit_mismatches",
        "numeric_mismatches",
        "affected_rows",
        "max_bf16_ulp",
    ):
        require(
            type(metric[key]) is int and metric[key] >= 0, "invalid count/ULP metric"
        )
    require(
        metric["numeric_mismatches"] <= metric["bit_mismatches"] <= metric["elements"]
        and metric["affected_rows"] <= rows,
        "invalid mismatch counts",
    )
    for key in ("max_abs", "mean_abs", "rms"):
        require(
            math.isfinite(metric[key]) and 0 <= metric[key] <= metric["max_abs"],
            "invalid error metric",
        )
    require(
        (metric["numeric_mismatches"] == 0)
        == (metric["max_bf16_ulp"] == metric["max_abs"] == 0)
        and (metric["bit_mismatches"] == 0) == (metric["affected_rows"] == 0),
        "metric equality/count disagreement",
    )


def analyze_summary(summary, manifest):
    require(summary["frozen_sources_verified"] is True, "source freeze failed")
    require(summary["weights"] == manifest["weight_sha256"], "weight receipts changed")
    keys = [
        (r["source_index"], r["rows"], r["seed"], r["site"]) for r in summary["checks"]
    ]
    require(keys == case_keys(manifest["targets"]), "matrix incomplete/reordered")
    maximum = dict.fromkeys(ARMS, 0)
    failed, mismatches = [], 0
    for row in summary["checks"]:
        require(
            row["rank"] == manifest["targets"][row["source_index"]]["rank"]
            and row["repeat_graph_guards_mutation_pass"] is True,
            "rank or execution gate failed",
        )
        for key in ("input_sha256", "changed_sha256"):
            check_digest(row[key])
        require(row["input_sha256"] != row["changed_sha256"], "vacuous changed input")
        require(
            len(row["oracle_sha256"]) == 2
            and set(row["oracle"]) == set(row["outputs"]) == set(ARMS),
            "missing input phase or arm",
        )
        for digest in row["oracle_sha256"]:
            check_digest(digest)
        passed = True
        for arm in ARMS:
            require(
                len(row["oracle"][arm]) == len(row["outputs"][arm]) == 2,
                "missing output phase",
            )
            require(
                row["outputs"][arm][0] != row["outputs"][arm][1],
                "vacuous output mutation",
            )
            for phase, metric in enumerate(row["oracle"][arm]):
                check_digest(row["outputs"][arm][phase])
                check_metric(metric, row["rows"])
                require(
                    (metric["bit_mismatches"] == 0)
                    == (row["outputs"][arm][phase] == row["oracle_sha256"][phase]),
                    "oracle hash/metric disagreement",
                )
                maximum[arm] = max(maximum[arm], metric["max_bf16_ulp"])
                passed &= metric["max_bf16_ulp"] <= 1
        require(passed == row["passed"], "incorrect numerical verdict")
        if not passed:
            failed.append(
                {k: row[k] for k in ("source_index", "rank", "rows", "seed", "site")}
            )
        require(len(row["cross_config"]) == 2, "missing paired phase")
        for phase, metric in enumerate(row["cross_config"]):
            check_metric(metric, row["rows"])
            require(
                (metric["bit_mismatches"] == 0)
                == (
                    row["outputs"]["control"][phase]
                    == row["outputs"]["geometry"][phase]
                ),
                "paired hash/metric disagreement",
            )
            mismatches += metric["bit_mismatches"]
    require(
        summary["status"] == ("failed" if failed else "complete"),
        "wrong terminal verdict",
    )
    return dict(
        numerical_pass=not failed,
        pairs=len(keys),
        oracle_max_bf16_ulp=maximum,
        pairwise_bit_mismatches=mismatches,
        failed_cases=failed,
    )


def verify_binaries(manifest, output, summary):
    require(len(summary["binaries"]) == 26, "incomplete binary set")
    for index, source in enumerate(manifest["targets"]):
        copied = output / f"source-{index}" / Path(source["source"]).name
        require(sha(copied) == source["source_sha256"], "copied source changed")
        for offset, label in enumerate(ARMS):
            binary = summary["binaries"][2 * index + offset]
            selected = binary["selected"]
            require(
                binary["source_index"] == index
                and binary["rank"] == source["rank"]
                and binary["arm"] == label
                and binary["source_sha256"] == source["source_sha256"],
                "binary/source binding changed",
            )
            require(
                selected["config"] == (CONTROL if label == "control" else GEOMETRY),
                "binary configuration changed",
            )
            files = binary_files(output, source, selected)
            require(
                all(binary[k] == value for k, value in files.items()),
                "binary bytes changed",
            )
            if label == "control":
                require(
                    selected["hash"] == source["control"]["triton_cache_hash"]
                    and binary["cubin_sha256"] == source["control"]["cubin_sha256"],
                    "control no longer matches original",
                )


def audit(manifest_path, arm):
    manifest = read_manifest(manifest_path)
    output = Path(manifest["outputs"][arm])
    target_path = output / "analysis.json"
    require(not target_path.exists(), "preserve prior audit")
    summary = json.loads((output / "summary.json").read_text())
    require(
        summary["manifest_sha256"] == sha(manifest_path) and summary["arm"] == arm,
        "manifest or process arm changed",
    )
    require(
        summary["prior_a"]
        == (prior_a(manifest, manifest_path) if arm == "b" else None),
        "prior-process gate changed",
    )
    require(summary["git_commit"] == manifest["git_commit"], "wrong source commit")
    verify_binaries(manifest, output, summary)
    analysis = analyze_summary(summary, manifest)
    report = dict(
        status="complete",
        scope=__doc__,
        manifest_sha256=sha(manifest_path),
        summary_sha256=sha(output / "summary.json"),
        binaries_verified=26,
        sources_verified=len(manifest["sources"]),
        original_files_verified=len(manifest["original_files"]),
        **analysis,
    )
    write_new(target_path, report)
    print(json.dumps(report), flush=True)
    return report["numerical_pass"]


def compare_processes(a, b):
    for key in (
        "weights",
        "checks",
        "binaries",
        "torch",
        "triton",
        "cuda",
        "git_commit",
        "manifest_sha256",
    ):
        require(a[key] == b[key], f"independent processes differ: {key}")


def pair_audit(manifest_path):
    manifest = read_manifest(manifest_path)
    records, receipts = [], []
    for arm in ("a", "b"):
        output = Path(manifest["outputs"][arm])
        report = json.loads((output / "analysis.json").read_text())
        summary_path = output / "summary.json"
        require(
            report["status"] == "complete"
            and report["numerical_pass"] is True
            and report["summary_sha256"] == sha(summary_path)
            and report["manifest_sha256"] == sha(manifest_path),
            "both processes and audits must pass",
        )
        summary = json.loads(summary_path.read_text())
        require(
            analyze_summary(summary, manifest)["numerical_pass"],
            "pair numerical gate failed",
        )
        verify_binaries(manifest, output, summary)
        require(
            summary["arm"] == arm
            and summary["prior_a"]
            == (prior_a(manifest, manifest_path) if arm == "b" else None),
            "process order receipts changed",
        )
        records.append(summary)
        receipts.append(
            dict(
                arm=arm,
                summary_sha256=sha(summary_path),
                analysis_sha256=sha(output / "analysis.json"),
            )
        )
    compare_processes(*records)
    report = dict(
        status="complete",
        scope=__doc__,
        numerical_pass=True,
        cross_process_exact=True,
        pairs=624,
        matched_pairs=312,
        manifest_sha256=sha(manifest_path),
        receipts=receipts,
    )
    write_new(Path(manifest["outputs"]["a"]).parent / "pair-analysis.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit", "compare"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--discovery", type=Path)
    parser.add_argument("--series-root", type=Path)
    parser.add_argument("--arm", choices=("a", "b"))
    args = parser.parse_args()
    if args.action == "prepare":
        if args.discovery is None or args.series_root is None:
            parser.error("prepare needs discovery and a fresh series root")
        prepare(args.discovery, args.manifest, args.series_root)
    elif args.action == "compare":
        pair_audit(args.manifest)
    else:
        if args.arm is None:
            parser.error("run/audit needs the prescribed process arm")
        passed = (run if args.action == "run" else audit)(args.manifest, args.arm)
        if not passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
