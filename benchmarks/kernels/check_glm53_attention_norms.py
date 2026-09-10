#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare saved GLM53 attention combo/split kernels, never serving caches.

Synthetic packed activations and real layer11 weights; source-exact recorded
launches, not frontend, TP execution, performance, or model-causality proof.
Historical combo coverage is from static-future callbacks, NOT a complete
graph-held binding inventory. The new split sources have graph-held receipts.
"""

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
from itertools import product
from pathlib import Path

from benchmarks.analyze_glm53_reduction_receipts import sha
from benchmarks.kernels.check_glm53_cached_rmsnorm import compare, oracle, tensor_sha
from benchmarks.kernels.check_glm53_rmsnorm_intervention import load_source
from slimserve.rmsnorm_diagnostic import single_launcher_receipt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "perf/results"
NEW_AUDIT = RESULTS / "2026-09-10/runtime-control/no-combo-first-workload-analysis.json"
OLD_AUDIT = (
    RESULTS
    / "2026-09-09/runtime-control/rmsnorm-noop-complete-graph-control-analysis.json"
)
OLD_MANIFEST = (
    RESULTS / "2026-09-09/rmsnorm-complete-graph-serving-caches/control/manifest.json"
)
MODEL = Path("/raid/weights/GLM-5.3-Flash-NVFP4-FP8-KDA-TP4")
ROWS = (1, 3, 16, 640, 7616)
SEEDS = (530901, 530902)
MAGNITUDES = (0.125, 1.0, 8.0)
LAYOUTS = {512: (1536, 512), 1536: (0, 1536), 128: (2048, 256)}
WEIGHTS = (
    "kv_a_layernorm.weight",
    "q_a_layernorm.weight",
    "indexer.k_norm.weight",
    "indexer.k_norm.bias",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_info(text):
    """Read logical extents from the body, never from rounded size hints."""
    functions = [
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("triton_")
    ]
    require(len(functions) == 1, "one generated kernel required")
    function = functions[0]
    widths = [
        ast.literal_eval(n.value)
        for n in ast.walk(function)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "r0_numel" for t in n.targets)
    ]
    hints = [
        ast.literal_eval(n.value)
        for n in ast.walk(function)
        if isinstance(n, ast.keyword) and n.arg == "size_hints"
    ]
    require(len(hints) == 1, "one size hint required")
    body = ast.Module(body=function.body, type_ignores=[])
    body_text = ast.unparse(body)
    return dict(
        kernel=function.name,
        widths=widths,
        size_hints=hints[0],
        args=[a.arg for a in function.args.args],
        body_sha256=hashlib.sha256(ast.dump(body).encode()).hexdigest(),
        bf16_intermediate_casts=body_text.count(".to(tl.bfloat16)"),
    )


def flat_config(config):
    return {
        **config["kwargs"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }


def load_checked(path, digest):
    require(sha(path) == digest, f"receipt changed: {path}")
    return json.loads(path.read_text())


def load_weights():
    import torch
    from safetensors import safe_open

    index = json.loads((MODEL / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    weights = []
    for suffix, width in zip(WEIGHTS, (512, 1536, 128, 128)):
        key = "model.language_model.layers.11.self_attn." + suffix
        with safe_open(MODEL / index[key], framework="pt", device="cpu") as f:
            weight = f.get_tensor(key)
        require(
            weight.dtype == torch.bfloat16 and weight.shape == (width,),
            "checkpoint weight shape/dtype mismatch",
        )
        weights.append(weight)
    return weights


def prepare(path):
    require(not path.exists(), "new manifest required")
    new = load_checked(
        NEW_AUDIT, "eb333fb93c2ac28218d46ca7799d95ce2378f73ebc0ccfe5d49138b6e2530c87"
    )
    old = load_checked(
        OLD_AUDIT, "b8a3f1223798174e1f1a4ca691b60e2b2b75357f4f7fa027cd0e073b343fe415"
    )
    original = json.loads(OLD_MANIFEST.read_text())
    original_root = Path(original["original_namespace"])
    records = []
    # Include every recorded callback source for the broader 4096 comparison.
    inventory = []
    seen = set()
    for row in old["launchers"]:
        key = (row["rank"], row["filename"])
        if key in seen:
            continue
        seen.add(key)
        source = original_root / row["filename"]
        require(
            sha(source) == original["original_files"][row["filename"]],
            "old source changed",
        )
        info = source_info(source.read_text())
        (selected,) = row["before"]
        entry = dict(
            rank=row["rank"],
            arm="combo",
            source=str(source),
            source_sha256=sha(source),
            info=info,
            selected=selected,
        )
        inventory.append(entry)
        if info["widths"] == [512, 1536, 128]:
            require(info["bf16_intermediate_casts"] == 0, "unexpected combo rounding")
            # This old cache's exact rank-local directory is in its frozen manifest.
            # Never search other caches for a binary with a convenient name.
            relative = (
                f"inductor_cache/triton/{row['rank']}/{selected['hash']}/"
                f"{info['kernel']}.cubin"
            )
            entry["cubin_sha256"] = original["original_files"][relative]
            require(
                sha(original_root / relative) == entry["cubin_sha256"],
                "old combo binary changed",
            )
            records.append(entry)
    seen = set()
    for row in new["reductions"]:
        key = (row["rank"], row["filename"])
        if key in seen:
            continue
        seen.add(key)
        source = Path(row["filename"])
        require(sha(source) == row["source_sha256"], "new source changed")
        info = source_info(source.read_text())
        entry = dict(
            rank=row["rank"],
            arm="split",
            source=str(source),
            source_sha256=sha(source),
            info=info,
            selected=dict(
                hash=row["selected"]["hash"],
                config=flat_config(row["selected"]["config"]),
            ),
            cubin_sha256=row["cubin_sha256"],
        )
        inventory.append(entry)
        if len(info["widths"]) == 1 and info["widths"][0] in LAYOUTS:
            require(info["bf16_intermediate_casts"] == 0, "unexpected split rounding")
            records.append(entry)
    for rank in range(4):
        combos = [r for r in records if r["rank"] == rank and r["arm"] == "combo"]
        splits = [r for r in records if r["rank"] == rank and r["arm"] == "split"]
        require(
            len(combos) == 1 and len(splits) == 3,
            "one combo/three splits per rank required",
        )
        require(
            {r["info"]["widths"][0] for r in splits} == set(LAYOUTS), "missing shape"
        )
    sources = {
        str(ROOT / name): sha(ROOT / name)
        for name in (
            "benchmarks/kernels/check_glm53_attention_norms.py",
            "benchmarks/analyze_glm53_reduction_receipts.py",
            "benchmarks/analyze_glm53_quality_pair.py",
            "benchmarks/kernels/check_glm53_cached_rmsnorm.py",
            "benchmarks/kernels/check_glm53_rmsnorm_intervention.py",
            "slimserve/rmsnorm_diagnostic.py",
            "vllm/model_executor/models/glm5_next.py",
            "vllm/model_executor/layers/mla.py",
            "vllm/model_executor/layers/glm5_next_indexer.py",
            "vllm/ir/ops/layernorm.py",
        )
    }
    for receipt in (
        NEW_AUDIT,
        OLD_AUDIT,
        OLD_MANIFEST,
        MODEL / "config.json",
        MODEL / "model.safetensors.index.json",
    ):
        sources[str(receipt)] = sha(receipt)
    for row in inventory:
        sources[row["source"]] = row["source_sha256"]
    sources.update(
        {str(ROOT / name): digest for name, digest in new["native_sha256"].items()}
    )
    import torch._inductor.codegen.triton as codegen
    import torch._inductor.config as config
    import torch._inductor.runtime.triton_heuristics as heuristics
    import triton.compiler.compiler as compiler

    for module in (codegen, config, heuristics, compiler):
        sources[module.__file__] = sha(module.__file__)
    matches = []
    for current in inventory:
        if current["arm"] != "split" or current["info"]["widths"] != [4096]:
            continue
        for previous in inventory:
            if (
                previous["arm"] == "combo"
                and previous["rank"] == current["rank"]
                and previous["info"]["body_sha256"] == current["info"]["body_sha256"]
            ):
                matches.append(
                    dict(
                        rank=current["rank"],
                        old_source=previous["source"],
                        new_source=current["source"],
                        old_config=previous["selected"]["config"],
                        new_config=current["selected"]["config"],
                        body_sha256=current["info"]["body_sha256"],
                    )
                )
    manifest = dict(
        schema=1,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        sources=sources,
        records=records,
        inventory=inventory,
        identical_4096_body_matches=matches,
        weight_sha256={k: tensor_sha(w) for k, w in zip(WEIGHTS, load_weights())},
        original_namespace=str(original_root),
        original_files=original["original_files"],
        rows=ROWS,
        seeds=SEEDS,
        magnitudes=MAGNITUDES,
        weights=WEIGHTS,
        expected_pairs=4 * len(ROWS) * len(SEEDS) * len(MAGNITUDES),
        gates=(
            "exact repeat/replay, intact input/weight/output guards, "
            "FP64 <=1 BF16 ULP for each norm"
        ),
        scope=__doc__,
    )
    verify(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                manifest=str(path),
                sha256=sha(path),
                sources=len(records),
                pairs=manifest["expected_pairs"],
            )
        )
    )


def verify(manifest):
    for name, digest in manifest["sources"].items():
        require(sha(name) == digest, f"frozen source/native/receipt changed: {name}")
    root = Path(manifest["original_namespace"])
    require(
        {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
        == set(manifest["original_files"]),
        "original cache inventory changed",
    )
    for name, digest in manifest["original_files"].items():
        require(sha(root / name) == digest, f"original cache changed: {name}")


def layernorm_oracle(x, weight, bias):
    import torch

    out = torch.empty_like(x)
    for source, target in zip(x.split(128), out.split(128)):
        centered = source.double() - source.double().mean(-1, keepdim=True)
        normalized = centered * torch.rsqrt(
            centered.square().mean(-1, keepdim=True) + 1e-6
        )
        target.copy_((normalized * weight.double() + bias.double()).bfloat16())
    return out


def packed_inputs(rows, seed, magnitude):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(rows, 2336, generator=generator) * magnitude).bfloat16()


def run_launches(launches, data, changed, weights):
    import torch

    rows = len(data)
    storage = torch.full((rows + 2, 2336), 123.0, dtype=torch.bfloat16, device="cuda")
    x = storage[1:-1]
    wg = [w.cuda() for w in weights]
    buffers = [
        torch.full((rows + 2, stride), 123.0, dtype=torch.bfloat16, device="cuda")
        for _, stride in LAYOUTS.values()
    ]
    outputs = [b[1:-1, :width] for b, width in zip(buffers, LAYOUTS)]

    def launch():
        stream = torch.cuda.current_stream().cuda_stream
        if "combo" in launches:
            launches["combo"](x, *wg, *outputs, rows, rows, rows, stream=stream)
        else:
            launches[512](x, wg[0], outputs[0], rows, 512, stream=stream)
            launches[1536](x, wg[1], outputs[1], rows, 1536, stream=stream)
            launches[128](x, wg[2], wg[3], outputs[2], rows, 128, stream=stream)

    def check_inputs(expected):
        require(torch.equal(x.cpu(), expected), "packed input mutated")
        require(torch.all(storage[[0, -1]] == 123).item(), "input guard mutated")
        require(
            all(torch.equal(w.cpu(), ref) for w, ref in zip(wg, weights)),
            "weight mutated",
        )
        for buffer, width in zip(buffers, LAYOUTS):
            require(
                torch.all(buffer[[0, -1]] == 123).item(), "output row guard mutated"
            )
            require(
                torch.all(buffer[1:-1, width:] == 123).item(),
                "output stride gap mutated",
            )

    x.copy_(data)
    launch()
    first = [t.cpu() for t in outputs]
    launch()
    require(
        all(torch.equal(t.cpu(), ref) for t, ref in zip(outputs, first)),
        "eager repeat mismatch",
    )
    check_inputs(data)
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream(device=x.device)
    with torch.cuda.graph(graph, stream=capture_stream):
        launch()
    x.copy_(changed)
    graph.replay()
    second = [t.cpu() for t in outputs]
    check_inputs(changed)
    launch()
    require(
        all(torch.equal(t.cpu(), ref) for t, ref in zip(outputs, second)),
        "changed eager/replay mismatch",
    )
    x.copy_(data)
    graph.replay()
    require(
        all(torch.equal(t.cpu(), ref) for t, ref in zip(outputs, first)),
        "original replay mismatch",
    )
    require(
        all(not torch.equal(a, b) for a, b in zip(first, second)),
        "vacuous changed-input check",
    )
    check_inputs(data)
    return first, second


def run(manifest_path, output):
    require(not output.exists(), "new output required; preserve all attempts")
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    require(
        not subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "GPU workload already active",
    )
    output.mkdir(parents=True)
    os.environ["TRITON_CACHE_DIR"] = str(output.resolve() / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(output.resolve() / "inductor")
    summary = dict(
        status="running",
        manifest_sha256=sha(manifest_path),
        checks=[],
        binaries=[],
        scope=__doc__,
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
        summary["weights"] = {k: tensor_sha(w) for k, w in zip(WEIGHTS, weights)}
        require(summary["weights"] == manifest["weight_sha256"], "weights changed")
        for rank in range(4):
            with torch.cuda.device(rank):
                arms = {"combo": {}, "split": {}}
                for n, record in enumerate(manifest["records"]):
                    if record["rank"] != rank:
                        continue
                    copied = output / f"source-{n}" / Path(record["source"]).name
                    copied.parent.mkdir()
                    shutil.copyfile(record["source"], copied)
                    template = load_source(
                        copied, record["info"]["kernel"], f"attention_norm_{n}"
                    )
                    saved = record["selected"]["config"]
                    config = triton.Config(
                        {
                            k: v
                            for k, v in saved.items()
                            if k not in ("num_warps", "num_stages")
                        },
                        num_warps=saved["num_warps"],
                        num_stages=saved["num_stages"],
                    )
                    # Recorded configuration only. No timing, autotune, or substitution.
                    launcher = template._precompile_config(config).make_launcher()
                    actual = single_launcher_receipt(launcher)
                    require(
                        actual == record["selected"],
                        "source/config did not reproduce recorded binary key",
                    )
                    cubin = (
                        Path(os.environ["TRITON_CACHE_DIR"])
                        / actual["hash"]
                        / (record["info"]["kernel"] + ".cubin")
                    )
                    require(
                        sha(cubin) == record["cubin_sha256"],
                        "cubin bytes differ from serving",
                    )
                    summary["binaries"].append(
                        dict(
                            rank=rank,
                            arm=record["arm"],
                            source_sha256=sha(copied),
                            selected=actual,
                            cubin_sha256=sha(cubin),
                        )
                    )
                    key = (
                        "combo"
                        if record["arm"] == "combo"
                        else record["info"]["widths"][0]
                    )
                    arms[record["arm"]][key] = launcher
                for rows in ROWS:
                    for seed in SEEDS:
                        for magnitude in MAGNITUDES:
                            original = packed_inputs(rows, seed, magnitude)
                            changed = packed_inputs(rows, seed + 100, magnitude)
                            references = []
                            for data in (original, changed):
                                references.append(
                                    [
                                        oracle(data[:, 1536:2048], weights[0]),
                                        oracle(data[:, :1536], weights[1]),
                                        layernorm_oracle(
                                            data[:, 2048:2176], weights[2], weights[3]
                                        ),
                                    ]
                                )
                            values = {
                                arm: run_launches(launches, original, changed, weights)
                                for arm, launches in arms.items()
                            }
                            metrics = {
                                arm: [
                                    [compare(t, ref) for t, ref in zip(phase, refs)]
                                    for phase, refs in zip(value, references)
                                ]
                                for arm, value in values.items()
                            }
                            pair = [
                                [compare(a, b) for a, b in zip(aa, bb)]
                                for aa, bb in zip(values["combo"], values["split"])
                            ]
                            passed = all(
                                m["max_bf16_ulp"] <= 1
                                for arm in metrics.values()
                                for phase in arm
                                for m in phase
                            )
                            summary["checks"].append(
                                dict(
                                    rank=rank,
                                    rows=rows,
                                    seed=seed,
                                    magnitude=magnitude,
                                    passed=passed,
                                    oracle=metrics,
                                    combo_vs_split=pair,
                                    repeat_graph_guards_mutation_pass=True,
                                    inputs=[tensor_sha(t) for t in (original, changed)],
                                    outputs={
                                        arm: [
                                            [tensor_sha(t) for t in phase]
                                            for phase in value
                                        ]
                                        for arm, value in values.items()
                                    },
                                )
                            )
                            save()
                print(
                    json.dumps(dict(rank=rank, completed_pairs=len(summary["checks"]))),
                    flush=True,
                )
        require(
            len(summary["checks"]) == manifest["expected_pairs"], "incomplete matrix"
        )
        verify(manifest)
        require(
            all(r["passed"] for r in summary["checks"]),
            "one-BF16-ULP FP64 gate failed; full matrix retained",
        )
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        save()


def audit(manifest_path, output):
    """Validate stored receipts/metrics after exit; no generated code or GPU use."""
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    path = output / "summary.json"
    summary = json.loads(path.read_text())
    require(summary["manifest_sha256"] == sha(manifest_path), "manifest changed")
    require(summary["weights"] == manifest["weight_sha256"], "weight receipt changed")
    require(len(summary["binaries"]) == 16, "incomplete binary matrix")
    expected_binaries = [
        dict(
            rank=r["rank"],
            arm=r["arm"],
            source_sha256=r["source_sha256"],
            selected=r["selected"],
            cubin_sha256=r["cubin_sha256"],
        )
        for rank in range(4)
        for r in manifest["records"]
        if r["rank"] == rank
    ]
    require(summary["binaries"] == expected_binaries, "binary receipts changed")
    for i, record in enumerate(manifest["records"]):
        require(
            sha(output / f"source-{i}" / Path(record["source"]).name)
            == record["source_sha256"],
            "copied source changed",
        )
        cubin = (
            output
            / "triton"
            / record["selected"]["hash"]
            / (record["info"]["kernel"] + ".cubin")
        )
        require(sha(cubin) == record["cubin_sha256"], "probe cubin changed")
    keys = [
        (r["rank"], r["rows"], r["seed"], r["magnitude"]) for r in summary["checks"]
    ]
    require(
        keys == list(product(range(4), ROWS, SEEDS, MAGNITUDES)),
        "matrix incomplete/reordered",
    )
    maximum = {arm: {width: 0 for width in LAYOUTS} for arm in ("combo", "split")}
    mismatches = {width: 0 for width in LAYOUTS}
    failed = []
    for row in summary["checks"]:
        require(row["repeat_graph_guards_mutation_pass"], "repeat/guard gate failed")
        passed = True
        for arm in ("combo", "split"):
            require(
                len(row["oracle"][arm]) == len(row["outputs"][arm]) == 2,
                "missing input phase",
            )
            for phase in range(2):
                require(
                    len(row["oracle"][arm][phase])
                    == len(row["outputs"][arm][phase])
                    == 3,
                    "missing norm output",
                )
                for i, width in enumerate(LAYOUTS):
                    metric = row["oracle"][arm][phase][i]
                    require(
                        metric["elements"] == row["rows"] * width,
                        "incorrect metric extent",
                    )
                    maximum[arm][width] = max(
                        maximum[arm][width], metric["max_bf16_ulp"]
                    )
                    passed &= metric["max_bf16_ulp"] <= 1
        require(passed == row["passed"], "incorrect numerical verdict")
        if not passed:
            failed.append({k: row[k] for k in ("rank", "rows", "seed", "magnitude")})
        require(len(row["combo_vs_split"]) == 2, "missing paired phase")
        for phase in range(2):
            require(len(row["combo_vs_split"][phase]) == 3, "missing paired output")
            for i, width in enumerate(LAYOUTS):
                metric = row["combo_vs_split"][phase][i]
                mismatch = metric["bit_mismatches"]
                require(
                    metric["elements"] == row["rows"] * width, "incorrect paired extent"
                )
                require(
                    (mismatch == 0)
                    == (
                        row["outputs"]["combo"][phase][i]
                        == row["outputs"]["split"][phase][i]
                    ),
                    "paired hash/metric disagreement",
                )
                mismatches[width] += mismatch
    require(
        summary["status"] == ("failed" if failed else "complete"),
        "wrong terminal verdict",
    )
    report = dict(
        status="complete",
        numerical_pass=not failed,
        summary_sha256=sha(path),
        manifest_sha256=sha(manifest_path),
        pairs=len(keys),
        original_files_verified=len(manifest["original_files"]),
        source_receipts_verified=len(manifest["sources"]),
        binaries_verified=16,
        oracle_max_bf16_ulp=maximum,
        pairwise_bit_mismatches=mismatches,
        failed_cases=failed,
        scope=__doc__,
    )
    target = output / "analysis.json"
    require(not target.exists(), "preserve previous audit")
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--audit", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.prepare:
        require(args.output is None, "preparation does not run a probe")
        prepare(args.manifest)
    else:
        require(args.output is not None, "probe output required")
        if args.audit:
            audit(args.manifest, args.output)
        else:
            run(args.manifest, args.output)


if __name__ == "__main__":
    main()
