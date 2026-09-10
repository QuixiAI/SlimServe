# SPDX-License-Identifier: Apache-2.0
"""CPU arithmetic experiments on the completed indexer LayerNorm128 matrix.

These are explicitly rounded CPU models, NOT reproductions of Triton reduction
trees, GPU rsqrt, compiler contraction, or saved kernel outputs. Consuming a
completed audit does not rerun its expired source freeze. No GPU initialization,
kernel installation, timing, tolerance change, or production qualification.
"""

import argparse
import hashlib
import json
import os
import subprocess
from itertools import product
from pathlib import Path

from benchmarks.analyze_glm53_reduction_receipts import sha
from benchmarks.kernels.check_glm53_attention_norms import (
    MAGNITUDES,
    ROWS,
    SEEDS,
    WEIGHTS,
    layernorm_oracle,
    load_checked,
    load_weights,
    mismatch_examples,
    packed_inputs,
    require,
)
from benchmarks.kernels.check_glm53_cached_rmsnorm import compare, tensor_sha

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "perf/results/2026-09-10"
PINS = {
    "summary": (
        "attention-norm-rank-private-probe/summary.json",
        "64c60102d391baffa7a4391257a9b742ed6932e6aea63cdb8cae77a478880c78",
    ),
    "audit": (
        "attention-norm-rank-private-probe/analysis.json",
        "0da742b0b8d7eb08651be0b32fff0b7c874aaa87781d01d47ce2ca3877e06bdb",
    ),
    "manifest": (
        "runtime-control/attention-rank-private-manifest.json",
        "29f934cd1cd6d695e0b62ba840926e31e6c7cc0212ae46d493269ee53736a12a",
    ),
    "supplement": (
        "runtime-control/attention-rank-private-supplement.json",
        "ca1f301a402d655b0e42db598437615864444b927ac723df4cb7db3f5d283f85",
    ),
}
MODELS = {
    "fp32_separate": "FP32 moments, centering, rsqrt, normalization, multiply/add.",
    "fp32_fused_affine": (
        "FP32 normalized value; affine in FP64 then rounded to FP32. "
        "Fused-affine emulation for these bounded BF16 weights, not GPU FMA."
    ),
    "fp32_norm_fp64_affine": "FP32 normalized value; FP64 multiply/add.",
    "fp32_moments_fp64_tail": (
        "FP32 mean/centered variance; recenter input and compute rsqrt, "
        "normalization and affine in FP64, including epsilon addition."
    ),
    "fp64_norm_rounded_fp32": (
        "FP64 moments/normalization; round normalized value to FP32, then FP64 affine."
    ),
    "fp64_control": "Full FP64 oracle arithmetic; reference/control only.",
}


def joined_cases(summary, audit, manifest, supplement):
    """Validate the closed evidence's joins; never call a frozen-source reader."""
    require(audit["status"] == supplement["status"] == "complete", "unclosed audit")
    require(summary["status"] == "failed", "historical failed probe required")
    require(audit["numerical_pass"] is False, "failed oracle must remain failed")
    for document in (summary, audit, supplement):
        require(document["manifest_sha256"] == PINS["manifest"][1], "manifest join")
    for document in (audit, supplement):
        require(document["summary_sha256"] == PINS["summary"][1], "summary join")
    require(supplement["audit_sha256"] == PINS["audit"][1], "audit join")
    require(summary["weights"] == manifest["weight_sha256"], "weight receipt join")
    require(
        (manifest["rows"], manifest["seeds"], manifest["magnitudes"])
        == (list(ROWS), list(SEEDS), list(MAGNITUDES)),
        "historical matrix changed",
    )
    checks = summary["checks"]
    keys = [(r["rank"], r["rows"], r["seed"], r["magnitude"]) for r in checks]
    require(keys == list(product(range(4), ROWS, SEEDS, MAGNITUDES)), "matrix order")
    require(audit["pairs"] == manifest["expected_pairs"] == len(checks), "pair count")
    require(
        all(r["repeat_graph_guards_mutation_pass"] is True for r in checks),
        "historical replay/guard failure",
    )
    failed = []
    for r in checks:
        passed = all(
            m["max_bf16_ulp"] <= 1
            for phases in r["oracle"].values()
            for phase in phases
            for m in phase
        )
        require(r["passed"] is passed, "historical pass flag disagrees")
        if not passed:
            failed.append({k: r[k] for k in ("rank", "rows", "seed", "magnitude")})
    require(failed == audit["failed_cases"], "failed-case join")
    require(len(failed) == supplement["failed_pairs"] == 20, "failed-case count")
    per_rank = [
        [{k: v for k, v in r.items() if k != "rank"} for r in checks if r["rank"] == i]
        for i in range(4)
    ]
    require(all(r == per_rank[0] for r in per_rank[1:]), "rank numerical drift")
    require(supplement["all_four_rank_records_exact"] is True, "rank join")
    return [r for r in checks if r["rank"] == 0]


def arithmetic_models(x, weight, bias):
    """Eager CPU tensor operations; each operation rounds to its tensor dtype."""
    import torch

    require(
        all(
            t.device.type == "cpu" and t.dtype == torch.bfloat16
            for t in (x, weight, bias)
        ),
        "CPU BF16 inputs required",
    )
    require(
        x.ndim == 2
        and x.shape[0] > 0
        and x.shape[1] == 128
        and weight.shape == bias.shape == (128,),
        "LayerNorm128 shape required",
    )
    require(all(torch.isfinite(t).all() for t in (x, weight, bias)), "finite inputs")
    x32, w32, b32 = x.float(), weight.float(), bias.float()
    mean32 = x32.mean(-1, keepdim=True)
    centered32 = x32 - mean32
    var32 = centered32.square().mean(-1, keepdim=True)
    norm32 = centered32 * torch.rsqrt(var32 + 1e-6)
    x64, w64, b64 = x.double(), weight.double(), bias.double()
    centered64 = x64 - x64.mean(-1, keepdim=True)
    norm64 = centered64 * torch.rsqrt(centered64.square().mean(-1, keepdim=True) + 1e-6)
    affine64 = norm32.double() * w64 + b64
    tail64 = (x64 - mean32.double()) * torch.rsqrt(var32.double() + 1e-6)
    return {
        "fp32_separate": norm32 * w32 + b32,
        "fp32_fused_affine": affine64.float(),
        "fp32_norm_fp64_affine": affine64,
        "fp32_moments_fp64_tail": tail64 * w64 + b64,
        "fp64_norm_rounded_fp32": norm64.float().double() * w64 + b64,
        "fp64_control": norm64 * w64 + b64,
    }


def modeled_outputs(x, weight, bias):
    import torch

    outputs = {name: torch.empty_like(x) for name in MODELS}
    for start in range(0, len(x), 128):
        values = arithmetic_models(x[start : start + 128], weight, bias)
        for name, value in values.items():
            outputs[name][start : start + 128].copy_(value.bfloat16())
    return outputs


def retained_examples(case, phase, x, weight, bias, reference):
    """Reconstruct every retained failing scalar, without inventing GPU values."""
    evidence = []
    for arm in ("combo", "split"):
        failures = case["mismatch_examples"][arm][phase][2]
        if failures is None:
            continue
        # The saved file keeps at most 16; this matrix's failures all fit.
        require(
            failures["count"] == len(failures["worst"]), "truncated historical failures"
        )
        for example in failures["worst"]:
            row, col = example["row"], example["column"]
            require(
                float(reference[row, col]) == example["reference"],
                "oracle scalar drift",
            )
            values = arithmetic_models(x[row : row + 1], weight, bias)
            exact = float(values["fp64_control"][0, col])
            b = float(bias[col])
            evidence.append(
                dict(
                    arm=arm,
                    **example,
                    fp64_result=exact,
                    cancellation_ratio=(abs(exact - b) + abs(b)) / abs(exact)
                    if exact
                    else None,
                    cpu_models={
                        name: dict(
                            unrounded=float(v[0, col]), bf16=float(v.bfloat16()[0, col])
                        )
                        for name, v in values.items()
                    },
                )
            )
    return evidence


def analyze():
    import torch

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "hide GPUs explicitly")
    require(not torch.cuda.is_initialized(), "CUDA must remain uninitialized")
    torch.set_num_threads(1)
    receipts = {str(RESULTS / name): digest for name, digest in PINS.values()}
    documents = {
        key: load_checked(RESULTS / name, digest)
        for key, (name, digest) in PINS.items()
    }
    cases = joined_cases(**documents)
    weights = load_weights()
    weight_hashes = {name: tensor_sha(w) for name, w in zip(WEIGHTS, weights)}
    require(weight_hashes == documents["summary"]["weights"], "real weight drift")
    weight, bias = weights[2:]
    records = []
    for case in cases:
        for phase in range(2):
            packed = packed_inputs(
                case["rows"], case["seed"] + 100 * phase, case["magnitude"]
            )
            digest = tensor_sha(packed)
            require(digest == case["inputs"][phase], "packed input drift")
            x = packed[:, 2048:2176]
            reference = layernorm_oracle(x, weight, bias)
            outputs = modeled_outputs(x, weight, bias)
            require(
                torch.equal(outputs["fp64_control"], reference), "FP64 control drift"
            )
            records.append(
                dict(
                    rows=case["rows"],
                    seed=case["seed"],
                    magnitude=case["magnitude"],
                    phase=phase,
                    packed_sha256=digest,
                    indexer_sha256=tensor_sha(x),
                    oracle_sha256=tensor_sha(reference),
                    models={
                        name: dict(
                            **compare(value, reference),
                            **mismatch_examples(value, reference),
                            sha256=tensor_sha(value),
                        )
                        for name, value in outputs.items()
                    },
                    historical_examples=retained_examples(
                        case, phase, x, weight, bias, reference
                    ),
                )
            )
        print(json.dumps({"completed_unique_cases": len(records) // 2}), flush=True)
    worst = documents["supplement"]["worst_example"]
    (case,) = [
        r
        for r in records
        if all(r[k] == worst[k] for k in ("rows", "seed", "magnitude", "phase"))
    ]
    (point,) = [
        r
        for r in case["historical_examples"]
        if all(r[k] == worst[k] for k in ("arm", "row", "column"))
    ]
    require(
        point["fp64_result"]
        == documents["supplement"]["worst_cancellation"]["fp64_result"],
        "worst FP64 scalar drift",
    )
    aggregate = {
        name: dict(
            elements=sum(r["models"][name]["elements"] for r in records),
            bit_mismatches=sum(r["models"][name]["bit_mismatches"] for r in records),
            above_one_ulp=sum(r["models"][name]["count"] for r in records),
            failed_phases=sum(r["models"][name]["count"] > 0 for r in records),
            max_bf16_ulp=max(r["models"][name]["max_bf16_ulp"] for r in records),
        )
        for name in MODELS
    }
    require(not torch.cuda.is_initialized(), "unexpected CUDA initialization")
    for path, digest in receipts.items():
        require(sha(path) == digest, "completed receipt changed during analysis")
    sources = [
        Path(__file__),
        ROOT / "benchmarks/kernels/check_glm53_attention_norms.py",
        ROOT / "benchmarks/kernels/check_glm53_cached_rmsnorm.py",
        ROOT / "benchmarks/analyze_glm53_reduction_receipts.py",
    ]
    return dict(
        status="complete",
        scope=__doc__,
        receipts=receipts,
        source_sha256={str(p): sha(p) for p in sources},
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        tracked_worktree_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT)
        ).hexdigest(),
        torch=torch.__version__,
        threads=torch.get_num_threads(),
        device="cpu",
        weights=weight_hashes,
        unique_cases=len(cases),
        phases=len(records),
        all_four_rank_records_exact=True,
        historical_indexer_oracle_pass=False,
        production_qualified=False,
        gpu_run=False,
        model_descriptions=MODELS,
        aggregate=aggregate,
        worst_historical_point=point,
        records=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Exclusive reservation preserves even failed attempts; do not overwrite.
    with args.output.open("x") as stream:
        try:
            result = analyze()
        except BaseException as error:
            json.dump({"status": "failed", "error": repr(error)}, stream, indent=2)
            stream.write("\n")
            raise
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha(args.output),
                "aggregate": result["aggregate"],
            }
        )
    )


if __name__ == "__main__":
    main()
