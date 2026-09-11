# SPDX-License-Identifier: Apache-2.0
"""One-process, fixed-config KDA gate arithmetic diagnostic, not a TPS test."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

from benchmarks.analyze_glm53_kda_choices import read_pinned
from benchmarks.analyze_glm53_quality_pair import require
from benchmarks.analyze_glm53_reduction_receipts import sha
from benchmarks.kernels.check_glm53_cached_rmsnorm import tensor_sha

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path("/raid/weights/GLM-5.3-Flash-NVFP4-FP8-KDA-TP4")
INVENTORY = (
    ROOT / "perf/results/2026-09-10/runtime-control/kda-disk-choice-analysis.json"
)
INVENTORY_SHA = "744500910b0930425294a1cf425cc05bf51a7a59a5f31d59639c2515f525984b"
KERNEL = "kda_gate_chunk_cumsum_vector_kernel"
CHUNK = "vllm/models/kimi_k3/amd/ops/third_party/kda/chunk.py"
LAYOUTS = ((1,), (63,), (64,), (65,), (1000,), (17, 63, 65, 855), (7616,))
SEEDS = (530901, 530902)
MAGNITUDES = (0.125, 1.0, 8.0)
WARPS = (8, 2)
PACKED_WIDTH = 3 * 16 * 128 + 128 + 2 * 128
WEIGHT_NAMES = tuple(
    f"model.language_model.layers.0.self_attn.{name}" for name in ("A_log", "dt_bias")
)
# A diagnostic FP32 integrity check, not a replacement for model or BF16 gates.
ATOL, RTOL = 1e-6, 2e-6


def matrix():
    return [
        dict(rank=rank, lengths=list(lengths), seed=seed, magnitude=magnitude)
        for rank, lengths, seed, magnitude in product(
            range(4), LAYOUTS, SEEDS, MAGNITUDES
        )
    ]


def save_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_weights():
    import torch
    from safetensors import safe_open

    with safe_open(
        MODEL / "f32-overrides.safetensors", framework="pt", device="cpu"
    ) as f:
        a, bias = (f.get_tensor(name) for name in WEIGHT_NAMES)
    require(a.dtype == bias.dtype == torch.float32, "FP32 repairs required")
    require(a.shape == (64,) and bias.shape == (8192,), "wrong weight geometry")
    return a, bias.reshape(64, 128)


def preparation():
    """Collect fresh source/evidence receipts, without starting a probe."""
    inventory = read_pinned(INVENTORY, INVENTORY_SHA)
    gate_rows = [r for r in inventory["comparisons"] if r["kernel"] == KERNEL]
    require(len(gate_rows) == 4, "incomplete gate inventory")
    receipts = {str(INVENTORY): INVENTORY_SHA}
    for row in gate_rows:
        for arm, warps in (("original", 8), ("fresh_shared", 2)):
            choice = row[arm]
            selected = choice["selected"]
            require(selected["kwargs"] == {"BS": 32}, "gate block changed")
            require(
                selected["num_warps"] == warps and selected["num_stages"] == 3,
                "gate config changed",
            )
            read_pinned(choice["path"], choice["sha256"])
            receipts[choice["path"]] = choice["sha256"]
    config = json.loads((MODEL / "config.json").read_text())["text_config"][
        "linear_attn_config"
    ]
    require(
        config["head_dim"] == 128 and config["num_heads"] == 64, "model heads changed"
    )
    require(
        config["gate_lower_bound"] == -5
        and not config.get("use_full_rank_gate", False),
        "wrong gate model",
    )
    swapset = json.loads((MODEL / "fp8-swapset.json").read_text())
    require(swapset["beta_rows"] == 128, "packed beta geometry changed")
    paths = [
        ROOT / CHUNK,
        Path(__file__),
        ROOT / "benchmarks/analyze_glm53_kda_choices.py",
        ROOT / "benchmarks/analyze_glm53_quality_pair.py",
        ROOT / "benchmarks/analyze_glm53_reduction_receipts.py",
        ROOT / "benchmarks/kernels/check_glm53_cached_rmsnorm.py",
        ROOT / "perf/glm53-kda-choice-protocol.md",
        ROOT / "vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py",
        ROOT / "slimserve/fp8_swapset.py",
        MODEL / "config.json",
        MODEL / "fp8-swapset.json",
        MODEL / "f32-overrides.safetensors",
    ]
    # Include the serving re-export and actual imported dependency tree, not
    # only the similarly named generic FLA KDA implementation.
    for folder in (
        "vllm/models/kimi_k3/amd/ops/third_party/kda",
        "vllm/models/kimi_k3/nvidia/ops/third_party/kda",
        "vllm/third_party/flash_linear_attention/ops",
        "vllm/triton_utils",
    ):
        paths.extend((ROOT / folder).glob("*.py"))
    site = Path(sys.prefix) / "lib/python3.12/site-packages"
    for relative in (
        "triton/runtime",
        "triton/compiler",
        "triton/language",
        "triton/backends/nvidia",
    ):
        paths.extend((site / relative).glob("*.py"))
    paths += [
        site / "triton/_C/libtriton.so",
        site / "triton/backends/nvidia/bin/ptxas",
        site / "triton/backends/nvidia/bin/ptxas-blackwell",
    ]
    for file in paths:
        receipts[str(file.resolve())] = sha(file)
    # Source unchanged from both completed model arms; no historical runtime
    # config or whole-cubin identity is implied by this source comparison.
    history = {}
    for commit in ("59ae0c88f", "ce6df61aa"):
        source = subprocess.check_output(["git", "show", f"{commit}:{CHUNK}"], cwd=ROOT)
        history[commit] = hashlib.sha256(source).hexdigest()
        require(
            history[commit] == sha(ROOT / CHUNK), "gate source changed since serving"
        )
    weights = load_weights()
    return dict(
        schema=1,
        sources=receipts,
        history=history,
        cases=matrix(),
        weight_sha256=[tensor_sha(w) for w in weights],
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        atol=ATOL,
        rtol=RTOL,
        warps=list(WARPS),
        packed_width=PACKED_WIDTH,
    )


def prepare(path):
    require(not path.exists(), "manifest exists")
    manifest = preparation()
    save_new(path, manifest)
    receipts = manifest["sources"]
    print(f"Prepared {len(matrix())} pairs; {len(receipts)} source/evidence hashes")


def verify(manifest):
    require(manifest["schema"] == 1 and manifest["cases"] == matrix(), "matrix changed")
    require(
        (manifest["atol"], manifest["rtol"]) == (ATOL, RTOL), "oracle criterion changed"
    )
    require(
        manifest["warps"] == list(WARPS) and manifest["packed_width"] == PACKED_WIDTH,
        "geometry changed",
    )
    for name, digest in manifest["sources"].items():
        require(sha(name) == digest, f"frozen source/evidence changed: {name}")
    require(
        [tensor_sha(w) for w in load_weights()] == manifest["weight_sha256"],
        "repair values changed",
    )


def sequence_metadata(lengths):
    require(
        bool(lengths) and all(type(n) is int and n > 0 for n in lengths),
        "invalid sequences",
    )
    starts, chunks = [0], []
    for seq, size in enumerate(lengths):
        starts.append(starts[-1] + size)
        chunks.extend((seq, chunk) for chunk in range((size + 63) // 64))
    return starts, chunks


def oracle(raw_g, raw_beta, a, bias, lengths):
    import torch

    gate = -5.0 * torch.sigmoid(
        a.double().exp()[None, None, :, None]
        * (raw_g.double() + bias.double()[None, None])
    )
    result = torch.empty_like(gate)
    start = 0
    for length in lengths:
        for offset in range(0, length, 64):
            lo, hi = start + offset, start + min(offset + 64, length)
            result[:, lo:hi] = gate[:, lo:hi].cumsum(1) / math.log(2)
        start += length
    return result, raw_beta.double().sigmoid()


def errors(actual, reference):
    import torch

    require(
        torch.isfinite(actual).all().item() and torch.isfinite(reference).all().item(),
        "nonfinite output",
    )
    delta = (actual.double() - reference).abs()
    scaled = delta / (ATOL + RTOL * reference.abs())
    return dict(
        max_abs=delta.max().item(),
        max_error_ratio=scaled.max().item(),
        pass_oracle=bool((scaled <= 1).all().item()),
    )


def pair_delta(a, b):
    import torch

    require(
        a.shape == b.shape and a.dtype == b.dtype == torch.float32, "FP32 pair required"
    )
    delta = a.double() - b.double()
    return dict(
        elements=a.numel(),
        bit_mismatches=(a.view(torch.int32) != b.view(torch.int32)).sum().item(),
        max_abs=delta.abs().max().item(),
        rms=delta.square().mean().sqrt().item(),
    )


def launch(jit, raw_g, raw_beta, a, bias, out, beta_out, starts, chunks, warps):
    require(warps in WARPS, "unprescribed warp count")
    return jit[(5, len(chunks), 16)](
        s=raw_g,
        raw_beta=raw_beta,
        A_log=a,
        g_bias=bias,
        o=out,
        beta_out=beta_out,
        cu_seqlens=starts,
        chunk_indices=chunks,
        cumsum_scale=1.4426950408889634,
        lower_bound=-5.0,
        beta=1.0,
        threshold=20.0,
        T=raw_g.shape[1],
        stride_beta_batch=raw_beta.stride(0),
        stride_beta_token=raw_beta.stride(1),
        stride_beta_head=raw_beta.stride(2),
        H=16,
        S=128,
        BT=64,
        BS=32,
        HAS_BIAS=True,
        IS_VARLEN=True,
        USE_LOWER_BOUND=True,
        num_warps=warps,
        num_stages=3,
    )


def gpu_query():
    return subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()


def gpu_config():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,power.limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()


def run(manifest_path, output):
    require(not output.exists(), "attempt already exists; no retries")
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "one prescribed GPU required"
    )
    require(not gpu_query(), "GPU workload already active")
    output.mkdir(parents=True)
    os.environ["TRITON_CACHE_DIR"] = str(output.resolve() / "triton")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(output.resolve() / "inductor")
    os.environ["TRITON_CACHE_AUTOTUNING"] = "0"
    save_new(
        output / "attempt.json",
        dict(manifest_sha256=sha(manifest_path), pid=os.getpid()),
    )
    summary = dict(
        status="running",
        pairs=0,
        records=[],
        binaries={},
        manifest_sha256=sha(manifest_path),
        gpu_config=gpu_config(),
    )
    (output / "binaries").mkdir()
    try:
        import torch
        from triton.runtime.jit import JITFunction

        from vllm.models.kimi_k3.amd.ops.third_party.kda.chunk import (
            kda_gate_chunk_cumsum_vector_kernel,
        )

        torch.set_num_threads(4)
        require(torch.cuda.get_device_capability() == (12, 0), "SM120 required")
        summary["gpu"] = torch.cuda.get_device_name()
        summary["torch"] = torch.__version__
        # Invoke the actual source JIT with explicit heuristics/config, never
        # Autotuner.run: no tuning, cache injection or serving policy change.
        jit = kda_gate_chunk_cumsum_vector_kernel.fn.fn
        require(
            isinstance(jit, JITFunction) and jit.__name__ == KERNEL, "wrong serving JIT"
        )
        all_a, all_bias = load_weights()
        for index, case in enumerate(matrix()):
            rank, lengths = case["rank"], case["lengths"]
            tokens = sum(lengths)
            generator = torch.Generator().manual_seed(case["seed"])
            g = (
                (
                    torch.randn((1, tokens, 16, 128), generator=generator)
                    * case["magnitude"]
                )
                .bfloat16()
                .cuda()
            )
            packed = torch.full(
                (1, tokens, PACKED_WIDTH), 3.5, dtype=torch.bfloat16, device="cuda"
            )
            beta = packed[:, :, 6144:6160]
            beta.copy_(
                (
                    torch.randn((1, tokens, 16), generator=generator)
                    * case["magnitude"]
                ).bfloat16()
            )
            a = all_a[rank * 16 : (rank + 1) * 16].cuda()
            bias = all_bias[rank * 16 : (rank + 1) * 16].cuda()
            starts_cpu, chunks_cpu = sequence_metadata(lengths)
            starts = torch.tensor(starts_cpu, dtype=torch.int32, device="cuda")
            chunks = torch.tensor(chunks_cpu, dtype=torch.int32, device="cuda")
            protected = [g, packed, a, bias, starts, chunks]
            guards, outputs, graphs, binary_ids = [], [], [], []
            for warps in WARPS:
                stores = [
                    torch.full((n + 64,), 12345.0, device="cuda")
                    for n in (g.numel(), beta.numel())
                ]
                pair = (stores[0][32:-32].view_as(g), stores[1][32:-32].view_as(beta))
                saved = [t.clone() for t in protected]
                compiled = launch(jit, g, beta, a, bias, *pair, starts, chunks, warps)
                eager = [t.clone() for t in pair]
                launch(jit, g, beta, a, bias, *pair, starts, chunks, warps)
                require(
                    all(torch.equal(x, y) for x, y in zip(eager, pair)),
                    "eager repetition mismatch",
                )
                require(
                    all(torch.equal(x, y) for x, y in zip(saved, protected)),
                    "input mutation",
                )
                cubin = compiled.asm["cubin"]
                digest = hashlib.sha256(cubin).hexdigest()
                if digest not in summary["binaries"]:
                    with (output / "binaries" / f"{digest}.cubin").open("xb") as f:
                        f.write(cubin)
                summary["binaries"][digest] = dict(
                    warps=warps,
                    stages=compiled.metadata.num_stages,
                    name=compiled.name,
                    hash=compiled.hash,
                )
                require(
                    compiled.metadata.num_warps == warps
                    and compiled.metadata.num_stages == 3,
                    "compiled geometry changed",
                )
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch(jit, g, beta, a, bias, *pair, starts, chunks, warps)
                graph.replay()
                require(
                    all(torch.equal(x, y) for x, y in zip(eager, pair)),
                    "graph mismatch",
                )
                require(
                    all(torch.equal(x, y) for x, y in zip(saved, protected)),
                    "graph input mutation",
                )
                guards.append(stores)
                outputs.append(pair)
                graphs.append(graph)
                binary_ids.append(digest)
            record = dict(index=index, **case, phases=[], binary_ids=binary_ids)
            for phase in range(2):
                if phase:
                    g.add_(0.125)
                    beta.add_(-0.25)
                    saved = [t.clone() for t in protected]
                    for graph in graphs:
                        graph.replay()
                    for warps, pair in zip(WARPS, outputs):
                        changed = [t.clone() for t in pair]
                        launch(jit, g, beta, a, bias, *pair, starts, chunks, warps)
                        require(
                            all(torch.equal(x, y) for x, y in zip(changed, pair)),
                            "changed-input graph mismatch",
                        )
                    require(
                        all(torch.equal(x, y) for x, y in zip(saved, protected)),
                        "changed input mutation",
                    )
                reference = oracle(g, beta, a, bias, lengths)
                phase_row = dict(
                    phase=phase,
                    arms=[],
                    pairwise=[pair_delta(x, y) for x, y in zip(*outputs)],
                )
                for warps, pair, stores in zip(WARPS, outputs, guards):
                    require(
                        all(
                            torch.all(t[:32] == 12345).item()
                            and torch.all(t[-32:] == 12345).item()
                            for t in stores
                        ),
                        "output guard overwritten",
                    )
                    phase_row["arms"].append(
                        dict(
                            warps=warps,
                            oracle=[errors(x, y) for x, y in zip(pair, reference)],
                            output_sha256=[tensor_sha(t.cpu()) for t in pair],
                        )
                    )
                record["phases"].append(phase_row)
                del reference
            require(
                all(
                    record["phases"][0]["arms"][arm]["output_sha256"]
                    != record["phases"][1]["arms"][arm]["output_sha256"]
                    for arm in range(2)
                ),
                "changed-input outputs did not change",
            )
            record["eager_repetition_graph_replay_mutation_guards_passed"] = True
            save_new(output / f"pair-{index:03d}.json", record)
            summary["records"].append(
                dict(
                    path=f"pair-{index:03d}.json",
                    sha256=sha(output / f"pair-{index:03d}.json"),
                )
            )
            summary["pairs"] += 1
            require(
                all(
                    o["pass_oracle"]
                    for p in record["phases"]
                    for arm in p["arms"]
                    for o in arm["oracle"]
                ),
                "FP32 oracle gate failed",
            )
            del graphs, outputs, guards, protected, saved, g, beta, packed
            del graph, pair, stores, eager
            if (index + 1) % 12 == 0:
                print(f"Completed {index + 1}/{len(matrix())} pairs", flush=True)
        verify(manifest)
        summary["status"] = "complete"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = repr(exc)
        raise
    finally:
        save_new(output / "summary.json", summary)


def audit(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    summary = json.loads((output / "summary.json").read_text())
    require(summary["manifest_sha256"] == sha(manifest_path), "manifest changed")
    require(not gpu_query(), "GPU not released")
    require(gpu_config() == summary["gpu_config"], "GPU configuration changed")
    records = [read_pinned(output / r["path"], r["sha256"]) for r in summary["records"]]
    complete = summary["status"] == "complete" and len(records) == len(matrix())
    require(summary["pairs"] == len(records), "pair count mismatch")
    require(
        not list((output / "triton").rglob("*.autotune.json")),
        "unexpected autotuning",
    )
    for digest in summary["binaries"]:
        require(
            sha(output / "binaries" / f"{digest}.cubin") == digest,
            "compiled binary changed",
        )
    for index, record in enumerate(records):
        audit_pair(record, index, summary["binaries"])
    oracle_passed = all(
        o["pass_oracle"]
        for record in records
        for phase in record["phases"]
        for arm in phase["arms"]
        for o in arm["oracle"]
    )
    require(not complete or oracle_passed, "complete result fails oracle")
    result = dict(
        status="complete" if complete else "terminal-failure",
        pairs=len(records),
        oracle_passed=oracle_passed if records else None,
        compiled_binaries_verified=len(summary["binaries"]),
        source_receipts_verified=len(manifest["sources"]),
        summary_sha256=sha(output / "summary.json"),
        changed_gate_elements=sum(
            p["pairwise"][0]["bit_mismatches"] for r in records for p in r["phases"]
        ),
        changed_beta_elements=sum(
            p["pairwise"][1]["bit_mismatches"] for r in records for p in r["phases"]
        ),
        gpu_processes_after="",
        limitation=__doc__,
    )
    save_new(output / "analysis.json", result)
    print(json.dumps(result, indent=2))


def audit_pair(record, index, binaries):
    expected = matrix()[index]
    require(
        record["index"] == index and {k: record[k] for k in expected} == expected,
        "wrong pair sequence",
    )
    require([p["phase"] for p in record["phases"]] == [0, 1], "missing phases")
    require(
        record["eager_repetition_graph_replay_mutation_guards_passed"] is True,
        "incomplete launch checks",
    )
    require(len(record["binary_ids"]) == 2, "missing binaries")
    for digest, warps in zip(record["binary_ids"], WARPS):
        require(
            binaries[digest]["warps"] == warps and binaries[digest]["stages"] == 3,
            "wrong compiled config",
        )
    for phase in record["phases"]:
        require([a["warps"] for a in phase["arms"]] == list(WARPS), "wrong arms")
        require(len(phase["pairwise"]) == 2, "missing pair comparison")
        for arm in phase["arms"]:
            require(
                len(arm["oracle"]) == 2 and len(arm["output_sha256"]) == 2,
                "incomplete output evidence",
            )
            for result in arm["oracle"]:
                ratio = result["max_error_ratio"]
                require(math.isfinite(ratio) and ratio >= 0, "invalid oracle error")
                require(result["pass_oracle"] == (ratio <= 1), "wrong oracle verdict")
        for slot, width in enumerate((2048, 16)):
            pair = phase["pairwise"][slot]
            require(
                pair["elements"] == sum(expected["lengths"]) * width,
                "wrong output extent",
            )
            count = pair["bit_mismatches"]
            require(
                type(count) is int and 0 <= count <= pair["elements"],
                "invalid mismatch count",
            )
            same_hash = (
                phase["arms"][0]["output_sha256"][slot]
                == phase["arms"][1]["output_sha256"][slot]
            )
            require(same_hash == (count == 0), "hash/bit verdict differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.manifest)
    else:
        require(args.output is not None, "output required")
        (run if args.action == "run" else audit)(args.manifest, args.output)


if __name__ == "__main__":
    main()
