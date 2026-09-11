#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Isolated mHC arithmetic-layout A/B at the pinned GLM-5.3 settings.

Both variants compile from this tree in one extension; serving stays untouched.
Graph timings rotate over 90 distinct FP32 parameter sets, like the model's
90 sites, so a single hot weight matrix is not the performance baseline.
"""

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

import torch


def build(directory, name="mhc_output_parallel_probe"):
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA_HOME must select the installed serving toolkit")
    version = subprocess.check_output(
        [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"], text=True
    )
    if f"release {torch.version.cuda}," not in version:
        raise RuntimeError(
            f"probe toolkit must match Torch CUDA {torch.version.cuda}: {version}"
        )

    root = Path(__file__).resolve().parents[2]
    directory.mkdir(parents=True, exist_ok=True)
    return load(
        name=name,
        sources=[str(Path(__file__).with_name(name + ".cu"))],
        extra_include_paths=[str(root / "csrc/quixicore/serving")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "-std=c++20",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-gencode=arch=compute_120f,code=sm_120f",
        ],
        build_directory=str(directory),
        verbose=True,
    )


def inputs(batch, seed, magnitude=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape):
        return torch.randn(shape, device="cuda", generator=g)

    return [
        (rand(batch, 4096) * magnitude).bfloat16(),
        (rand(batch, 4, 4096) * magnitude).bfloat16(),
        torch.sigmoid(rand(batch, 4)) * 2,
        rand(batch, 4, 4).softmax(-1),
        rand(24, 16384) * 0.02,
        torch.tensor([0.3, 0.2, 0.1], device="cuda"),
        rand(24) * 0.1,
        (1 + rand(4096) * 0.1).bfloat16(),
    ]


def compare(reference, candidate):
    stats = []
    for index, (ref, got) in enumerate(zip(reference, candidate)):
        delta = (got.float() - ref.float()).abs()
        row = {
            "max_abs": delta.max().item(),
            "rms_abs": delta.square().mean().sqrt().item(),
            "different_fraction": (got != ref).float().mean().item(),
        }
        stats.append(row)
        if index == 0:
            assert torch.equal(got, ref), "post-mixed BF16 residual changed"
        elif got.dtype == torch.float32:
            torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-6)
        else:
            scale = ref.float().abs().amax(-1, keepdim=True).clamp_min(1e-6)
            assert (delta / scale).max().item() < 2**-7
    return stats


def graph_checks(extension, batch):
    data = inputs(batch, 501)
    # Prime both paths outside capture; retain outputs from the captured calls.
    extension.run(*data, True, False, False)
    extension.run(*data, True, False, True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reference = extension.run(*data, True, False, False)
        candidate = extension.run(*data, True, False, True)
    checks = []
    for seed in range(3):
        for target, fresh in zip(data, inputs(batch, seed + 601)):
            target.copy_(fresh)
        graph.replay()
        checks.append(compare(reference, candidate))
    return checks


def installed_baseline_checks(extension, batch):
    from vllm.quixicore.ops import quixicore_ops

    data = inputs(batch, 701)
    x, residual, post, comb, fn, scale, base, norm = data
    checks = []
    for fused in (False, True):
        for with_norm in (False, True):
            constants = [1e-5, 1e-6, 1e-6, 2.0, 20, norm if with_norm else None, 1e-5]
            if fused:
                reference = quixicore_ops.dsv4_mhc_fused_post_pre(
                    x, residual, post, comb, fn, scale, base, *constants
                )
            else:
                reference = [
                    residual,
                    *quixicore_ops.dsv4_mhc_pre(residual, fn, scale, base, *constants),
                ]
            isolated = extension.run(*data, fused, with_norm, False)
            if len(reference) != len(isolated):
                raise ValueError("isolated baseline output count differs from serving")
            if not all(torch.equal(a, b) for a, b in zip(reference, isolated)):
                raise ValueError(
                    "isolated baseline differs from the installed serving operator"
                )
            checks.append({"fused": fused, "norm": with_norm, "bit_exact": True})
    return checks


def checkpoint_parameters(model):
    """Load only the 90 actual mHC sites, not the model's large linear weights."""
    from contextlib import ExitStack

    from safetensors import safe_open

    index = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    sites, identity = [], []
    with ExitStack() as stack:
        overrides = stack.enter_context(
            safe_open(model / "f32-overrides.safetensors", framework="pt", device="cpu")
        )
        shards = {}
        for layer in range(45):
            for kind in ("attn", "ffn"):
                prefix = f"model.language_model.layers.{layer}.hc_{kind}"
                filename = index[prefix + "_fn"]
                if filename not in shards:
                    shards[filename] = stack.enter_context(
                        safe_open(model / filename, framework="pt", device="cpu")
                    )
                # Serving upcasts the checkpoint's BF16 fn; only base/scale
                # are restored from the native FP32 sidecar.
                tensors = [
                    shards[filename].get_tensor(prefix + "_fn").float(),
                    overrides.get_tensor(prefix + "_scale"),
                    overrides.get_tensor(prefix + "_base"),
                ]
                identity.append(
                    {
                        "site": prefix,
                        "sha256": [
                            hashlib.sha256(t.numpy().tobytes()).hexdigest()
                            for t in tensors
                        ],
                    }
                )
                sites.append([t.cuda() for t in tensors])
    return sites, identity


def measure(extension, data, candidate, repeats):
    # Same work sequence and buffers for each variant. Every parameter set is
    # distinct; graph allocators retain its output storage until replay ends.
    for row in data:
        extension.run(*row, True, False, candidate)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for row in data:
            extension.run(*row, True, False, candidate)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(repeats):
            graph.replay()
        b.record()
        b.synchronize()
        samples.append(a.elapsed_time(b) * 1000 / (repeats * len(data)))
    return {"graph_us": samples, "median_us": statistics.median(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--model",
        type=Path,
        help="also check and time all 90 actual mHC parameter sets",
    )
    args = parser.parse_args()
    if not args.batch or min(args.seeds, args.repeats, *args.batch) < 1:
        parser.error("batch sizes, seeds and repeats must be positive")
    if max(args.batch) > 8:
        parser.error("this probe covers only the cooperative decode range <= 8")
    if not args.build_only:
        if args.output is None or args.output.exists():
            parser.error("a run requires a new --output path")
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        ).strip()
        if active:
            parser.error(f"GPUs already have compute processes: {active}")
    extension = build(args.build_dir)
    if args.build_only:
        return
    root = Path(__file__).resolve().parents[2]
    sources = [
        root / "csrc/quixicore/serving/mhc_ampere.cuh",
        Path(__file__).with_name("mhc_output_parallel.cuh"),
        Path(__file__).with_suffix(".py"),
        Path(__file__).with_name("mhc_output_parallel_probe.cu"),
    ]
    result = {
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "torch": torch.__version__,
        "isolated_binary_sha256": hashlib.sha256(
            Path(extension.__file__).read_bytes()
        ).hexdigest(),
        "serving_binary_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root / "vllm").glob("_quixicore_C*.so")
        },
        "gpu": torch.cuda.get_device_name(),
        "sinkhorn_iterations": 20,
        "settings": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "checks": [],
        "graph_checks": [],
        "checkpoint_checks": [],
        "installed_baseline_checks": [],
        "timings": [],
        "status": "running",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    try:
        parameters = None
        if args.model:
            parameters, result["checkpoint_parameters"] = checkpoint_parameters(
                args.model
            )
            save()
        for batch in args.batch:
            result["installed_baseline_checks"].append(
                {
                    "batch": batch,
                    "checks": installed_baseline_checks(extension, batch),
                }
            )
            save()
            for seed in range(args.seeds):
                for magnitude in [0.01, 1.0, 100.0]:
                    data = inputs(batch, seed, magnitude)
                    for fused in [False, True]:
                        for norm in [False, True]:
                            baseline = extension.run(*data, fused, norm, False)
                            candidate = extension.run(*data, fused, norm, True)
                            row = {
                                "batch": batch,
                                "seed": seed,
                                "magnitude": magnitude,
                                "fused": fused,
                                "norm": norm,
                            }
                            result["checks"].append(row)
                            save()
                            row["errors"] = compare(baseline, candidate)
                    save()
            print(f"batch {batch}: arithmetic checks passed", flush=True)
            result["graph_checks"].append(
                {"batch": batch, "errors": graph_checks(extension, batch)}
            )
            if parameters:
                for site, weights in enumerate(parameters):
                    for magnitude in (0.01, 1.0, 100.0):
                        data = inputs(batch, site + 100, magnitude)
                        data[4:7] = weights
                        row = {"batch": batch, "site": site, "magnitude": magnitude}
                        result["checkpoint_checks"].append(row)
                        save()
                        row["errors"] = compare(
                            extension.run(*data, True, False, False),
                            extension.run(*data, True, False, True),
                        )
                save()
            if not args.check_only:
                data = [inputs(batch, seed + 100) for seed in range(90)]
                if parameters:
                    for inputs_row, weights in zip(data, parameters):
                        inputs_row[4:7] = weights
                row = {"batch": batch, "sites": len(data)}
                # A/B/A identifies drift instead of selecting a favorable run.
                for label, candidate in [("A", False), ("B", True), ("A2", False)]:
                    row[label] = measure(extension, data, candidate, args.repeats)
                result["timings"].append(row)
                print(json.dumps(row), flush=True)
                save()
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        save()
        raise
    save()


if __name__ == "__main__":
    main()
