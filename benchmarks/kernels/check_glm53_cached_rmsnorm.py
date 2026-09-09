#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay source-bound cached RMSNorm configs without changing serving caches.

Diagnostic only: synthetic activations, three real checkpoint norm vectors,
exact generated source and Inductor compilation metadata. Cross-config equality
is an observation, not an assumed gate or proof of full-model causality.
"""

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROWS = (1, 16, 640, 7616)
SEEDS = (530901, 530902)
SITES = (
    (0, "input_layernorm", 0.125),
    (22, "post_attention_layernorm", 1.0),
    (44, "post_attention_layernorm", 8.0),
)
INPLACE_ARGS = ("in_out_ptr0", "in_ptr0", "xnumel", "r0_numel", "XBLOCK", "R0_BLOCK")
TRIPLE_ARGS = (
    "in_ptr0",
    "in_ptr1",
    "out_ptr1",
    "out_ptr2",
    "out_ptr3",
    "xnumel",
    "r0_numel",
    "XBLOCK",
    "R0_BLOCK",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_layout(text, kernel):
    functions = [
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name == kernel
    ]
    if len(functions) != 1:
        raise ValueError("expected exactly one named kernel")
    arguments = tuple(arg.arg for arg in functions[0].args.args)
    if arguments == INPLACE_ARGS:
        return "inplace"
    if arguments == TRIPLE_ARGS:
        return "triple"
    raise ValueError(f"unrecognized generated kernel signature: {arguments}")


def checked_config(config):
    keys = ("XBLOCK", "R0_BLOCK", "num_warps", "num_stages")
    values = tuple(config[key] for key in keys)
    if values not in ((1, 1024, 8, 1), (1, 4096, 16, 1)):
        raise ValueError(f"unexpected recorded configuration: {values}")
    return dict(zip(keys, values))


def compare(a, b):
    import torch

    if a.shape != b.shape or a.dtype != torch.bfloat16 or b.dtype != a.dtype:
        raise ValueError("matching BF16 tensors required")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("finite BF16 tensors required")
    # Chunk metrics so the largest serving shape stays comfortably bounded.
    count = numeric = affected_rows = ulp = 0
    maximum = absolute = squared = 0.0
    for aa, bb in zip(a.split(128), b.split(128)):
        bits_a, bits_b = aa.view(torch.int16), bb.view(torch.int16)
        changed = bits_a != bits_b
        count += changed.sum().item()
        affected_rows += changed.any(-1).sum().item()
        numeric += (aa != bb).sum().item()
        distance = aa.double() - bb.double()
        maximum = max(maximum, distance.abs().max().item())
        absolute += distance.abs().sum().item()
        squared += distance.square().sum().item()

        def ordered(bits):
            integer = bits.int() & 0xFFFF
            magnitude = integer & 0x7FFF
            return torch.where(integer >= 0x8000, -magnitude, magnitude)

        ulp = max(ulp, (ordered(bits_a) - ordered(bits_b)).abs().max().item())
    return dict(
        elements=a.numel(),
        bit_mismatches=count,
        numeric_mismatches=numeric,
        affected_rows=affected_rows,
        max_bf16_ulp=ulp,
        max_abs=maximum,
        mean_abs=absolute / a.numel(),
        rms=(squared / a.numel()) ** 0.5,
    )


def oracle(x, weight):
    import torch

    output = torch.empty_like(x)
    for source, target in zip(x.split(128), output.split(128)):
        wide = source.double()
        inverse = torch.rsqrt(wide.square().mean(-1, keepdim=True) + 1e-5)
        target.copy_((wide * inverse * weight.double()).bfloat16())
    return output


def make_inputs(rows, seed, magnitude):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(rows, 4096, generator=generator) * magnitude).bfloat16()


def tensor_sha(tensor):
    import torch

    return hashlib.sha256(
        tensor.contiguous().view(-1).view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def run_configuration(launcher, layout, x, changed, weight):
    import torch

    rows = x.shape[0]
    wg = weight.cuda()
    storage = torch.full((rows + 2, 4096), 123.0, dtype=torch.bfloat16, device="cuda")
    working = storage[1:-1]
    guards = [storage]
    if layout == "triple":
        guards += [torch.full_like(storage, 123.0) for _ in range(3)]
        destinations = [buffer[1:-1] for buffer in guards[1:]]
    else:
        destinations = [working]

    def launch():
        call_args = [working, wg]
        if layout == "triple":
            call_args += destinations
        launcher(*call_args, rows, 4096, stream=torch.cuda.current_stream().cuda_stream)

    outputs = []
    graph = torch.cuda.CUDAGraph()
    for iteration, data in enumerate((x, x, changed)):
        working.copy_(data)
        if iteration == 2:
            with torch.cuda.graph(graph):
                launch()
            # Change input after capture; reset in-place data outside graph.
            working.copy_(x)
            graph.replay()
            torch.cuda.synchronize()
            if compare(destinations[0].cpu(), outputs[0])["bit_mismatches"]:
                raise ValueError("graph original input differs from eager")
            working.copy_(changed)
            graph.replay()
        else:
            launch()
        torch.cuda.synchronize()
        if not all(torch.all(buffer[[0, -1]] == 123).item() for buffer in guards):
            raise ValueError("output guard overwritten")
        if not torch.equal(wg.cpu(), weight):
            raise ValueError("weight mutated")
        if layout == "triple" and not torch.equal(working.cpu(), data):
            raise ValueError("read-only input mutated")
        output = destinations[0].cpu()
        if any(compare(t.cpu(), output)["bit_mismatches"] for t in destinations[1:]):
            raise ValueError("three generated outputs differ")
        if iteration == 1 and compare(output, outputs[0])["bit_mismatches"]:
            raise ValueError("same-config eager repeat differs")
        outputs.append(output)
    if compare(outputs[0], outputs[2])["bit_mismatches"] == 0:
        raise ValueError("changed-input check vacuous")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist; preserve every attempt")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        parser.error("GPUs already have compute processes")
    args.output.mkdir(parents=True)
    # Import only copies of generated source; both caches are private to this run.
    os.environ["TRITON_CACHE_DIR"] = str(args.output.resolve() / "triton-cache")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(
        args.output.resolve() / "inductor-cache"
    )
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    result = dict(
        status="running",
        diagnostic_only=True,
        method=__doc__,
        probe_sha256=sha(__file__),
        audit_sha256=sha(args.audit),
        git_head=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        cases=dict(rows=ROWS, seeds=SEEDS, sites=SITES, changed_seed_offset=100),
        sources=[],
        weights=[],
        checks=[],
        limitation=(
            "Synthetic inputs, not archived pre-RMS activations; "
            "no serving TPS or causality claim"
        ),
    )

    def save():
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    receipts = {}
    try:
        import torch
        import triton
        from safetensors import safe_open
        from torch._inductor.runtime.autotune_cache import AutotuneCache

        result.update(
            torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda
        )
        if torch.cuda.device_count() != 4:
            raise ValueError("requires the four rank-matched SM120 devices")
        result["devices"] = [str(torch.cuda.get_device_properties(i)) for i in range(4)]
        if any(torch.cuda.get_device_capability(i) != (12, 0) for i in range(4)):
            raise ValueError("requires SM120")
        audit = json.loads(args.audit.read_text())
        if len(audit["reduction_width_changes"]) != 4:
            raise ValueError("expected exactly four source-bound reduction changes")
        weights = []
        index_path = args.model / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())["weight_map"]
        result["weight_index_sha256"] = sha(index_path)
        for layer, norm, magnitude in SITES:
            key = f"model.language_model.layers.{layer}.{norm}.weight"
            with safe_open(args.model / index[key], framework="pt", device="cpu") as f:
                weight = f.get_tensor(key)
            if weight.dtype != torch.bfloat16 or weight.shape != (4096,):
                raise ValueError("expected actual BF16 H4096 norm weight")
            weights.append(weight)
            result["weights"].append(dict(key=key, sha256=tensor_sha(weight)))

        seen_ranks = set()
        for entry in audit["reduction_width_changes"]:
            if len(entry["sources"]) != 1:
                raise ValueError("ambiguous source/config binding")
            bound = entry["sources"][0]
            relative = Path(bound["relative"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid relative source path")
            rank = int(bound["device_indices"][0])
            if rank in seen_ranks or rank not in range(4):
                raise ValueError("expected one source per rank")
            seen_ranks.add(rank)
            for base in map(Path, audit["paths"]):
                source = base / relative
                key = AutotuneCache._prepare_key(str(source))
                config_path = Path(
                    AutotuneCache._make_local_cache_key(str(source.parent), key)
                )
                if str(config_path.relative_to(base)) != entry["relative"]:
                    raise ValueError("source/config key mismatch")
                if sha(source) != bound["source_sha256"]:
                    raise ValueError("source changed since audit")
                if (
                    sha(config_path)
                    != entry["file_sha256"][audit["paths"].index(str(base))]
                ):
                    raise ValueError("config changed since audit")
                for path in (source, config_path):
                    receipts[str(path)] = sha(path)
            (kernel_name,) = bound["kernels"]
            layout = source_layout(source.read_text(), kernel_name)
            copied = args.output / f"rank{rank}" / relative.name
            copied.parent.mkdir()
            shutil.copyfile(source, copied)
            with torch.cuda.device(rank):
                spec = importlib.util.spec_from_file_location(
                    f"glm53_rms_rank{rank}", copied
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                autotuner = getattr(module, kernel_name)
                launchers = []
                record = dict(
                    rank=rank,
                    source=str(source),
                    sha256=sha(copied),
                    kernel=kernel_name,
                    layout=layout,
                    configurations=[],
                )
                result["sources"].append(record)
                for saved in entry["configs"]:
                    config = checked_config(saved)
                    cfg = triton.Config(
                        {k: config[k] for k in ("XBLOCK", "R0_BLOCK")},
                        num_warps=config["num_warps"],
                        num_stages=config["num_stages"],
                    )
                    # Preserve Inductor's exact signature, divisibility attributes,
                    # constants and compile options. Never call its autotune/run hook.
                    compiled = autotuner._precompile_config(cfg)
                    launcher = compiled.make_launcher()
                    record["configurations"].append(
                        dict(
                            **config,
                            recorded_cache_hash=saved["triton_cache_hash"],
                            compiled_cache_hash=launcher.cache_hash,
                        )
                    )
                    if launcher.cache_hash != saved["triton_cache_hash"]:
                        raise ValueError(
                            "recompiled kernel hash differs from serving artifact"
                        )
                    launchers.append(launcher)
                save()
                for rows in ROWS:
                    for seed in SEEDS:
                        for site, (_, _, magnitude) in enumerate(SITES):
                            weight = weights[site]
                            x = make_inputs(rows, seed, magnitude)
                            changed = make_inputs(rows, seed + 100, magnitude)
                            references = [oracle(data, weight) for data in (x, changed)]
                            outputs = [
                                run_configuration(launcher, layout, x, changed, weight)
                                for launcher in launchers
                            ]
                            checks = dict(
                                rank=rank,
                                rows=rows,
                                seed=seed,
                                site=site,
                                input_sha256=tensor_sha(x),
                                changed_sha256=tensor_sha(changed),
                                guards_repeat_graph_and_mutation_pass=True,
                                cross_config=[],
                                oracle=[],
                            )
                            for i, output_index in enumerate((0, 2)):
                                checks["cross_config"].append(
                                    compare(
                                        outputs[0][output_index],
                                        outputs[1][output_index],
                                    )
                                )
                                checks["oracle"].append(
                                    [
                                        compare(arm[output_index], references[i])
                                        for arm in outputs
                                    ]
                                )
                            result["checks"].append(checks)
                            save()
                            if any(
                                metric["max_bf16_ulp"] > 1
                                for pair in checks["oracle"]
                                for metric in pair
                            ):
                                raise ValueError(
                                    "predeclared one-BF16-ULP oracle gate failed"
                                )
                            print(
                                json.dumps(
                                    {
                                        k: checks[k]
                                        for k in (
                                            "rank",
                                            "rows",
                                            "seed",
                                            "site",
                                            "cross_config",
                                        )
                                    }
                                ),
                                flush=True,
                            )
                torch.cuda.synchronize()
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["original_receipts"] = receipts
        result["original_sources_and_configs_unchanged"] = all(
            sha(p) == value for p, value in receipts.items()
        )
        if not result["original_sources_and_configs_unchanged"]:
            result["status"] = "failed"
        save()
    if result["status"] != "complete":
        raise RuntimeError("diagnostic failed")


if __name__ == "__main__":
    main()
