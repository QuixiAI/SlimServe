# SPDX-License-Identifier: Apache-2.0
"""Fixed-config W/U recompute arithmetic; no serving or performance promotion.

Identity cases have an exact algebraic oracle. Conditioned random triangular
cases record float64-reference error without declaring a model accuracy gate.
Cross-config bit equality is always an observation, never an assumed result.
"""

import argparse
import hashlib
import json
import os
from itertools import product
from pathlib import Path

from benchmarks.analyze_glm53_quality_pair import require
from benchmarks.analyze_glm53_reduction_receipts import sha
from benchmarks.kernels import check_glm53_kda_gate as gate
from benchmarks.kernels.check_glm53_cached_rmsnorm import compare as compare_bf16
from benchmarks.kernels.check_glm53_cached_rmsnorm import tensor_sha

KERNEL = "recompute_w_u_fwd_kernel"
WARPS = (4, 8)
OUTPUTS = ("w", "u", "kg")


def compare(a, b):
    # The existing 128-row default is useful for host probes. These larger
    # GPU-side comparisons amortize launch/synchronization overhead while
    # keeping each metrics tile to one million elements.
    return compare_bf16(a, b, chunk_rows=8192)


def matrix():
    rows = []
    for rank, lengths in product(range(4), gate.LAYOUTS):
        rows.append(
            dict(
                rank=rank,
                lengths=list(lengths),
                seed=530901,
                magnitude=1.0,
                regime="identity",
            )
        )
        rows.extend(
            dict(
                rank=rank,
                lengths=list(lengths),
                seed=seed,
                magnitude=magnitude,
                regime="conditioned",
            )
            for seed, magnitude in product(gate.SEEDS, gate.MAGNITUDES)
        )
    return rows


def prepare(path):
    require(not path.exists(), "manifest exists")
    shared = gate.preparation()
    sources = shared["sources"]
    sources[str(Path(__file__).resolve())] = sha(__file__)
    completed = gate.ROOT / "perf/results/2026-09-10/kda-gate-v1/analysis.json"
    gate.read_pinned(
        completed, "dbb92de6bc14f00c86ba8d19a8a680ab8d5ad82ed696906d300a3432e5d9bf70"
    )
    sources[str(completed)] = sha(completed)
    inventory = gate.read_pinned(gate.INVENTORY, gate.INVENTORY_SHA)
    targets = [r for r in inventory["comparisons"] if r["kernel"] == KERNEL]
    require(len(targets) == 4, "missing recompute ranks")
    for row in targets:
        for label, warps in (("original", 4), ("fresh_shared", 8)):
            choice = row[label]
            config = choice["selected"]
            require(
                config["kwargs"] == {}
                and config["num_warps"] == warps
                and config["num_stages"] == 3,
                "unqualified recompute geometry",
            )
            gate.read_pinned(choice["path"], choice["sha256"])
            sources[choice["path"]] = choice["sha256"]
    manifest = dict(
        schema="kda-recompute-v1",
        cases=matrix(),
        warps=list(WARPS),
        sources=sources,
        history=shared["history"],
        git_commit=shared["git_commit"],
        targets=targets,
    )
    gate.save_new(path, manifest)
    print(f"Prepared {len(matrix())} pairs; {len(sources)} frozen receipts")


def verify(manifest):
    require(manifest["schema"] == "kda-recompute-v1", "wrong schema")
    require(
        manifest["cases"] == matrix() and manifest["warps"] == list(WARPS),
        "plan changed",
    )
    for name, digest in manifest["sources"].items():
        require(sha(name) == digest, f"frozen source changed: {name}")


def make_inputs(case, *, device):
    import torch

    lengths, tokens = case["lengths"], sum(case["lengths"])
    gen = torch.Generator().manual_seed(case["seed"])
    shape = (1, tokens, 16, 128)
    k = torch.randn(shape, generator=gen)
    k = (k / k.norm(dim=-1, keepdim=True)).bfloat16()
    v = (torch.randn(shape, generator=gen) * case["magnitude"]).bfloat16()
    A = torch.zeros((1, tokens, 16, 64), dtype=torch.bfloat16)
    starts, chunks = gate.sequence_metadata(lengths)
    for seq, chunk in chunks:
        lo = starts[seq] + chunk * 64
        count = min(64, starts[seq + 1] - lo)
        tile = torch.eye(64)[None].repeat(16, 1, 1)
        if case["regime"] == "conditioned":
            tile += 0.015625 * torch.randn((16, 64, 64), generator=gen).tril(-1)
        A[0, lo : lo + count] = tile[:, :count].permute(1, 0, 2).bfloat16()
    if case["regime"] == "identity":
        beta = torch.full((1, tokens, 16), 0.5)
        g = torch.zeros(shape)
    else:
        all_a, all_bias = gate.load_weights()
        rank = case["rank"]
        raw_g = (torch.randn(shape, generator=gen) * case["magnitude"]).bfloat16()
        raw_beta = torch.randn((1, tokens, 16), generator=gen).bfloat16()
        # Mathematical gate fixtures use the real repaired shard, not claimed
        # to be captured model activations or the previous probe's bit images.
        g, beta = gate.oracle(
            raw_g,
            raw_beta,
            all_a[rank * 16 : (rank + 1) * 16],
            all_bias[rank * 16 : (rank + 1) * 16],
            lengths,
        )
        g, beta = g.float(), beta.float()
    return dict(
        k=k.to(device),
        v=v.to(device),
        A=A.to(device),
        beta=beta.to(device),
        gk=g.to(device),
        starts=torch.tensor(starts, dtype=torch.int32, device=device),
        chunks=torch.tensor(chunks, dtype=torch.int32, device=device),
    )


def reference(inputs, lengths):
    """Batched float64 dot oracle with the source's BF16 operand round points."""
    import torch

    k, v, beta, g, A = (inputs[name] for name in ("k", "v", "beta", "gk", "A"))
    starts, chunks = gate.sequence_metadata(lengths)
    dev = k.device
    lo = torch.tensor([starts[s] + c * 64 for s, c in chunks], device=dev)
    end = torch.tensor([starts[s + 1] for s, c in chunks], device=dev)
    ix = lo[:, None] + torch.arange(64, device=dev)
    valid = ix < end[:, None]
    ix = ix.clamp_max(k.shape[1] - 1)

    def blocks(t):
        value = t[0, ix].double()
        mask = valid.reshape(*valid.shape, *([1] * (value.ndim - 2)))
        return value.masked_fill(~mask, 0)

    kk, vv, bb, gg = [blocks(t) for t in (k, v, beta, g)]
    aa = blocks(A).permute(0, 2, 1, 3)
    # For comparison against mathematical FP64, these two intermediate values
    # are narrowed exactly where the serving source narrows tensor-core operands.
    vb = (vv * bb[..., None]).bfloat16().double().permute(0, 2, 1, 3)
    kb = (kk * bb[..., None] * gg.exp2()).bfloat16().double().permute(0, 2, 1, 3)
    u = (aa @ vb).permute(0, 2, 1, 3)[valid].unsqueeze(0)
    w = (aa @ kb).permute(0, 2, 1, 3)[valid].unsqueeze(0)
    last = (end - lo).clamp_max(64) - 1
    g_last = gg[torch.arange(len(chunks), device=dev), last]
    kg = (kk * (g_last[:, None] - gg).exp2())[valid].unsqueeze(0)
    return w, u, kg


def launch(jit, inputs, outputs, warps):
    require(warps in WARPS, "unprescribed warps")
    k, v, beta, A, gk = (inputs[n] for n in ("k", "v", "beta", "A", "gk"))
    w, u, kg = outputs
    return jit[(len(inputs["chunks"]), 16)](
        q=None,
        k=k,
        qg=None,
        kg=kg,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        gk=gk,
        cu_seqlens=inputs["starts"],
        chunk_indices=inputs["chunks"],
        T=k.shape[1],
        H=16,
        K=128,
        V=128,
        BT=64,
        BK=64,
        BV=64,
        DOT_PRECISION="ieee",
        STORE_QG=False,
        STORE_KG=True,
        IS_VARLEN=True,
        num_warps=warps,
        num_stages=3,
    )


def metrics(actual, expected):
    import torch

    require(torch.isfinite(expected).all().item(), "nonfinite reference")
    result = compare(actual.reshape(-1, 128), expected.bfloat16().reshape(-1, 128))
    result["fp64_max_abs"] = (actual.double() - expected).abs().max().item()
    return result


def run(manifest_path, output):
    require(not output.exists(), "attempt exists; no retries")
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "GPU0 required")
    require(not gate.gpu_query(), "another GPU workload is active")
    output.mkdir(parents=True)
    (output / "binaries").mkdir()
    os.environ.update(
        TRITON_CACHE_DIR=str(output.resolve() / "triton"),
        TORCHINDUCTOR_CACHE_DIR=str(output.resolve() / "inductor"),
        TRITON_CACHE_AUTOTUNING="0",
    )
    summary = dict(
        status="running",
        manifest_sha256=sha(manifest_path),
        gpu_config=gate.gpu_config(),
        records=[],
        binaries={},
    )
    gate.save_new(
        output / "attempt.json",
        dict(pid=os.getpid(), manifest_sha256=sha(manifest_path)),
    )
    try:
        import torch
        from triton.runtime.jit import JITFunction

        from vllm.models.kimi_k3.amd.ops.third_party.kda.chunk import (
            recompute_w_u_fwd_kernel,
        )

        torch.set_num_threads(4)
        require(torch.cuda.get_device_capability() == (12, 0), "SM120 required")
        jit = recompute_w_u_fwd_kernel.fn.fn
        require(
            isinstance(jit, JITFunction) and jit.__name__ == KERNEL, "wrong serving JIT"
        )
        for index, case in enumerate(matrix()):
            inputs = make_inputs(case, device="cuda")
            baseline = {name: t.clone() for name, t in inputs.items()}
            outputs, guards, graphs, binary_ids = [], [], [], []
            for warps in WARPS:
                stores = [
                    torch.full(
                        (inputs["k"].numel() + 64,),
                        123,
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    for _ in OUTPUTS
                ]
                values = tuple(t[32:-32].view_as(inputs["k"]) for t in stores)
                compiled = launch(jit, inputs, values, warps)
                digest = hashlib.sha256(compiled.asm["cubin"]).hexdigest()
                if digest not in summary["binaries"]:
                    with (output / "binaries" / f"{digest}.cubin").open("xb") as f:
                        f.write(compiled.asm["cubin"])
                require(
                    compiled.metadata.num_warps == warps
                    and compiled.metadata.num_stages == 3,
                    "wrong compiled config",
                )
                summary["binaries"][digest] = dict(
                    warps=warps, stages=3, hash=compiled.hash
                )
                first = [t.clone() for t in values]
                launch(jit, inputs, values, warps)
                require(
                    all(torch.equal(a, b) for a, b in zip(first, values)),
                    "eager repeat mismatch",
                )
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch(jit, inputs, values, warps)
                graph.replay()
                require(
                    all(torch.equal(a, b) for a, b in zip(first, values)),
                    "graph mismatch",
                )
                outputs.append(values)
                graphs.append(graph)
                guards.append(stores)
                binary_ids.append(digest)
            record = dict(index=index, **case, phases=[], binary_ids=binary_ids)
            for phase in range(2):
                if phase:
                    inputs["k"].mul_(0.5)
                    inputs["v"].mul_(-0.5)
                    baseline["k"].mul_(0.5)
                    baseline["v"].mul_(-0.5)
                    for graph in graphs:
                        graph.replay()
                    for warps, values in zip(WARPS, outputs):
                        changed = [t.clone() for t in values]
                        launch(jit, inputs, values, warps)
                        require(
                            all(torch.equal(a, b) for a, b in zip(changed, values)),
                            "changed graph mismatch",
                        )
                require(
                    all(torch.equal(t, baseline[name]) for name, t in inputs.items()),
                    "input mutation",
                )
                refs = reference(inputs, case["lengths"])
                row = dict(
                    phase=phase,
                    arms=[],
                    pairwise=[
                        compare(a.reshape(-1, 128), b.reshape(-1, 128))
                        for a, b in zip(*outputs)
                    ],
                )
                for warps, values, stores in zip(WARPS, outputs, guards):
                    require(
                        all(
                            torch.all(t[:32] == 123).item()
                            and torch.all(t[-32:] == 123).item()
                            for t in stores
                        ),
                        "guard overwritten",
                    )
                    row["arms"].append(
                        dict(
                            warps=warps,
                            reference=[metrics(a, b) for a, b in zip(values, refs)],
                            hashes=[tensor_sha(t.cpu()) for t in values],
                        )
                    )
                record["phases"].append(row)
                del refs
            record["repeat_replay_mutation_guards_passed"] = True
            gate.save_new(output / f"pair-{index:03d}.json", record)
            summary["records"].append(
                dict(
                    path=f"pair-{index:03d}.json",
                    sha256=sha(output / f"pair-{index:03d}.json"),
                )
            )
            if case["regime"] == "identity":
                require(
                    all(
                        r["bit_mismatches"] == 0
                        for p in record["phases"]
                        for arm in p["arms"]
                        for r in arm["reference"]
                    ),
                    "exact identity oracle failed",
                )
            require(
                all(
                    record["phases"][0]["arms"][a]["hashes"]
                    != record["phases"][1]["arms"][a]["hashes"]
                    for a in range(2)
                ),
                "unchanged outputs after input change",
            )
            del inputs, baseline, outputs, guards, graphs, first, values, stores, graph
            if (index + 1) % 14 == 0:
                print(f"Completed {index + 1}/{len(matrix())} pairs", flush=True)
        verify(manifest)
        summary["status"] = "complete"
    except BaseException as exc:
        summary.update(status="failed", error=repr(exc))
        raise
    finally:
        gate.save_new(output / "summary.json", summary)


def audit_record(record, index, binaries, *, allow_identity_failure=False):
    case = matrix()[index]
    require(
        record["index"] == index and {k: record[k] for k in case} == case, "wrong case"
    )
    require(
        record["repeat_replay_mutation_guards_passed"] is True, "launch check failed"
    )
    require([p["phase"] for p in record["phases"]] == [0, 1], "missing phase")
    require(len(record["binary_ids"]) == 2, "missing binary")
    for digest, warps in zip(record["binary_ids"], WARPS):
        require(
            binaries[digest]["warps"] == warps and binaries[digest]["stages"] == 3,
            "wrong binary",
        )
    for phase in record["phases"]:
        require([a["warps"] for a in phase["arms"]] == list(WARPS), "wrong arms")
        require(len(phase["pairwise"]) == 3, "missing outputs")
        for arm in phase["arms"]:
            require(
                len(arm["reference"]) == len(arm["hashes"]) == 3, "missing reference"
            )
        for slot, pair in enumerate(phase["pairwise"]):
            require(pair["elements"] == sum(case["lengths"]) * 2048, "wrong extent")
            require(
                0 <= pair["bit_mismatches"] <= pair["elements"], "wrong mismatch count"
            )
            same = phase["arms"][0]["hashes"][slot] == phase["arms"][1]["hashes"][slot]
            require(same == (pair["bit_mismatches"] == 0), "hash/count mismatch")
        if case["regime"] == "identity" and not allow_identity_failure:
            require(
                all(
                    r["bit_mismatches"] == 0
                    for a in phase["arms"]
                    for r in a["reference"]
                ),
                "identity failure",
            )


def audit(manifest_path, output):
    manifest = json.loads(manifest_path.read_text())
    verify(manifest)
    summary = json.loads((output / "summary.json").read_text())
    require(summary["manifest_sha256"] == sha(manifest_path), "manifest changed")
    require(
        not gate.gpu_query() and summary["gpu_config"] == gate.gpu_config(),
        "GPU release/config failed",
    )
    require(not list((output / "triton").rglob("*.autotune.json")), "unexpected tuning")
    for digest in summary["binaries"]:
        require(
            sha(output / "binaries" / f"{digest}.cubin") == digest, "binary changed"
        )
    records = []
    for index, entry in enumerate(summary["records"]):
        require(entry["path"] == f"pair-{index:03d}.json", "wrong record path")
        record = gate.read_pinned(output / entry["path"], entry["sha256"])
        audit_record(
            record,
            index,
            summary["binaries"],
            allow_identity_failure=summary["status"] == "failed",
        )
        records.append(record)
    complete = summary["status"] == "complete" and len(records) == len(matrix())
    result = dict(
        status="complete" if complete else "terminal-failure",
        pairs=len(records),
        identity_oracle_passed=all(
            metric["bit_mismatches"] == 0
            for record in records
            if record["regime"] == "identity"
            for phase in record["phases"]
            for arm in phase["arms"]
            for metric in arm["reference"]
        )
        if records
        else None,
        sources_verified=len(manifest["sources"]),
        binaries_verified=len(summary["binaries"]),
        summary_sha256=sha(output / "summary.json"),
        elements_per_output=sum(
            p["pairwise"][0]["elements"] for r in records for p in r["phases"]
        ),
        changed={
            name: sum(
                p["pairwise"][i]["bit_mismatches"] for r in records for p in r["phases"]
            )
            for i, name in enumerate(OUTPUTS)
        },
        gpu_processes_after="",
        limitation=__doc__,
    )
    gate.save_new(output / "analysis.json", result)
    print(json.dumps(result, indent=2))


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
