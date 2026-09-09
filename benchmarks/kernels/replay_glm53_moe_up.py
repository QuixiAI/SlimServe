#!/usr/bin/env python3
"""Replay captured layer3 gate/up inputs; vary only saved assignment ordering."""

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch

    import vllm._custom_ops  # noqa: F401 -- register installed native operators
    from vllm.scalar_type import scalar_types

    torch.set_num_threads(1)
    result = {
        "diagnostic_only": True,
        "status": "running",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ),
        "script_sha256": sha(Path(__file__)),
        "native_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in sorted((ROOT / "vllm").glob("*_C*.so"))
        },
        "capture": str(args.capture.resolve()),
        "protocol": "all4 ranks; 3 saved layouts each; 8 eager repeats per layout; "
        "one graph per rank, layouts 1/2/3/3/2/1/1/2/3; no timing claim",
        "workers": [],
    }

    def save():
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save()

    def bits(t):
        return t.contiguous().view(torch.uint8)

    def metrics(a, b):
        av, bv = a.float(), b.float()
        d = bv - av
        indices = (
            (bits(a).reshape(*a.shape, 2) != bits(b).reshape(*b.shape, 2))
            .any(-1)
            .nonzero()
        )
        return {
            "changed_elements": len(indices),
            "elements": a.numel(),
            "max_abs": float(d.abs().max()),
            "rms_delta": float(d.square().mean().sqrt()),
            "first_changed_coordinates": indices[:16].tolist(),
            "first_values": [
                [float(av[tuple(i)]), float(bv[tuple(i)])]
                for i in indices[:16].tolist()
            ],
        }

    journals = sorted((args.capture / "trace").glob("model-*.jsonl"))
    assert len(journals) == 4
    for rank, journal in enumerate(journals):
        rows = [json.loads(line) for line in journal.read_text().splitlines()]
        tensors = {(r["match"], r["stage"]): r for r in rows if r["kind"] == "tensor"}
        devices = {r["device"] for r in rows if r["kind"] == "tensor"}
        assert devices == {f"cuda:{rank}"}
        torch.cuda.set_device(rank)
        assert torch.cuda.get_device_capability() == (12, 0)
        params = [r for r in rows if r["kind"] == "moe_gemm" and r["stage"] == "up"]
        assert len(params) == 3
        assert all(
            {k: v for k, v in p.items() if k != "match"}
            == {k: v for k, v in params[0].items() if k != "match"}
            for p in params
        )
        meta = params[0]
        assert meta["b_q_type_id"] == scalar_types.float4_e2m1f.id
        p = meta["parameters"]
        assert (
            p["size_m"],
            p["size_n"],
            p["size_k"],
            p["top_k"],
            p["moe_block_size"],
        ) == (640, 1024, 4096, 8, 32)
        assert not p["use_atomic_add"] and p["use_fp32_reduce"]
        files = {
            (r["match"], r["stage"]): r for r in rows if r["kind"] == "moe_snapshot"
        }

        def load(stage, match=1, files=files, tensors=tensors, rank=rank):
            stage = "moe3.up." + stage
            info = files[match, stage]
            path = Path(info["path"])
            assert sha(path) == info["file_sha256"]
            t = torch.load(path, map_location="cpu", weights_only=True)
            assert (
                hashlib.sha256(
                    memoryview(t.reshape(-1).view(torch.uint8).numpy())
                ).hexdigest()
                == tensors[match, stage]["sha256"]
            )
            return t.cuda(rank)

        for stage in (
            "input",
            "b_qweight",
            "b_scales",
            "global_scale",
            "topk_weights",
            "workspace_before",
            "workspace_after",
            "expert_ids",
            "padded_count",
        ):
            assert (
                len({tensors[m, "moe3.up." + stage]["sha256"] for m in (1, 2, 3)}) == 1
            )
        x, w, s, g = [
            load(k) for k in ("input", "b_qweight", "b_scales", "global_scale")
        ]
        topw, workspace = load("topk_weights"), load("workspace_before")
        assert not torch.count_nonzero(workspace)
        experts, count = load("expert_ids"), load("padded_count")
        layouts = [load("sorted_ids", m) for m in (1, 2, 3)]
        expected = [load("output", m) for m in (1, 2, 3)]
        # Restore the production allocation capacities. Only saved, defined
        # prefixes enter computation; unused capacity gets harmless sentinels.
        capacity = 640 * 8 + 288 * 31
        sorted_ids = torch.full((capacity,), 5120, dtype=torch.int32, device=rank)
        expert_ids = torch.full(
            ((capacity + 31) // 32,), -1, dtype=torch.int32, device=rank
        )
        expert_ids[: experts.numel()].copy_(experts)
        output = torch.empty_like(expected[0])
        optional = {
            key: None
            if spec is None
            else torch.empty(
                spec["shape"],
                dtype=getattr(torch, spec["dtype"].removeprefix("torch.")),
                device=rank,
            )
            for key, spec in meta["optional"].items()
        }

        def call(
            x=x,
            output=output,
            w=w,
            s=s,
            g=g,
            optional=optional,
            workspace=workspace,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            count=count,
            topw=topw,
            p=p,
            meta=meta,
        ):
            return torch.ops._moe_C.moe_wna16_marlin_gemm(
                x,
                output,
                w,
                optional["b_bias"],
                s,
                optional["a_scales"],
                g,
                optional["b_qzeros"],
                optional["g_idx"],
                optional["perm"],
                workspace,
                sorted_ids,
                expert_ids,
                count,
                topw,
                p["moe_block_size"],
                p["top_k"],
                p["mul_topk_weights"],
                meta["b_q_type_id"],
                p["size_m"],
                p["size_n"],
                p["size_k"],
                p["is_k_full"],
                p["use_atomic_add"],
                p["use_fp32_reduce"],
                p["is_zp_float"],
                p["thread_k"],
                p["thread_n"],
                p["blocks_per_sm"],
            )

        worker = {
            "rank": rank,
            "pid": rows[0]["pid"],
            "journal_sha256": sha(journal),
            "gemm_metadata": meta,
            "eager": [],
            "graph": [],
        }
        result["workers"].append(worker)
        for m, layout in enumerate(layouts, 1):
            sorted_ids[: layout.numel()].copy_(layout)
            outputs = []
            for repeat in range(8):
                # Alternate the previous output; detect unintended accumulation.
                output.fill_(float("nan") if repeat % 2 else 123)
                got = call().clone()
                torch.cuda.synchronize()
                assert torch.isfinite(got).all()
                lock_ok = not bool(torch.count_nonzero(workspace))
                same = torch.equal(bits(got), bits(expected[m - 1]))
                repeated = not outputs or torch.equal(bits(got), bits(outputs[0]))
                item = {
                    "layout": m,
                    "repeat": repeat + 1,
                    "locks_zero": lock_ok,
                    "matches_serving_capture_bits": same,
                    "matches_first_repeat_bits": repeated,
                }
                if not same:
                    item["vs_capture"] = metrics(expected[m - 1], got)
                worker["eager"].append(item)
                if not outputs:
                    outputs.append(got)
            save()
        graph = torch.cuda.CUDAGraph()
        # torch.cuda.graph's implicit capture stream is cached once process-wide;
        # a multi-device replay must supply a stream on this rank's device.
        capture_stream = torch.cuda.Stream(device=rank)
        capture_stream.wait_stream(torch.cuda.current_stream(rank))
        with torch.cuda.graph(graph, stream=capture_stream):
            graph_output = call()
        for m in (1, 2, 3, 3, 2, 1, 1, 2, 3):
            sorted_ids[: layouts[m - 1].numel()].copy_(layouts[m - 1])
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            worker["graph"].append(
                {
                    "layout": m,
                    "matches_serving_capture_bits": torch.equal(
                        bits(graph_output), bits(expected[m - 1])
                    ),
                    "locks_zero": not bool(torch.count_nonzero(workspace)),
                }
            )
        worker["layout_pairs"] = [
            {"layouts": [a + 1, b + 1], **metrics(expected[a], expected[b])}
            for a, b in itertools.combinations(range(3), 2)
        ]
        worker["all_exact"] = all(
            r["matches_serving_capture_bits"] and r["locks_zero"]
            for r in worker["eager"] + worker["graph"]
        )
        print(
            json.dumps(
                {
                    "rank": rank,
                    "all_exact": worker["all_exact"],
                    "layout_pairs": worker["layout_pairs"],
                }
            ),
            flush=True,
        )
        save()
        del call, load
        del (
            graph,
            graph_output,
            outputs,
            expected,
            layouts,
            x,
            w,
            s,
            g,
            topw,
            workspace,
            output,
        )
        torch.cuda.empty_cache()
    assert result["native_sha256"] == {
        p: sha(ROOT / p) for p in result["native_sha256"]
    }
    result["status"] = "complete"
    result["all_exact"] = all(w["all_exact"] for w in result["workers"])
    save()


if __name__ == "__main__":
    main()
