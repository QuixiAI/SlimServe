# SPDX-License-Identifier: Apache-2.0
"""Fixed native-control/canonical-tie replay of actual layer23 captures.

Each original GPU: each selector gets one warmup, five eager and five graph
calls. Both use canonical output order. Preserve ALL88 output matrices. This
is a correctness/repeatability diagnostic, not a benchmark or quality claim.
"""

import argparse
import json
import subprocess
from pathlib import Path

import torch

from benchmarks.kernels.replay_glm53_indexer import sha, tensor_sha, verified_selection


def canonical_selection(logits, starts, ends, k=512):
    """Independent CPU score-descending, ID-ascending oracle; zeros compare equal."""
    assert logits.device.type == "cpu" and logits.dtype == torch.float32
    assert starts.dtype == ends.dtype == torch.int32
    rows, columns = logits.shape
    assert starts.shape == ends.shape == (rows,)
    assert (starts == 0).all() and ((ends >= 0) & (ends <= columns)).all()
    result = torch.full((rows, k), -1, dtype=torch.int32)
    for row, end in enumerate(ends.tolist()):
        assert torch.isfinite(logits[row, :end]).all()
        selected = logits[row, :end].argsort(descending=True, stable=True)[:k]
        result[row, : len(selected)] = selected.sort().values.int()
    return result


def replay_device(captured, expected, device, label, operation, canonicalize):
    rows, columns = captured["logits"].shape
    observations, outputs = [], []
    with torch.cuda.device(device):
        logits, starts, ends = (
            captured[k].to(device) for k in ("logits", "starts", "ends")
        )
        indices = torch.empty_like(captured["indices"], device=device)
        undefined = torch.arange(columns, device=device)[None, :] >= ends[:, None]

        def call():
            operation(logits, starts, ends, indices, rows, columns, 1, 512)
            canonicalize(indices)

        def observe(mode, repeat, callback):
            logits.masked_fill_(undefined, float("nan") if repeat % 2 == 0 else 123.0)
            before = logits.view(torch.uint8).clone()
            indices.fill_(-777)
            callback()
            assert torch.equal(before, logits.view(torch.uint8))
            actual = indices.cpu()
            _, actual_ties = verified_selection(
                captured["logits"], captured["starts"], captured["ends"], actual
            )
            assert actual_ties == 1
            ordered = actual.where(actual >= 0, 2147483647)
            assert torch.equal(ordered, ordered.sort(dim=1).values)
            changed = (actual != expected).any(dim=1).nonzero().flatten().tolist()
            assert changed in ([], [6329])
            if label == "canonical-ties":
                assert not changed
            observations.append(
                dict(
                    selector=label,
                    mode=mode,
                    repeat=repeat,
                    undefined_poison="NaN" if repeat % 2 == 0 else "123",
                    all_score_sets_valid=True,
                    canonical_oracle_equal=not changed,
                    changed_rows_vs_canonical=changed,
                    sha256=tensor_sha(actual),
                )
            )
            outputs.append(actual)

        observe("warmup", 0, call)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
        torch.cuda.current_stream(device).wait_stream(stream)
        for mode, callback in (("eager", call), ("graph", graph.replay)):
            for repeat in range(5):
                observe(mode, repeat, callback)
        torch.cuda.synchronize(device)
        del graph, stream
    return observations, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference-native", required=True, type=Path)
    parser.add_argument("--native-comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    assert not active, f"GPU compute processes already active: {active}"
    from slimserve.canonical_indexer import canonicalize, enabled
    from vllm import _custom_ops as ops

    assert not enabled(), "replay must not inherit serving diagnostic flags"
    assert torch.cuda.device_count() == 4
    assert all(torch.cuda.get_device_capability(i) == (12, 0) for i in range(4))
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    run = args.run.resolve()
    receipt = json.loads((run / "summary.json").read_text())
    assert receipt["status"] == "complete" and receipt["diagnostic_only"]
    assert receipt["git_commit"] == "4cead10f6fcb12a199f1b65c47f28929cbf271e2"
    native = "vllm/_C_stable_libtorch.abi3.so"
    assert sha(args.reference_native) == receipt["runtime"]["native_sha256"][native]
    proof = json.loads(args.native_comparison.read_text())
    assert proof["before_binary"]["sha256"] == sha(args.reference_native)
    assert proof["candidate_binary"]["sha256"] == sha(root / native)
    assert not proof["changed_common_functions"] and not proof["removed_functions"]
    assert proof["identical_common_functions"] == proof["before_functions"]
    journals = {}
    for path in (run / "trace").glob("index-*.jsonl"):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert events[0]["capture_layer"] == 23
        assert events[0]["selection_order"] == "canonical-pool-id"
        forward = next(e for e in events if e["kind"] == "forward_begin")
        device = int(forward["device"].split(":")[-1])
        assert device not in journals
        assert len([e for e in events if e["kind"] == "request_complete"]) == 3
        journals[device] = path, events
    assert set(journals) == set(range(4))
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="running",
        diagnostic_only=True,
        protocol=(
            "each original GPU, each selector:1 warmup+5 eager+5 graph; "
            "canonical output order"
        ),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        implementation_sha256={
            name: sha(root / name)
            for name in (
                "benchmarks/kernels/replay_glm53_indexer_ties.py",
                "benchmarks/kernels/replay_glm53_indexer.py",
                "slimserve/canonical_indexer.py",
                "slimserve/canonical_indexer_kernel.py",
                "vllm/_custom_ops.py",
                "csrc/libtorch_stable/sampler.cu",
            )
        },
        native_sha256=sha(root / native),
        reference_native_sha256=sha(args.reference_native),
        native_comparison=dict(
            path=str(args.native_comparison), sha256=sha(args.native_comparison)
        ),
        source_receipt=dict(
            path=str(run / "summary.json"), sha256=sha(run / "summary.json")
        ),
        torch=torch.__version__,
        devices=[],
    )

    def save():
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    for device in range(4):
        path, events = journals[device]
        captured, files = {}, []
        for stage in ("logits", "starts", "ends", "indices"):
            items = [
                e
                for e in events
                if e["kind"] == "tensor"
                and e["chunk"] == 1
                and e["stage"] == f"layer-23.call-0.{stage}"
            ]
            assert [e["match"] for e in items] == [1, 2, 3]
            if stage != "indices":
                assert len({e["sha256"] for e in items}) == 1
            item = items[0]
            archive = Path(item["archive"]["path"])
            assert sha(archive) == item["archive"]["sha256"]
            assert archive.stat().st_size == item["archive"]["bytes"]
            value = torch.load(archive, weights_only=True, map_location="cpu")
            assert tensor_sha(value) == item["sha256"]
            captured[stage] = value
            files.append(dict(stage=stage, path=str(archive), sha256=sha(archive)))
        _, ties = verified_selection(**captured)
        assert ties == 1 and captured["logits"].shape == (7616, 1904)
        expected = canonical_selection(
            *(captured[k] for k in ("logits", "starts", "ends"))
        )
        assert 994 in expected[6329] and 1398 not in expected[6329]
        observations, outputs = [], []
        for label, operation in (
            ("native", ops.top_k_per_row_prefill),
            ("canonical-ties", ops.glm53_top_k_per_row_prefill),
        ):
            recorded, matrices = replay_device(
                captured, expected, device, label, operation, canonicalize
            )
            observations.extend(recorded)
            outputs.extend(matrices)
        archive = args.output / f"gpu-{device}-all22-outputs.pt"
        with archive.open("xb") as stream:
            torch.save(torch.stack(outputs), stream)
        counts = {
            label: len({r["sha256"] for r in observations if r["selector"] == label})
            for label in ("native", "canonical-ties")
        }
        result["devices"].append(
            dict(
                device=device,
                journal=str(path),
                journal_sha256=sha(path),
                inputs=files,
                canonical_oracle_sha256=tensor_sha(expected),
                observations=observations,
                distinct_output_sets=counts,
                output_archive=dict(
                    path=str(archive), sha256=sha(archive), bytes=archive.stat().st_size
                ),
            )
        )
        save()
        print(
            json.dumps(
                dict(
                    device=device,
                    verified_calls=len(observations),
                    distinct_output_sets=counts,
                )
            ),
            flush=True,
        )
    assert sha(root / native) == result["native_sha256"]
    for name, digest in result["implementation_sha256"].items():
        assert sha(root / name) == digest
    result["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
