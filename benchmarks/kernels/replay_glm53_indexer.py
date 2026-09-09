#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay captured first-layer GLM indexer inputs; no timing/quality claim.

Fixed protocol: each original GPU, one warmup + five eager + five graph calls,
alternating NaN/123 in undefined tails. Save EVERY output. The captured input
has no ambiguous cutoff ties, so selected sets must equal the validated capture;
raw output order is allowed to vary. Never substitute a new serving start.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_sha(value):
    return hashlib.sha256(
        memoryview(value.contiguous().reshape(-1).view(torch.uint8).numpy())
    ).hexdigest()


def verified_selection(logits, starts, ends, indices, k=512):
    """Independent CPU score-set, bounds, uniqueness and cutoff-tie checks."""
    rows, columns = logits.shape
    assert columns >= k and logits.dtype == torch.float32
    assert starts.shape == ends.shape == (rows,)
    assert starts.dtype == ends.dtype == indices.dtype == torch.int32
    assert indices.shape == (rows, k)
    assert bool((starts == 0).all())
    assert bool(((ends >= 0) & (ends <= columns)).all())
    visible = torch.arange(columns)[None, :] < ends[:, None]
    assert bool(torch.isfinite(logits[visible]).all())
    assert bool((logits[~visible] == 0).all())
    valid = torch.arange(k)[None, :] < ends.clamp(max=k)[:, None]
    assert torch.equal(indices >= 0, valid)
    assert bool((indices[~valid] == -1).all())
    assert bool(((indices < ends[:, None]) | ~valid).all())
    ordered = indices.sort(dim=1).values
    assert bool(((ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] == -1)).all())
    values = logits.gather(1, indices.clamp_min(0).long()).masked_fill(
        ~valid, -torch.inf
    )
    best = logits.masked_fill(~visible, -torch.inf).topk(k, dim=1).values
    assert torch.equal(values.sort(dim=1, descending=True).values, best)
    cutoff = best[:, -1]
    greater = ((logits > cutoff[:, None]) & visible).sum(dim=1)
    equal = ((logits == cutoff[:, None]) & visible).sum(dim=1)
    ambiguous = (ends > k) & (equal > k - greater)
    return ordered, int(ambiguous.sum())


def replay_device(captured, expected, device, operation):
    rows, columns = captured["logits"].shape
    records, outputs = [], []
    with torch.cuda.device(device):
        logits, starts, ends = (
            captured[k].to(device) for k in ("logits", "starts", "ends")
        )
        indices = torch.empty_like(captured["indices"], device=device)
        undefined = torch.arange(columns, device=device)[None, :] >= ends[:, None]

        def call():
            operation(logits, starts, ends, indices, rows, columns, 1, 512)

        def observed_call(mode, repeat, callback):
            poison = float("nan") if repeat % 2 == 0 else 123.0
            logits.masked_fill_(undefined, poison)
            before = logits.view(torch.uint8).clone()
            indices.fill_(-777)
            callback()
            assert torch.equal(before, logits.view(torch.uint8)), (
                "native modified logits"
            )
            actual = indices.cpu()
            assert torch.equal(actual.sort(dim=1).values, expected), (
                "selected set changed"
            )
            changed = actual != captured["indices"]
            records.append(
                {
                    "mode": mode,
                    "repeat": repeat,
                    "undefined_tail_poison": "NaN" if repeat % 2 == 0 else "123",
                    "selected_set_equals_validated_capture": True,
                    "changed_order_positions_vs_capture": int(changed.sum()),
                    "changed_order_rows_vs_capture": int(changed.any(dim=1).sum()),
                    "sha256": tensor_sha(actual),
                }
            )
            outputs.append(actual)

        observed_call("warmup", 0, call)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.graph(graph, stream=stream):
            call()
        torch.cuda.current_stream(device).wait_stream(stream)
        for mode, callback in (("eager", call), ("graph", graph.replay)):
            for repeat in range(5):
                observed_call(mode, repeat, callback)
        torch.cuda.synchronize(device)
        del graph, stream
    return records, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPU compute processes already active: {active}")
    from slimserve.index_journal import enabled
    from vllm import _custom_ops as ops

    assert not enabled(), "replay must run without the serving observer"
    assert torch.cuda.device_count() == 4
    assert all(torch.cuda.get_device_capability(i) == (12, 0) for i in range(4))
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    run = args.run.resolve()
    receipt = json.loads((run / "summary.json").read_text())
    assert receipt["status"] == "complete" and receipt["diagnostic_only"]
    native = "vllm/_C_stable_libtorch.abi3.so"
    assert sha(root / native) == receipt["runtime"]["native_sha256"][native]
    paths = list((run / "trace").glob("index-*.jsonl"))
    assert len(paths) == 4
    journals = {}
    for path in paths:
        events = [json.loads(line) for line in path.read_text().splitlines()]
        forward = next(e for e in events if e["kind"] == "forward_begin")
        device = int(forward["device"].split(":")[-1])
        assert device not in journals
        assert forward["tokens"] == 7616 and forward["computed_tokens"] == 0
        journals[device] = path, events
    assert set(journals) == set(range(4))
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "running",
        "diagnostic_only": True,
        "protocol": (
            "each original GPU: 1 warmup +5 eager +5 graph; poison NaN/123 tails"
        ),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "implementation_sha256": {
            name: sha(root / name)
            for name in (
                "benchmarks/kernels/replay_glm53_indexer.py",
                "vllm/_custom_ops.py",
                "slimserve/index_journal.py",
            )
        },
        "native_sha256": sha(root / native),
        "torch": torch.__version__,
        "source_receipt": {
            "path": str(run / "summary.json"),
            "sha256": sha(run / "summary.json"),
        },
        "devices": [],
    }

    def save():
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    for device in range(4):
        path, events = journals[device]
        assert len([e for e in events if e["kind"] == "request_complete"]) == 3
        captured, inputs = {}, []
        for stage in ("logits", "starts", "ends", "indices"):
            matches = [
                e
                for e in events
                if e["kind"] == "tensor"
                and e["chunk"] == 1
                and e["stage"] == f"layer-03.call-0.{stage}"
            ]
            assert [m["match"] for m in matches] == [1, 2, 3]
            if stage != "indices":
                assert len({m["sha256"] for m in matches}) == 1
            item = matches[0]
            archive = Path(item["archive"]["path"])
            assert sha(archive) == item["archive"]["sha256"]
            value = torch.load(archive, weights_only=True, map_location="cpu")
            assert tensor_sha(value) == item["sha256"]
            captured[stage] = value
            inputs.append(
                {"stage": stage, "path": str(archive), "sha256": sha(archive)}
            )
        expected, ambiguous = verified_selection(**captured)
        assert ambiguous == 0, "protocol requires unambiguous captured score sets"
        rows, columns = captured["logits"].shape
        assert (rows, columns) == (7616, 1904)
        records, outputs = replay_device(
            captured, expected, device, ops.top_k_per_row_prefill
        )
        output = args.output / f"gpu-{device}-all11-outputs.pt"
        with output.open("xb") as stream:
            torch.save(torch.stack(outputs), stream)
        result["devices"].append(
            {
                "device": device,
                "gpu": torch.cuda.get_device_name(device),
                "journal_sha256": sha(path),
                "inputs": inputs,
                "observations": records,
                "output_archive": {
                    "path": str(output),
                    "sha256": sha(output),
                    "bytes": output.stat().st_size,
                },
            }
        )
        save()
        print(
            json.dumps(
                {
                    "device": device,
                    "verified_calls": len(records),
                    "distinct_output_orders": len({r["sha256"] for r in records}),
                }
            ),
            flush=True,
        )
    result["status"] = "complete"
    assert sha(root / native) == result["native_sha256"]
    save()


if __name__ == "__main__":
    main()
