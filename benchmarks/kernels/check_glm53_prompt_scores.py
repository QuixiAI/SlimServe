# SPDX-License-Identifier: Apache-2.0
"""One prescribed CUDA process: prompt-score exactness, memory and timing.

No model projection, serving qualification, retry or alternative sampler. Uses
the actual Sampler, including its compiled inclusive-rank reduction.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

from vllm.v1.sample.prompt_logprobs import MODES, gather_prompt_logprobs
from vllm.v1.sample.sampler import Sampler

SOURCES = (
    __file__,
    "vllm/v1/sample/prompt_logprobs.py",
    "vllm/v1/sample/sampler.py",
    "vllm/v1/sample/ops/logprobs.py",
    "tests/slimserve/test_prompt_score_chunks.py",
    "perf/glm53-prompt-score-protocol.md",
)


def sources():
    return {
        str(Path(name).resolve()): hashlib.sha256(Path(name).read_bytes()).hexdigest()
        for name in SOURCES
    }


def gpu_query(fields, kind="gpu"):
    return subprocess.check_output(
        ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()


def cases():
    result = []
    for rows in (1, 639, 1024, 1025, 2051, 7616):
        for mode in MODES:
            for count in (0, 5):
                result.append(
                    dict(
                        rows=rows,
                        vocab=154880,
                        mode=mode,
                        count=count,
                        dtype="bfloat16",
                        equal=False,
                    )
                )
    for dtype in ("float16", "float32"):
        for mode in ("raw_logprobs", "raw_logits"):
            result.append(
                dict(
                    rows=1025,
                    vocab=154880,
                    mode=mode,
                    count=5,
                    dtype=dtype,
                    equal=False,
                )
            )
    for vocab, count in ((154880, 5), (17, 17)):
        for mode in MODES:
            result.append(
                dict(
                    rows=1025,
                    vocab=vocab,
                    mode=mode,
                    count=count,
                    dtype="bfloat16",
                    equal=True,
                )
            )
    return result


def reference(logits, targets, case):
    scores = (
        logits.float()
        if case["mode"].endswith("logits")
        else Sampler.compute_logprobs(logits)
    )
    return Sampler.gather_logprobs(scores, case["count"], targets)


def invoke(arm, logits, targets, case):
    if arm == "control":
        return reference(logits, targets, case)
    return gather_prompt_logprobs(
        logits, targets, case["count"], case["mode"], sampler=Sampler
    )


def measured(arm, logits, targets, case):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin = time.perf_counter()
    start.record()
    output = invoke(arm, logits, targets, case)
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - begin) * 1000
    record = dict(
        cuda_ms=start.elapsed_time(end),
        wall_ms=wall_ms,
        allocated_before=before,
        peak_extra_bytes=torch.cuda.max_memory_allocated() - before,
    )
    host = [tensor.cpu() for tensor in output[:3]]
    del output
    return host, record


def same(actual, expected):
    return all(
        a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        for a, b in zip(actual, expected, strict=True)
    )


def allocation_trace(arm, logits, targets, case, directory):
    # This is separate from timing and occurs only after both paths are warm.
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.memory._record_memory_history(max_entries=10000)
    try:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            record_shapes=True,
            with_stack=True,
        ) as profile:
            output = invoke(arm, logits, targets, case)
            torch.cuda.synchronize()
            del output
        snapshot = torch.cuda.memory._snapshot()
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    profile.export_chrome_trace(str(directory / f"{arm}-allocation-trace.json"))
    events = [
        dict(
            name=e.name,
            shapes=e.input_shapes,
            device_memory_usage=e.device_memory_usage,
            stack=e.stack,
        )
        for e in profile.events()
        if e.device_memory_usage >= 100 * 1024**2
    ]
    allocations = [
        event
        for device in snapshot["device_traces"]
        for event in device
        if event["action"] == "alloc" and event["size"] >= 100 * 1024**2
    ]
    return dict(
        operators=events,
        allocations=allocations,
        scope="isolated warmed sampler; not attribution of live-server OOM",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="started",
        sources=sources(),
        cases=cases(),
        results=[],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        pid=os.getpid(),
        repetitions=3,
        warmups_per_arm=1,
        arm_order=["control", "chunked"],
        caches={
            key: os.environ.get(key)
            for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
        },
    )

    def save():
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        assert all(result["caches"].values()), "private caches required"
        assert not gpu_query("pid", "compute-apps"), "GPU workload already active"
        fields = "uuid,name,driver_version,power.limit"
        result["hardware_before"] = gpu_query(fields)
        assert torch.cuda.device_count() == 1
        assert torch.cuda.get_device_capability() == (12, 0)
        result["device"] = torch.cuda.get_device_name()
        assert "RTX PRO 6000" in result["device"]
        torch.manual_seed(530910)
        for number, case in enumerate(result["cases"]):
            # Both offset rows and padded vocabulary test noncontiguous head views.
            rows, vocab = case["rows"], case["vocab"]
            backing = torch.empty(
                (rows + 2, vocab + 8),
                device="cuda",
                dtype=getattr(torch, case["dtype"]),
            )
            if case["equal"]:
                backing.fill_(1)
            else:
                backing.normal_()
            original = backing.clone()
            logits = backing[1:-1, 3 : 3 + vocab]
            targets = torch.arange(rows, device="cuda", dtype=torch.int64) % vocab
            row = dict(case=case, measurements={}, exact=False)
            result["results"].append(row)
            expected = None
            for arm in result["arm_order"]:
                # Exactly one full untimed warmup per arm and case, including JIT.
                warm = invoke(arm, logits, targets, case)
                torch.cuda.synchronize()
                host = [tensor.cpu() for tensor in warm[:3]]
                del warm
                if expected is None:
                    expected = host
                assert same(host, expected), (number, arm, "warmup differs")
                samples = []
                row["measurements"][arm] = samples
                for repeat in range(3):
                    actual, timing = measured(arm, logits, targets, case)
                    samples.append(timing)
                    assert same(actual, expected), (number, arm, repeat, "differs")
                if case["equal"]:
                    assert torch.equal(expected[2], torch.full_like(expected[2], vocab))
            assert torch.equal(backing, original), "input or guards changed"
            if rows == 7616 and case["mode"] == "raw_logprobs" and case["count"] == 0:
                row["allocation_traces"] = {
                    arm: allocation_trace(arm, logits, targets, case, args.output)
                    for arm in result["arm_order"]
                }
            row["exact"] = True
            print(
                json.dumps(
                    dict(
                        case=number,
                        **case,
                        exact=True,
                        medians={
                            arm: statistics.median(v["cuda_ms"] for v in samples)
                            for arm, samples in row["measurements"].items()
                        },
                        peaks={
                            arm: max(v["peak_extra_bytes"] for v in samples)
                            for arm, samples in row["measurements"].items()
                        },
                    )
                ),
                flush=True,
            )
            del backing, original, logits, targets
            save()
        result["hardware_after"] = gpu_query(fields)
        assert result["hardware_before"] == result["hardware_after"]
        assert sources() == result["sources"], "source changed during probe"
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
