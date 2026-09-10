# SPDX-License-Identifier: Apache-2.0
"""Qualified correction plus diagnostic-only scheduler/stream envelope checks.

No change to the AOT-qualified leaf dispatch. Both control and correction have
the same host checks; timings are diagnostics, not production baselines.
"""

import threading
from functools import wraps

from benchmarks.kernels.audit_glm53_indexer_correction_graphs import inventory
from benchmarks.kernels.check_glm53_attention_norms import require
from benchmarks.kernels.glm53_geometry_serving import ServingGeometry
from benchmarks.kernels.glm53_indexer_correction_loader import IndexerCorrectionLoader


def check_envelope(record, capacity):
    require(
        type(capacity) is int
        and capacity == 8192
        and type(record["max_num_tokens"]) is int
        and 0 < record["max_num_tokens"] <= capacity
        and record["scheduler_max_tokens"] == record["max_num_tokens"]
        and all(
            type(n) is int and 0 < n <= record["max_num_tokens"]
            for n in record["capture_sizes"]
        )
        and record["data_parallel_size"] == record["decode_context_parallel_size"] == 1
        and record["use_ubatching"] is False
        and record["sequence_parallel"] is False,
        "indexer correction requires bounded serialized TP4 scheduler/capture",
    )


def runtime_envelope(runner, capacity):
    record = dict(
        max_num_tokens=runner.max_num_tokens,
        scheduler_max_tokens=runner.scheduler_config.max_num_batched_tokens,
        capture_sizes=list(runner.cudagraph_batch_sizes),
        data_parallel_size=runner.parallel_config.data_parallel_size,
        decode_context_parallel_size=runner.parallel_config.decode_context_parallel_size,
        use_ubatching=runner.parallel_config.use_ubatching,
        sequence_parallel=runner.compilation_config.pass_config.enable_sp,
    )
    check_envelope(record, capacity)
    return record


class ServingIndexerCorrection(ServingGeometry):
    def __init__(self, rank, manifest, path):
        super().__init__(
            rank,
            manifest,
            path,
            loader_type=IndexerCorrectionLoader,
            graph_inventory=inventory,
            root_only=True,
        )
        self.streams["runtime-envelope"] = (
            self.folder / "runtime-envelope.jsonl"
        ).open("x")
        self.runner = None
        self.owner_thread = self.last_stream = self.live_stream = None
        self.observed = set()

    def snapshot(self, phase):
        import torch

        envelope = runtime_envelope(self.runner, self.manifest["selection_capacity"])
        require(
            envelope == self.summary["runtime_envelope"], "runtime envelope changed"
        )
        report = super().snapshot(phase)
        guards = []
        for binding in self.loader.controller.owners.values():
            if "adapter" not in binding:
                continue
            adapter = binding["adapter"]
            adapter.verify()
            storage = adapter.storage
            require(
                bool(torch.all(storage[0] == 171))
                and bool(torch.all(storage[-1] == 171)),
                "serving selection arena guard overwritten",
            )
            guards.append(dict(address=adapter.address, bytes=storage.numel()))
        require(
            len(guards) == (2 if self.mode == "correction" else 0),
            "arena coverage differs",
        )
        self.summary.setdefault("arena_snapshots", {})[phase] = guards
        self.save()
        return report

    def observe_padding(self, tokens, result):
        import torch

        padded, ubatch, dp_tokens = result[1].num_tokens, result[2], result[3]
        require(
            type(tokens) is int
            and type(padded) is int
            and 0
            < tokens
            <= padded
            <= self.summary["runtime_envelope"]["max_num_tokens"]
            and not ubatch
            and dp_tokens is None,
            "actual padded batch exceeds arena or enables parallel microbatches",
        )
        thread = threading.get_ident()
        require(
            self.owner_thread in (None, thread),
            "concurrent worker thread is unqualified",
        )
        self.owner_thread = thread
        stream = int(torch.cuda.current_stream().cuda_stream)
        phase = "live" if self.summary["status"] == "capture-qualified" else "startup"
        if phase == "live":
            require(self.live_stream in (None, stream), "live forward stream changed")
            self.live_stream = stream
        barrier = self.last_stream is not None and self.last_stream != stream
        if barrier:
            require(
                not torch.cuda.is_current_stream_capturing(),
                "stream transition inside capture",
            )
            # Startup/profile/capture may use different streams. Establish ordering
            # explicitly at transitions only; steady-state batches do not synchronize.
            torch.cuda.synchronize()
        self.last_stream = stream
        key = (phase, tokens, padded, stream)
        if key not in self.observed or barrier:
            self.observed.add(key)
            self.emit(
                "runtime-envelope",
                dict(
                    phase=phase,
                    tokens=tokens,
                    padded=padded,
                    stream=stream,
                    thread=thread,
                    transition_barrier=barrier,
                ),
            )

    def install(self, runner):
        self.runner = runner
        self.summary["runtime_envelope"] = runtime_envelope(
            runner, self.manifest["selection_capacity"]
        )
        self.save()
        original = runner._determine_batch_execution_and_padding

        @wraps(original)
        def padding(*args, **kwargs):
            try:
                result = original(*args, **kwargs)
                self.observe_padding(args[0] if args else kwargs["num_tokens"], result)
                return result
            except BaseException as error:
                self.fail("runtime-envelope", error)
                raise

        super().install(runner)
        runner._determine_batch_execution_and_padding = padding

        def restore():
            require(
                runner._determine_batch_execution_and_padding is padding,
                "foreign padding hook change",
            )
            runner._determine_batch_execution_and_padding = original

        self.stack.callback(restore)


def check_runtime_records(manifest, summary, records):
    """Independent CPU audit of configuration, guards and observed padded batches."""
    envelope = summary["runtime_envelope"]
    check_envelope(envelope, manifest["selection_capacity"])
    phases = ("before-forward", "capture-before", "capture-after")
    arenas = summary["arena_snapshots"]
    require(set(arenas) == set(phases), "missing arena snapshot")
    reference = arenas[phases[0]]
    require(
        all(arenas[p] == reference for p in phases)
        and len(reference) == (2 if manifest["mode"] == "correction" else 0)
        and len({r["address"] for r in reference}) == len(reference)
        and all(
            type(r["address"]) is int
            and r["address"] > 0
            and r["bytes"] == (manifest["selection_capacity"] + 2) * 128
            for r in reference
        ),
        "arena lifetime/address/size changed",
    )
    require(
        records
        and {r["phase"] for r in records} == {"startup", "live"}
        and len({r["thread"] for r in records}) == 1
        and len({r["stream"] for r in records if r["phase"] == "live"}) == 1,
        "runtime thread/stream coverage incomplete",
    )
    last_stream, live = None, False
    for row in records:
        require(
            type(row["tokens"]) is int
            and type(row["padded"]) is int
            and 0 < row["tokens"] <= row["padded"] <= envelope["max_num_tokens"],
            "observed padding outside arena",
        )
        require(not live or row["phase"] == "live", "startup after live execution")
        live = row["phase"] == "live"
        require(
            row["transition_barrier"]
            is (last_stream is not None and last_stream != row["stream"]),
            "missing/unexpected stream transition barrier",
        )
        last_stream = row["stream"]
    return dict(
        observed_batches=len(records),
        max_padded=max(r["padded"] for r in records),
        single_live_stream=True,
        arena_bindings=len(reference),
    )
