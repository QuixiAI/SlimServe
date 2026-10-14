# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import functools
import os
from collections.abc import Callable

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput


def make_output_copy_stream(device: torch.device | str | None) -> torch.Stream:
    """Stream for output and side-channel copies that would otherwise
    overlap with compute on a second stream.

    On MPS this returns the CURRENT (producing) stream — which is the only
    stream that exists: torch.Stream(mps) always returns stream_id 0, so a
    dedicated copy stream buys no overlap there, and the cross-stream
    choreography has parked the engine (see make_completion_event). This
    helper is the single place that platform decision lives — consumers
    compare their copy stream against the main stream instead of testing
    device types. CUDA/ROCm get a dedicated stream to overlap the
    transfer."""
    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type == "mps":
        return torch.accelerator.current_stream(dev)
    return torch.Stream(device)


def _producing_stream(main_stream: torch.Stream) -> torch.Stream:
    """The stream the producer just ran on: the current stream, queried
    live. Falls back to the handed stream only where it cannot be queried
    (the MPS single-stream case keeps its handed producing stream)."""
    if _is_metal_platform():
        return main_stream
    try:
        return torch.accelerator.current_stream(main_stream.device)
    except Exception:
        return main_stream


@functools.cache
def _is_metal_platform() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_metal()


# Ops kill-switch: =1 replaces the Metal completion event below with a full
# torch.mps.synchronize() in the output wait. Costs the drafter-tail overlap
# (~3-4% decode throughput measured on dsv4-xxs-1) but removes the event
# from the completion path entirely if the parked-event wedge ever recurs.
_METAL_DRAIN = os.environ.get("VLLM_QC_ASYNC_OUT_DRAIN", "0") == "1"


def make_completion_event():
    """Completion marker for an output copy.

    On Metal: a native torch.mps.Event, or None when the drain kill-switch
    is armed. Deliberately NOT the generic torch.Event() machinery: MPS
    exposes exactly one stream (torch.Stream(mps) always returns
    stream_id 0), and the generic cross-"stream" record/wait_stream
    choreography has parked the engine forever in MPSEvent::synchronize on
    timing-sensitive cold boots — GPU idle, signal never delivered
    (vllm/platforms/metal_compat.py has the incident history). Elsewhere: a
    generic torch.Event for the dedicated copy stream."""
    if _is_metal_platform():
        return None if _METAL_DRAIN else torch.mps.Event()
    return torch.Event()


def record_completion_event(event, output_stream: torch.Stream) -> None:
    """Record a make_completion_event marker on the copy's stream.

    The native MPS event records argless on the current (only) stream;
    the generic event records on the handed output stream; None (drain
    mode) records nothing — sync_completion_event drains instead."""
    if event is None:
        return
    if _is_metal_platform():
        event.record()
    else:
        event.record(output_stream)


def sync_completion_event(event) -> None:
    """Wait for a make_completion_event marker (None = drain the stream)."""
    if event is None:
        torch.mps.synchronize()
    else:
        event.synchronize()


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.Stream,
        copy_stream: torch.Stream,
        check_ep_fault: bool = False,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock;
        # a native mps event (or drain sentinel) on Metal — see
        # make_completion_event for why the generic machinery is unsafe there.
        self.copy_event = make_completion_event()
        self._has_fault: torch.Tensor | None = None
        # (host int64[40], event) from the post_update row guard, attached by
        # the runner after the kernel is enqueued; checked in get_output so
        # a corrupted sampler buffer is reported before the scheduler can act
        # on it. See model_runner._post_update_guard_failure.
        self.post_update_guard: tuple[torch.Tensor, torch.cuda.Event] | None = None
        self.post_update_guard_check: Callable[[torch.Tensor, int], None] | None = None

        # On MPS, make_output_copy_stream hands out the producing stream
        # (the only one that exists), so the copies enqueue with no stream
        # switching at all. CUDA/ROCm arrive with a dedicated copy stream
        # and retain the overlap.
        # The producing stream is whatever is current NOW, never a cached
        # value: the sampler ran on it, and this copy must wait for it
        # (see stream() below for the corruption a stale cache caused).
        main_stream = _producing_stream(main_stream)
        copy_on_main_stream = copy_stream == main_stream
        output_stream = main_stream if copy_on_main_stream else copy_stream
        with (
            contextlib.nullcontext()
            if copy_on_main_stream
            else stream(output_stream, main_stream)
        ):
            if not copy_on_main_stream:
                output_stream.wait_stream(main_stream)

            self.sampled_token_ids = async_copy_to_np(sampler_output.sampled_token_ids)
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            record_completion_event(self.copy_event, output_stream)

    def get_output(self) -> ModelRunnerOutput:
        sync_completion_event(self.copy_event)

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
        if self.post_update_guard is not None:
            host, event = self.post_update_guard
            # post_update is enqueued on the main stream right after the
            # sampler; the copy event above already waited for the sampler,
            # so this adds only the tiny kernel itself.
            event.synchronize()
            if int(host[0]) != 0 and self.post_update_guard_check is not None:
                self.post_update_guard_check(host, len(num_sampled_tokens))
        row_len = (
            self.sampled_token_ids.shape[1] if self.sampled_token_ids.ndim == 2 else 1
        )
        for i, num_tokens in enumerate(num_sampled_tokens):
            if num_tokens < 0 or num_tokens > row_len:
                raise RuntimeError(
                    f"sampler output corrupted: num_sampled[{i}]={num_tokens} on "
                    f"a {row_len}-wide sampled_token_ids row (host copy); the "
                    "per-step sampler buffers were overwritten before the output "
                    "copy ran (perf/optimization_status.md 2026-09-01). Counts: "
                    f"{num_sampled_tokens[:16]}"
                )
        for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
            del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict

        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: "
                "one or more ranks timed out during dispatch/combine. "
                f"Mask: {mask.cpu().tolist()}"
            )

        return self.model_runner_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: torch.Tensor,
        is_valid: torch.Tensor | None,
        main_stream: torch.Stream,
        copy_stream: torch.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        self.is_valid = is_valid
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock;
        # native mps event (or drain sentinel) on Metal.
        self.copy_event = make_completion_event()

        # Same single-place platform decision as AsyncOutput: on MPS the
        # handed copy stream IS the main stream (make_output_copy_stream).
        main_stream = _producing_stream(main_stream)
        copy_on_main_stream = copy_stream == main_stream
        output_stream = main_stream if copy_on_main_stream else copy_stream
        with (
            contextlib.nullcontext()
            if copy_on_main_stream
            else stream(output_stream, main_stream)
        ):
            if not copy_on_main_stream:
                output_stream.wait_stream(main_stream)
            self.pooler_output_cpu = self.pooler_output.to("cpu", non_blocking=True)
            if self.is_valid is not None:
                self.is_valid_cpu = self.is_valid.to("cpu", non_blocking=True)
            else:
                self.is_valid_cpu = None
            record_completion_event(self.copy_event, output_stream)

    def get_output(self) -> ModelRunnerOutput:
        pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        sync_completion_event(self.copy_event)
        if self.is_valid_cpu is not None:
            is_valid_cpu = self.is_valid_cpu.tolist()
            for i, is_valid in enumerate(is_valid_cpu):
                if not is_valid:
                    pooler_output[i] = None
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.Stream, from_stream: torch.Stream | None = None):
    """Lightweight accelerator stream context manager.

    Restores the stream that was ACTUALLY current on entry, never a cached
    `from_stream`. Restoring a stale cached stream was the GLM-5.2/MI300X
    corruption root cause (2026-09-01): the runner had cached the default
    stream as "main" before vLLM installed its dedicated per-process stream,
    so every output copy exited this context by parking the main thread on
    the null stream. Everything launched until the next explicit stream
    switch (post_update, the attention metadata builder, parts of the next
    forward) then ran on a second hardware queue with no ordering against
    the sampler and the model - kernels read buffers before they were
    written and saw whatever the allocator had recycled (the MoE router's
    expert ids in the sampler's count buffer). `from_stream` is accepted
    for call-site compatibility and only used when no accelerator stream
    can be queried.
    """
    try:
        prev = torch.accelerator.current_stream(to_stream.device)
    except Exception:
        prev = from_stream
    try:
        torch.accelerator.set_stream(to_stream)
        yield
    finally:
        if prev is not None:
            torch.accelerator.set_stream(prev)
