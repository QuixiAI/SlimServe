# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import functools
import os

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsTensors, ModelRunnerOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput


@functools.cache
def _is_metal_platform() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_metal()


# Ops kill-switch: =1 replaces the Metal completion event below with a full
# torch.mps.synchronize() in get_output(). Costs the drafter-tail overlap
# (~3-4% decode throughput measured on dsv4-xxs-1) but removes the event
# from the completion path entirely if the parked-event wedge ever recurs.
_METAL_DRAIN = os.environ.get("VLLM_QC_ASYNC_OUT_DRAIN", "0") == "1"


def _metal_completion_event():
    """Completion marker for the single MPS stream.

    None when the drain kill-switch is armed; otherwise a native
    torch.mps.Event recorded on the current (only) stream. Deliberately NOT
    the generic torch.Event()/torch.Stream() machinery: MPS exposes exactly
    one stream (torch.Stream(mps) always returns stream_id 0), and the
    cross-"stream" record/wait_stream/set_stream choreography the CUDA path
    uses has parked the engine forever in MPSEvent::synchronize on
    timing-sensitive boots — GPU idle, signal never delivered (see
    vllm/platforms/metal_compat.py for the incident history).
    """
    if _METAL_DRAIN:
        return None
    return torch.mps.Event()


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
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        # On Metal this is a native torch.mps.Event on the one real stream
        # and the copy_stream/main_stream handles are unused — see
        # _metal_completion_event for why the generic path is unsafe there.
        self._on_metal = _is_metal_platform()
        self.copy_event = (
            _metal_completion_event() if self._on_metal else torch.Event()
        )
        self._has_fault: torch.Tensor | None = None

        with (
            contextlib.nullcontext()
            if self._on_metal
            else stream(copy_stream, main_stream)
        ):
            if not self._on_metal:
                copy_stream.wait_stream(main_stream)

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
            if self._on_metal:
                if self.copy_event is not None:
                    self.copy_event.record()
            else:
                self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        if self.copy_event is None:
            torch.mps.synchronize()
        else:
            self.copy_event.synchronize()

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
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
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        # Native mps event on Metal — see _metal_completion_event.
        self._on_metal = _is_metal_platform()
        self.copy_event = (
            _metal_completion_event() if self._on_metal else torch.Event()
        )

        with (
            contextlib.nullcontext()
            if self._on_metal
            else stream(copy_stream, main_stream)
        ):
            if not self._on_metal:
                copy_stream.wait_stream(main_stream)
            self.pooler_output_cpu = self.pooler_output.to("cpu", non_blocking=True)
            if self.is_valid is not None:
                self.is_valid_cpu = self.is_valid.to("cpu", non_blocking=True)
            else:
                self.is_valid_cpu = None
            if self._on_metal:
                if self.copy_event is not None:
                    self.copy_event.record()
            else:
                self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        if self.copy_event is None:
            torch.mps.synchronize()
        else:
            self.copy_event.synchronize()
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
def stream(to_stream: torch.Stream, from_stream: torch.Stream):
    """Lightweight accelerator stream context manager."""
    try:
        torch.accelerator.set_stream(to_stream)
        yield
    finally:
        torch.accelerator.set_stream(from_stream)
