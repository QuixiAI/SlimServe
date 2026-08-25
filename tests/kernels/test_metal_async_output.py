# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression for the MPS async-output cross-stream cold-start hang.

The platform decision lives in make_output_copy_stream: on MPS it hands
back the producing stream, and the copy classes detect that by stream
equality — no device-type checks in shared worker code. These tests pin
the helper's contract and both copy classes (the pooling twin included)
on the helper-provided stream.
"""

import torch

try:
    import pytest

    pytestmark = pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="requires Apple Metal (MPS)",
    )
except ModuleNotFoundError:
    pass


def test_make_output_copy_stream_is_producing_stream_on_mps() -> None:
    from vllm.v1.worker.gpu.async_utils import make_output_copy_stream

    dev = torch.device("mps")
    copy_stream = make_output_copy_stream(dev)
    assert copy_stream == torch.accelerator.current_stream(dev), (
        "on MPS the output copy stream must be the producing stream; a "
        "cold cross-stream event hand-off can lose its completion signal"
    )


def test_make_completion_event_is_native_mps_event() -> None:
    """The generic torch.Event machinery is the observed signal-loss point
    on Metal (MPSEvent::synchronize park); the completion marker must be
    the native torch.mps.Event unless the drain kill-switch is armed."""
    from vllm.v1.worker.gpu.async_utils import _METAL_DRAIN, make_completion_event

    event = make_completion_event()
    if _METAL_DRAIN:
        assert event is None
    else:
        assert isinstance(event, torch.mps.Event)
        assert not isinstance(event, torch.Event)


def test_metal_async_output_stays_on_producer_stream() -> None:
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput, make_output_copy_stream
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    dev = torch.device("mps")
    sampled = torch.tensor([[37]], dtype=torch.int64, device="mps")
    sampler = SamplerOutput(sampled, None, None, None)
    runner_output = ModelRunnerOutput(req_ids=["r"], req_id_to_index={"r": 0})
    main_stream = torch.accelerator.current_stream(dev)

    pending = AsyncOutput(
        model_runner_output=runner_output,
        sampler_output=sampler,
        num_sampled_tokens=torch.tensor([1], dtype=torch.int32, device="mps"),
        main_stream=main_stream,
        copy_stream=make_output_copy_stream(dev),
    )
    assert pending.get_output().sampled_token_ids == [[37]]


def test_metal_async_pooling_output_stays_on_producer_stream() -> None:
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import (
        AsyncPoolingOutput,
        make_output_copy_stream,
    )

    dev = torch.device("mps")
    pooled = torch.tensor([[1.0, 2.0]], dtype=torch.float32, device="mps")
    runner_output = ModelRunnerOutput(req_ids=["r"], req_id_to_index={"r": 0})
    main_stream = torch.accelerator.current_stream(dev)

    pending = AsyncPoolingOutput(
        model_runner_output=runner_output,
        pooler_output=pooled,
        is_valid=None,
        main_stream=main_stream,
        copy_stream=make_output_copy_stream(dev),
    )
    out = pending.get_output().pooler_output
    assert out is not None and torch.equal(out[0], pooled[0].cpu())
