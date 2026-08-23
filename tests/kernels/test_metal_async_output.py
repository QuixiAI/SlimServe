# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression for the MPS async-output cross-stream cold-start hang."""

import torch

try:
    import pytest

    pytestmark = pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="requires Apple Metal (MPS)",
    )
except ModuleNotFoundError:
    pass


class _PoisonCopyStream:
    def wait_stream(self, _stream) -> None:
        raise AssertionError("MPS output must not use a cross-stream wait")


def test_metal_async_output_stays_on_producer_stream() -> None:
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    sampled = torch.tensor([[37]], dtype=torch.int64, device="mps")
    sampler = SamplerOutput(sampled, None, None, None)
    runner_output = ModelRunnerOutput(req_ids=["r"], req_id_to_index={"r": 0})
    main_stream = torch.accelerator.current_stream(torch.device("mps"))

    pending = AsyncOutput(
        model_runner_output=runner_output,
        sampler_output=sampler,
        num_sampled_tokens=torch.tensor([1], dtype=torch.int32, device="mps"),
        main_stream=main_stream,
        copy_stream=_PoisonCopyStream(),  # type: ignore[arg-type]
    )
    assert pending.get_output().sampled_token_ids == [[37]]
