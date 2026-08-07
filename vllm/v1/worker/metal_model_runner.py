# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The V1 GPU model runner on PyTorch-MPS.

Only the two device-specific override points the base class documents need
replacing. Input batching, sampling, KV-cache management and the model forward
are all inherited and run unchanged on ``mps`` tensors.
"""

import torch

from vllm.logger import init_logger
from vllm.platforms.metal import gpu_core_count
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


class MetalModelRunner(GPUModelRunner):
    def _init_device_properties(self) -> None:
        # vLLM's heuristics want an SM count; the Apple GPU core count is the
        # closest honest analogue.
        self.num_sms = gpu_core_count()

    def _sync_device(self) -> None:
        torch.mps.synchronize()
