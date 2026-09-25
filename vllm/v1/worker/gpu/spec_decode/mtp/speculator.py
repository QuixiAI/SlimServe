# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import replace
from vllm.model_executor.models.utils import get_draft_quant_config
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


class MTPSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        # MTP weights may come from a different checkpoint than the target.
        # Construct the draft with its own quantization metadata; inheriting
        # the target's exclusion list can turn a packed draft head into an
        # unquantized layer and reject its scale tensors at load time.
        spec = self.vllm_config.speculative_config
        assert spec is not None
        draft_config = replace(
            self.vllm_config,
            quant_config=get_draft_quant_config(self.vllm_config),
            kernel_config=replace(
                self.vllm_config.kernel_config,
                moe_backend=spec.moe_backend or self.vllm_config.kernel_config.moe_backend,
            ),
        )
        return load_eagle_model(target_model, draft_config)
