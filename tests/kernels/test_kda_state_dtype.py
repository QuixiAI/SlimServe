# SPDX-License-Identifier: Apache-2.0
"""The KDA recurrent state dtype follows ``mamba_ssm_cache_dtype``: fp32
unless the cache config names another dtype (bf16 halves the per-step state
traffic on the rtx6000 GLM-5.3-Flash record)."""

import torch

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator


def test_kda_state_dtype_defaults_to_fp32_recurrent_state():
    conv, ssm = MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto")
    assert conv == torch.bfloat16 and ssm == torch.float32
    conv, ssm = MambaStateDtypeCalculator.kda_state_dtype(torch.bfloat16, "auto", "auto")
    assert conv == torch.bfloat16 and ssm == torch.float32


def test_kda_state_dtype_honours_the_ssm_cache_dtype():
    conv, ssm = MambaStateDtypeCalculator.kda_state_dtype(
        torch.bfloat16, "auto", "bfloat16"
    )
    assert conv == torch.bfloat16 and ssm == torch.bfloat16
    conv, ssm = MambaStateDtypeCalculator.kda_state_dtype(
        torch.bfloat16, "auto", "float32"
    )
    assert ssm == torch.float32
