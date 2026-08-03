# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config parser for the GGUF models this fork serves.

The previous version delegated to `HFConfigParser`, which meant a config.json
had to exist somewhere -- either in the GGUF's directory or via
`--hf-config-path`. It does not: the builders assemble the whole config from
GGUF metadata alone, validated field-for-field against the reference
checkpoint.

Dispatch is on `general.architecture`, because the two supported models share
nothing at the metadata level: GLM-5.2-Vision is `glm-dsa` plus an mmproj with
`clip.vision.*`, DeepSeek-V4-Flash is `deepseek4` with no vision at all.
"""

from pathlib import Path

from transformers import PretrainedConfig

from vllm.transformers_utils.config_parser_base import ConfigParserBase
from vllm.transformers_utils.gguf_native import build_config_from_gguf
from vllm.transformers_utils.gguf_utils import gguf_architecture


class GGUFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        architecture = gguf_architecture(str(model))
        if architecture == "deepseek4":
            from vllm.transformers_utils.gguf_deepseek4 import (
                build_deepseek4_config_from_gguf,
            )

            config = build_deepseek4_config_from_gguf(str(model))
        elif architecture == "kimi-k3":
            from vllm.transformers_utils.gguf_kimi_k3 import (
                build_kimi_k3_config_from_gguf,
            )

            config = build_kimi_k3_config_from_gguf(str(model))
        else:
            config = build_config_from_gguf(str(model))
        return config.to_dict(), config
