# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config parser for the GLM-5.2-Vision GGUF pair.

The previous version delegated to `HFConfigParser`, which meant a config.json
had to exist somewhere -- either in the GGUF's directory or via
`--hf-config-path`. It does not: `build_config_from_gguf` assembles the whole
`Glm5vConfig` (text + vision) from `glm-dsa.*` and the mmproj's `clip.vision.*`
metadata, validated field-for-field against the reference checkpoint.
"""

from pathlib import Path

from transformers import PretrainedConfig

from vllm.transformers_utils.config_parser_base import ConfigParserBase
from vllm.transformers_utils.gguf_native import build_config_from_gguf


class GGUFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        config = build_config_from_gguf(str(model))
        return config.to_dict(), config
