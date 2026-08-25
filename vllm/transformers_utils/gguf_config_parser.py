# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config parser for the GGUF models this fork serves.

The previous version delegated to `HFConfigParser`, which meant a config.json
had to exist somewhere -- either in the GGUF's directory or via
`--hf-config-path`. It does not: the builders assemble the whole config from
GGUF metadata alone, validated field-for-field against the reference
checkpoint.

Dispatch is on `general.architecture`, because the supported artifacts share
nothing at the metadata level. Standalone DSpark drafts use the `dflash` or
`dflash-draft` schemas and carry no tokenizer vocabulary.
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
        elif architecture == "dflash-draft":
            from vllm.transformers_utils.gguf_kimi_k3_dspark import (
                build_kimi_k3_dspark_config_from_gguf,
            )

            config = build_kimi_k3_dspark_config_from_gguf(str(model))
        elif architecture == "dflash":
            # Three published drafters share the `dflash` architecture string:
            # the DeepSeek-V4 0731 DSpark drafter (MoE, MLA-shaped, carries
            # `dflash.expert_count`), the DFlash 2 drafter (carries
            # `dflash.selector_rank`), and the Muse-Glimmer drafter (dense
            # GQA, neither key). Route on those schema differences.
            from vllm.transformers_utils.gguf_utils import gguf_reader

            reader = gguf_reader(str(model))
            if "dflash.expert_count" in reader.fields:
                from vllm.transformers_utils.gguf_dflash import (
                    build_dflash_config_from_gguf,
                )

                config = build_dflash_config_from_gguf(str(model))
            elif "dflash.selector_rank" in reader.fields:
                from vllm.transformers_utils.gguf_qwen35 import (
                    build_qwen38_dflash2_config_from_gguf,
                )

                config = build_qwen38_dflash2_config_from_gguf(str(model))
            else:
                from vllm.transformers_utils.gguf_muse_glimmer import (
                    build_muse_glimmer_dflash_config_from_gguf,
                )

                config = build_muse_glimmer_dflash_config_from_gguf(str(model))
        elif architecture == "muse-glimmer":
            from vllm.transformers_utils.gguf_muse_glimmer import (
                build_muse_glimmer_config_from_gguf,
            )

            config = build_muse_glimmer_config_from_gguf(str(model))
        elif architecture == "kimi-k3":
            from vllm.transformers_utils.gguf_kimi_k3 import (
                build_kimi_k3_config_from_gguf,
            )

            config = build_kimi_k3_config_from_gguf(str(model))
        elif architecture == "qwen35":
            from vllm.transformers_utils.gguf_qwen35 import (
                build_qwen35_config_from_gguf,
            )

            config = build_qwen35_config_from_gguf(str(model))
        elif architecture == "glm-dsa":
            config = build_config_from_gguf(str(model))
        elif architecture == "qwen35":
            from vllm.transformers_utils.gguf_qwen35 import (
                build_qwen35_config_from_gguf,
            )

            config = build_qwen35_config_from_gguf(str(model))
        else:
            raise ValueError(f"Unsupported GGUF architecture: {architecture}")
        return config.to_dict(), config
