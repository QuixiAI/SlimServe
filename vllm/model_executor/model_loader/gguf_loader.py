# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF loader for GLM-5.2-Vision.

This repo serves exactly one model on exactly one GPU configuration, so this
loader is deliberately not general. It takes a local path to the first shard
and hands the weights to the glm-dsa adapter. There is no remote download, no
`<repo_id>:<quant_type>` resolution, and no adapter registry -- every one of
those paths was dead code that only ever produced misleading errors on the way
to loading this model.
"""

import os
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    # Deferred: quantization.gguf.config pulls in models.utils, which imports
    # this package. Only needed for a cast.
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig

logger = init_logger(__name__)


class GGUFModelLoader(BaseModelLoader):
    """Loads the GLM-5.2-Vision GGUF from a local shard path."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                "Model loader extra config is not supported for load format "
                f"{load_config.load_format}"
            )

    def _gguf_path(self, model_config: ModelConfig) -> str:
        path = model_config.model_weights or model_config.model
        if not os.path.isfile(path):
            raise ValueError(
                f"Expected a local .gguf file, got {path!r}. This loader is "
                "specialised for the local GLM-5.2-Vision shards; pass the "
                "path to shard 00001-of-00006."
            )
        return path

    def _prepare_adapter(self, model_config: ModelConfig):
        # Imported here, not at module scope: the adapter pulls in
        # models.utils, which imports this package.
        from vllm.model_executor.model_loader.gguf_adapters import (
            GlmDsaGGUFAdapter,
        )

        adapter = GlmDsaGGUFAdapter(model_config.hf_config)
        adapter.prepare_loading(self._gguf_path(model_config), model_config)
        return adapter

    def download_model(self, model_config: ModelConfig) -> None:
        self._gguf_path(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        adapter = self._prepare_adapter(model_config)
        model.load_weights(adapter.prepare_weights(model_config))

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        adapter = self._prepare_adapter(model_config)
        vllm_config.model_config.hf_config = model_config.hf_config
        logger.debug(
            "GGUF unquantized modules: %s", adapter.load_spec.unquantized_modules
        )
        vllm_config.quant_config = cast("GGUFConfig", vllm_config.quant_config)
        vllm_config.quant_config.unquantized_modules.extend(
            adapter.load_spec.unquantized_modules
        )

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, prefix=prefix)
            model.load_weights(adapter.prepare_weights(model_config))
            process_weights_after_loading(model, model_config, target_device)
        return model
