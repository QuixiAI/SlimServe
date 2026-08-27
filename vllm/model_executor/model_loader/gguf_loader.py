# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF loader for SlimServe's supported model artifacts.

This loader is deliberately not general. It takes a local GGUF path and
dispatches by ``general.architecture`` to the adapters required by the
registered GLM-5.2-Vision, Kimi K3, DSpark, and DeepSeek-V4-Flash artifacts.
"""

import os
import time
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.ep_weight_filter import (
    compute_local_expert_ids,
)
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.bootstamp import bootstamp
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
                "restricted to SlimServe's registered local artifacts."
            )
        return path

    def _prepare_adapter(self, model_config: ModelConfig):
        # Imported here, not at module scope: the adapters pull in models.utils,
        # which imports this package.
        from vllm.model_executor.model_loader.gguf_adapters import (
            Deepseek4GGUFAdapter,
            DFlashGGUFAdapter,
            GlmDsaGGUFAdapter,
            KimiK3DSparkGGUFAdapter,
            KimiK3GGUFAdapter,
            MuseGlimmerDFlashGGUFAdapter,
            MuseGlimmerGGUFAdapter,
            Qwen35GGUFAdapter,
        )
        from vllm.transformers_utils.gguf_utils import gguf_architecture

        path = self._gguf_path(model_config)
        # Dispatch on the file rather than the config: at this point the config
        # was itself built from the GGUF, so the architecture string is the one
        # authority both agree on.
        architecture = gguf_architecture(path)
        adapter_cls: type
        if architecture == "deepseek4":
            adapter_cls = Deepseek4GGUFAdapter
        elif architecture == "dflash-draft":
            adapter_cls = KimiK3DSparkGGUFAdapter
        elif architecture == "dflash":
            # Same routing as the config parser: the DeepSeek-V4 DSpark
            # drafter carries expert keys, the DFlash 2 drafter carries the
            # selector metadata, the Muse-Glimmer drafter neither.
            from vllm.transformers_utils.gguf_utils import gguf_reader

            fields = gguf_reader(path).fields
            if "dflash.expert_count" in fields:
                adapter_cls = DFlashGGUFAdapter
            elif "dflash.selector_rank" in fields:
                from vllm.model_executor.model_loader.gguf_adapters import (
                    DFlash2QwenGGUFAdapter,
                )

                adapter_cls = DFlash2QwenGGUFAdapter
            else:
                adapter_cls = MuseGlimmerDFlashGGUFAdapter
        elif architecture == "muse-glimmer":
            adapter_cls = MuseGlimmerGGUFAdapter
        elif architecture == "qwen35":
            from vllm.model_executor.model_loader.gguf_adapters import (
                Qwen35GGUFAdapter,
            )

            adapter_cls = Qwen35GGUFAdapter
        elif architecture == "kimi-k3":
            adapter_cls = KimiK3GGUFAdapter
        elif architecture == "glm-dsa":
            adapter_cls = GlmDsaGGUFAdapter
        elif architecture == "qwen35":
            adapter_cls = Qwen35GGUFAdapter
        else:
            raise ValueError(f"Unsupported GGUF architecture: {architecture}")
        adapter = adapter_cls(model_config.hf_config)
        adapter.prepare_loading(self._gguf_path(model_config), model_config)
        if architecture == "kimi-k3":
            self._configure_ep_expert_shard(adapter, model_config)
        return adapter

    @staticmethod
    def _configure_ep_expert_shard(adapter, model_config: ModelConfig) -> None:
        """Give GGUF adapters the global expert ids owned by this EP rank."""
        from vllm.config import get_current_vllm_config
        from vllm.distributed import (
            get_dp_group,
            get_pcp_group,
            get_tensor_model_parallel_rank,
        )

        parallel_config = get_current_vllm_config().parallel_config
        if not model_config.is_moe or not parallel_config.enable_expert_parallel:
            return
        if parallel_config.enable_eplb:
            raise ValueError("Fused GGUF expert loading does not support EPLB")

        dp_size = parallel_config.data_parallel_size
        pcp_size = parallel_config.prefill_context_parallel_size
        tp_size = parallel_config.tensor_parallel_size
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        pcp_rank = get_pcp_group().rank_in_group if pcp_size > 1 else 0
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        ep_size = dp_size * pcp_size * tp_size
        ep_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + tp_rank
        local_expert_ids = compute_local_expert_ids(
            model_config.get_num_experts(),
            ep_size,
            ep_rank,
            placement=parallel_config.expert_placement_strategy,
        )
        adapter.set_local_expert_ids(local_expert_ids)
        if local_expert_ids is not None:
            logger.info_once(
                "GGUF EP shard: ep_size=%d, ep_rank=%d, loading %d/%d experts",
                ep_size,
                ep_rank,
                len(local_expert_ids),
                model_config.get_num_experts(),
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._gguf_path(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        adapter = self._prepare_adapter(model_config)
        model.load_weights(  # type: ignore[operator]
            self._timed_weights(adapter, model_config)
        )

    def _timed_weights(self, adapter, model_config: ModelConfig):
        """Yield weights while splitting producer time from consumer time.

        Producer time is spent inside the adapter iterator (mmap reads, dequant
        and reshape on the CPU); the rest of the wall-clock of
        ``model.load_weights`` is the consumer (name mapping and H2D copies).
        """
        gen = adapter.prepare_weights(model_config)
        produced = 0
        producer_s = 0.0
        while True:
            start = time.perf_counter()
            try:
                item = next(gen)
            except StopIteration:
                break
            producer_s += time.perf_counter() - start
            produced += 1
            yield item
        bootstamp(
            f"gguf producer: {produced} tensors, {producer_s:.2f}s inside iterator"
        )

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        adapter = self._prepare_adapter(model_config)
        logger.debug(
            "GGUF unquantized modules: %s", adapter.load_spec.unquantized_modules
        )
        vllm_config.quant_config = cast("GGUFConfig", vllm_config.quant_config)
        vllm_config.quant_config.unquantized_modules.extend(
            adapter.load_spec.unquantized_modules
        )

        target_device = torch.device(device_config.device)  # type: ignore[arg-type]
        with set_default_torch_dtype(model_config.dtype):  # type: ignore[arg-type]
            start = time.perf_counter()
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
            bootstamp(f"gguf load: initialize_model {time.perf_counter() - start:.2f}s")
            from vllm.distributed.parallel_state import get_tp_group

            tp_device_comm = get_tp_group().device_communicator
            start_async_init = getattr(tp_device_comm, "start_async_init", None)
            if start_async_init is not None:
                start_async_init()
            start = time.perf_counter()
            model.load_weights(  # type: ignore[operator]
                self._timed_weights(adapter, model_config)
            )
            bootstamp(
                f"gguf load: load_weights total {time.perf_counter() - start:.2f}s"
            )
            start = time.perf_counter()
            process_weights_after_loading(model, model_config, target_device)
            bootstamp(
                "gguf load: process_weights_after_loading "
                f"{time.perf_counter() - start:.2f}s"
            )
        return model
