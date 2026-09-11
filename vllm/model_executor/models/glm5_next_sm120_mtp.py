# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3-Flash MTP (NextN) draft model.

The checkpoint's layer ``num_hidden_layers`` is a one-step multi-token
predictor: ``enorm``/``hnorm`` + ``eh_proj`` fold the token embedding and the
target's final hidden state into one stream, a plain-residual DSA decoder
block (its own sparse MLA, pooled indexer and MoE; no mHC, no KDA) refines
it, and ``shared_head.norm`` feeds the target's LM head. Mirrors
``deepseek_mtp.py``: forward returns ``(pre_norm_hidden, post_norm_hidden)``,
``compute_logits`` applies the head to the normalized pre-norm hidden.

NVFP4 conversions such as RedHatAI's keep the head in
``model_mtp.safetensors`` outside the shard index; the loader reads that
file alone when it exists (``allow_patterns_overrides``).
"""

import os
from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_mtp import SharedHead
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2MixtureOfExperts,
    DeepseekV2MoE,
)
from vllm.model_executor.models.glm5_next import Glm5NextSM120MTPBlock
from vllm.model_executor.models.utils import (
    get_spec_layer_idx_from_weight_name,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)

MTP_SIDECAR_FILE = "model_mtp.safetensors"
_DEBUG_TEACHER_FORCED = os.environ.get("SLIMSERVE_MTP_DEBUG", "0") == "1"


def topk_buffer_width(config) -> int:
    """Pooled-indexer output width: expanded pools + tail, padded to 32."""
    width = config.index_topk + config.index_kpool - 1
    return (width + 31) // 32 * 32


class Glm5NextMultiTokenPredictorLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            topk_buffer_width(config),
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=quant_config
        )
        # Prefix stays the layer's so the compressed-tensors targets for this
        # layer match; checkpoint names are rewritten onto ``mtp_block``.
        self.mtp_block = Glm5NextSM120MTPBlock(
            config,
            vllm_config,
            prefix=prefix,
            topk_indices_buffer=topk_indices_buffer,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert inputs_embeds is not None
        # Position 0 has no previous token: zero its embedding (reference MTP).
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )
        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        hidden_states = residual + hidden_states  # pre-final-norm
        # Recycle the post-norm hidden into the next draft step; the logits
        # path normalizes the pre-norm element itself (one norm each).
        return hidden_states, self.shared_head(hidden_states)


class Glm5NextMultiTokenPredictor(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        # The loader hands the draft the serving VllmConfig (its model_config
        # is the target's VL config, no num_hidden_layers); the draft's own
        # promoted text config lives on speculative_config.
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): Glm5NextMultiTokenPredictorLayer(
                    vllm_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )
        self._mtp_layers = list(self.layers.values())
        self._mtp_mla_attns = [
            layer.mtp_block.self_attn.mla_attn for layer in self._mtp_layers
        ]
        self.logits_processor = LogitsProcessor(config.vocab_size)

    def set_skip_topk(self, skip: bool) -> None:
        """index_share_for_mtp_iteration: step 0 computes top-k, later steps
        reuse the indices step 0 wrote into the shared buffer."""
        for mla_attn in self._mtp_mla_attns:
            mla_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor) -> None:
        """Gather each request's last-token top-k rows to the buffer front."""
        num_slots = slot_ids.numel()
        for mla_attn in self._mtp_mla_attns:
            buf = mla_attn.topk_indices_buffer
            assert buf is not None
            buf[:num_slots] = buf[slot_ids]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self._mtp_layers[current_step_idx](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        mtp_layer = self._mtp_layers[spec_step_idx % self.num_mtp_layers]
        return self.logits_processor(
            mtp_layer.shared_head.head, mtp_layer.shared_head(hidden_states)
        )


class Glm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts):
    """Draft model entry (architecture ``Glm5NextMTPModel``)."""

    hf_to_vllm_prefix = {"model.language_model.": "model."}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = Glm5NextMultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.set_moe_parameters()
        # Conversions that keep the head outside the shard index: read only
        # that file (the loader would otherwise filter it out and stream the
        # whole target checkpoint for nothing).
        model_path = vllm_config.model_config.model
        sidecar = os.path.join(model_path, MTP_SIDECAR_FILE)
        self.allow_patterns_overrides: list[str] | None = (
            [MTP_SIDECAR_FILE] if os.path.isfile(sidecar) else None
        )
        if self.allow_patterns_overrides:
            logger.info("glm5_next_mtp: draft weights from %s", sidecar)

    def set_moe_parameters(self) -> None:
        self.num_moe_layers = self.config.num_nextn_predict_layers
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers.values():
            mlp = layer.mtp_block.mlp
            if isinstance(mlp, DeepseekV2MoE):
                example_moe = mlp
                self.moe_mlp_layers.append(mlp)
                self.moe_layers.append(mlp.experts)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        if _DEBUG_TEACHER_FORCED and input_ids is not None and input_ids.shape[0] > 8:
            self._log_teacher_forced(out[0], input_ids, spec_step_idx)
        return out

    @torch.no_grad()
    def _log_teacher_forced(
        self, hidden: torch.Tensor, input_ids: torch.Tensor, spec_step_idx: int
    ) -> None:
        """Diagnostic (SLIMSERVE_MTP_DEBUG=1): on a multi-token draft pass the
        draft's input at row i is the token at position i+1 and its target
        is the token at i+2, i.e. input_ids[i + 1]; report top-1 agreement."""
        logits = self.model.compute_logits(hidden, spec_step_idx)
        if logits is None:
            return
        pred = logits.argmax(dim=-1)[:-1]
        gold = input_ids[1:]
        acc = (pred == gold).float().mean().item()
        logger.info(
            "glm5_next_mtp teacher-forced: %d rows, top-1 next-token agreement %.3f",
            int(gold.numel()),
            acc,
        )
        dump_dir = os.environ.get("SLIMSERVE_MTP_DEBUG_DIR")
        if dump_dir and get_tensor_model_parallel_rank() == 0:
            # Draft argmax per row for the offline draft-vs-target comparison.
            os.makedirs(dump_dir, exist_ok=True)
            self._dump_idx = getattr(self, "_dump_idx", 0) + 1
            torch.save(
                {
                    "input_ids": input_ids.cpu(),
                    "draft_argmax": logits.argmax(dim=-1).cpu(),
                },
                os.path.join(dump_dir, f"draft-{self._dump_idx:03d}.pt"),
            )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    # Checkpoint names for the MTP layer, e.g.
    #   model.language_model.layers.45.self_attn.q_a_proj.weight
    #   model.language_model.layers.45.mlp.experts.7.gate_proj.weight_scale
    #   model.language_model.layers.45.shared_head.norm.weight
    # The block's parameters live under ``mtp_block``; the layer-level
    # pieces (enorm/hnorm/eh_proj/shared_head) do not.
    _LAYER_LEVEL = ("enorm", "hnorm", "eh_proj", "shared_head")

    @classmethod
    def strip_hf_prefix(cls, name: str) -> str:
        for pref, new in cls.hf_to_vllm_prefix.items():
            if name.startswith(pref):
                return new + name[len(pref) :]
        return name

    @classmethod
    def rewrite_spec_layer_name(cls, spec_layer: int, name: str) -> str:
        name = cls.strip_hf_prefix(name)
        head = f"model.layers.{spec_layer}."
        if not name.startswith(head):
            return name
        rest = name[len(head) :]
        if rest.startswith("embed_tokens"):
            return "model." + rest  # shared with the target
        if rest.split(".", 1)[0] in cls._LAYER_LEVEL:
            return name
        return head + "mtp_block." + rest

    stacked_params_mapping = [
        ("fused_qkv_a_proj", "q_a_proj", 0),
        ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
        ("fused_qkv_a_proj", "indexer.wk", 2),
        ("fused_qkv_a_proj", "indexer.index_kpool_compress_gate", 3),
        ("fused_qkv_a_proj", "indexer.weights_proj", 4),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        expert_params_mapping = fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
        )
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()
        for name, weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            # The helper matches "model.layers.<n>." only; strip the VL
            # checkpoint's "model.language_model." first.
            name = self.strip_hf_prefix(name)
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            if spec_layer is None:
                continue
            name = self.rewrite_spec_layer_name(spec_layer, name)
            if ".mlp.experts." in name:
                for param_name, ckpt_name, expert_id, shard_id in expert_params_mapping:
                    if ckpt_name not in name:
                        continue
                    tgt = name.replace(ckpt_name, param_name)
                    if tgt not in params_dict:
                        continue
                    param = params_dict[tgt]
                    param.weight_loader(
                        param, weight, tgt, shard_id=shard_id, expert_id=expert_id
                    )
                    loaded.add(tgt)
                    break
                else:
                    logger.warning_once("glm5_next_mtp: unmatched expert %s", name)
                continue
            mapped = False
            for target, ckpt_name, shard_id in self.stacked_params_mapping:
                token = f".{ckpt_name}."
                if token not in name:
                    continue
                tgt = name.replace(ckpt_name, target)
                if tgt not in params_dict:
                    continue
                param = params_dict[tgt]
                param.weight_loader(param, weight, shard_id)
                loaded.add(tgt)
                mapped = True
                break
            if mapped:
                continue
            if name not in params_dict:
                logger.warning_once("glm5_next_mtp: unmatched weight %s", name)
                continue
            param = params_dict[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, weight)
            loaded.add(name)

        start = self.model.mtp_start_layer_idx
        for layer_idx in range(start, start + self.model.num_mtp_layers):
            if not any(f"model.layers.{layer_idx}." in n for n in loaded):
                raise ValueError(
                    f"MTP layer {layer_idx} has no weights in the checkpoint; "
                    f"NVFP4 conversions ship it as {MTP_SIDECAR_FILE}"
                )
        return loaded
