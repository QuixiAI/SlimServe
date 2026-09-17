# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import (
    AutoRegressiveSpeculator,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


class MTPSpeculator(AutoRegressiveSpeculator):
    share_mtp_topk_indices: bool = False

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        draft_model = load_eagle_model(target_model, self.vllm_config)
        spec_config = self.vllm_config.speculative_config
        draft_hf_config = (
            spec_config.draft_model_config.hf_text_config
            if spec_config is not None
            else None
        )
        # index_share_for_mtp_iteration: draft step 0 computes the MTP
        # layer's own sparse top-k, steps 1+ skip the indexer and reuse those
        # rows (compacted to each request's last token).
        self.share_mtp_topk_indices = bool(
            getattr(draft_hf_config, "index_share_for_mtp_iteration", False)
            and hasattr(draft_model.model, "set_skip_topk")
            and hasattr(draft_model.model, "compact_topk_indices")
        )
        # The GLM-5.3 head runs its experts and head only on the rows the
        # speculator selects (the request tails on the prefill step).
        self.selects_output_rows = bool(
            getattr(draft_model, "selects_output_rows", False)
        )
        return draft_model

    def on_prefill_begin(self, num_reqs: int) -> None:
        # Unconditional, so a step that died midway cannot leave reuse mode on.
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)

    def on_prefill_end(self, num_reqs: int) -> None:
        # Step 0 wrote a top-k row for every query token of the multi-token
        # batch; steps 1+ read rows [0, num_reqs).
        if self.share_mtp_topk_indices and self.num_speculative_steps > 1:
            self.model.model.compact_topk_indices(self.last_token_indices[:num_reqs])

    def on_multi_step_decode_begin(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(True)

    def on_multi_step_decode_end(self, num_reqs: int) -> None:
        if self.share_mtp_topk_indices:
            self.model.model.set_skip_topk(False)
