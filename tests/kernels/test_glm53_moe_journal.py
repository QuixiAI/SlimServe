# SPDX-License-Identifier: Apache-2.0
"""Actual layer3 weights, synthetic640-token inputs/routes; observer parity only."""

import json
from pathlib import Path

import pytest
import torch

import vllm._custom_ops as ops
from benchmarks.kernels.profile_glm53_marlin import load_layer
from slimserve.model_journal import ACTIVE, ModelJournal
from slimserve.moe_journal import (
    instrument_marlin_gemm,
    instrument_moe_sum,
    instrument_router,
)
from slimserve.score_journal import ScoreJournal
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.quixicore.ops import quixicore_ops as qc
from vllm.scalar_type import scalar_types

MODEL = Path("/raid/weights/GLM-5.3-Flash-NVFP4")
pytestmark = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (12, 0)
        and MODEL.is_dir()
    ),
    reason="requires SM120 and pinned GLM53 NVFP4 weights",
)


def test_native_marlin_and_shared_sum_observers_preserve_bits(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "1")
    monkeypatch.setenv("SLIMSERVE_GLM53_MOE_JOURNAL", "1")
    monkeypatch.setattr(
        ops, "moe_wna16_marlin_gemm", instrument_marlin_gemm(ops.moe_wna16_marlin_gemm)
    )
    monkeypatch.setattr(
        qc, "moe_sum_add", staticmethod(instrument_moe_sum(qc.moe_sum_add))
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "schema": 1,
                "prompt_ids": list(range(640)),
                "max_matches": 3,
                "output_directory": str(tmp_path / "trace"),
            }
        )
    )
    score = ScoreJournal(config)
    journal = ModelJournal(score)
    weights, _ = load_layer(MODEL, 3, 0)
    w13, s13, g13, w2, s2, g2, workspace = weights
    ids = ((torch.arange(640 * 8, device="cuda") * 37) % 288).reshape(640, 8).int()
    topk_weights = torch.full((640, 8), 2.5 / 8, device="cuda", dtype=torch.float32)
    logits = torch.zeros(640, 288, device="cuda")

    @instrument_router
    def router(hidden_states, router_logits):
        return topk_weights, ids

    def call(x, shared):
        selected_weights, selected_ids = router(x, logits)

        def reduce(moe_out, out, topk_ids, expert_map):
            qc.moe_sum_add(moe_out, shared, out)
            return out

        return fused_marlin_moe(
            x,
            w13,
            w2,
            None,
            None,
            s13,
            s2,
            selected_weights,
            selected_ids,
            scalar_types.float4_e2m1f.id,
            global_scale1=g13,
            global_scale2=g2,
            workspace=workspace,
            clamp_limit=10.0,
            moe_sum=reduce,
        )

    try:
        for match, seed in enumerate((871, 872), 1):
            generator = torch.Generator(device="cuda").manual_seed(seed)
            x = torch.randn(
                640, 4096, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            shared = torch.randn(
                640, 4096, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            expected = call(x, shared).clone()
            repeated = call(x, shared)
            assert torch.equal(
                expected.view(torch.uint8), repeated.view(torch.uint8)
            ), "untraced repeat differs"
            # This is a MoE-only synthetic fixture, not a complete model trace.
            journal.begin(f"synthetic-{match}")
            journal.operations = 8
            token = ACTIVE.set(journal)
            try:
                actual = call(x, shared)
            finally:
                ACTIVE.reset(token)
            assert torch.isfinite(actual).all()
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            assert journal.records == 29
            assert journal.moe_capture.completed == set(range(1, match + 1))
            journal.active = None
        assert journal.moe_capture.saved_bytes < 2 * 1024**3
        assert len(list(journal.moe_capture.directory.glob("*-parameter.pt"))) == 6
    finally:
        journal.close()
        score.close()
