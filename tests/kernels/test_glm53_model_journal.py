# SPDX-License-Identifier: Apache-2.0
"""Synthetic compile-opaque trace qualification, not a model quality test."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks.kernels.benchmark_mhc_output_parallel import inputs
from slimserve.model_journal import ACTIVE, install_model_journal, instrument_mhc
from vllm.model_executor.layers import glm5_next_mhc_ops as mhc
from vllm.quixicore.ops import quixicore_ops as qc
from vllm.utils.torch_utils import direct_register_custom_op

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)),
    reason="requires SM120",
)


def test_compiled_all90_sites_trace_without_changing_output(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "1")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "prompt_ids": list(range(640)),
                "max_matches": 3,
                "output_directory": str(tmp_path / "trace"),
            }
        )
    )
    monkeypatch.setenv("SLIMSERVE_GLM53_SCORE_JOURNAL", str(config_path))
    library = torch.library.Library("slimserve_mhc_journal_test", "FRAGMENT")
    for name in ("glm5_mhc_pre", "glm5_mhc_fused_post_pre", "glm5_mhc_post"):
        direct_register_custom_op(
            op_name=name,
            op_func=instrument_mhc(getattr(mhc, name)),
            fake_impl=getattr(mhc, f"_{name}_fake"),
            target_lib=library,
        )
    ops = torch.ops.slimserve_mhc_journal_test

    def forward(residual, fn, scale, base):
        p, c, x = ops.glm5_mhc_pre(residual, fn, scale, base, 1e-5, 1e-6, 2.0, 20)
        for _ in range(89):
            residual, p, c, x = ops.glm5_mhc_fused_post_pre(
                x,
                residual,
                p,
                c,
                fn,
                scale,
                base,
                1e-5,
                1e-6,
                2.0,
                20,
            )
        out = ops.glm5_mhc_post(x, residual, p, c)
        return out[:, 0, :].contiguous()

    saved = (
        qc.get_glm53_mhc_prefill_tc(),
        qc.get_dsv4_mhc_mode(),
        qc.get_dsv4_mhc_prefill_min_t(),
    )
    try:
        qc.set_glm53_mhc_prefill_tc(0)
        qc.set_dsv4_mhc_mode(2)
        qc.set_dsv4_mhc_prefill_min_t(64)
        data = inputs(640, 771)
        residual, fn, scale, base = data[1], data[4].bfloat16(), data[5], data[6]
        # Shrink the residual mixing parameters to keep the synthetic90-site
        # recurrence finite. These are not model checkpoint parameters.
        scale = torch.zeros_like(scale)
        base = torch.zeros_like(base)
        base[:8] = -8.0
        compiled = torch.compile(forward, backend="inductor", fullgraph=True)
        expected = forward(residual, fn, scale, base)
        actual = compiled(residual, fn, scale, base)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        runner = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="glm5_next"),
                hf_text_config=SimpleNamespace(hidden_size=4096, num_hidden_layers=45),
            ),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=4, pipeline_parallel_size=1
            ),
            speculative_config=None,
            num_prompt_logprobs={"request": 0},
            requests={
                "request": SimpleNamespace(
                    prompt_token_ids=list(range(640)), num_computed_tokens=0
                )
            },
            input_batch=SimpleNamespace(num_reqs=1),
            _model_forward=lambda **kw: compiled(kw["inputs_embeds"], fn, scale, base),
        )
        install_model_journal(runner)
        for match, seed in enumerate((772, 773), 1):
            fresh = inputs(640, seed)[1]
            expected = compiled(fresh, fn, scale, base).clone()
            output = runner._model_forward(
                input_ids=torch.arange(640, device="cuda"),
                positions=torch.arange(640, device="cuda"),
                inputs_embeds=fresh,
            )
            assert torch.isfinite(output).all()
            assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
            assert ACTIVE.get() is None
            path = next((tmp_path / "trace").glob("model-*.jsonl"))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert rows[-1] == {
                "kind": "complete",
                "match": match,
                "operations": 91,
                "tensor_records": 996,
            }
            assert len([r for r in rows if r["kind"] == "operation"]) == match * 91
        runner._slimserve_score_journal.close()
    finally:
        qc.set_glm53_mhc_prefill_tc(saved[0])
        qc.set_dsv4_mhc_mode(saved[1])
        qc.set_dsv4_mhc_prefill_min_t(saved[2])
