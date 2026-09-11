# SPDX-License-Identifier: Apache-2.0
"""Opt-in GLM53 forward fingerprints through existing opaque mHC operations.

No new GPU operations or arithmetic. Disabled registration returns the original
callables unchanged. Enabled copies deliberately synchronize execution and are
not eligible for throughput comparisons.
"""

import hashlib
import inspect
import json
import os
from contextvars import ContextVar
from functools import wraps

from slimserve.glm53_ordering import enabled as native_order_enabled

ACTIVE = ContextVar("slimserve_glm53_model_journal", default=None)
OP_NAMES = ("glm5_mhc_pre", "glm5_mhc_fused_post_pre", "glm5_mhc_post")


def enabled():
    value = os.environ.get("SLIMSERVE_GLM53_MODEL_JOURNAL", "0")
    if value not in ("0", "1"):
        raise ValueError("SLIMSERVE_GLM53_MODEL_JOURNAL must be 0 or 1")
    return value == "1"


def instrument_mhc(function):
    if not enabled():
        return function
    if function.__name__ not in OP_NAMES:
        raise ValueError("unknown model-journal operation")
    signature = inspect.signature(function)

    @wraps(function)
    def traced(*args, **kwargs):
        journal = ACTIVE.get()
        if journal is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        journal.before_op(function.__name__, bound.arguments)
        result = function(*args, **kwargs)
        journal.after_op(function.__name__, result)
        return result

    return traced


class ModelJournal:
    def __init__(self, score_journal):
        from slimserve.moe_journal import enabled as moe_enabled

        self.prompt_ids = score_journal.prompt_ids
        self.limit = score_journal.limit
        self.matches = self.operations = self.records = 0
        self.active = self.pending = None
        self.moe_enabled = moe_enabled()
        self.moe_capture = None
        self.path = score_journal.path.with_name(f"model-{os.getpid()}.jsonl")
        self.stream = self.path.open("x", buffering=1)
        self.write(
            {
                "kind": "header",
                "schema": 1,
                "diagnostic_only": True,
                "pid": os.getpid(),
                "max_matches": self.limit,
                "max_tensor_bytes": 1024**3,
                "max_tensor_records_per_match": 1024,
                "expected_operations": 91,
                "first_moe_snapshots": self.moe_enabled,
                "small_route_order": (
                    "canonical-native"
                    if native_order_enabled()
                    or os.getenv("SLIMSERVE_GLM53_STABLE_ROUTE", "0") == "1"
                    else "atomic-native"
                ),
                "large_alignment_order": (
                    "canonical-native"
                    if native_order_enabled()
                    or os.getenv("SLIMSERVE_GLM53_STABLE_ALIGN", "0") == "1"
                    else "atomic-plus-sort"
                    if os.getenv("SLIMSERVE_GLM53_CANONICAL_MOE", "0") == "1"
                    else "atomic"
                ),
                "max_moe_dump_bytes_per_worker": 2 * 1024**3,
                "score_journal": str(score_journal.path),
                "score_header": json.loads(
                    score_journal.path.read_text().splitlines()[0]
                ),
            }
        )

    def write(self, row):
        self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    def record(self, stage, tensor):
        import torch

        if self.active is None or self.records >= 1024:
            raise ValueError("model trace inactive or record bound exceeded")
        nbytes = tensor.numel() * tensor.element_size()
        if not 0 < nbytes <= 1024**3:
            raise ValueError("model trace tensor byte bound exceeded")
        host = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256(
            memoryview(host.reshape(-1).view(torch.uint8).numpy())
        ).hexdigest()
        self.write(
            {
                "kind": "tensor",
                "match": self.matches,
                "stage": stage,
                "device": str(tensor.device),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "nbytes": nbytes,
                "sha256": digest,
            }
        )
        self.records += 1
        return host

    def begin(self, request_id):
        if self.active is not None or self.matches >= self.limit:
            raise ValueError("extra or overlapping model trace")
        self.matches += 1
        self.operations = self.records = 0
        self.active = request_id
        self.write({"kind": "begin", "match": self.matches, "request_id": request_id})

    def before_op(self, name, arguments):
        import torch

        expected = OP_NAMES[
            0 if self.operations == 0 else 1 if self.operations < 90 else 2
        ]
        if (
            self.active is None
            or self.pending is not None
            or self.operations > 90
            or name != expected
        ):
            raise ValueError("model trace operation order changed")
        self.pending = name
        self.write(
            {
                "kind": "operation",
                "match": self.matches,
                "site": self.operations,
                "name": name,
                "parameters": {
                    k: v
                    for k, v in arguments.items()
                    if not isinstance(v, torch.Tensor)
                },
            }
        )
        for key, tensor in arguments.items():
            if isinstance(tensor, torch.Tensor):
                self.record(f"site-{self.operations:03d}.input.{key}", tensor)

    def after_op(self, name, result):
        if self.active is None or self.pending != name:
            raise ValueError("model trace operation completion changed")
        names = {
            OP_NAMES[0]: ("post_mix", "comb_mix", "layer_input"),
            OP_NAMES[1]: ("residual", "post_mix", "comb_mix", "layer_input"),
            OP_NAMES[2]: ("streams",),
        }[name]
        values = result if isinstance(result, tuple) else (result,)
        if len(values) != len(names):
            raise ValueError("model trace result arity changed")
        for key, tensor in zip(names, values):
            self.record(f"site-{self.operations:03d}.output.{key}", tensor)
        self.pending = None
        self.operations += 1

    def finish(self):
        if self.moe_enabled and (
            self.moe_capture is None or self.matches not in self.moe_capture.completed
        ):
            raise ValueError("missing first-MoE capture")
        if self.active is None or self.pending is not None or self.operations != 91:
            raise ValueError("incomplete model trace")
        self.write(
            {
                "kind": "complete",
                "match": self.matches,
                "operations": self.operations,
                "tensor_records": self.records,
            }
        )
        self.active = None

    def close(self):
        self.stream.close()


def install_model_journal(runner):
    """Wrap this runner instance once; no per-step hook at all when disabled."""
    if not enabled():
        return
    from slimserve.score_journal import ScoreJournal

    config = runner.model_config.hf_config
    text_config = runner.model_config.hf_text_config
    if (
        text_config.hidden_size != 4096
        or text_config.num_hidden_layers != 45
        or runner.parallel_config.tensor_parallel_size != 4
        or runner.parallel_config.pipeline_parallel_size != 1
        or runner.speculative_config is not None
    ):
        raise ValueError("model journal requires GLM53 TP4 no-spec")
    score = ScoreJournal.from_env(getattr(config, "model_type", None))
    if score is None:
        raise ValueError("model journal requires a bounded score journal")
    runner._slimserve_score_journal = score
    journal = ModelJournal(score)
    original = runner._model_forward

    @wraps(original)
    def traced(**kwargs):
        import torch

        matches = [
            rid
            for rid in runner.num_prompt_logprobs
            if runner.requests[rid].prompt_token_ids == journal.prompt_ids
        ]
        if not matches:
            return original(**kwargs)
        if len(matches) != 1 or runner.input_batch.num_reqs != 1:
            raise ValueError("model journal requires one unmixed request")
        rid = matches[0]
        if (
            runner.requests[rid].num_computed_tokens != 0
            or runner.num_prompt_logprobs[rid] != 0
        ):
            raise ValueError("model journal requires an uncached whole prompt")
        input_ids, embeds, positions = (
            kwargs.get(k) for k in ("input_ids", "inputs_embeds", "positions")
        )
        tokens = input_ids if input_ids is not None else embeds
        if tokens is None or tokens.shape[0] != 640:
            raise ValueError("model journal requires the exact 640-token forward")
        if tokens.device.type != "cuda" or torch.cuda.is_current_stream_capturing():
            raise ValueError("model journal requires uncaptured CUDA prefill")
        journal.begin(rid)
        if input_ids is not None:
            if input_ids.cpu().tolist() != journal.prompt_ids:
                raise ValueError("model inputs differ from requested token IDs")
            journal.record("forward.input_ids", input_ids)
        if embeds is not None:
            journal.record("forward.inputs_embeds", embeds)
        journal.record("forward.positions", positions)
        token = ACTIVE.set(journal)
        try:
            result = original(**kwargs)
            if not isinstance(result, torch.Tensor) or result.shape != (640, 4096):
                raise ValueError("model journal requires the complete GLM53 output")
            journal.record("forward.output", result)
            journal.record("forward.prompt_head_rows", result[:639])
            journal.finish()
            return result
        finally:
            ACTIVE.reset(token)

    runner._model_forward = traced
