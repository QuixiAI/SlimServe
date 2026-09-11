# SPDX-License-Identifier: Apache-2.0
"""Bounded first-MoE snapshots; disabled decorators are identity functions."""

import hashlib
import inspect
import os
import re
from functools import wraps

from slimserve.model_journal import ACTIVE
from slimserve.model_journal import enabled as model_journal_enabled
from slimserve.score_journal import private_directory, private_open

MAX_DUMP_BYTES = 2 * 1024**3


def enabled():
    flag = os.environ.get("SLIMSERVE_GLM53_MOE_JOURNAL", "0")
    if flag not in ("0", "1"):
        raise ValueError("SLIMSERVE_GLM53_MOE_JOURNAL must be 0 or 1")
    if flag == "1" and not model_journal_enabled():
        raise ValueError("MoE journal requires the bounded model journal")
    return flag == "1"


class MoECapture:
    def __init__(self, journal):
        self.journal = journal
        self.seen = set()
        self.gemm_calls = {}
        self.completed = set()
        self.saved_bytes = 0
        self.directory = journal.path.parent / f"moe-{os.getpid()}"
        private_directory(self.directory, exist_ok=False)

    def snapshot(self, stage, tensor, parameter=False):
        import torch

        if re.fullmatch(r"(?:router|up|down|sum)\.[a-z_]+", stage) is None:
            raise ValueError("invalid MoE snapshot stage")
        match = self.journal.matches
        key = match, stage
        if key in self.seen:
            raise ValueError("duplicate first-MoE stage")
        self.seen.add(key)
        host = self.journal.record("moe3." + stage, tensor)
        if parameter and match != 1:
            return
        # Ensure Torch serialization cannot retain an oversized backing storage.
        nbytes = host.numel() * host.element_size()
        if host.storage_offset() or host.untyped_storage().nbytes() != nbytes:
            host = host.clone()
        if self.saved_bytes + nbytes + 65536 > MAX_DUMP_BYTES:
            raise ValueError("first-MoE worker dump byte bound exceeded")
        suffix = "parameter" if parameter else f"match-{match}"
        path = self.directory / f"{stage}-{suffix}.pt"
        with private_open(path, binary=True) as stream:
            torch.save(host, stream)
        size = path.stat().st_size
        if size > nbytes + 65536:
            raise ValueError("unexpected tensor serialization overhead")
        self.saved_bytes += size
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        self.journal.write(
            {
                "kind": "moe_snapshot",
                "match": match,
                "stage": "moe3." + stage,
                "path": str(path),
                "file_bytes": size,
                "file_sha256": digest.hexdigest(),
                "parameter": parameter,
                "worker_saved_bytes": self.saved_bytes,
            }
        )


def _capture():
    journal = ACTIVE.get()
    if journal is None or journal.operations != 8 or journal.pending is not None:
        return None
    if journal.moe_capture is None:
        journal.moe_capture = MoECapture(journal)
    return journal.moe_capture


def _decorate(function, operation):
    if not enabled():
        return function
    signature = inspect.signature(function)

    @wraps(function)
    def traced(*args, **kwargs):
        capture = _capture()
        if capture is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return operation(capture, bound.arguments, lambda: function(*args, **kwargs))

    return traced


def _router(capture, args, call):
    if args["hidden_states"].shape != (640, 4096) or args["router_logits"].shape != (
        640,
        288,
    ):
        raise ValueError("first-MoE router shape changed")
    capture.snapshot("router.input", args["hidden_states"])
    capture.snapshot("router.logits", args["router_logits"])
    weights, ids = result = call()
    if weights.shape != (640, 8) or ids.shape != (640, 8):
        raise ValueError("first-MoE routes changed")
    capture.snapshot("router.weights", weights)
    capture.snapshot("router.ids", ids)
    return result


def instrument_router(function):
    return _decorate(function, _router)


def _marlin(capture, args, call):
    import torch

    match = capture.journal.matches
    index = capture.gemm_calls.get(match, 0)
    if index not in (0, 1):
        raise ValueError("extra first-MoE GEMM")
    stage = "up" if index == 0 else "down"
    expected = ((640, 1024, 4096, 8), (5120, 4096, 512, 1))[index]
    if (
        tuple(args[k] for k in ("size_m", "size_n", "size_k", "top_k")) != expected
        or args["moe_block_size"] != 32
        or args["input"].dtype != torch.bfloat16
        or args["use_atomic_add"]
        or not args["use_fp32_reduce"]
        or args["topk_weights"].shape != (640, 8)
    ):
        raise ValueError("first-MoE Marlin recipe changed")
    optional = ("b_bias", "a_scales", "b_qzeros", "g_idx", "perm")
    if any(args[k] is not None and args[k].numel() != 0 for k in optional):
        raise ValueError("unexpected optional tensor in first-MoE Marlin")
    capture.gemm_calls[match] = index + 1
    capture.journal.write(
        {
            "kind": "moe_gemm",
            "match": match,
            "stage": stage,
            "parameters": {
                k: args[k]
                for k in (
                    "size_m",
                    "size_n",
                    "size_k",
                    "top_k",
                    "moe_block_size",
                    "mul_topk_weights",
                    "is_k_full",
                    "use_atomic_add",
                    "use_fp32_reduce",
                    "is_zp_float",
                    "thread_k",
                    "thread_n",
                    "blocks_per_sm",
                )
            },
            "b_q_type_id": args["b_q_type"].id,
            "optional": {
                k: None
                if args[k] is None
                else {"shape": list(args[k].shape), "dtype": str(args[k].dtype)}
                for k in optional
            },
            "global_scale_present": args["global_scale"] is not None,
        }
    )
    for key in ("input", "b_qweight", "b_scales", "global_scale"):
        if args[key] is not None:
            capture.snapshot(stage + "." + key, args[key], parameter=key != "input")
    capture.snapshot(stage + ".workspace_before", args["workspace"])
    # Only defined alignment prefixes count as evidence, not uninitialized
    # capacity beyond the actual padded token/block counts.
    used = int(args["num_tokens_past_padded"].item())
    block = args["moe_block_size"]
    if not (
        0 < used <= args["sorted_token_ids"].numel()
        and used % block == 0
        and used // block <= args["expert_ids"].numel()
    ):
        raise ValueError("invalid first-MoE alignment extent")
    capture.snapshot(stage + ".sorted_ids", args["sorted_token_ids"][:used])
    capture.snapshot(stage + ".expert_ids", args["expert_ids"][: used // block])
    capture.snapshot(stage + ".padded_count", args["num_tokens_past_padded"])
    capture.snapshot(stage + ".topk_weights", args["topk_weights"])
    result = call()
    capture.snapshot(stage + ".output", result)
    capture.snapshot(stage + ".workspace_after", args["workspace"])
    return result


def instrument_marlin_gemm(function):
    return _decorate(function, _marlin)


def _sum(capture, args, call):
    match = capture.journal.matches
    if (
        capture.gemm_calls.get(match) != 2
        or match in capture.completed
        or args["x"].shape != (640, 8, 4096)
        or args["shared"].shape != (640, 4096)
    ):
        raise ValueError("first-MoE shared sum changed")
    capture.snapshot("sum.input", args["x"])
    capture.snapshot("sum.shared", args["shared"])
    result = call()
    capture.snapshot("sum.output", args["out"])
    capture.completed.add(match)
    capture.journal.write(
        {
            "kind": "moe_complete",
            "match": match,
            "gemm_calls": 2,
            "worker_saved_bytes": capture.saved_bytes,
        }
    )
    return result


def instrument_moe_sum(function):
    return _decorate(function, _sum)
