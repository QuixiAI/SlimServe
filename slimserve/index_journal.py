# SPDX-License-Identifier: Apache-2.0
"""Bounded read-only selector trace for one repeated 8199-token GLM request.

All hooks are absent when disabled. Copies/hashes are diagnostic synchronization;
no GPU values are changed. Undefined logit tails are zeroed ONLY in private CPU
copies before hashing. Save one configured layer's first chunk for offline replay.
"""

import hashlib
import inspect
import json
import os
import re
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import torch

from slimserve.glm53_ordering import enabled as native_order_enabled

ACTIVE = ContextVar("glm53_index_request", default=None)
LAYER = ContextVar("glm53_index_layer", default=None)
LAYERS = tuple(range(3, 44, 4))
MAX_FILE_BYTES = 1024**3
MAX_TENSOR_BYTES = 128 * 1024**2


def enabled():
    path = os.environ.get("SLIMSERVE_GLM53_INDEX_JOURNAL")
    if path and (
        os.environ.get("SLIMSERVE_GLM53_MODEL_JOURNAL") != "1"
        or (
            not native_order_enabled()
            and os.environ.get("SLIMSERVE_GLM53_CANONICAL_MOE") != "1"
        )
    ):
        raise ValueError(
            "index journal requires bounded model journal and canonical MoE"
        )
    return bool(path)


def cpu_logits(logits, starts, ends):
    """Keep exactly the native operation's defined row ranges, without aliasing."""
    if logits.dim() != 2:
        raise ValueError("invalid indexer logit dimensions")
    host = logits.detach().to(device="cpu", copy=True).contiguous()
    lo, hi = (t.detach().cpu().contiguous() for t in (starts, ends))
    rows, columns = host.shape
    if (
        host.dtype != torch.float32
        or lo.dtype != torch.int32
        or hi.dtype != torch.int32
        or lo.shape != (rows,)
        or hi.shape != (rows,)
        or not bool(((lo >= 0) & (hi >= lo) & (hi <= columns)).all())
    ):
        raise ValueError("invalid indexer logit ranges")
    col = torch.arange(columns)
    host.masked_fill_((col[None, :] < lo[:, None]) | (col[None, :] >= hi[:, None]), 0)
    return host, lo, hi


class IndexJournal:
    def __init__(self, config_path):
        raw = config_path.read_bytes()
        config = json.loads(raw)
        if (
            not isinstance(config, dict)
            or set(config) - {"capture_layer"}
            != {"schema", "prompt_ids", "max_matches", "output_directory"}
            or type(config["schema"]) is not int
            or config["schema"] != 1
            or not isinstance(config["prompt_ids"], list)
            or len(config["prompt_ids"]) != 8199
            or any(type(i) is not int or i < 0 for i in config["prompt_ids"])
            or type(config["max_matches"]) is not int
            or not 1 <= config["max_matches"] <= 3
            or not isinstance(config["output_directory"], str)
            or not config["output_directory"]
            or type(config.get("capture_layer", 3)) is not int
            or config.get("capture_layer", 3) not in LAYERS
        ):
            raise ValueError(
                "index journal requires 8199 token IDs, 1..3 matches "
                "and a DSA capture layer"
            )
        self.prompt_ids = config["prompt_ids"]
        self.limit = config["max_matches"]
        self.capture_layer = config.get("capture_layer", 3)
        self.match = self.cursor = self.chunk = self.saved_bytes = 0
        self.request = self.device = None
        self.in_forward = False
        self.layers = []
        self.layer_calls = 0
        self.records = set()
        directory = Path(config["output_directory"])
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"index-{os.getpid()}.jsonl"
        self.stream = self.path.open("x", buffering=1)
        self.archive = directory / f"index-{os.getpid()}"
        self.archive.mkdir(exist_ok=False)
        root = Path(__file__).resolve().parents[1]
        sources = (
            "slimserve/canonical_indexer.py",
            "slimserve/canonical_indexer_kernel.py",
            "slimserve/index_journal.py",
            "vllm/_custom_ops.py",
            "vllm/model_executor/layers/glm5_next_indexer.py",
            "vllm/v1/worker/gpu_model_runner.py",
            "slimserve/canonical_moe.py",
            "slimserve/glm53_ordering.py",
            "slimserve/canonical_moe_kernel.py",
            "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py",
        )
        self.write(
            {
                "kind": "header",
                "schema": 1,
                "diagnostic_only": True,
                "pid": os.getpid(),
                "config_sha256": hashlib.sha256(raw).hexdigest(),
                "prompt_ids": self.prompt_ids,
                "max_matches": self.limit,
                "max_chunks_per_match": 8,
                "expected_layers": list(LAYERS),
                "capture_layer": self.capture_layer,
                "max_tensor_bytes": MAX_TENSOR_BYTES,
                "max_worker_archive_bytes": MAX_FILE_BYTES,
                "undefined_logit_tails": "zeroed in private CPU copies only",
                "selection_order": (
                    "canonical-pool-id"
                    if native_order_enabled()
                    or os.getenv("SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER") == "1"
                    else "native"
                ),
                "selection_ties": (
                    "smaller-pool-id"
                    if native_order_enabled()
                    or os.getenv("SLIMSERVE_GLM53_CANONICAL_INDEX_TIES") == "1"
                    else "native"
                ),
                "selection_order_implementation": (
                    "native-bitonic"
                    if native_order_enabled()
                    or os.getenv("SLIMSERVE_GLM53_CANONICAL_INDEX_FUSED") == "1"
                    else "post-sort"
                    if os.getenv("SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER") == "1"
                    else "native"
                ),
                "implementation_sha256": {
                    p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                    for p in sources
                },
            }
        )

    def write(self, item):
        self.stream.write(json.dumps(item, separators=(",", ":")) + "\n")

    def begin(self, request, computed, tokens, device):
        if self.in_forward:
            raise ValueError("overlapping index forward")
        if computed == 0:
            if self.request is not None or self.match >= self.limit:
                raise ValueError("extra or overlapping index request")
            self.match += 1
            self.cursor = self.chunk = 0
            self.request = request
            self.device = str(device)
            self.write(
                {"kind": "request_begin", "match": self.match, "request_id": request}
            )
        if (
            request != self.request
            or computed != self.cursor
            or str(device) != self.device
            or not 1 <= tokens <= 8192
            or computed + tokens > 8199
            or self.chunk >= 8
        ):
            raise ValueError("index request chunk contract changed")
        self.chunk += 1
        self.in_forward = True
        self.tokens = tokens
        self.layers = []
        self.write(
            {
                "kind": "forward_begin",
                "match": self.match,
                "chunk": self.chunk,
                "computed_tokens": computed,
                "tokens": tokens,
                "device": self.device,
            }
        )

    def snapshot(self, stage, tensor, save=False):
        if not self.in_forward or re.fullmatch(r"[a-z0-9_.-]+", stage) is None:
            raise ValueError("invalid index snapshot stage")
        key = self.match, self.chunk, stage
        if key in self.records:
            raise ValueError("duplicate index snapshot")
        self.records.add(key)
        nbytes = tensor.numel() * tensor.element_size()
        if not 0 < nbytes <= MAX_TENSOR_BYTES:
            raise ValueError("index snapshot exceeds byte bound")
        host = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256(
            memoryview(host.reshape(-1).view(torch.uint8).numpy())
        ).hexdigest()
        item = {
            "kind": "tensor",
            "match": self.match,
            "chunk": self.chunk,
            "stage": stage,
            "shape": list(host.shape),
            "dtype": str(host.dtype),
            "nbytes": nbytes,
            "sha256": digest,
        }
        if save:
            if host.storage_offset() or host.untyped_storage().nbytes() != nbytes:
                host = host.clone()
            if self.saved_bytes + nbytes + 65536 > MAX_FILE_BYTES:
                raise ValueError("index archive exceeds worker byte bound")
            path = self.archive / f"match-{self.match}-chunk-{self.chunk}-{stage}.pt"
            with path.open("xb") as stream:
                torch.save(host, stream)
            size = path.stat().st_size
            if size > nbytes + 65536:
                raise ValueError("unexpected index archive overhead")
            self.saved_bytes += size
            with path.open("rb") as stream:
                file_sha = hashlib.file_digest(stream, "sha256").hexdigest()
            item["archive"] = {
                "path": str(path),
                "bytes": size,
                "sha256": file_sha,
                "worker_saved_bytes": self.saved_bytes,
            }
        self.write(item)

    def finish(self):
        if not self.in_forward or tuple(self.layers) != LAYERS:
            raise ValueError("incomplete GLM indexer layer coverage")
        self.in_forward = False
        self.cursor += self.tokens
        self.write(
            {
                "kind": "forward_complete",
                "match": self.match,
                "chunk": self.chunk,
                "computed_tokens": self.cursor,
                "layers": self.layers,
            }
        )
        if self.cursor == 8199:
            self.write(
                {
                    "kind": "request_complete",
                    "match": self.match,
                    "chunks": self.chunk,
                    "worker_saved_bytes": self.saved_bytes,
                }
            )
            self.request = None

    def close(self):
        self.stream.close()


def instrument_pooled_indexer(function):
    if not enabled():
        return function
    signature = inspect.signature(function)

    @wraps(function)
    def traced(*args, **kwargs):
        journal = ACTIVE.get()
        if journal is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        prefix = bound.arguments["k_cache_prefix"]
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", prefix)
        if match is None or LAYER.get() is not None:
            raise ValueError("unrecognized or nested indexer layer")
        layer = int(match.group(1))
        if len(journal.layers) >= len(LAYERS) or layer != LAYERS[len(journal.layers)]:
            raise ValueError("indexer layer order changed")
        journal.layers.append(layer)
        journal.layer_calls = 0
        journal.layer_rows = 0
        token = LAYER.set(layer)
        try:
            result = function(*args, **kwargs)
            if journal.layer_calls == 0 or journal.layer_rows != journal.tokens:
                raise ValueError("incomplete indexer selector row coverage")
            return result
        finally:
            LAYER.reset(token)

    return traced


def instrument_topk(function):
    if not enabled():
        return function
    signature = inspect.signature(function)

    @wraps(function)
    def traced(*args, **kwargs):
        journal, layer = ACTIVE.get(), LAYER.get()
        if journal is None:
            return function(*args, **kwargs)
        if layer is None or journal.layer_calls >= 8:
            raise ValueError("missing indexer layer or excess selector calls")
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        logits, starts, ends, indices = (
            a[k] for k in ("logits", "cu_seqlen_ks", "cu_seqlen_ke", "raw_topk_indices")
        )
        rows = a["num_rows"]
        if (
            not 1 <= rows <= journal.tokens
            or logits.dim() != 2
            or logits.shape[0] != rows
            or not 1 <= logits.shape[1] <= 2049
            or a["stride0"] != logits.stride(0)
            or a["stride1"] != 1
            or logits.stride(1) != 1
            or a["topk_tokens"] != 512
            or indices.shape != (rows, 512)
            or indices.dtype != torch.int32
            or any(
                str(t.device) != journal.device for t in (logits, starts, ends, indices)
            )
        ):
            raise ValueError("long-context index selector shape/recipe changed")
        stage = f"layer-{layer:02d}.call-{journal.layer_calls}"
        save = (
            layer == journal.capture_layer
            and journal.chunk == 1
            and journal.layer_calls == 0
        )
        journal.layer_calls += 1
        journal.layer_rows += rows
        if journal.layer_rows > journal.tokens:
            raise ValueError("excess indexer selector rows")
        host, lo, hi = cpu_logits(logits, starts, ends)
        journal.snapshot(stage + ".logits", host, save)
        journal.snapshot(stage + ".starts", lo, save)
        journal.snapshot(stage + ".ends", hi, save)
        result = function(*args, **kwargs)
        journal.snapshot(stage + ".indices", indices, save)
        return result

    return traced


def install_index_journal(runner):
    if not enabled():
        return
    config = runner.model_config.hf_text_config
    if (
        config.hidden_size != 4096
        or config.num_hidden_layers != 45
        or runner.parallel_config.tensor_parallel_size != 4
        or runner.parallel_config.pipeline_parallel_size != 1
        or runner.speculative_config is not None
    ):
        raise ValueError("index journal requires GLM53 TP4 no-spec")
    journal = IndexJournal(Path(os.environ["SLIMSERVE_GLM53_INDEX_JOURNAL"]))
    runner._slimserve_index_journal = journal
    original = runner._model_forward

    @wraps(original)
    def traced(**kwargs):
        requests = [
            rid
            for rid in runner.num_prompt_logprobs
            if runner.requests[rid].prompt_token_ids == journal.prompt_ids
        ]
        if not requests:
            if journal.request is not None:
                raise ValueError("incomplete long-context request disappeared")
            return original(**kwargs)
        if len(requests) != 1 or runner.input_batch.num_reqs != 1:
            raise ValueError("index journal requires one unmixed request")
        rid = requests[0]
        if runner.num_prompt_logprobs[rid] != 0:
            raise ValueError("index journal requires target-only prompt scores")
        input_ids, embeds, positions = (
            kwargs.get(k) for k in ("input_ids", "inputs_embeds", "positions")
        )
        tokens = input_ids if input_ids is not None else embeds
        if (
            tokens is None
            or not tokens.is_cuda
            or torch.cuda.is_current_stream_capturing()
        ):
            raise ValueError("index journal requires uncaptured CUDA prefill")
        computed = runner.requests[rid].num_computed_tokens
        journal.begin(rid, computed, tokens.shape[0], tokens.device)
        if positions.cpu().tolist() != list(
            range(computed, computed + tokens.shape[0])
        ):
            raise ValueError("index request positions changed")
        if input_ids is not None:
            if (
                input_ids.cpu().tolist()
                != journal.prompt_ids[computed : computed + tokens.shape[0]]
            ):
                raise ValueError("index request token IDs changed")
            journal.snapshot("forward.input_ids", input_ids)
        if embeds is not None:
            journal.snapshot("forward.inputs_embeds", embeds)
        journal.snapshot("forward.positions", positions)
        token = ACTIVE.set(journal)
        try:
            result = original(**kwargs)
            if not isinstance(result, torch.Tensor) or result.shape != (
                tokens.shape[0],
                4096,
            ):
                raise ValueError("long-context model output changed")
            journal.snapshot("forward.output", result)
            journal.finish()
            return result
        finally:
            ACTIVE.reset(token)

    runner._model_forward = traced
