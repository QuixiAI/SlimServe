# SPDX-License-Identifier: Apache-2.0
"""Opt-in, bounded prompt-score fingerprints; diagnostic synchronization only.

Never changes tensor values. Copies one stage at a time to CPU and retains
only metadata/digests, not model tensors. Must not be used for performance claims.
"""

import hashlib
import json
import os
from pathlib import Path

STAGES = (
    "prompt_head_input",
    "logits",
    "scores",
    "target_token_ids",
    "selected_logprobs",
)


class ScoreJournal:
    @classmethod
    def from_env(cls, model_type):
        path = os.environ.get("SLIMSERVE_GLM53_SCORE_JOURNAL")
        if not path:
            return None
        if model_type not in ("glm5_next", "glm5_next_text"):
            raise ValueError("score journal is scoped to the GLM53 diagnostic")
        return cls(Path(path))

    def __init__(self, path):
        raw = path.read_bytes()
        config = json.loads(raw)
        if not isinstance(config, dict) or set(config) != {
            "schema",
            "prompt_ids",
            "max_matches",
            "output_directory",
        }:
            raise ValueError("invalid score journal configuration fields")
        ids, limit = config["prompt_ids"], config["max_matches"]
        if (
            type(config["schema"]) is not int
            or config["schema"] != 1
            or not isinstance(ids, list)
            or len(ids) != 640
            or any(type(t) is not int or t < 0 for t in ids)
            or type(limit) is not int
            or not 1 <= limit <= 8
            or not isinstance(config["output_directory"], str)
            or not config["output_directory"]
        ):
            raise ValueError("score journal requires 640 token IDs and 1..8 matches")
        self.prompt_ids, self.limit = ids, limit
        self.matches, self.active = 0, None
        directory = Path(config["output_directory"])
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"score-{os.getpid()}.jsonl"
        self.stream = self.path.open("x", buffering=1)
        root = Path(__file__).resolve().parents[1]
        source_paths = (
            "slimserve/canonical_indexer.py",
            "slimserve/canonical_indexer_kernel.py",
            "slimserve/index_journal.py",
            "vllm/model_executor/layers/glm5_next_indexer.py",
            "slimserve/score_journal.py",
            "vllm/v1/worker/gpu_model_runner.py",
            "vllm/model_executor/models/glm5_next.py",
            "vllm/v1/sample/sampler.py",
            "slimserve/model_journal.py",
            "vllm/model_executor/layers/glm5_next_mhc_ops.py",
            "slimserve/moe_journal.py",
            "vllm/_custom_ops.py",
            "vllm/quixicore/ops.py",
            "vllm/model_executor/layers/fused_moe/router/fused_moe_router.py",
            "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py",
            "slimserve/canonical_moe.py",
            "slimserve/canonical_moe_kernel.py",
            "vllm/model_executor/layers/fused_moe/router/glm_route_align.py",
            "csrc/quixicore/serving/glm_moe_routing.cuh",
            "csrc/quixicore/serving/glm_moe_stable_align.cuh",
            "vllm/model_executor/layers/fused_moe/router/glm_stable_align.py",
            "csrc/quixicore/tm_cuda/tm_cuda_serving.cu",
        )
        self._write(
            {
                "kind": "header",
                "schema": 1,
                "pid": os.getpid(),
                "diagnostic_only": True,
                "config_sha256": hashlib.sha256(raw).hexdigest(),
                "implementation_sha256": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in source_paths
                },
                "prompt_ids": ids,
                "max_matches": limit,
                "stages": STAGES,
                "max_tensor_bytes": 1024**3,
            }
        )

    def _write(self, row):
        self.stream.write(json.dumps(row, separators=(",", ":")) + "\n")

    def begin(self, prompt_ids, start_idx, num_logits, num_logprobs, request_id):
        if prompt_ids != self.prompt_ids:
            return None
        if self.active is not None or self.matches >= self.limit:
            raise ValueError("unexpected extra or overlapping score trace match")
        if start_idx != 0 or num_logits != 639 or num_logprobs != 0:
            raise ValueError(
                "score trace requires one complete uncached 640-token prompt"
            )
        self.matches += 1
        self.active = []
        self._write({"kind": "begin", "match": self.matches, "request_id": request_id})
        return self.matches

    def record(self, match, stage, tensor):
        import torch

        if (
            match != self.matches
            or self.active is None
            or len(self.active) >= len(STAGES)
            or stage != STAGES[len(self.active)]
        ):
            raise ValueError("score trace stage order changed")
        nbytes = tensor.numel() * tensor.element_size()
        if tensor.ndim == 0 or tensor.shape[0] != 639 or nbytes > 1024**3:
            raise ValueError("score trace shape or byte bound exceeded")
        if stage in ("target_token_ids", "selected_logprobs") and tensor.numel() != 639:
            raise ValueError("score trace requires one target and score per row")
        # Blocking D2H is deliberate instrumentation. This trace is never a
        # timing baseline and cannot, by itself, exonerate a hidden race.
        host = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256(memoryview(host.view(torch.uint8).numpy())).hexdigest()
        self._write(
            {
                "kind": "tensor",
                "match": match,
                "stage": stage,
                "device": str(tensor.device),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "nbytes": nbytes,
                "sha256": digest,
                # Small final vectors allow exact comparison with HTTP results.
                "values": host.tolist()
                if stage in ("target_token_ids", "selected_logprobs")
                else None,
            }
        )
        self.active.append(stage)

    def finish(self, match):
        if match != self.matches or self.active != list(STAGES):
            raise ValueError("incomplete score trace")
        self._write({"kind": "complete", "match": match})
        self.active = None

    def close(self):
        self.stream.close()
