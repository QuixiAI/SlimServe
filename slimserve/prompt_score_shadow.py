# SPDX-License-Identifier: Apache-2.0
"""Opt-in same-live-input scoring comparison, never a timing/default policy.

Runs the existing full scorer and (above 1024 rows) the unchanged chunked scorer.
Blocking CPU copies deliberately verify input preservation and exact output bits.
Only small score outputs and hashes are retained, never the vocabulary matrices.
"""

import hashlib
import json
import os
from pathlib import Path

FLAG = "SLIMSERVE_GLM53_PROMPT_SCORE_SHADOW"
ROOT = Path(__file__).resolve().parents[1]
VOCAB = 154880
MAX_ROWS = 8192
COPY_BYTES = 32 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prompt_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def validate_environment():
    if not os.getenv(FLAG):
        return
    require(
        os.getenv("SLIMSERVE_GLM53_PROMPT_SCORE_CHUNKS") == "1",
        "shadow requires chunks=1",
    )
    allowed = {FLAG, "SLIMSERVE_GLM53_PROMPT_SCORE_CHUNKS"}
    conflicts = [
        key
        for key, value in os.environ.items()
        if key.startswith("SLIMSERVE_GLM53_")
        and key not in allowed
        and value not in ("", "0")
    ]
    require(
        not conflicts,
        f"shadow cannot combine with other model diagnostics: {conflicts}",
    )
    require(
        os.getenv("VLLM_FORCE_AOT_LOAD", "0") in ("", "0"),
        "shadow uses normal compilation",
    )
    require(os.getenv("VLLM_GLM5_MHC_PREFILL_TC", "0") == "0", "shadow preserves TC0")
    directory = Path(os.environ[FLAG]).resolve()
    require(
        directory.is_relative_to(ROOT / "perf/results"),
        "shadow output must be under perf/results",
    )


def validate_plan(plan):
    if not os.getenv(FLAG):
        return
    validate_environment()
    engine = plan.engine
    options = engine.get("compilation_config", {}).get("inductor_compile_config", {})
    require(
        plan.profile_id == "glm53-nvfp4-4"
        and plan.platform == "rtx6000"
        and plan.gpus == 4
        and plan.quant.name == "NVFP4"
        and (plan.weight_recipe or {}).get("id")
        == "glm53-redhatai-nvfp4-fp8-kda-tp4-v1"
        and engine.get("tensor_parallel_size") == 4
        and not engine.get("enable_expert_parallel", False)
        and not plan.speculative
        and engine.get("moe_backend") == "marlin"
        and engine.get("kv_cache_dtype", "auto") == "auto"
        and engine.get("dtype", "auto") in ("auto", "bfloat16", "bf16")
        and not options.get("deterministic")
        and options.get("combo_kernels") is not False,
        "shadow requires the unchanged GLM53 SM120 TP4 recipe/compiler policy",
    )


def device_check(logits):
    import torch

    require(
        logits.is_cuda and torch.cuda.get_device_capability(logits.device) == (12, 0),
        "shadow requires SM120 CUDA logits",
    )


def tensor_digest(tensor):
    """Hash logical tensor bytes using at most COPY_BYTES of host staging."""
    import torch

    require(tensor.ndim > 0 and tensor.shape[0] > 0, "empty shadow tensor")
    bytes_per_row = tensor[0].numel() * tensor.element_size()
    require(0 < bytes_per_row <= COPY_BYTES, "shadow row exceeds host staging bound")
    rows = max(1, COPY_BYTES // bytes_per_row)
    digest = hashlib.sha256()
    for start in range(0, tensor.shape[0], rows):
        host = tensor[start : start + rows].detach().cpu().contiguous()
        digest.update(memoryview(host.view(torch.uint8).numpy()))
        del host
    return digest.hexdigest()


class PromptScoreShadow:
    @classmethod
    def from_env(cls, runner):
        if not os.getenv(FLAG):
            return None
        import torch

        validate_environment()
        config, parallel = runner.model_config.hf_text_config, runner.parallel_config
        require(
            config.model_type in ("glm5_next", "glm5_next_text")
            and config.hidden_size == 4096
            and config.num_hidden_layers == 45
            and parallel.tensor_parallel_size == 4
            and parallel.pipeline_parallel_size == 1
            and not parallel.enable_expert_parallel
            and runner.speculative_config is None
            and runner.dtype is torch.bfloat16
            and runner.kv_cache_dtype is torch.bfloat16,
            "shadow worker must match fixed GLM53 TP4 BF16 recipe",
        )
        return cls(Path(os.environ[FLAG]), parallel.rank)

    def __init__(self, directory, rank):
        require(type(rank) is int and 0 <= rank < 4, "invalid shadow rank")
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"shadow-rank{rank}-pid{os.getpid()}.jsonl"
        self.stream = self.path.open("x", buffering=1)
        self.calls = 0
        sources = (
            "slimserve/prompt_score_shadow.py",
            "vllm/v1/worker/gpu_model_runner.py",
            "vllm/v1/sample/prompt_logprobs.py",
            "vllm/v1/sample/sampler.py",
            "vllm/v1/sample/ops/logprobs.py",
        )
        self.write(
            dict(
                kind="header",
                schema=1,
                rank=rank,
                pid=os.getpid(),
                diagnostic_only=True,
                copy_bytes=COPY_BYTES,
                max_rows=MAX_ROWS,
                vocab=VOCAB,
                sources={
                    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                    for name in sources
                },
            )
        )

    def write(self, row):
        self.stream.write(
            json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n"
        )

    def gather(
        self,
        logits,
        targets,
        count,
        mode,
        *,
        sampler,
        prompt_ids,
        start_idx,
        request_id,
    ):
        import torch

        from vllm.v1.sample.prompt_logprobs import CHUNK_ROWS, gather_prompt_logprobs

        self.calls += 1
        require(self.calls <= 512, "shadow exceeds prescribed call bound")
        self.write(
            dict(
                kind="begin",
                call=self.calls,
                request_id=request_id,
                prompt_sha256=prompt_digest(prompt_ids),
                prompt_tokens=len(prompt_ids),
                start_idx=start_idx,
                rows=logits.shape[0],
            )
        )
        try:
            device_check(logits)
            require(
                logits.ndim == 2
                and 0 < logits.shape[0] <= MAX_ROWS
                and logits.shape[1] == VOCAB
                and logits.dtype is torch.bfloat16
                and targets.dtype is torch.int64
                and targets.shape == (logits.shape[0],)
                and targets.device == logits.device
                and type(count) is int
                and 0 <= count <= 5
                and mode == "raw_logprobs"
                and type(start_idx) is int
                and start_idx >= 0
                and start_idx + logits.shape[0] <= len(prompt_ids) - 1,
                "shadow input contract changed",
            )
            require(
                targets.cpu().tolist()
                == prompt_ids[start_idx + 1 : start_idx + 1 + logits.shape[0]],
                "shadow target offsets changed",
            )
            paired = logits.shape[0] > CHUNK_ROWS
            before = (tensor_digest(logits), tensor_digest(targets)) if paired else None
            # Reference is exactly the unchunked runner's existing operations.
            scores = sampler.compute_logprobs(logits)
            reference = sampler.gather_logprobs(scores, count, targets)
            del scores
            require(
                reference.cu_num_generated_tokens is None, "unexpected packed reference"
            )
            if paired:
                require(
                    before == (tensor_digest(logits), tensor_digest(targets)),
                    "reference mutated inputs",
                )
                actual = gather_prompt_logprobs(
                    logits, targets, count, mode, sampler=sampler
                )
                require(
                    before == (tensor_digest(logits), tensor_digest(targets)),
                    "chunked scorer mutated inputs",
                )
            else:
                # Preserve normal small-request dispatch. These rows are NOT
                # reported as evidence for the chunked scorer.
                actual = reference
            require(
                actual.cu_num_generated_tokens is None, "unexpected packed candidate"
            )
            require(
                torch.isfinite(reference.logprobs).all().item()
                and torch.isfinite(actual.logprobs).all().item(),
                "nonfinite prompt scores",
            )
            fields = []
            for expected, observed in zip(reference[:3], actual[:3], strict=True):
                require(
                    expected.shape == observed.shape
                    and expected.dtype == observed.dtype,
                    "shadow output geometry changed",
                )
                expected_sha, actual_sha = (
                    tensor_digest(expected),
                    tensor_digest(observed),
                )
                require(expected_sha == actual_sha, "shadow output bits differ")
                fields.append(
                    dict(
                        shape=list(observed.shape),
                        dtype=str(observed.dtype),
                        reference_sha256=expected_sha,
                        chunked_sha256=actual_sha,
                    )
                )
            self.write(
                dict(
                    kind="complete",
                    call=self.calls,
                    paired=paired,
                    count=count,
                    mode=mode,
                    input_sha256=before,
                    outputs=fields,
                    token_ids=actual.logprob_token_ids[:, 0].cpu().tolist(),
                    logprobs=actual.logprobs[:, 0].cpu().tolist(),
                    ranks=actual.selected_token_ranks.cpu().tolist(),
                )
            )
            return actual
        except BaseException as error:
            self.write(dict(kind="failed", call=self.calls, error=repr(error)))
            raise
