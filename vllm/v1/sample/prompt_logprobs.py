# SPDX-License-Identifier: Apache-2.0
"""Opt-in bounded row-wise GLM53 prompt scoring; not a profile default.

Keep the full, unchanged lm_head result. Only split the subsequent independent
row operations, using the existing sampler's log_softmax/top-k/rank functions.

The native-order diagnostic passed exact-score/memory checks; production-order
rollout stopped on the unchanged control's historical quality gate. Full records:
docs/glm53-flash-sm120-review.md, "Qualification record map". No speedup claim.
"""

import torch

from vllm.v1.outputs import LogprobsTensors

CHUNK_ROWS = 1024
MODES = ("raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits")


def gather_prompt_logprobs(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    num_logprobs: int,
    mode: str,
    *,
    sampler,
    chunk_rows: int = CHUNK_ROWS,
) -> LogprobsTensors:
    """Bound score/conversion temporaries, not the requested output size.

    No sampling processors apply to prompt tokens. ``processed_*`` therefore
    has the same meaning as ``raw_*``, as in GPUModelRunner. The caller still
    owns the projection, TP gather, journal and asynchronous CPU transfer.
    """
    if (
        logits.ndim != 2
        or not logits.is_floating_point()
        or logits.shape[0] <= 0
        or logits.shape[1] <= 0
        or target_token_ids.shape != (logits.shape[0],)
        or target_token_ids.dtype != torch.int64
        or target_token_ids.device != logits.device
        or type(num_logprobs) is not int
        or not 0 <= num_logprobs <= logits.shape[1]
        or mode not in MODES
        or type(chunk_rows) is not int
        or chunk_rows <= 0
    ):
        raise ValueError("invalid prompt-score matrix, targets, mode or chunk bound")
    rows = logits.shape[0]
    output = None
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        piece = logits[start:end]
        scores = (
            piece.to(torch.float32)
            if mode in ("raw_logits", "processed_logits")
            else sampler.compute_logprobs(piece)
        )
        part = sampler.gather_logprobs(
            scores, num_logprobs, target_token_ids[start:end]
        )
        if part.cu_num_generated_tokens is not None:
            raise ValueError("prompt scoring requires flat per-row outputs")
        if rows <= chunk_rows:
            return part
        if output is None:
            output = LogprobsTensors(
                *(t.new_empty((rows, *t.shape[1:])) for t in part[:3])
            )
        for destination, source in zip(output[:3], part[:3]):
            destination[start:end].copy_(source)
        # Do not retain the preceding chunk's score matrix while creating the
        # next one. Outputs own independent, requested-size storage only.
        del scores, part, piece, source
    assert output is not None
    return output
