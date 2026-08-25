# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.v1.worker.gpu.async_utils import make_output_copy_stream, stream
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


class StructuredOutputsWorker:
    def __init__(self, max_num_logits: int, vocab_size: int, device: torch.device):
        self.logits_indices = torch.zeros(
            max_num_logits, dtype=torch.int32, device=device
        )
        self.grammar_bitmask = torch.zeros(
            (max_num_logits, cdiv(vocab_size, 32)), dtype=torch.int32, device=device
        )
        self.device = device
        # On MPS this is the producing stream (make_output_copy_stream):
        # the bitmask/mapping staging otherwise cold-starts a cross-stream
        # hand-off on the first structured-output request mid-serve.
        self.copy_stream = make_output_copy_stream(device)

    def apply_grammar_bitmask(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        grammar_req_ids: list[str],
        grammar_bitmask: np.ndarray,
    ) -> None:
        if not grammar_req_ids:
            return

        # Construct bitmask -> logits mapping
        mapping: list[int] = []
        req_ids = input_batch.req_ids
        cu_num_logits = input_batch.cu_num_logits_np.tolist()
        req_id_to_idx = {req_id: i for i, req_id in enumerate(req_ids)}
        for grammar_req_id in grammar_req_ids:
            req_idx = req_id_to_idx[grammar_req_id]
            logits_start_idx = cu_num_logits[req_idx]
            logits_end_idx = cu_num_logits[req_idx + 1]
            mapping.extend(range(logits_start_idx, logits_end_idx))

        self.apply_grammar_bitmask_rows(logits, mapping, grammar_bitmask)

    def apply_grammar_bitmask_rows(
        self,
        logits: torch.Tensor,
        mapping: list[int],
        grammar_bitmask: np.ndarray,
        target_token_ids: torch.Tensor | None = None,
    ) -> None:
        """Apply packed target-vocabulary masks to selected logits rows.

        ``target_token_ids`` maps columns of a reduced draft vocabulary to
        target token IDs. It is omitted for ordinary target-vocabulary logits.
        """
        if not mapping:
            return

        # Asynchronously copy the bitmask to GPU.
        current_stream = torch.accelerator.current_stream(self.device)
        with stream(self.copy_stream, current_stream):
            bitmask = async_copy_to_gpu(
                grammar_bitmask, out=self.grammar_bitmask[: grammar_bitmask.shape[0]]
            )

        # Asynchronously copy the mapping to GPU.
        with stream(self.copy_stream, current_stream):
            logits_indices = torch.tensor(
                mapping,
                dtype=torch.int32,
                device="cpu",
                pin_memory=self.device.type != "mps",
            )
            logits_indices = self.logits_indices[: len(mapping)].copy_(
                logits_indices, non_blocking=True
            )

        # Ensure all async copies are complete before launching the kernel.
        if self.copy_stream != current_stream:
            current_stream.wait_stream(self.copy_stream)

        num_masks = bitmask.shape[0]
        assert num_masks == len(mapping)
        vocab_size = logits.shape[-1]
        from vllm.v1.worker.gpu.sample.gumbel import _use_native_sample_kernels

        if target_token_ids is not None:
            target_token_ids = target_token_ids.to(logits.device, dtype=torch.int64)
            masks = bitmask[:num_masks]
            words = masks[:, target_token_ids // 32]
            allowed = ((words >> (target_token_ids % 32)) & 1).bool()
            selected = logits.index_select(0, logits_indices.to(torch.int64))
            selected.masked_fill_(~allowed, float("-inf"))
            logits.index_copy_(0, logits_indices.to(torch.int64), selected)
        elif logits.device.type == "mps":
            masks = bitmask[:num_masks]
            bit_indices = torch.arange(vocab_size, device=logits.device)
            words = masks[:, bit_indices // 32]
            allowed = ((words >> (bit_indices % 32)) & 1).bool()
            selected = logits.index_select(0, logits_indices.to(torch.int64))
            selected.masked_fill_(~allowed, float("-inf"))
            logits.index_copy_(0, logits_indices.to(torch.int64), selected)
        elif _use_native_sample_kernels() and logits.dtype in (
            torch.float32,
            torch.bfloat16,
            torch.float16,
        ):
            from vllm.quixicore import quixicore_ops

            quixicore_ops.v2_grammar_bitmask(logits, logits_indices, bitmask, num_masks)
        else:
            BLOCK_SIZE = 8192
            grid = (num_masks, cdiv(vocab_size, BLOCK_SIZE))
            _apply_grammar_bitmask_kernel[grid](
                logits,
                logits.stride(0),
                logits_indices,
                bitmask,
                bitmask.stride(0),
                vocab_size,
                BLOCK_SIZE=BLOCK_SIZE,
            )

        # Ensure the copy stream waits for the device tensors to finish being used
        # before it re-uses or deallocates them
        if self.copy_stream != current_stream:
            self.copy_stream.wait_stream(current_stream)


# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/main/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
@triton.jit
def _apply_grammar_bitmask_kernel(
    logits_ptr,
    logits_stride,
    logits_indices_ptr,
    bitmask_ptr,
    bitmask_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    bitmask_idx = tl.program_id(0)
    logits_idx = tl.load(logits_indices_ptr + bitmask_idx)

    # Load the bitmask.
    block_id = tl.program_id(1)
    bitmask_offset = (block_id * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
    packed_bitmask = tl.load(
        bitmask_ptr + bitmask_idx * bitmask_stride + bitmask_offset,
        mask=bitmask_offset < bitmask_stride,
    )
    # Unpack the bitmask.
    bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
    bitmask = bitmask.reshape(BLOCK_SIZE)

    # Apply the bitmask to the logits.
    block_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    tl.store(
        logits_ptr + logits_idx * logits_stride + block_offset,
        -float("inf"),
        mask=bitmask & (block_offset < vocab_size),
    )
