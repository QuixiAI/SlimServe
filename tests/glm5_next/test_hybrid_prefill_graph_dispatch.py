# SPDX-License-Identifier: Apache-2.0
"""Real V1 runner/dispatcher regression for one-token hybrid prompt tails."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.config import CompilationConfig, CUDAGraphMode
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def make_runner(computed, prompts, hybrid=True, mode=CUDAGraphMode.FULL_DECODE_ONLY):
    compilation = CompilationConfig(
        cudagraph_mode=mode, cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32],
        max_cudagraph_capture_size=32,
    )
    parallel = SimpleNamespace(data_parallel_size=1, tensor_parallel_size=8)
    config = SimpleNamespace(
        compilation_config=compilation, num_speculative_tokens=0,
        lora_config=None, scheduler_config=SimpleNamespace(max_num_seqs=64),
        parallel_config=parallel,
        observability_config=SimpleNamespace(cudagraph_metrics=False),
    )
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(mode, 1)
    return SimpleNamespace(
        uniform_decode_query_len=1, model_config=SimpleNamespace(
            is_encoder_decoder=False, is_hybrid=hybrid),
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu=np.array(computed),
            num_prompt_tokens=np.array(prompts), lora_id_to_lora_request={}),
        compilation_config=compilation, parallel_config=parallel,
        vllm_config=config, cudagraph_dispatcher=dispatcher,
        _is_uniform_decode=GPUModelRunner._is_uniform_decode,
        _pad_for_sequence_parallelism=lambda n: n,
    )


def dispatch(runner, count, **kwargs):
    return GPUModelRunner._determine_batch_execution_and_padding(
        runner, num_tokens=count, num_reqs=count,
        num_scheduled_tokens_np=np.ones(count, dtype=np.int32),
        max_num_scheduled_tokens=1, use_cascade_attn=False, **kwargs,
    )[0]


@pytest.mark.parametrize("count", [1, 8, 16, 32])
@pytest.mark.parametrize("computed,prompt", [(0, 1), (9216, 9217)])
def test_hybrid_single_token_prefill_cannot_replay_decode_graph(
    count, computed, prompt,
):
    runner = make_runner([computed] * count, [prompt] * count)
    # This is the same split used by GDNAttentionMetadataBuilder. The graph
    # state-index buffer is refreshed only when num_prefills == 0.
    metadata = SimpleNamespace(
        max_query_len=1, num_reqs=count, num_actual_tokens=count,
        query_start_loc_cpu=torch.arange(count + 1),
        is_prefilling=torch.ones(count, dtype=torch.bool),
    )
    assert split_decodes_and_prefills(
        metadata, decode_threshold=1, treat_short_extends_as_decodes=False,
    ) == (0, count, 0, count)
    assert dispatch(runner, count) != CUDAGraphMode.FULL


@pytest.mark.parametrize("count", [1, 8, 16, 32])
def test_actual_decode_keeps_full_graph(count):
    runner = make_runner([9217] * count, [9217] * count)
    assert dispatch(runner, count) == CUDAGraphMode.FULL


def test_mixed_decode_and_prompt_tail_excludes_full():
    runner = make_runner([9500] * 7 + [9216], [9217] * 8)
    assert dispatch(runner, 8) != CUDAGraphMode.FULL


def test_dummy_capture_override_and_nonhybrid_are_unchanged():
    runner = make_runner([0], [1])
    assert dispatch(runner, 1, force_uniform_decode=True) == CUDAGraphMode.FULL
    runner.model_config.is_hybrid = False
    assert dispatch(runner, 1) == CUDAGraphMode.FULL


def test_padded_inactive_input_rows_do_not_disable_decode():
    runner = make_runner([9217, 0], [9217, 12069])
    assert dispatch(runner, 1) == CUDAGraphMode.FULL
