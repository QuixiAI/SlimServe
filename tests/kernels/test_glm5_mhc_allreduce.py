# SPDX-License-Identifier: Apache-2.0
"""The GLM mHC transition fused with the tensor-parallel all-reduce
(quixicore/serving/glm5_mhc_allreduce.cuh through the custom all-reduce)
against the plain all-reduce followed by the split Triton transition, eager
and under CUDA graph capture. Needs 2 or 4 visible GPUs with peer access."""

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs at least two GPUs",
)

HIDDEN, HC, MIXES = 4096, 4, 24
RMS_EPS, HC_EPS, POST_MULT, ITERS, NORM_EPS = 1e-5, 1e-6, 2.0, 20, 1e-5
TOKENS = (1, 2, 4, 7, 8, 9, 33, 64)  # group, block and token-chunk boundaries


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _inputs(tokens: int, rank: int, device):
    common = torch.Generator(device=device).manual_seed(1234 + tokens)
    mine = torch.Generator(device=device).manual_seed(99 * (rank + 1) + tokens)
    x = torch.randn((tokens, HIDDEN), generator=mine, device=device).bfloat16()
    residual = torch.randn(
        (tokens, HC, HIDDEN), generator=common, device=device
    ).bfloat16()
    post = torch.rand((tokens, HC), generator=common, device=device) * 2
    comb = torch.rand((tokens, HC, HC), generator=common, device=device)
    fn = torch.randn((MIXES, HC * HIDDEN), generator=common, device=device) * 0.02
    scale = torch.rand((3,), generator=common, device=device) + 0.5
    base = torch.randn((MIXES,), generator=common, device=device) * 0.1
    weight = (torch.rand((HIDDEN,), generator=common, device=device) + 0.5).bfloat16()
    return x, residual, post, comb, fn, scale, base, weight


def _compare(fused, reference, label: str) -> None:
    res, post, comb, layer = fused
    res_ref, post_ref, comb_ref, layer_ref = reference
    torch.testing.assert_close(res, res_ref, atol=0, rtol=0, msg=f"{label}: residual")
    torch.testing.assert_close(post, post_ref.view_as(post), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(comb, comb_ref.view_as(comb), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(
        layer.float(), layer_ref.float(), atol=2e-2, rtol=2**-7, msg=f"{label}: layer"
    )


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["VLLM_CUSTOM_AR_ALLOW_PCIE"] = "1"
    os.environ["VLLM_CUSTOM_AR_MAX_SIZE_MB"] = "64"
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        tensor_model_parallel_all_reduce,
    )
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.model_executor.layers.glm5_next_mhc_triton import mhc_transition
    from vllm.quixicore.ops import quixicore_ops

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_distributed_environment(
        world_size, rank, f"tcp://127.0.0.1:{port}", rank, backend="nccl"
    )
    with set_current_vllm_config(VllmConfig()):
        ensure_model_parallel_initialized(world_size, 1)
    try:
        ca_comm = get_tp_group().device_communicator.ca_comm
        if ca_comm is None or ca_comm.disabled:
            raise RuntimeError("custom all-reduce is disabled on this box")
        if not quixicore_ops.has_glm5_mhc_allreduce():
            raise RuntimeError("this build has no glm5_mhc_allreduce binding")

        def reference(x, residual, post, comb, fn, scale, base, weight):
            full = tensor_model_parallel_all_reduce(x)
            return mhc_transition(
                full,
                residual,
                post,
                comb,
                fn,
                scale,
                base,
                RMS_EPS,
                HC_EPS,
                POST_MULT,
                ITERS,
                weight,
                NORM_EPS,
            )

        # The serving policy fuses small batches inside captured graphs only;
        # the kernel itself is verified eagerly and up to its own limit.
        ca_comm._GLM5_MHC_FUSE_TOKENS = ca_comm._GLM5_MHC_MAX_TOKENS
        ca_comm._GLM5_MHC_FUSE_EAGER = True

        def fused(x, residual, post, comb, fn, scale, base, weight):
            out = ca_comm.fused_all_reduce_glm5_mhc(
                x,
                residual,
                post,
                comb.view(-1, HC, HC),
                fn,
                scale,
                base,
                weight,
                RMS_EPS,
                HC_EPS,
                POST_MULT,
                ITERS,
                NORM_EPS,
            )
            assert out is not None, "fused path declined an eligible input"
            return out

        for tokens in TOKENS:
            for use_norm in (False, True):
                x, residual, post, comb, fn, scale, base, weight = _inputs(
                    tokens, rank, device
                )
                w = weight if use_norm else None
                ref = reference(x, residual, post, comb, fn, scale, base, w)
                out = fused(x, residual, post, comb, fn, scale, base, w)
                # The comb coefficients arrive on the deferred side stream;
                # every consumer joins first (the model's forward ends with
                # one before any capture starts).
                ca_comm.join_glm5_mhc()
                torch.cuda.synchronize()
                _compare(out, ref, f"eager T={tokens} norm={use_norm}")
                # Twice more on the same inputs: the arrival counters must
                # come back to zero between launches.
                for _ in range(2):
                    out = fused(x, residual, post, comb, fn, scale, base, w)
                    ca_comm.join_glm5_mhc()
                    torch.cuda.synchronize()
                    _compare(out, ref, f"repeat T={tokens} norm={use_norm}")

        # Graph capture: the registered-buffer path with replays.
        tokens = 4
        x, residual, post, comb, fn, scale, base, weight = _inputs(tokens, rank, device)
        ref = reference(x, residual, post, comb, fn, scale, base, weight)
        graph = torch.cuda.CUDAGraph()
        with ca_comm.capture(), torch.cuda.graph(graph):
            captured = fused(x, residual, post, comb, fn, scale, base, weight)
            # The deferred sinkhorn's side stream must rejoin the capture.
            ca_comm.join_glm5_mhc()
        for replay in range(3):
            graph.replay()
            torch.cuda.synchronize()
            _compare(captured, ref, f"graph replay {replay}")
        torch.distributed.barrier()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.parametrize(
    "world_size", [n for n in (2, 4) if torch.cuda.device_count() >= n]
)
def test_fused_allreduce_transition(world_size: int) -> None:
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)
