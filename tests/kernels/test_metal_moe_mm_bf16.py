# SPDX-License-Identifier: Apache-2.0
"""bf16 activations over the Metal tiled MoE GEMMs. ``_fused_moe_gguf``
routes bf16 prefill batches through the fp16 iq2_xxs w13 / q2_K down tiles
with one cast in and one cast out (GLM-5.3-Flash runs bf16; before the
route existed its prefill fell through to the per-slot decode GEMVs). The
tile route must agree with the GEMV route on the same bf16 input, be
deterministic, return bf16, and match the fp16-input route up to the
input rounding."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple Metal (MPS)"
)
DEV = "mps"
QT_IQ2_XXS = 16
QT_Q2_K = 10


def _finite_half_bytes(*shape) -> torch.Tensor:
    vals = (torch.rand(*shape, dtype=torch.float32) * 0.5 + 0.25).to(torch.float16)
    return vals.view(torch.uint8).reshape(*shape[:-1], shape[-1] * 2)


def _make_iq2_xxs(e: int, n_rows: int, k: int) -> torch.Tensor:
    nb = k // 256
    blocks = torch.randint(0, 256, (e, n_rows, nb, 66), dtype=torch.uint8)
    blocks[..., 0:2] = _finite_half_bytes(e, n_rows, nb, 1)
    return blocks.reshape(e, n_rows, nb * 66).contiguous().to(DEV)


def _make_q2_k(e: int, n_rows: int, k: int) -> torch.Tensor:
    nb = k // 256
    blocks = torch.randint(0, 256, (e, n_rows, nb, 84), dtype=torch.uint8)
    blocks[..., 80:84] = _finite_half_bytes(e, n_rows, nb, 2)
    return blocks.reshape(e, n_rows, nb * 84).contiguous().to(DEV)


def _run(x, w1, w2, topk_weights, topk_ids):
    from vllm.model_executor.layers.quantization.gguf.fused_moe import (
        _fused_moe_gguf,
    )

    return _fused_moe_gguf(
        x,
        w1,
        w2,
        topk_weights,
        topk_ids,
        QT_IQ2_XXS,
        QT_Q2_K,
        "silu",
        None,
        None,
        None,
        None,
    )


@pytest.mark.parametrize("tokens", [64, 300])
def test_bf16_tile_route_matches_gemv(monkeypatch, tokens):
    from vllm.quixicore import quixicore_ops

    if not quixicore_ops.is_available():
        pytest.skip("quixicore Metal extension not built")
    torch.manual_seed(tokens)
    E, hidden, inter, topk = 16, 256, 256, 8
    w1 = _make_iq2_xxs(E, 2 * inter, hidden)
    w2 = _make_q2_k(E, hidden, inter)
    ids = torch.rand(tokens, E).argsort(dim=1)[:, :topk].to(torch.int32)
    ids = ids.contiguous().to(DEV)
    weights = torch.softmax(torch.randn(tokens, topk), dim=1).to(DEV)
    # Random weight bytes are far louder than real experts; keep the fp16
    # intermediates in range (the reference absmax is asserted below).
    x = (torch.randn(tokens, hidden) * 0.002).to(torch.bfloat16).to(DEV)

    monkeypatch.setenv("VLLM_QC_MOE_MM_MIN_TOKENS", "32")
    mm = _run(x, w1, w2, weights, ids)
    mm_again = _run(x, w1, w2, weights, ids)
    mm16 = _run(x.to(torch.float16), w1, w2, weights, ids)
    monkeypatch.setenv("VLLM_QC_MOE_MM_MIN_TOKENS", "1000000")
    vec = _run(x, w1, w2, weights, ids)
    torch.mps.synchronize()

    assert mm.dtype == torch.bfloat16 and mm.shape == x.shape
    assert torch.equal(mm, mm_again)
    assert torch.isfinite(mm.float()).all()
    scale = vec.float().abs().max().item() + 1e-6
    assert torch.isfinite(vec.float()).all() and scale < 2e4, scale
    err_vec = (mm.float() - vec.float()).abs().max().item()
    assert err_vec / scale <= 5e-2, (err_vec, scale, tokens)
    assert mm16.dtype == torch.float16
    err16 = (mm.float() - mm16.float()).abs().max().item()
    assert err16 / scale <= 2e-2, (err16, scale, tokens)
