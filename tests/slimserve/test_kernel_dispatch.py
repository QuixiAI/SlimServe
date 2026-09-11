# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for optional native paths; no CUDA launch or model load."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "cuda,capability,disabled,symbol,expected",
    [
        (False, 120, False, True, False),
        (True, 75, False, True, False),
        (True, 80, False, True, True),
        (True, 120, False, True, True),
        (True, 120, True, True, False),
        (True, 120, False, False, False),
    ],
)
def test_decode_gemm_dispatch(
    monkeypatch, fp8, cuda, capability, disabled, symbol, expected
):
    from vllm.model_executor.layers import utils
    from vllm.quixicore.ops import quixicore_ops as qc

    suffix = "_fp8" if fp8 else ""
    enabled = getattr(utils, f"decode_gemm{suffix}_enabled")
    calls = []
    monkeypatch.setattr(
        utils,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: cuda,
            has_device_capability=lambda minimum: capability >= minimum,
        ),
    )
    monkeypatch.setenv(
        f"SLIMSERVE_DECODE_GEMM{suffix.upper()}", "0" if disabled else "1"
    )
    monkeypatch.setattr(
        qc, f"has_decode_gemm{suffix}", lambda: calls.append(True) or symbol
    )
    enabled.cache_clear()
    try:
        assert enabled() is expected
        assert bool(calls) is (cuda and capability >= 80 and not disabled)
    finally:
        enabled.cache_clear()


@pytest.mark.parametrize("symbol", [False, True])
def test_swapab_missing_symbol_uses_triton(monkeypatch, symbol):
    from vllm.quixicore.ops import quixicore_ops as qc
    from vllm.v1.attention.backends.mla import quixicore_mla_sparse_prefill as pf

    q = SimpleNamespace(
        shape=(2048, 16, 512),
        dtype=torch.bfloat16,
        device="cuda:0",
        is_cuda=True,
        is_contiguous=lambda: True,
        data_ptr=lambda: 256,
    )
    kv = torch.empty(1, 64, 512, dtype=torch.bfloat16)
    table = torch.zeros(1, 1, dtype=torch.int32)
    indices = torch.zeros(1, 1, dtype=torch.int32)
    lengths = torch.ones(1, dtype=torch.int32)
    native_out, fallback_out = object(), object()
    calls = []

    class TritonKernel:
        def __getitem__(self, grid):
            assert grid == (2048,)
            return lambda *args, **kwargs: calls.append("triton")

    def has(name):
        assert name == "mla_prefill_bf16_sparse_nope_sm120"
        return symbol

    monkeypatch.setattr(pf, "SWAPAB_ENABLED", True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (12, 0))
    monkeypatch.setattr(torch, "empty_like", lambda value: fallback_out)
    monkeypatch.setattr(pf, "_sparse_mla_prefill_kernel", TritonKernel())
    monkeypatch.setattr(qc, "has", has)
    monkeypatch.setattr(
        qc,
        "mla_prefill_bf16_sparse_nope_sm120",
        lambda *args: calls.append("native") or native_out,
    )
    out = pf.sparse_mla_prefill_nope(q, kv, table, indices, lengths, 64, 0.0625)
    assert out is (native_out if symbol else fallback_out)
    assert calls == (["native"] if symbol else ["triton"])
