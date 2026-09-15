# SPDX-License-Identifier: Apache-2.0
"""The hetero-quant shard GEMVs may share a concurrent region only when every
shard writes its column slice in place: a ring copy inside the region is a
TORCH_CHECK. The Python gate must mirror the native strided-output rule
(q8_0 NR batch kernel at M 2..4, q4_K NR chunks at M <= 8, batch 1 always);
the subprocess check runs the native op inside a region for M = 1..8."""

import os
import subprocess
import sys

import pytest
import torch
from gguf import GGMLQuantizationType as WT

from vllm.model_executor.layers.quantization.gguf import linear as gl

K = 4096


def _shard(rows, t):
    return (torch.empty(rows, 8), t)


def test_gate_mirrors_the_native_strided_rule(monkeypatch):
    monkeypatch.setenv("VLLM_QC_Q8_NR", "1")
    monkeypatch.delenv("VLLM_QC_Q4K_NR", raising=False)
    monkeypatch.delenv("VLLM_QC_Q4K_NR_MM", raising=False)
    shards = [_shard(16384, WT.Q4_K), _shard(8448, WT.Q8_0)]
    for batch in range(1, 9):
        x = torch.empty(batch, K, dtype=torch.bfloat16)
        assert gl._metal_shard_region_ok(x, shards) is (batch <= 4), batch
    # q4_K alone rides its 2/4/8-row chunks at any M <= 8
    for batch in range(1, 9):
        x = torch.empty(batch, K, dtype=torch.bfloat16)
        assert gl._metal_shard_region_ok(x, [_shard(16384, WT.Q4_K)]) is True
    # q8_0 without the NR opt-in copies at every M > 1
    monkeypatch.setenv("VLLM_QC_Q8_NR", "0")
    x = torch.empty(2, K, dtype=torch.bfloat16)
    assert gl._metal_shard_region_ok(x, [_shard(8448, WT.Q8_0)]) is False
    assert gl._metal_shard_region_ok(x[:1], [_shard(8448, WT.Q8_0)]) is True
    # other vector formats: batch 1 only
    assert gl._metal_shard_region_ok(x, [_shard(64, WT.Q6_K)]) is False


_CHILD = r"""
import sys, torch
from gguf import GGMLQuantizationType as WT
from vllm.quixicore.ops import quixicore_ops as qc
from vllm.model_executor.layers.quantization.gguf import ops
from vllm.model_executor.layers.quantization.gguf import linear as gl
K, n0, n1 = 4096, 64, 96
g = torch.Generator().manual_seed(0)
def q8(rows):
    nb = K // 32
    b = torch.randint(0, 256, (rows, nb, 34), dtype=torch.uint8, generator=g)
    sc = (torch.rand(rows, nb, 1, generator=g) * 0.5 + 0.25).to(torch.float16)
    b[:, :, 0:2] = sc.view(torch.uint8).reshape(rows, nb, 2)
    return b.reshape(rows, nb * 34).contiguous().to("mps")
w0, w1 = q8(n0), q8(n1)
shards = [(w0, WT.Q8_0), (w1, WT.Q8_0)]
for batch in range(1, 9):
    x = (torch.randn(batch, K, generator=g) * 0.2).to(torch.bfloat16).to("mps")
    dense = torch.cat([ops.ggml_mul_mat_vec_a8(w0, x, WT.Q8_0, n0),
                       ops.ggml_mul_mat_vec_a8(w1, x, WT.Q8_0, n1)], dim=1)
    out = torch.empty(batch, n0 + n1, dtype=x.dtype, device="mps")
    gate = gl._metal_shard_region_ok(x, shards)
    qc.concurrent_begin()
    try:
        ops.ggml_mul_mat_vec_a8(w0, x, WT.Q8_0, n0, out=out[:, :n0])
        ops.ggml_mul_mat_vec_a8(w1, x, WT.Q8_0, n1, out=out[:, n0:])
        ok = True
    except RuntimeError as err:
        ok = False
        assert "concurrent region" in str(err), err
    finally:
        qc.concurrent_end()
    torch.mps.synchronize()
    assert ok == gate, (batch, ok, gate)
    if ok:
        assert torch.equal(out, dense), batch
print("REGION-OK")
"""


def test_native_region_matches_gate():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal only")
    from vllm.quixicore.ops import quixicore_ops as qc

    if not qc.is_available() or not qc.has("qc_concurrent_begin"):
        pytest.skip("concurrent regions not built")
    env = dict(os.environ, VLLM_QC_Q8_NR="1")
    res = subprocess.run(
        [sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True
    )
    assert res.returncode == 0 and "REGION-OK" in res.stdout, res.stderr[-2000:]
