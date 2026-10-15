# SPDX-License-Identifier: Apache-2.0
"""Metal q8_0 decode GEMV in the llama.cpp mul_mv geometry (qgemv_q8_0_nr)
against an exact fp32 dequant-dot reference and the generic row-walk kernel
(VLLM_QC_Q8_NR=0 in a subprocess). The route is opt-in per profile, so this
module opts the test process in at import (the launcher reads the switch once
per process, before any q8_0 GEMV runs). The NR kernel accumulates int8*y in
fp32 per lane (no per-element half rounding), so it is closer to the exact
reference than the generic walk; both must sit within bf16-output error."""

import os
import subprocess
import sys

# Opt the test process into the q8_0 NR route before any q8_0 GEMV launches.
os.environ.setdefault("VLLM_QC_Q8_NR", "1")

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Metal only"
)
DEV = "mps"


def _make_q8_0(rows, k, seed):
    g = torch.Generator().manual_seed(seed)
    nb = k // 32
    blocks = torch.randint(0, 256, (rows, nb, 34), dtype=torch.uint8, generator=g)
    sc = (torch.rand(rows, nb, 1, generator=g) * 0.5 + 0.25).to(torch.float16)
    blocks[:, :, 0:2] = sc.view(torch.uint8).reshape(rows, nb, 2)
    return blocks.reshape(rows, nb * 34).contiguous()


def _dequant(w, rows, k):
    nb = k // 32
    b = w.view(rows, nb, 34)
    d = b[:, :, 0:2].contiguous().view(torch.float16).float()  # [rows, nb, 1]
    q = b[:, :, 2:].contiguous().view(torch.int8).float()  # [rows, nb, 32]
    return (d * q).reshape(rows, k)


@pytest.mark.parametrize(
    "rows,k", [(8192, 4096), (4096, 16384), (2048, 4096), (6, 1024)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_q8_0_nr_matches_exact_reference(rows, k, dtype):
    from vllm.quixicore.ops import quixicore_ops as qc

    if not qc.has_kernel("qgemv_q8_0_nr_2x4_bfloat16"):
        pytest.skip("q8_0 nr kernel not built")
    w = _make_q8_0(rows, k, seed=rows + k)
    g = torch.Generator().manual_seed(1)
    x = (torch.randn(1, k, generator=g) * 0.2).to(dtype)
    ref = (x.float() @ _dequant(w, rows, k).T)
    got = qc.ggml_mul_mat_vec_a8(w.to(DEV), x.to(DEV), 8, rows).float().cpu()
    scale = ref.abs().max().item()
    err = (got - ref).abs().max().item() / scale
    eps = 2**-7 if dtype == torch.bfloat16 else 2**-10
    assert err <= 4 * eps, (err, eps)


def test_q8_0_nr_is_at_least_as_exact_as_generic_walk():
    """The generic kernel's error vs the exact reference bounds the NR
    kernel's (computed in a subprocess with the route pinned off, since the
    launcher reads its kill switch once per process)."""
    from vllm.quixicore.ops import quixicore_ops as qc

    if not qc.has_kernel("qgemv_q8_0_nr_2x4_bfloat16"):
        pytest.skip("q8_0 nr kernel not built")
    rows, k = 4096, 8192
    w = _make_q8_0(rows, k, seed=7)
    g = torch.Generator().manual_seed(2)
    x = (torch.randn(1, k, generator=g) * 0.2).to(torch.bfloat16)
    ref = x.float() @ _dequant(w, rows, k).T
    got = qc.ggml_mul_mat_vec_a8(w.to(DEV), x.to(DEV), 8, rows).float().cpu()
    err_nr = (got - ref).abs().mean().item()
    torch.save((w, x), "/tmp/q8nr_case.pt")
    code = (
        "import torch; from vllm.quixicore.ops import quixicore_ops as qc; "
        "w, x = torch.load('/tmp/q8nr_case.pt'); "
        f"y = qc.ggml_mul_mat_vec_a8(w.to('mps'), x.to('mps'), 8, {rows}); "
        "torch.save(y.float().cpu(), '/tmp/q8nr_generic.pt')"
    )
    env = dict(os.environ, VLLM_QC_Q8_NR="0")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)
    generic = torch.load("/tmp/q8nr_generic.pt")
    err_generic = (generic - ref).abs().mean().item()
    assert err_nr <= err_generic * 1.05, (err_nr, err_generic)


@pytest.mark.parametrize("m", [2, 3, 4])
@pytest.mark.parametrize("rows,k", [(8192, 4096), (4096, 16384), (6, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_q8_0_nr_mb_rows_bit_identical_to_batch_1(m, rows, k, dtype):
    """The verify-width kernel (qgemv_q8_0_nr_mb, M=2..4) walks the same
    blocks in the same order with the same reductions as the batch-1 NR
    kernel, so every output row must equal the batch-1 result on that row
    alone: a K=1 speculative verify reproduces decode numerics exactly."""
    from vllm.quixicore.ops import quixicore_ops as qc

    if not qc.has_kernel("qgemv_q8_0_nr_2x4_mb2_bfloat16"):
        pytest.skip("q8_0 nr mb kernel not built")
    w = _make_q8_0(rows, k, seed=rows + k + m).to(DEV)
    g = torch.Generator().manual_seed(m)
    x = (torch.randn(m, k, generator=g) * 0.2).to(dtype).to(DEV)
    got = qc.ggml_mul_mat_vec_a8(w, x, 8, rows)
    assert got.shape == (m, rows)
    for r in range(m):
        one = qc.ggml_mul_mat_vec_a8(w, x[r : r + 1].contiguous(), 8, rows)
        assert torch.equal(got[r], one[0]), (m, r, rows, k, dtype)
    ref = x.float().cpu() @ _dequant(w.cpu(), rows, k).T
    err = (got.float().cpu() - ref).abs().max().item() / ref.abs().max().item()
    eps = 2**-7 if dtype == torch.bfloat16 else 2**-10
    assert err <= 4 * eps, (err, eps)
