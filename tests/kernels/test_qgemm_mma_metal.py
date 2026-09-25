"""Metal small-M MMA GEMM (qgemm_mma via ggml_mul_mat_mma): q8_0 / q4_K x bf16
row-major, M <= 32. Oracle: a torch matmul over the dequantized weights with
the same half operands the kernel uses (bf16 X -> half, q8_0 -> half), fp32
accumulate; the kernel must stay within half-operand rounding of it, and a
unit-column-stride column-slice output must equal the contiguous one."""

import pytest
import torch


def _qc():
    from vllm.quixicore.ops import quixicore_ops

    if not torch.backends.mps.is_available() or not quixicore_ops.is_available():
        pytest.skip("requires Apple Metal (MPS) and the quixicore extension")
    if not quixicore_ops.has("ggml_mul_mat_mma"):
        pytest.skip("ggml_mul_mat_mma not built")
    return quixicore_ops


def _q8_0(N: int, K: int, g: torch.Generator) -> torch.Tensor:
    w = torch.randint(0, 256, (N, K // 32 * 34), generator=g, dtype=torch.uint8)
    blk = w.view(N, K // 32, 34)
    blk[..., 0] = torch.randint(
        0, 256, blk[..., 0].shape, generator=g, dtype=torch.uint8
    )
    blk[..., 1] = 0x14 + torch.randint(
        0, 4, blk[..., 1].shape, generator=g, dtype=torch.uint8
    )
    return w


def _dequant(w: torch.Tensor, N: int, K: int) -> torch.Tensor:
    blk = w.view(N, K // 32, 34)
    d = blk[..., :2].contiguous().view(torch.float16).float()
    q = blk[..., 2:].contiguous().view(torch.int8).float()
    return (q * d).reshape(N, K)


def _q4_K(N: int, K: int, g: torch.Generator) -> torch.Tensor:
    w = torch.randint(0, 256, (N, K // 256 * 144), generator=g, dtype=torch.uint8)
    blk = w.view(N, K // 256, 144)
    # finite d / dmin (half), random 6-bit scales and nibbles
    blk[..., 0] = torch.randint(
        0, 256, blk[..., 0].shape, generator=g, dtype=torch.uint8
    )
    blk[..., 1] = 0x14 + torch.randint(
        0, 4, blk[..., 1].shape, generator=g, dtype=torch.uint8
    )
    blk[..., 2] = torch.randint(
        0, 256, blk[..., 2].shape, generator=g, dtype=torch.uint8
    )
    blk[..., 3] = 0x10 + torch.randint(
        0, 4, blk[..., 3].shape, generator=g, dtype=torch.uint8
    )
    return w


def _dequant_q4_K(w: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """ggml's q4_K layout: d, dmin (half), 12 bytes of 6-bit scale/min pairs
    for 8 sub-blocks of 32, then 128 bytes of nibbles (low nibbles = even
    sub-block, high = odd, per 64-value pair)."""
    blk = w.view(N, K // 256, 144)
    d = blk[..., 0:2].contiguous().view(torch.float16).float()  # (N, nb, 1)
    dmin = blk[..., 2:4].contiguous().view(torch.float16).float()
    sc = blk[..., 4:16].int()  # (N, nb, 12)
    qs = blk[..., 16:144].int()  # (N, nb, 128)
    scales = torch.empty(N, K // 256, 8)
    mins = torch.empty(N, K // 256, 8)
    for j in range(8):
        if j < 4:
            scales[..., j] = (sc[..., j] & 63).float()
            mins[..., j] = (sc[..., j + 4] & 63).float()
        else:
            scales[..., j] = (
                (sc[..., j + 4] & 0x0F) | ((sc[..., j - 4] >> 6) << 4)
            ).float()
            mins[..., j] = (
                (sc[..., j + 4] >> 4) | ((sc[..., j] >> 6) << 4)
            ).float()
    out = torch.empty(N, K // 256, 8, 32)
    for j in range(8):
        byte = qs[..., (j // 2) * 32 : (j // 2) * 32 + 32]
        q = ((byte >> (4 * (j % 2))) & 0x0F).float()
        out[..., j, :] = (
            q * (d * scales[..., j : j + 1]) - dmin * mins[..., j : j + 1]
        )
    return out.reshape(N, K)


@pytest.mark.parametrize(
    "N,K", [(8192, 4096), (4096, 16384), (2112, 4096), (4096, 2048)]
)
@pytest.mark.parametrize("M", [1, 5, 8, 9, 16, 17, 24, 32])
@pytest.mark.parametrize("fmt", [8, 12])  # q8_0, q4_K
def test_mma_matches_half_operand_reference(N, K, M, fmt):
    qc = _qc()
    g = torch.Generator().manual_seed(7)
    if fmt == 8:
        w = _q8_0(N, K, g)
        wd = _dequant(w, N, K)
    else:
        w = _q4_K(N, K, g)
        wd = _dequant_q4_K(w, N, K)
    x = (torch.randn(M, K, generator=g) * 0.5).to(torch.bfloat16)
    ref = x.to(torch.float16).float() @ wd.to(torch.float16).float().T
    y = qc.ggml_mul_mat_mma(w.to("mps"), x.to("mps"), fmt, N).cpu().float()
    assert y.shape == (M, N)
    rel = (y - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    assert rel < 1e-2, rel


def test_mma_strided_output_in_place():
    qc = _qc()
    g = torch.Generator().manual_seed(8)
    N, K, M = 4096, 4096, 16
    w = _q8_0(N, K, g).to("mps")
    x = (torch.randn(M, K, generator=g) * 0.5).to(torch.bfloat16).to("mps")
    y = qc.ggml_mul_mat_mma(w, x, 8, N).clone()
    wide = torch.full((M, N + 96), 7.0, dtype=torch.bfloat16, device="mps")
    out = qc.ggml_mul_mat_mma(w, x, 8, N, out=wide[:, 96:])
    assert out.data_ptr() == wide[:, 96:].data_ptr()
    assert torch.equal(wide[:, 96:].contiguous().cpu(), y.cpu())
    assert torch.equal(
        wide[:, :96].cpu(), torch.full((M, 96), 7.0, dtype=torch.bfloat16)
    )


def test_mma_rejects_ineligible_shapes():
    qc = _qc()
    g = torch.Generator().manual_seed(9)
    w = _q8_0(4096, 4096, g).to("mps")
    with pytest.raises(RuntimeError):
        qc.ggml_mul_mat_mma(
            w, torch.zeros(33, 4096, dtype=torch.bfloat16, device="mps"), 8, 4096
        )
    with pytest.raises(RuntimeError):
        qc.ggml_mul_mat_mma(
            w, torch.zeros(4, 4096, dtype=torch.float16, device="mps"), 8, 4096
        )


@pytest.mark.parametrize("N,K", [(8192, 4096), (2112, 4096), (4096, 2048)])
@pytest.mark.parametrize("M", [8, 16, 24, 32])
@pytest.mark.parametrize("fmt", [8, 12])
@pytest.mark.parametrize("variant", [2, 3, 4])
def test_mma_variants_match(N, K, M, fmt, variant):
    """The r16k32 / r8k64 / r16k64 variants (2026-09-17) match the half-operand
    reference and the r8k32 original (only the fp32 K-stage order differs)."""
    qc = _qc()
    g = torch.Generator().manual_seed(11)
    if fmt == 8:
        w = _q8_0(N, K, g)
        wd = _dequant(w, N, K)
    else:
        w = _q4_K(N, K, g)
        wd = _dequant_q4_K(w, N, K)
    x = (torch.randn(M, K, generator=g) * 0.5).to(torch.bfloat16)
    ref = x.to(torch.float16).float() @ wd.to(torch.float16).float().T
    wm, xm = w.to("mps"), x.to("mps")
    base = qc.ggml_mul_mat_mma(wm, xm, fmt, N, None, 1).cpu().float()
    y = qc.ggml_mul_mat_mma(wm, xm, fmt, N, None, variant).cpu().float()
    scale = ref.abs().max().item() + 1e-9
    assert (y - ref).abs().max().item() / scale < 1e-2
    # vs the original: a different K-slice count reorders the fp32 partial
    # sums, so the bf16 output may differ by one ulp (2^-8 relative).
    assert (y - base).abs().max().item() / scale < 8e-3


def test_mma_variant_strided_and_fallback():
    qc = _qc()
    g = torch.Generator().manual_seed(12)
    N, K, M = 4096, 4096, 32
    w = _q8_0(N, K, g).to("mps")
    x = (torch.randn(M, K, generator=g) * 0.5).to(torch.bfloat16).to("mps")
    y = qc.ggml_mul_mat_mma(w, x, 8, N, None, 4).clone()
    wide = torch.full((M, N + 96), 7.0, dtype=torch.bfloat16, device="mps")
    out = qc.ggml_mul_mat_mma(w, x, 8, N, wide[:, 96:], 4)
    assert out.data_ptr() == wide[:, 96:].data_ptr()
    assert torch.equal(wide[:, 96:].contiguous().cpu(), y.cpu())
    # N % 64 != 0 falls back to the (8, 32) kernel for the r16 variants
    w2 = _q8_0(2080, K, g).to("mps")
    a = qc.ggml_mul_mat_mma(w2, x, 8, 2080, None, 1)
    b = qc.ggml_mul_mat_mma(w2, x, 8, 2080, None, 2)
    assert torch.equal(a.cpu(), b.cpu())
