# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from benchmarks.kernels import check_glm53_kda_recompute as probe


def test_fixed_matrix_contains_exact_and_conditioned_cases():
    cases = probe.matrix()
    assert len(cases) == 196
    assert sum(c["regime"] == "identity" for c in cases) == 28
    assert len({json.dumps(c, sort_keys=True) for c in cases}) == 196
    assert probe.WARPS == (4, 8)


def test_larger_comparison_chunks_preserve_metrics():
    import torch

    gen = torch.Generator().manual_seed(15)
    a = torch.randn((259, 128), generator=gen).bfloat16()
    b = (a.float() + 0.03).bfloat16()
    assert probe.compare(a, b) == probe.compare_bf16(a, b)
    for chunk_rows in (0, -1, True):
        with pytest.raises(ValueError, match="chunk size"):
            probe.compare_bf16(a, b, chunk_rows=chunk_rows)


@pytest.mark.parametrize("lengths", [[1], [65], [17, 63, 65, 855]])
def test_identity_reference_is_exact_at_ragged_boundaries(lengths):
    import torch

    case = dict(rank=0, lengths=lengths, seed=530901, magnitude=1.0, regime="identity")
    inputs = probe.make_inputs(case, device="cpu")
    w, u, kg = probe.reference(inputs, lengths)
    assert torch.equal(w.bfloat16(), inputs["k"] * 0.5)
    assert torch.equal(u.bfloat16(), inputs["v"] * 0.5)
    assert torch.equal(kg.bfloat16(), inputs["k"])
    inputs["k"].mul_(0.5)
    inputs["v"].mul_(-0.5)
    w, u, kg = probe.reference(inputs, lengths)
    assert torch.equal(w.bfloat16(), inputs["k"] * 0.5)
    assert torch.equal(u.bfloat16(), inputs["v"] * 0.5)
    assert torch.equal(kg.bfloat16(), inputs["k"])


def test_batched_reference_matches_independent_per_chunk_math():
    import torch

    lengths = [3, 65]
    case = dict(rank=0, lengths=lengths, seed=530901, magnitude=1.0, regime="identity")
    inputs = probe.make_inputs(case, device="cpu")
    gen = torch.Generator().manual_seed(79)
    inputs["A"].copy_(torch.randn(inputs["A"].shape, generator=gen).bfloat16())
    inputs["gk"].copy_(-torch.rand(inputs["gk"].shape, generator=gen))
    inputs["beta"].copy_(torch.rand(inputs["beta"].shape, generator=gen))
    batched = probe.reference(inputs, lengths)
    for offset, count in ((0, 3), (3, 64), (67, 1)):
        sl = slice(offset, offset + count)
        k, v, beta, g = [inputs[n][0, sl].double() for n in ("k", "v", "beta", "gk")]
        A = inputs["A"][0, sl, :, :count].double().permute(1, 0, 2)
        kb = (k * beta[..., None] * g.exp2()).bfloat16().double()
        vb = (v * beta[..., None]).bfloat16().double()
        expected = (
            (A @ kb.permute(1, 0, 2)).permute(1, 0, 2),
            (A @ vb.permute(1, 0, 2)).permute(1, 0, 2),
            k * (g[-1:] - g).exp2(),
        )
        for whole, block in zip(batched, expected):
            torch.testing.assert_close(whole[0, sl], block, rtol=1e-12, atol=1e-12)


def test_actual_jit_and_launch_match_the_serving_wrapper():
    import torch
    from triton.runtime.jit import JITFunction

    from vllm.models.kimi_k3.amd.ops.third_party.kda.chunk import (
        recompute_w_u_fwd_kernel,
    )

    assert isinstance(recompute_w_u_fwd_kernel.fn.fn, JITFunction)
    inputs = probe.make_inputs(probe.matrix()[0], device="cpu")
    outputs = [torch.empty_like(inputs["k"]) for _ in range(3)]
    calls = []

    class Capture:
        def __getitem__(self, grid):
            return lambda **kw: calls.append((grid, kw))

    probe.launch(Capture(), inputs, outputs, 4)
    grid, args = calls[0]
    assert grid == (1, 16)
    assert args["q"] is None and args["qg"] is None
    assert args["kg"] is outputs[2]
    assert not args["STORE_QG"] and args["STORE_KG"] and args["IS_VARLEN"]
    assert [args[k] for k in ("H", "K", "V", "BT", "BK", "BV")] == [
        16,
        128,
        128,
        64,
        64,
        64,
    ]
    assert args["num_warps"] == 4 and args["num_stages"] == 3
    assert args["DOT_PRECISION"] == "ieee"
    with pytest.raises(ValueError, match="unprescribed"):
        probe.launch(Capture(), inputs, outputs, 2)


def make_record():
    return dict(
        index=0,
        **probe.matrix()[0],
        binary_ids=["old", "new"],
        repeat_replay_mutation_guards_passed=True,
        phases=[
            dict(
                phase=p,
                arms=[
                    dict(
                        warps=w,
                        hashes=["a", "b", "c"],
                        reference=[dict(bit_mismatches=0)] * 3,
                    )
                    for w in (4, 8)
                ],
                pairwise=[dict(elements=2048, bit_mismatches=0)] * 3,
            )
            for p in range(2)
        ],
    )


def test_auditor_rejects_incomplete_wrong_or_identity_failing_records():
    bins = dict(old=dict(warps=4, stages=3), new=dict(warps=8, stages=3))
    probe.audit_record(make_record(), 0, bins)
    for mutate in (
        lambda r: r.update(rank=1),
        lambda r: r["phases"].pop(),
        lambda r: r["binary_ids"].pop(),
        lambda r: r["phases"][0]["pairwise"][0].update(elements=1),
        lambda r: r["phases"][0]["pairwise"][0].update(bit_mismatches=1),
        lambda r: r["phases"][0]["arms"][0]["reference"][0].update(bit_mismatches=1),
    ):
        row = make_record()
        mutate(row)
        with pytest.raises(ValueError):
            probe.audit_record(row, 0, bins)


def test_real_preparation_and_conditioned_inputs_on_cpu(tmp_path):
    import torch

    if not probe.gate.INVENTORY.exists():
        pytest.skip("local campaign evidence is not installed")
    path = tmp_path / "manifest.json"
    probe.prepare(path)
    manifest = json.loads(path.read_text())
    probe.verify(manifest)
    inputs = probe.make_inputs(probe.matrix()[1], device="cpu")
    assert inputs["A"].dtype == inputs["k"].dtype == torch.bfloat16
    assert inputs["gk"].dtype == inputs["beta"].dtype == torch.float32
    assert torch.all(inputs["gk"] <= 0)
    for value in probe.reference(inputs, [1]):
        assert torch.isfinite(value).all()
    manifest["warps"] = [4, 2]
    with pytest.raises(ValueError, match="plan changed"):
        probe.verify(manifest)


def test_failed_identity_can_be_closed_without_clearing_its_failure():
    bins = dict(old=dict(warps=4, stages=3), new=dict(warps=8, stages=3))
    row = make_record()
    row["phases"][0]["arms"][0]["reference"][0]["bit_mismatches"] = 1
    with pytest.raises(ValueError, match="identity failure"):
        probe.audit_record(row, 0, bins)
    probe.audit_record(row, 0, bins, allow_identity_failure=True)
    assert row["phases"][0]["arms"][0]["reference"][0]["bit_mismatches"] == 1
