# SPDX-License-Identifier: Apache-2.0
import copy
import json
import math

import pytest

from benchmarks.kernels import check_glm53_kda_gate as probe


def test_fixed_complete_matrix_and_packed_beta_extent():
    rows = probe.matrix()
    assert len(rows) == 168
    assert len({json.dumps(r, sort_keys=True) for r in rows}) == 168
    assert probe.PACKED_WIDTH == 6528
    assert probe.WARPS == (8, 2)
    starts, chunks = probe.sequence_metadata([17, 63, 65, 855])
    assert starts == [0, 17, 80, 145, 1000]
    assert chunks[:5] == [(0, 0), (1, 0), (2, 0), (2, 1), (3, 0)]
    assert chunks[-1] == (3, 13)
    for lengths in ([], [0], [-1], [True]):
        with pytest.raises(ValueError):
            probe.sequence_metadata(lengths)


def test_float64_reference_resets_at_chunk_and_sequence_boundaries():
    import torch

    gate = torch.zeros((1, 130, 16, 128), dtype=torch.bfloat16)
    beta = torch.zeros((1, 130, 16), dtype=torch.bfloat16)
    a, bias = torch.zeros(16), torch.zeros(16, 128)
    result, beta_out = probe.oracle(gate, beta, a, bias, [65, 65])
    assert result.dtype == torch.float64
    for start in (0, 64, 65, 129):
        assert result[0, start, 0, 0].item() == -2.5 / math.log(2)
    assert result[0, 63, 0, 0].item() == -160 / math.log(2)
    assert torch.all(beta_out == 0.5)
    metrics = probe.errors(result.float(), result)
    assert metrics["pass_oracle"]
    assert not probe.errors(result.float() + 1, result)["pass_oracle"]
    with pytest.raises(ValueError, match="nonfinite"):
        probe.errors(result.float() * math.nan, result)


def test_bit_comparison_and_direct_launch_arguments():
    import torch

    g = torch.zeros(1, 65, 16, 128, dtype=torch.bfloat16)
    packed = torch.zeros(1, 65, probe.PACKED_WIDTH, dtype=torch.bfloat16)
    beta = packed[:, :, 6144:6160]
    a, bias = torch.zeros(16), torch.zeros(16, 128)
    out, beta_out = g.float(), beta.float()
    starts, chunks = torch.tensor([0, 65]), torch.tensor([[0, 0], [0, 1]])
    calls = []

    class FakeJIT:
        def __getitem__(self, grid):
            return lambda **kw: calls.append((grid, kw))

    probe.launch(FakeJIT(), g, beta, a, bias, out, beta_out, starts, chunks, 8)
    grid, args = calls[0]
    assert grid == (5, 2, 16)
    assert args["stride_beta_token"] == 6528
    assert args["stride_beta_head"] == 1
    assert (args["BT"], args["BS"], args["num_warps"], args["num_stages"]) == (
        64,
        32,
        8,
        3,
    )
    assert args["HAS_BIAS"] and args["IS_VARLEN"] and args["USE_LOWER_BOUND"]
    assert args["lower_bound"] == -5
    x = torch.ones(2, 2)
    assert probe.pair_delta(x, x)["bit_mismatches"] == 0
    y = x.clone()
    y[0, 0] += 0.125
    assert probe.pair_delta(x, y)["bit_mismatches"] == 1


def test_actual_serving_jit_wrapper_is_cpu_inspectable():
    from triton.runtime.jit import JITFunction

    from vllm.models.kimi_k3.amd.ops.third_party.kda.chunk import (
        kda_gate_chunk_cumsum_vector_kernel,
    )

    jit = kda_gate_chunk_cumsum_vector_kernel.fn.fn
    assert isinstance(jit, JITFunction)
    assert jit.__name__ == probe.KERNEL
    assert jit.fn.__code__.co_filename.endswith(probe.CHUNK)


def record():
    row = dict(
        index=0,
        **probe.matrix()[0],
        binary_ids=["old", "new"],
        phases=[],
        eager_repetition_graph_replay_mutation_guards_passed=True,
    )
    for phase in range(2):
        row["phases"].append(
            dict(
                phase=phase,
                pairwise=[dict(elements=n, bit_mismatches=0) for n in (2048, 16)],
                arms=[
                    dict(
                        warps=w,
                        output_sha256=["a", "b"],
                        oracle=[
                            dict(max_error_ratio=0.5, pass_oracle=True),
                            dict(max_error_ratio=0.5, pass_oracle=True),
                        ],
                    )
                    for w in (8, 2)
                ],
            )
        )
    return row


def test_audit_does_not_trust_summary_verdict():
    binaries = dict(old=dict(warps=8, stages=3), new=dict(warps=2, stages=3))
    probe.audit_pair(record(), 0, binaries)
    for mutate in (
        lambda r: r.update(index=1),
        lambda r: r["phases"].pop(),
        lambda r: r["phases"][0]["arms"][0]["oracle"][0].update(pass_oracle=False),
        lambda r: r["phases"][0]["pairwise"][0].update(elements=1),
        lambda r: r["phases"][0]["pairwise"][0].update(bit_mismatches=1),
        lambda r: r.update(eager_repetition_graph_replay_mutation_guards_passed=False),
    ):
        altered = copy.deepcopy(record())
        mutate(altered)
        with pytest.raises(ValueError):
            probe.audit_pair(altered, 0, binaries)


def test_real_preparation_and_source_checks_without_gpu(tmp_path):
    if not probe.INVENTORY.exists():
        pytest.skip("local campaign evidence is not installed")
    manifest = tmp_path / "manifest.json"
    probe.prepare(manifest)
    data = json.loads(manifest.read_text())
    probe.verify(data)
    assert data["history"]["59ae0c88f"] == data["history"]["ce6df61aa"]
    with pytest.raises(FileExistsError):
        probe.save_new(manifest, {})
    data["warps"] = [8, 4]
    with pytest.raises(ValueError, match="geometry"):
        probe.verify(data)
