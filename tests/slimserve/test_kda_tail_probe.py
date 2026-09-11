# SPDX-License-Identifier: Apache-2.0
import pytest

from benchmarks.kernels import glm53_kda_tail as tail


def case(regime, lengths):
    return dict(rank=0, seed=530901, magnitude=1.0, regime=regime, lengths=lengths)


def test_fixed_matrix_and_mixed_output_types():
    assert len(tail.matrix()) == 224
    assert sum(c["regime"] != "conditioned" for c in tail.matrix()) == 56
    specs = tail.output_spec("state", [17, 63, 65, 855])
    assert [dtype for _, dtype in specs] == ["bfloat16", "bfloat16", "float32"]
    assert specs[0][0] == (1, 18, 16, 128, 128)
    assert specs[2][0] == (4, 16, 128, 128)


@pytest.mark.parametrize("stage", ["state", "output"])
@pytest.mark.parametrize("regime", ["identity", "basis"])
@pytest.mark.parametrize("lengths", [[1], [65], [3, 65, 129]])
def test_exact_algebraic_cases_match_independent_chunk_reference(
    stage, regime, lengths
):
    import torch

    spec = case(regime, lengths)
    inputs = tail.make_inputs(stage, spec, "cpu")
    for _ in range(2):
        calculated = (
            tail.state_reference if stage == "state" else tail.output_reference
        )(inputs, lengths)
        exact = tail.exact_reference(stage, spec, inputs)
        for actual, expected in zip(calculated, exact):
            assert torch.equal(actual.to(expected.dtype), expected)
        tail.mutate(stage, inputs)


def test_accumulation_counts_at_chunk_and_sequence_boundaries():
    import torch

    spec = case("basis", [257, 65])
    inputs = tail.make_inputs("state", spec, "cpu")
    snapshots, values, final = tail.exact_reference("state", spec, inputs)
    assert torch.count_nonzero(snapshots[0, 0]) == 0
    assert snapshots[0, 2, 0, 1, 0] == 1
    assert snapshots[0, 4, 0, 1, 0] == 2
    assert final[0, 0, 1, 0] == 3
    assert final[1, 0, 1, 0] == 1
    assert torch.count_nonzero(snapshots[0, 5]) == 0
    assert torch.equal(values, inputs["v"])


def test_fp32_state_is_not_silently_narrowed_to_bf16():
    import torch

    a = torch.ones((16, 128, 128), dtype=torch.float32)
    b = a.clone()
    b[0, 0, 0] += 2**-20
    assert torch.equal(a.bfloat16(), b.bfloat16())
    assert tail.compare(a, b)["bit_mismatches"] == 1
    with pytest.raises(ValueError, match="shape/dtype"):
        tail.compare(a, b.bfloat16())


@pytest.mark.parametrize("stage", ["state", "output"])
def test_real_jit_and_serving_launch_contract(stage):
    import torch

    assert tail.jit_for(stage).__name__ == tail.NAMES[stage]
    inputs = tail.make_inputs(stage, case("identity", [3, 65]), "cpu")
    outputs = tuple(
        torch.empty(shape, dtype=getattr(torch, dtype))
        for shape, dtype in tail.output_spec(stage, [3, 65])
    )
    calls = []

    class Capture:
        def __getitem__(self, grid):
            return lambda **kwargs: calls.append((grid, kwargs))

    tail.launch(stage, Capture(), inputs, outputs, tail.CONFIGS[stage][0])
    grid, args = calls[0]
    assert args["H"] == 16 and args["K"] == args["V"] == 128
    assert args["BT"] == 64 and args["IS_VARLEN"]
    if stage == "state":
        assert grid == (4, 32)
        assert args["ht"].dtype == torch.float32
        assert args["BV"] == 32 and args["USE_INITIAL_STATE"]
        assert args["USE_GK"] and not args["USE_G"]
        assert args["STORE_FINAL_STATE"] and args["SAVE_NEW_VALUE"] and args["USE_EXP2"]
        assert args["chunk_offsets"].tolist() == [0, 1, 3]
    else:
        assert grid == (2, 3, 16)
        assert args["BK"] == args["BV"] == 64
        assert args["scale"] == 128**-0.5
