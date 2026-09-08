"""Small-k sampling against an explicit stable-order probability oracle.

Top-k retains every kth-value tie; top-p sorts by ascending (logit, token ID).
Shared noise is indexed by vocabulary ID. No tie case is exempt from equality.
Use CPU FP64 unnormalized masses for the oracle: FP32 softmax/cumsum can put the
midpoint of 154880 equal tokens above 0.5 and incorrectly drop one fewer token.
This defines mathematical sampling semantics, not bitwise PyTorch reductions.
The CPU oracle is also independent of the GPU reduction kernels being checked.
"""

import numpy as np
import pytest
import torch

from vllm.quixicore.ops import quixicore_ops

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and quixicore_ops.has_topk_sample()),
    reason="needs CUDA and the QuixiCore topk_sample op",
)
DEV = "cuda"
V = 154880
K = 32


def reference(logits, k, p, noise):
    values = logits.cpu().numpy().astype(np.float64)
    ids = np.argsort(values, axis=-1, kind="stable")
    values = np.take_along_axis(values, ids, axis=-1)
    threshold = np.take_along_axis(
        values, (values.shape[1] - k.cpu().numpy())[:, None], axis=-1
    )
    values[values < threshold] = -np.inf
    mass = np.exp(values - values[:, -1:])
    if p is not None:
        budget = (1 - p.cpu().numpy().astype(np.float64)[:, None]) * mass.sum(
            -1, keepdims=True
        )
        discard = mass.cumsum(-1) <= budget
        discard[:, -1] = False
        mass[discard] = 0
    masked = np.zeros_like(mass)
    np.put_along_axis(masked, ids, mass, axis=-1)
    sampled = (masked / noise.cpu().numpy()).argmax(-1)
    return torch.from_numpy(sampled).to(logits.device)


@pytest.mark.parametrize("batch", [1, 3, 16])
@pytest.mark.parametrize("trial", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("noise_dtype", [torch.float32, torch.float64])
def test_matches_reference_with_shared_vocabulary_noise(batch, trial, noise_dtype):
    generator = torch.Generator(device=DEV).manual_seed(100 * batch + trial)
    logits = torch.randn(batch, V, device=DEV, generator=generator) * (1 + trial)
    if trial % 2:
        logits[:, :50] = -float("inf")
    if trial == 2:
        logits[:, 7] = logits[:, 11] = logits.cpu().amax(dim=-1).to(DEV)
    if trial == 3:
        logits = logits.round()  # Cutoff groups can greatly exceed 32.
    if trial == 4:
        logits = logits.bfloat16().float()  # Actual output-projection precision.
    k = torch.randint(
        1, K + 1, (batch,), device=DEV, dtype=torch.int32, generator=generator
    )
    p = (
        None
        if trial == 0
        else torch.rand(batch, device=DEV, generator=generator) * 0.7 + 0.3
    )
    noise = torch.empty(batch, V, device=DEV, dtype=noise_dtype).exponential_(
        generator=generator
    )
    got = quixicore_ops.topk_sample(logits, k, p, noise)
    assert torch.equal(got, reference(logits, k, p, noise))


@pytest.mark.parametrize("vocab", [512, 513, V])
@pytest.mark.parametrize("p_value", [None, 0.0, 0.02, 0.5, 0.95, 1.0])
def test_unbounded_ties_and_strictly_higher_values(vocab, p_value):
    generator = torch.Generator(device=DEV).manual_seed(901)
    logits = torch.zeros(4, vocab, device=DEV)
    # All ties; small strict prefix; a nucleus boundary above kth;
    # and one finite token with every other vocabulary entry masked.
    logits[1, 1:10] = 1
    logits[2, 1:10] = 3
    logits[2, 21:31] = 2
    logits[3] = -float("inf")
    logits[3, 31] = 0
    k = torch.tensor([1, 20, 32, 32], dtype=torch.int32, device=DEV)
    p = None if p_value is None else torch.full((4,), p_value, device=DEV)
    noise = torch.empty_like(logits).exponential_(generator=generator)
    got = quixicore_ops.topk_sample(logits, k, p, noise)
    assert torch.equal(got, reference(logits, k, p, noise))


def test_local_overflow_and_noncontiguous_rows():
    generator = torch.Generator(device=DEV).manual_seed(311)
    logits = torch.full((6, V), -float("inf"), device=DEV)[::2]
    # More than 32 ties inside a single candidate partition; other partitions
    # contain higher values. Discarded ties must still contribute their mass.
    logits[:, 100:3100] = 0
    logits[:, 70000:70009] = 1
    k = torch.tensor([10, 20, 32], dtype=torch.int32, device=DEV)
    p = torch.tensor([0.5, 0.95, 1.0], device=DEV)
    noise = torch.empty((3, V), device=DEV).exponential_(generator=generator)
    assert not logits.is_contiguous()
    got = quixicore_ops.topk_sample(logits, k, p, noise)
    assert torch.equal(got, reference(logits, k, p, noise))


def test_signed_zero_is_one_tie_group():
    generator = torch.Generator(device=DEV).manual_seed(991)
    logits = torch.zeros(3, V, device=DEV)
    logits[:, ::2] = -0.0
    k = torch.tensor([1, 20, 32], dtype=torch.int32, device=DEV)
    p = torch.tensor([0.1, 0.5, 0.95], device=DEV)
    noise = torch.empty_like(logits).exponential_(generator=generator)
    got = quixicore_ops.topk_sample(logits, k, p, noise)
    assert torch.equal(got, reference(logits, k, p, noise))


@pytest.mark.parametrize("vocab", [512, 513, V])
def test_uniform_noise_exposes_exact_nucleus_boundary(vocab):
    logits = torch.zeros(6, vocab, device=DEV)
    k = torch.full((6,), 20, dtype=torch.int32, device=DEV)
    p = torch.tensor([0.0, 0.02, 0.1, 0.5, 0.95, 1.0], device=DEV)
    noise = torch.ones_like(logits)
    # All retained scores tie, so argmax must be the lowest retained ID.
    # Random noise alone rarely catches a one-token cutoff error.
    got = quixicore_ops.topk_sample(logits, k, p, noise)
    assert torch.equal(got, reference(logits, k, p, noise))
    if vocab % 2 == 0:
        assert got[3].item() == vocab // 2  # Exact half of a uniform mass.


def test_every_k_uses_the_entire_uniform_tie_group():
    vocab = 1000
    logits = torch.zeros(32, vocab, device=DEV)
    k = torch.arange(1, 33, dtype=torch.int32, device=DEV)
    p = torch.linspace(0, 1, 32, device=DEV)
    expected = ((1 - p.double()) * vocab).floor().long().clamp(max=vocab - 1)
    got = quixicore_ops.topk_sample(logits, k, p, torch.ones_like(logits))
    assert torch.equal(got, expected)


def test_fp64_noise_is_not_silently_narrowed():
    logits = torch.full((1, 512), -float("inf"), device=DEV)
    logits[0, 3] = logits[0, 9] = 0
    k = torch.tensor([2], dtype=torch.int32, device=DEV)
    noise = torch.ones((1, 512), device=DEV, dtype=torch.float64)
    noise[0, 3] = 1 + 1e-9  # Both become 1 if narrowed to float32.
    got = quixicore_ops.topk_sample(logits, k, None, noise)
    assert got.item() == 9
    assert torch.equal(got, reference(logits, k, None, noise))


def test_seed_determinism_includes_large_tie_groups():
    from vllm.v1.sample.ops.topk_topp_sampler import small_topk_sample

    logits = torch.zeros(3, V, device=DEV)
    k = torch.tensor([1, 4, 20], device=DEV, dtype=torch.int32)
    p = torch.tensor([1.0, 0.5, 0.95], device=DEV)

    def draw():
        generators = {
            i: torch.Generator(device=DEV).manual_seed(11 + i) for i in range(3)
        }
        return small_topk_sample(logits, generators, k, p, False)

    expected = draw()
    for _ in range(5):
        assert torch.equal(draw(), expected)


def test_graph_replay_uses_live_logits_and_noise():
    generator = torch.Generator(device=DEV).manual_seed(778)
    logits = torch.randn(3, 512, device=DEV, generator=generator)
    k = torch.tensor([1, 20, 32], dtype=torch.int32, device=DEV)
    p = torch.tensor([0.95, 0.5, 1.0], device=DEV)
    noise = torch.empty_like(logits).exponential_(generator=generator)
    quixicore_ops.topk_sample(logits, k, p, noise)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = quixicore_ops.topk_sample(logits, k, p, noise)
    for tied in (False, True, False):
        logits.normal_(generator=generator)
        if tied:
            logits.round_()
        noise.exponential_(generator=generator)
        graph.replay()
        assert torch.equal(got, reference(logits, k, p, noise))


def test_empty_batch_and_wrong_noise_shape():
    logits = torch.empty((0, 512), device=DEV)
    k = torch.empty((0,), dtype=torch.int32, device=DEV)
    out = quixicore_ops.topk_sample(logits, k, None, logits)
    assert out.shape == (0,) and out.dtype == torch.int64
    with pytest.raises(RuntimeError, match="noise"):
        quixicore_ops.topk_sample(
            torch.zeros((1, 512), device=DEV),
            torch.ones((1,), dtype=torch.int32, device=DEV),
            None,
            torch.ones((1, K), device=DEV),
        )


@pytest.mark.parametrize("vocab", [513, V])
def test_nonfinite_dummy_inputs_never_emit_out_of_range_ids(vocab):
    # Dummy startup inputs need index safety, not meaningful probabilities.
    logits = torch.zeros(6, vocab, device=DEV)
    logits[0] = float("nan")
    logits[1] = float("inf")
    logits[2] = -float("inf")
    logits[3, 7] = float("nan")
    logits[4, 11] = float("inf")
    logits[5, ::3] = float("nan")
    k = torch.tensor([1, 20, 32, 20, 32, 1], dtype=torch.int32, device=DEV)
    for p in (None, torch.full((6,), 0.95, device=DEV)):
        got = quixicore_ops.topk_sample(logits, k, p, torch.ones_like(logits))
        assert ((got >= 0) & (got < vocab)).all().item()
