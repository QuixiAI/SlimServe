"""The QuixiCore small-k sampler reproduces vLLM's reference top-k / top-p masks
and, given the same exponential noise, the reference's sampled token."""

import pytest
import torch

from vllm.quixicore.ops import quixicore_ops
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and quixicore_ops.has_topk_sample()),
    reason="needs CUDA and the QuixiCore topk_sample op",
)
DEV = "cuda"
V = 154880
K = 32


def _cases(batch, trial):
    g = torch.Generator(device=DEV).manual_seed(100 * batch + trial)
    logits = torch.randn(batch, V, device=DEV, generator=g) * (1 + trial)
    if trial % 2:
        logits[:, :50] = -float("inf")  # masked entries
    if trial == 2:
        logits[:, 7] = logits[:, 11] = logits.max(dim=-1).values  # ties at the top
    if trial == 3:
        logits = logits.round()  # ties everywhere
    k = torch.randint(1, K + 1, (batch,), device=DEV, dtype=torch.int32, generator=g)
    p = None if trial == 0 else torch.rand(batch, device=DEV, generator=g) * 0.7 + 0.3
    noise = torch.empty(batch, K, device=DEV).exponential_(generator=g)
    return logits, k, p, noise


@pytest.mark.parametrize("batch", [1, 3, 16])
@pytest.mark.parametrize("trial", [0, 1, 2, 3])
def test_matches_reference_with_shared_noise(batch, trial):
    logits, k, p, noise = _cases(batch, trial)
    got = quixicore_ops.topk_sample(logits, k, p, noise)
    ref_probs = apply_top_k_top_p_pytorch(logits.clone(), k, p).softmax(-1)
    for r in range(batch):
        tok = int(got[r])
        kth = logits[r].topk(int(k[r])).values[-1]
        if int((logits[r] >= kth).sum()) > K:
            # More ties at the k-th value than the window holds: the kernel
            # keeps 32 of them where the reference keeps all (its top-p mass
            # differs), so only the top-k set is checked. Documented deviation.
            assert logits[r, tok] >= kth
            continue
        kept = ref_probs[r] > 0
        vmin = logits[r][kept].min()
        if int((logits[r][kept] == vmin).sum()) < int((logits[r] == vmin).sum()):
            # The top-p boundary falls inside a tie group: which members the
            # reference keeps is its sort order's choice, so the noise lanes
            # do not line up. The kernel keeps the same multiset of values.
            assert logits[r, tok] >= vmin
            continue
        # The kernel's noise lane j belongs to the j-th largest logit of the
        # row; place the same noise on those tokens for the reference.
        top = logits[r].topk(K)
        q_full = torch.full((V,), 1e30, device=DEV)
        q_full[top.indices] = noise[r]
        ref_tok = int((ref_probs[r] / q_full).argmax())
        assert ref_probs[r, tok] > 0, "sampled a masked token"
        # Ties share a logit; either member is a valid pick.
        assert tok == ref_tok or logits[r, tok] == logits[r, ref_tok]


def test_topk_only_and_p_one_keep_the_reference_set():
    logits = torch.randn(4, V, device=DEV) * 4
    k = torch.tensor([1, 5, 20, 32], device=DEV, dtype=torch.int32)
    p = torch.ones(4, device=DEV)
    noise = torch.ones(4, K, device=DEV)  # equal noise: the argmax is the top logit
    for pp in (None, p):
        got = quixicore_ops.topk_sample(logits, k, pp, noise)
        assert torch.equal(got.cpu(), logits.argmax(-1).cpu())


def test_python_wrapper_is_seed_deterministic_and_stays_in_the_kept_set():
    from vllm.v1.sample.ops.topk_topp_sampler import small_topk_sample

    logits = torch.randn(3, V, device=DEV) * 3
    k = torch.tensor([1, 4, 20], device=DEV, dtype=torch.int32)
    p = torch.tensor([1.0, 0.5, 0.9], device=DEV)

    def draw():
        gens = {i: torch.Generator(device=DEV).manual_seed(11 + i) for i in range(3)}
        return small_topk_sample(logits, gens, k, p, False)

    a, b = draw(), draw()
    assert torch.equal(a, b)
    ref = apply_top_k_top_p_pytorch(logits.clone(), k, p).softmax(-1)
    assert (ref.gather(1, a.unsqueeze(1)) > 0).all()
    assert int(a[0]) == int(logits[0].argmax())  # k = 1
    # Unseeded rows draw batched noise; still inside the kept set.
    c = small_topk_sample(logits, {}, k, p, False)
    assert (ref.gather(1, c.unsqueeze(1)) > 0).all()
