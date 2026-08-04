# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A backend that cannot serve a rank's head count must lose selection.

AITER's gfx942 MLA decode ships as pre-assembled code objects with the query
head count baked in, so it runs only multiples and divisors of 16. Kimi K3 at
TP8 gives 12 heads per rank. Before this was declared at selection time, the
selector picked AITER anyway and the engine died on an assert inside the layer
constructor -- with a message telling the user to change tensor_parallel_size,
even though TRITON_MLA was sitting right behind it in the priority list and
handles 12 heads by masking its 16-wide tile.
"""

import pytest

from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLABackend
from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend

# 96 attention heads shared over the tensor-parallel ranks K3 is served on.
KIMI_K3_HEADS = 96


@pytest.mark.parametrize(
    ("tensor_parallel_size", "aiter_fits"),
    [(2, True), (6, True), (8, False)],
)
def test_aiter_declines_the_head_counts_it_cannot_run(tensor_parallel_size, aiter_fits):
    heads_per_rank = KIMI_K3_HEADS // tensor_parallel_size
    assert AiterMLABackend.supports_num_heads(heads_per_rank) is aiter_fits


def test_triton_mla_covers_the_head_counts_aiter_declines():
    """Falling through is only useful if something behind it accepts."""
    for tensor_parallel_size in (2, 6, 8):
        heads_per_rank = KIMI_K3_HEADS // tensor_parallel_size
        assert TritonMLABackend.supports_num_heads(heads_per_rank)


def test_backends_accept_every_head_count_by_default():
    """The hook is opt-in; a backend that says nothing must not be filtered."""
    from vllm.v1.attention.backend import AttentionBackend

    assert AttentionBackend.supports_num_heads(12)
    assert AttentionBackend.supports_num_heads(1)
