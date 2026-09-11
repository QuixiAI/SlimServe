# SPDX-License-Identifier: Apache-2.0

import pytest

from benchmarks.measure_glm5_tier_duplicate_pages import measure, measure_shared


@pytest.mark.parametrize("owners", [1, 4])
def test_existing_index_duplicates_attention_but_keeps_tails_independent(owners):
    result = measure(owners)
    assert result["distinct_attention_pages"] == 224 + 28
    assert result["attention_page_copies"] == owners * (224 + 28)
    assert result["tail_page_copies"] == owners * 4
    assert result["disk_page_writes"] == owners * 256
    assert result["snapshots"][-1]["resumable"] == owners
    assert result["snapshots"][-1]["pending_writes"] == 0


@pytest.mark.parametrize("owners,blocks", [(0, 224), (1, 7), (1, 225)])
def test_accounting_rejects_unsupported_shapes(owners, blocks):
    with pytest.raises(ValueError, match="multiple-of-eight"):
        measure(owners, blocks)


@pytest.mark.parametrize("families", [1, 2, 4])
def test_independent_prefix_families_only_share_between_replay_rounds(families):
    result = measure(owners=8, blocks=8, prefix_families=families)
    assert result["distinct_attention_pages"] == families * 9
    assert result["attention_page_copies"] == 8 * 9
    assert result["tail_page_copies"] == 8 * 4
    assert result["snapshots"][-1]["resumable"] == 8
    assert result["avoidable_attention_copies_if_safely_shared"] == (8 - families) * 9


@pytest.mark.parametrize("families", [0, 5])
def test_invalid_family_count_is_rejected(families):
    with pytest.raises(ValueError, match="prefix_families"):
        measure(owners=4, prefix_families=families)


@pytest.mark.parametrize("families", [1, 2, 4])
def test_shared_pool_reduces_copies_without_sharing_private_tails(families):
    original = measure(owners=8, blocks=8, prefix_families=families)
    shared = measure_shared(owners=8, blocks=8, prefix_families=families)
    assert shared["attention_page_copies"] == original["distinct_attention_pages"]
    assert shared["tail_page_copies"] == original["tail_page_copies"]
    assert shared["disk_page_writes"] == families * 9 + 8 * 4
    assert shared["owners_with_all_pages_ready"] == 8
    assert shared["final_pool_stats"]["host_used"] == families * 9 + 8 * 4
