"""Sub-row arithmetic for the host-resident main KV planner
(docs/host_resident_kv_design.md).

The residency splits every logical block into rows that must be whole
multiples of 16 tokens. The natural attention block is the GDN state page
over the indexer's 64 B/token, so it halves with every TP doubling; the
aligner therefore holds it at 12,688 tokens (main_kv_block_tokens), whose
13 x 976 split is the validated geometry. 6,352 (the natural TP8 block)
splits only into 1 or 397 rows."""

from types import SimpleNamespace

from vllm.v1.core.kv_cache_utils import main_kv_gpu_rows, main_kv_sub_blocks


def _cfg(**extra):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra)
    )


def test_validated_block_splits_into_thirteen_rows():
    assert main_kv_sub_blocks(_cfg(main_kv_sub_blocks=8), 12688) == 13
    assert 12688 // 13 == 976 and 976 % 16 == 0


def test_natural_tp8_block_has_no_usable_split():
    # 6352 = 16 x 397: the only legal splits are 1 (a whole 19.5 MB row at
    # TP8) or 397 (16-token rows). This is why the aligner holds the block.
    assert main_kv_sub_blocks(_cfg(main_kv_sub_blocks=8), 6352) == 1
    assert main_kv_sub_blocks(_cfg(main_kv_sub_blocks=300), 6352) == 397


def test_defaults_without_a_transfer_config():
    bare = SimpleNamespace(kv_transfer_config=None)
    assert main_kv_sub_blocks(bare, 12688) == 13
    assert main_kv_gpu_rows(bare) == 16
    assert main_kv_gpu_rows(_cfg(main_kv_gpu_rows=104)) == 104
