# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global Muse attention must retain history in both kernels and allocation."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("one_request", [False, True])
def test_fused_metadata_preserves_independent_groups(one_request):
    from vllm.model_executor.models.muse_glimmer import _fused_layer_metadata

    groups = [
        SimpleNamespace(
            block_table=torch.tensor([[i, i + 1], [i + 2, i + 3]], dtype=torch.int32),
            seq_lens_gpu=torch.tensor([19, 11], dtype=torch.int32),
            slot_mapping=torch.tensor([i * 64 + 18, (i + 2) * 64 + 10]),
        )
        for i in (1, 9, 17)
    ]
    metadata = dict(zip(("local-a", "local-b", "full-a"), groups))
    metadata["local-a2"] = groups[0]
    names = ["local-a", "local-b", "full-a", "local-a2"]
    tables, lengths, slots = _fused_layer_metadata(metadata, names, 2, one_request)
    for i in range(3):
        expected_table = groups[i].block_table
        expected_lengths = groups[i].seq_lens_gpu
        if one_request:
            expected_table = expected_table[:1].expand(2, -1)
            expected_lengths = torch.tensor([18, 19], dtype=torch.int32)
        torch.testing.assert_close(tables[i], expected_table)
        torch.testing.assert_close(lengths[i], expected_lengths)
        torch.testing.assert_close(slots[i], groups[i].slot_mapping)
    assert tables[0] is tables[3]
    assert lengths[0] is lengths[3]
    assert slots[0] is slots[3]


def test_global_attention_clears_inherited_window_for_cache_planner(monkeypatch):
    from vllm.model_executor.models import muse_glimmer as muse

    class FakeAttention(torch.nn.Module):
        def __init__(self, *args, cache_config, per_layer_sliding_window, **kwargs):
            super().__init__()
            self.sliding_window = (
                cache_config.sliding_window
                if per_layer_sliding_window is None
                else per_layer_sliding_window
            )
            self.impl = SimpleNamespace(sliding_window=self.sliding_window)

    for name in (
        "QKVParallelLinear",
        "RowParallelLinear",
        "ColumnParallelLinear",
        "RMSNorm",
        "get_rope",
    ):
        monkeypatch.setattr(muse, name, lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(muse, "Attention", FakeAttention)
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=32768,
        rope_parameters={},
        sliding_window=2048,
    )
    cache = SimpleNamespace(sliding_window=2048)
    local = muse.MuseGlimmerAttention(config, 0, cache)
    global_layer = muse.MuseGlimmerAttention(config, 3, cache)
    assert local.attn.sliding_window == local.attn.impl.sliding_window == 2048
    assert global_layer.attn.impl.sliding_window is None
    assert global_layer.attn.sliding_window is None
