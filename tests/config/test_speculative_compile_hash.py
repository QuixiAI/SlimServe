# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block drafters (DFlash/DSpark) compile a graph shaped by the draft block
size, so num_speculative_tokens must separate their compile caches."""

from types import SimpleNamespace

from vllm.config.speculative import SpeculativeConfig


def _hash(method, k):
    cfg = SimpleNamespace(
        method=method, draft_model_config=None, num_speculative_tokens=k
    )
    return SpeculativeConfig.compute_hash(cfg)


def test_block_drafters_key_the_compile_cache_by_block_size():
    assert _hash("dflash", 3) != _hash("dflash", 4)
    assert _hash("dspark", 5) != _hash("dspark", 7)
    assert _hash("dflash", 3) == _hash("dflash", 3)


def test_autoregressive_drafters_do_not():
    assert _hash("mtp", 1) == _hash("mtp", 3)
