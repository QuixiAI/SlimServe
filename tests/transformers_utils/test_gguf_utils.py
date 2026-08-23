# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.transformers_utils import gguf_utils


class _ScalarField:
    def __init__(self, value):
        self.parts = [np.asarray([value])]
        self._value = value

    def contents(self):
        return self._value


class _Reader:
    def __init__(self, values):
        self.values = values

    def get_field(self, key):
        value = self.values.get(key)
        return None if value is None else _ScalarField(value)


def test_extract_vision_config_uses_decoded_scalar_contents(monkeypatch):
    keys = gguf_utils.Keys
    values = {
        keys.Clip.PROJECTOR_TYPE: "muse-glimmer",
        keys.ClipVision.EMBEDDING_LENGTH: 1536,
        keys.ClipVision.FEED_FORWARD_LENGTH: 8960,
        keys.ClipVision.BLOCK_COUNT: 50,
        keys.ClipVision.Attention.HEAD_COUNT: 16,
        keys.ClipVision.IMAGE_SIZE: 896,
        keys.ClipVision.PATCH_SIZE: 14,
        keys.ClipVision.Attention.LAYERNORM_EPS: 1e-5,
    }
    monkeypatch.setattr(gguf_utils, "gguf_reader", lambda _path: _Reader(values))

    config = gguf_utils.extract_vision_config_from_gguf("mmproj-kquant.gguf")

    assert config is not None
    assert config.hidden_size == 1536
    assert config.intermediate_size == 8960
    assert config.num_hidden_layers == 50
    assert config.num_attention_heads == 16
    assert config.image_size == 896
    assert config.patch_size == 14
    assert config.layer_norm_eps == 1e-5
