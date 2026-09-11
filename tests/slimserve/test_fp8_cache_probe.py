# SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def probe():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/kernels/benchmark_glm53_fp8_cache.py"
    )
    spec = importlib.util.spec_from_file_location("fp8_cache_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rotation_is_byte_sized_not_a_fixed_copy_count(probe):
    for weight_bytes, copies in ((1024 * 4096, 97), (4096 * 512, 193)):
        assert 24 * weight_bytes < probe.L2_BYTES
        assert probe.rotation_count(weight_bytes) == copies
        assert copies * weight_bytes > 3 * probe.L2_BYTES
    with pytest.raises(ValueError):
        probe.rotation_count(0)


def test_oracle_rounds_dequant_weight_before_fp64_matmul(probe):
    q = torch.ones(128, 128).to(torch.float8_e4m3fn)
    scale = torch.tensor([[1.007]])
    expected = torch.full((128, 128), 1.007).bfloat16().double()
    assert torch.equal(probe.oracle_weight(q, scale), expected)


@pytest.mark.parametrize("rank", [0, 3])
def test_tp4_shared_expert_slices_and_gate_up_order(probe, monkeypatch, rank):
    gate = (
        torch.arange(1, 5)
        .repeat_interleave(512)[:, None]
        .expand(2048, 4096)
        .to(torch.float8_e4m3fn)
    )
    up = (
        torch.arange(5, 9)
        .repeat_interleave(512)[:, None]
        .expand(2048, 4096)
        .to(torch.float8_e4m3fn)
    )
    down = (
        torch.arange(10, 14)
        .repeat_interleave(512)[None, :]
        .expand(4096, 2048)
        .to(torch.float8_e4m3fn)
    )
    scales = torch.arange(16, dtype=torch.float32)[:, None].expand(16, 32).contiguous()
    tensors = {
        "gate_proj.weight": gate,
        "up_proj.weight": up,
        "down_proj.weight": down,
        "gate_proj.weight_scale": scales,
        "up_proj.weight_scale": scales + 100,
        "down_proj.weight_scale": scales.t().contiguous(),
    }

    class Source:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get_tensor(self, key):
            prefix = "model.language_model.layers.3.mlp.shared_experts."
            assert key.startswith(prefix)
            return tensors[key[len(prefix) :]]

    monkeypatch.setattr(probe, "safe_open", lambda *args, **kwargs: Source())
    result = probe.load_weights(Path("unused"), 3, rank)
    q, s = result["gate_up"]
    assert q.shape == (1024, 4096) and s.shape == (8, 32)
    assert (q[:512].float() == rank + 1).all()
    assert (q[512:].float() == rank + 5).all()
    assert torch.equal(s[:4], scales[rank * 4 : (rank + 1) * 4])
    assert torch.equal(s[4:], scales[rank * 4 : (rank + 1) * 4] + 100)
    q, s = result["down"]
    assert q.shape == (4096, 512) and s.shape == (32, 4)
    assert (q.float() == rank + 10).all()
    assert torch.equal(s, scales.t()[:, rank * 4 : (rank + 1) * 4])
