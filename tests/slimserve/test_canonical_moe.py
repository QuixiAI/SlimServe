# SPDX-License-Identifier: Apache-2.0
import pytest

from slimserve.canonical_moe import enabled, geometry


def test_flag_is_off_and_requires_bounded_journal(monkeypatch):
    monkeypatch.delenv("SLIMSERVE_GLM53_NATIVE_ORDER", raising=False)
    monkeypatch.delenv("SLIMSERVE_GLM53_CANONICAL_MOE", raising=False)
    assert not enabled()
    monkeypatch.setenv("SLIMSERVE_GLM53_CANONICAL_MOE", "yes")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        enabled()
    monkeypatch.setenv("SLIMSERVE_GLM53_CANONICAL_MOE", "1")
    monkeypatch.delenv("SLIMSERVE_GLM53_MODEL_JOURNAL", raising=False)
    with pytest.raises(ValueError, match="bounded model journal"):
        enabled()
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "1")
    assert enabled()


@pytest.mark.parametrize("tokens", [0, 8193, True, 1.0])
def test_geometry_rejects_out_of_scope(tokens):
    with pytest.raises(ValueError):
        geometry(tokens, 8, 288, 32)


def test_geometry_matches_capacity_and_extreme_skew():
    assert geometry(1, 8, 288, 8) == (64, 8, 8)
    assert geometry(640, 8, 288, 32) == (14048, 439, 640)
    assert geometry(7616, 8, 288, 64) == (79072, 1236, 7616)
    for args in [(640, 7, 288, 32), (640, 8, 256, 32), (640, 8, 288, 128)]:
        with pytest.raises(ValueError):
            geometry(*args)
