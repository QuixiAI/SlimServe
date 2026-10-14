# SPDX-License-Identifier: Apache-2.0
"""Adapter bookkeeping on the resident DiT: one detach undoes one attach, and
`attached()` leaves nothing behind when an attach or the stage raises. The
module is loaded with MLX and the DiT stubbed (it only touches `dit.w`,
`dit.split_k` and `dit.lora` here), so this runs without a Mac."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

LORA_PY = (
    Path(__file__).resolve().parents[2] / "slimserve" / "video" / "ltx25" / "lora.py"
)


@pytest.fixture
def lora_mod(monkeypatch):
    mlx = types.ModuleType("mlx")
    core = types.ModuleType("mlx.core")
    core.array = object
    mlx.core = core
    dit = types.ModuleType("slimserve.video.ltx25.dit")
    dit.LTX25DiT = object
    checkpoints = types.ModuleType("slimserve.video.ltx25.checkpoints")
    for name, module in (
        ("mlx", mlx),
        ("mlx.core", core),
        ("slimserve.video.ltx25.dit", dit),
        ("slimserve.video.ltx25.checkpoints", checkpoints),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_lora_under_test", LORA_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _DiT:
    def __init__(self):
        self.w = {"blocks.0.attn.to_q.weight": None, "blocks.0.ff.weight": None}
        self.split_k = {}
        self.lora = {}


def _adapter(lora_mod, name, targets=("blocks.0.attn.to_q",)):
    lora = lora_mod.Lora.__new__(lora_mod.Lora)
    lora.path = Path(f"{name}.safetensors")
    lora.pairs = {t: (object(), object()) for t in targets}
    return lora


def test_one_detach_undoes_one_attach(lora_mod):
    dit, a, b = _DiT(), _adapter(lora_mod, "a"), _adapter(lora_mod, "b")
    a.attach(dit, 1.0)
    a.attach(dit, 0.5)
    b.attach(dit, 0.3)
    a.detach(dit)
    terms = dit.lora["blocks.0.attn.to_q"]
    assert [t[2] for t in terms] == [1.0, 0.3]  # the most recent a went
    a.detach(dit)
    b.detach(dit)
    assert dit.lora == {}
    a.detach(dit)  # nothing attached: a no-op
    assert dit.lora == {}


def test_attached_cleans_up_when_the_body_raises(lora_mod):
    dit, a, b = _DiT(), _adapter(lora_mod, "a"), _adapter(lora_mod, "b")
    with pytest.raises(RuntimeError), lora_mod.attached(dit, [(a, 1.0), (b, 0.5)]):
        assert len(dit.lora["blocks.0.attn.to_q"]) == 2
        raise RuntimeError("a window failed")
    assert dit.lora == {}


def test_attached_cleans_up_when_a_later_attach_fails(lora_mod):
    dit, a = _DiT(), _adapter(lora_mod, "a")
    bad = _adapter(lora_mod, "bad", targets=("blocks.9.missing",))
    with (
        pytest.raises(KeyError, match="not in the DiT"),
        lora_mod.attached(dit, [(a, 1.0), (bad, 1.0)]),
    ):
        pass
    assert dit.lora == {}


def test_attached_keeps_adapters_attached_outside_it(lora_mod):
    dit, a = _DiT(), _adapter(lora_mod, "a")
    a.attach(dit, 1.0)  # e.g. a request-wide user LoRA
    with lora_mod.attached(dit, [(a, 0.25)]):
        assert [t[2] for t in dit.lora["blocks.0.attn.to_q"]] == [1.0, 0.25]
    assert [t[2] for t in dit.lora["blocks.0.attn.to_q"]] == [1.0]
