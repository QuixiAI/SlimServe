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


PIPELINE_PY = LORA_PY.with_name("pipeline.py")


def _unguarded_loop_attaches(source: str) -> list[int]:
    """Lines where pipeline.py attaches adapters in a loop outside any `try`.

    A single `Lora.attach` is atomic (it validates every target before adding
    any term), so a lone attach followed by `try/finally: detach` is safe. A
    loop is not: if adapter #2 raises, #1 is already on the resident DiT, so
    loops must run inside the `try` whose `finally` detaches (or use
    `lora.attached()`, which does exactly that)."""
    import ast

    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    bad = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "attach"
        ):
            continue
        in_loop = in_try = False
        child, parent = node, parents.get(node)
        while parent is not None and not isinstance(
            parent, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if isinstance(parent, (ast.For, ast.While)) and child in parent.body:
                in_loop = True
            if isinstance(parent, ast.Try) and child in parent.body:
                in_try = True
            child, parent = parent, parents.get(parent)
        if in_loop and not in_try:
            bad.append(node.lineno)
    return bad


def test_pipeline_never_attaches_adapters_in_a_loop_outside_a_try():
    assert _unguarded_loop_attaches(PIPELINE_PY.read_text()) == []


def test_the_loop_check_catches_the_alpha_pattern():
    leaky = (
        "def alpha(dit, adapters):\n"
        "    for lora, strength in adapters:\n"
        "        lora.attach(dit, strength)\n"
        "    try:\n"
        "        pass\n"
        "    finally:\n"
        "        for lora, _ in adapters:\n"
        "            lora.detach(dit)\n"
    )
    assert _unguarded_loop_attaches(leaky) == [3]


# ---- the user LoRA cache ------------------------------------------------------


def _loader(lora_mod, loads):
    def load(key):
        loads.append(Path(key).name)
        return _adapter(lora_mod, Path(key).stem)

    return load


def test_user_lora_cache_is_bounded_lru(lora_mod, tmp_path):
    loads = []
    cache = lora_mod.LoraCache(2)
    load = _loader(lora_mod, loads)
    a = cache.get(tmp_path / "a.safetensors", load)
    cache.get(tmp_path / "b.safetensors", load)
    assert cache.get(tmp_path / "a.safetensors", load) is a  # hit; a is now newest
    cache.get(tmp_path / "c.safetensors", load)  # evicts b, the least recent
    assert len(cache) == 2
    assert tmp_path / "a.safetensors" in cache
    assert tmp_path / "b.safetensors" not in cache
    cache.get(tmp_path / "b.safetensors", load)
    assert loads == ["a.safetensors", "b.safetensors", "c.safetensors", "b.safetensors"]


def test_evicting_releases_the_reference_but_an_attached_adapter_detaches(
    lora_mod, tmp_path
):
    import gc
    import weakref

    cache = lora_mod.LoraCache(1)
    load = _loader(lora_mod, [])
    dit = _DiT()
    a = cache.get(tmp_path / "a.safetensors", load)
    a.attach(dit, 1.0)
    cache.get(tmp_path / "b.safetensors", load)  # a evicted while attached
    a.detach(dit)
    assert dit.lora == {}
    ref = weakref.ref(a)
    del a
    gc.collect()
    assert ref() is None  # nothing but the request held it


def test_user_lora_cache_size_comes_from_the_environment(lora_mod, monkeypatch):
    monkeypatch.delenv("SLIMSERVE_VIDEO_LORA_CACHE", raising=False)
    assert lora_mod.LoraCache().max_entries == lora_mod.DEFAULT_USER_LORA_CACHE
    monkeypatch.setenv("SLIMSERVE_VIDEO_LORA_CACHE", "7")
    assert lora_mod.LoraCache().max_entries == 7
    assert lora_mod.LoraCache(0).max_entries == 0
    with pytest.raises(ValueError):
        lora_mod.LoraCache(-1)


def test_a_zero_sized_cache_keeps_nothing(lora_mod, tmp_path):
    loads = []
    cache = lora_mod.LoraCache(0)
    load = _loader(lora_mod, loads)
    cache.get(tmp_path / "a.safetensors", load)
    cache.get(tmp_path / "a.safetensors", load)
    assert len(cache) == 0 and loads == ["a.safetensors", "a.safetensors"]


def test_the_engine_has_no_unbounded_adapter_dict():
    source = PIPELINE_PY.read_text()
    assert "user_lora_cache[" not in source
    assert "self.user_lora_cache = lora_mod.LoraCache(" in source
