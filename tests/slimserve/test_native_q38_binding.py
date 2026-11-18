"""The ctypes binding of the q38 engine must match the library's struct layouts.

Loading the library runs the size checks in `slimserve.native.q38._load`; no
GPU work happens. Skipped where the engine is not built (it is macOS/Metal
only: `native/q38/build.sh`)."""
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[2] / "native" / "q38" / "build" / "libq38.dylib"


@pytest.mark.skipif(not LIB.exists(), reason="native/q38 is not built on this machine")
def test_binding_matches_library_layouts():
    import ctypes as C

    from slimserve.native import q38

    lib = q38._load(LIB)  # raises on a struct size mismatch
    cfg = q38._PoolConfig()
    lib.q38_pool_config_default(C.byref(cfg))
    assert cfg.max_streams == 8
    assert cfg.big_every >= -1 and cfg.big_rows >= 16
    assert 0 < cfg.depth <= 7
