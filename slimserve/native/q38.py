"""ctypes binding for the q38 engine (native/q38): Qwen3.8-27B on Apple Metal.

The library drives ONE sequence at a time. `Engine` owns the weights and the
GPU state; `Generator` owns the decode loop (plain, or speculative with the
model's MTP head) and the prefix snapshots that make repeated prompts cheap.
Nothing here is thread-safe: call from one thread.
"""
from __future__ import annotations

import ctypes as C
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence


class _Config(C.Structure):
    _fields_ = [("max_ctx", C.c_int), ("max_rows", C.c_int), ("block_size", C.c_int), ("serial", C.c_int)]


class _Sampling(C.Structure):
    _fields_ = [("temperature", C.c_float), ("top_k", C.c_int), ("top_p", C.c_float)]


class _GenConfig(C.Structure):
    _fields_ = [
        ("sampling", _Sampling),
        ("draft", _Sampling),
        ("depth", C.c_int),
        ("adaptive", C.c_float),
        ("chunk", C.c_int),
        ("seed", C.c_uint64),
        ("eos", C.c_int32 * 4),
        ("n_eos", C.c_int),
        ("snapshot_prompt_end", C.c_int),
        ("max_snapshot_bytes", C.c_int64),
    ]


class _GenStats(C.Structure):
    _fields_ = [
        ("prompt_tokens", C.c_int64),
        ("reused_tokens", C.c_int64),
        ("prefill_ms", C.c_double),
        ("cycles", C.c_int64),
        ("emitted", C.c_int64),
        ("drafted", C.c_int64),
        ("kept", C.c_int64),
        ("target_ms", C.c_double),
        ("draft_ms", C.c_double),
        ("snapshots", C.c_int),
        ("snapshot_bytes", C.c_int64),
    ]


class _PoolConfig(C.Structure):
    _fields_ = [("draft", _Sampling), ("depth", C.c_int), ("adaptive", C.c_float), ("max_streams", C.c_int),
                ("max_cache_bytes", C.c_int64), ("mtp_prompt_tail", C.c_int),
                ("big_every", C.c_int), ("big_rows", C.c_int)]


class _PoolRequest(C.Structure):
    _fields_ = [
        ("tokens", C.POINTER(C.c_int32)),
        ("n", C.c_int),
        ("embeds", C.c_void_p),
        ("n_emb", C.c_int),
        ("pos3", C.c_void_p),
        ("snap_at", C.POINTER(C.c_int)),
        ("n_snap", C.c_int),
        ("sampling", _Sampling),
        ("seed", C.c_uint64),
        ("eos", C.c_int32 * 4),
        ("n_eos", C.c_int),
        ("max_tokens", C.c_int64),
    ]


class _PoolEvent(C.Structure):
    _fields_ = [("stream", C.c_int), ("n", C.c_int), ("tokens", C.c_int32 * 8), ("done", C.c_int),
                ("prompt_done", C.c_int), ("reused", C.c_int64)]


class _PoolStats(C.Structure):
    _fields_ = [("cycles", C.c_int64), ("rows", C.c_int64), ("emitted", C.c_int64), ("drafted", C.c_int64),
                ("kept", C.c_int64), ("target_ms", C.c_double), ("draft_ms", C.c_double),
                ("snapshots", C.c_int), ("idle_sequences", C.c_int), ("cache_bytes", C.c_int64)]


@dataclass
class PoolEvent:
    stream: int
    tokens: list[int]
    done: int          # 0 running, 1 end of sequence, 2 max_tokens
    prompt_done: bool  # first event after the prompt is in
    reused: int        # prompt tokens taken from a live sequence or a snapshot


@dataclass
class GenStats:
    prompt_tokens: int
    reused_tokens: int
    prefill_ms: float
    cycles: int
    emitted: int
    drafted: int
    kept: int
    target_ms: float
    draft_ms: float
    snapshots: int
    snapshot_bytes: int


def _load(lib_path: str | Path) -> C.CDLL:
    lib = C.CDLL(str(lib_path))
    p, i, i64 = C.c_void_p, C.c_int, C.c_int64
    sig = {
        "q38_create": (p, [C.c_char_p, C.c_char_p, C.POINTER(_Config), C.c_char_p, i]),
        "q38_destroy": (None, [p]),
        "q38_position": (i64, [p]),
        "q38_vocab": (i, [p]),
        "q38_load_mtp": (i, [p, C.c_char_p, C.c_char_p, i]),
        "q38_has_mtp": (i, [p]),
        "q38_load_dspine": (i, [p, C.c_char_p, C.c_char_p, i]),
        "q38_has_dspine": (i, [p]),
        "q38_gen_config_default": (None, [C.POINTER(_GenConfig)]),
        "q38_gen_create": (p, [p, C.POINTER(_GenConfig)]),
        "q38_gen_free": (None, [p]),
        "q38_gen_prompt": (i, [p, C.POINTER(C.c_int32), i, C.POINTER(i), i]),
        "q38_gen_prompt_mm": (i, [p, C.POINTER(C.c_int32), i, C.c_void_p, i, C.c_void_p, C.POINTER(i), i]),
        "q38_hidden": (i, [p]),
        "q38_gen_next": (i, [p, C.POINTER(C.c_int32), i]),
        "q38_gen_reseed": (None, [p, C.c_uint64]),
        "q38_gen_set_sampling": (None, [p, C.POINTER(_Sampling)]),
        "q38_gen_set_eos": (None, [p, C.POINTER(C.c_int32), i]),
        "q38_gen_get_stats": (None, [p, C.POINTER(_GenStats)]),
        "q38_pool_config_default": (None, [C.POINTER(_PoolConfig)]),
        "q38_pool_create": (p, [p, C.POINTER(_PoolConfig)]),
        "q38_pool_free": (None, [p]),
        "q38_pool_open": (i, [p, C.POINTER(_PoolRequest)]),
        "q38_pool_close": (None, [p, i]),
        "q38_pool_step": (i, [p, C.POINTER(_PoolEvent), i]),
        "q38_pool_active": (i, [p]),
        "q38_pool_get_stats": (None, [p, C.POINTER(_PoolStats)]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    sizes = (C.c_int * 4)()
    lib.q38_abi_sizes.restype, lib.q38_abi_sizes.argtypes = None, [C.POINTER(C.c_int)]
    lib.q38_abi_sizes(sizes)
    mine = [C.sizeof(t) for t in (_Config, _Sampling, _GenConfig, _GenStats)]
    if list(sizes) != mine:
        raise RuntimeError(f"libq38 struct layout {list(sizes)} does not match this binding {mine}; rebuild both")
    lib.q38_pool_abi_sizes.restype, lib.q38_pool_abi_sizes.argtypes = None, [C.POINTER(C.c_int)]
    lib.q38_pool_abi_sizes(sizes)
    mine = [C.sizeof(t) for t in (_PoolConfig, _PoolRequest, _PoolEvent, _PoolStats)]
    if list(sizes) != mine:
        raise RuntimeError(f"libq38 pool struct layout {list(sizes)} does not match this binding {mine}; rebuild both")
    return lib


class Engine:
    """Weights + GPU state. `metallibs` is a ':'-separated search list."""

    def __init__(self, lib_path, pack, metallibs, *, mtp_pack=None, dspine_pack=None, max_ctx=32768,
                 max_rows=8):
        self._lib = _load(lib_path)
        cfg = _Config(max_ctx=max_ctx, max_rows=max_rows, block_size=16, serial=0)
        err = C.create_string_buffer(512)
        self._h = self._lib.q38_create(str(pack).encode(), str(metallibs).encode(), C.byref(cfg), err, len(err))
        if not self._h:
            raise RuntimeError(f"q38_create: {err.value.decode(errors='replace')}")
        if mtp_pack and self._lib.q38_load_mtp(self._h, str(mtp_pack).encode(), err, len(err)):
            self.close()
            raise RuntimeError(f"q38_load_mtp: {err.value.decode(errors='replace')}")
        if dspine_pack and self._lib.q38_load_dspine(self._h, str(dspine_pack).encode(), err, len(err)):
            self.close()
            raise RuntimeError(f"q38_load_dspine: {err.value.decode(errors='replace')}")
        self.max_ctx = max_ctx

    @property
    def vocab(self) -> int:
        return self._lib.q38_vocab(self._h)

    @property
    def hidden(self) -> int:
        return self._lib.q38_hidden(self._h)

    @property
    def has_mtp(self) -> bool:
        return bool(self._lib.q38_has_mtp(self._h))

    @property
    def has_dspine(self) -> bool:
        """The DSpine block drafter is loaded; generators and pools draft with it."""
        return bool(self._lib.q38_has_dspine(self._h))

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.q38_destroy(self._h)
            self._h = None

    def __del__(self):
        self.close()


class Generator:
    """Decode loop over an Engine's sequence. One live Generator per Engine."""

    def __init__(self, engine: Engine, *, depth=2, adaptive=0.5, temperature=1.0, top_k=20, top_p=0.95,
                 seed=42, eos: Sequence[int] = (248046, 248044), chunk=8, max_snapshot_bytes=16 << 30):
        self._e = engine
        self._lib = engine._lib
        cfg = _GenConfig()
        self._lib.q38_gen_config_default(C.byref(cfg))
        cfg.sampling = _Sampling(temperature, top_k, top_p)
        cfg.depth = depth if engine.has_mtp else 0
        cfg.adaptive = adaptive
        if engine.has_dspine:  # blocks of up to 7 under the marginal rule (a negative adaptive is its rate)
            cfg.depth = 7 if depth > 0 else 0
            cfg.adaptive = adaptive if adaptive < 0 else -0.06
        cfg.chunk = chunk
        cfg.seed = seed
        cfg.n_eos = len(eos)
        for k, t in enumerate(eos):
            cfg.eos[k] = t
        cfg.max_snapshot_bytes = max_snapshot_bytes
        self._g = self._lib.q38_gen_create(engine._h, C.byref(cfg))
        if not self._g:
            raise RuntimeError("q38_gen_create failed (check depth / eos / chunk)")
        self._buf = (C.c_int32 * 8)()

    def prompt(self, tokens: Sequence[int], snapshot_at: Sequence[int] = (), seed: int | None = None,
               embeds=None, pos3=None) -> None:
        """Make `tokens` the sequence, reusing the longest cached prefix.

        Image rows: a negative token takes its input embedding from `embeds`
        (a C-contiguous uint16 array [k, hidden] of BF16 bits, the k-th
        negative token taking row k); the negative value is the row's identity
        for prefix reuse. `pos3` (int32 [n, 3]) gives each token's (t, h, w)
        rotary position; omit both for text.
        """
        n = len(tokens)
        if n < 1 or n >= self._e.max_ctx:
            raise ValueError(f"prompt of {n} tokens does not fit the engine context {self._e.max_ctx}")
        if seed is not None:
            self._lib.q38_gen_reseed(self._g, seed)
        arr = (C.c_int32 * n)(*tokens)
        marks = sorted(m for m in set(snapshot_at) if 0 < m <= n)
        marr = (C.c_int * max(1, len(marks)))(*marks)
        if embeds is None and pos3 is None:
            rc = self._lib.q38_gen_prompt(self._g, arr, n, marr, len(marks))
        else:
            n_emb = 0 if embeds is None else int(embeds.shape[0])
            if n_emb and (embeds.dtype.itemsize != 2 or embeds.shape[1] != self._e.hidden
                          or not embeds.flags["C_CONTIGUOUS"]):
                raise ValueError("embeds must be a C-contiguous [k, hidden] array of BF16 bits")
            if pos3 is not None and (pos3.shape != (n, 3) or pos3.dtype.itemsize != 4
                                     or not pos3.flags["C_CONTIGUOUS"]):
                raise ValueError("pos3 must be a C-contiguous int32 [n, 3] array")
            rc = self._lib.q38_gen_prompt_mm(
                self._g, arr, n, embeds.ctypes.data if n_emb else None, n_emb,
                pos3.ctypes.data if pos3 is not None else None, marr, len(marks))
        if rc:
            raise RuntimeError(f"q38_gen_prompt failed: {rc}")

    def set_sampling(self, temperature: float, top_k: int, top_p: float) -> None:
        """Target sampling for the prompts that follow (temperature 0 = most likely token)."""
        sp = _Sampling(float(temperature), int(top_k), float(top_p))
        self._lib.q38_gen_set_sampling(self._g, C.byref(sp))

    def set_eos(self, ids: Sequence[int]) -> None:
        ids = list(ids)[:4]
        arr = (C.c_int32 * max(1, len(ids)))(*ids)
        self._lib.q38_gen_set_eos(self._g, arr, len(ids))

    def next(self, cap: int = 8) -> list[int]:
        """One decode cycle: 1..cap tokens, [] after end of sequence."""
        n = self._lib.q38_gen_next(self._g, self._buf, max(1, min(cap, 8)))
        if n < 0:
            raise RuntimeError(f"q38_gen_next failed: {n}")
        return list(self._buf[:n])

    def stream(self, max_tokens: int) -> Iterator[list[int]]:
        left = max_tokens
        while left > 0:
            out = self.next(left)
            if not out:
                return
            left -= len(out)
            yield out

    def stats(self) -> GenStats:
        s = _GenStats()
        self._lib.q38_gen_get_stats(self._g, C.byref(s))
        return GenStats(*(getattr(s, f) for f, _ in _GenStats._fields_))

    def close(self) -> None:
        if getattr(self, "_g", None):
            self._lib.q38_gen_free(self._g)
            self._g = None

    def __del__(self):
        self.close()


class Pool:
    """Several generations sharing one Engine (native/q38/include/q38/pool.h).

    Every `step()` is one pass over the weights carrying rows of all running
    streams: decode rows (with MTP drafts while few streams decode) and prompt
    chunks. Prefix reuse (idle sequences and snapshots) is shared by all
    streams. One live Pool per Engine, and not together with a Generator.
    """

    BUSY = -7

    def __init__(self, engine: Engine, *, max_streams=8, depth=2, adaptive=0.5, max_cache_bytes=16 << 30,
                 big_every: int | None = None):
        self._e = engine
        self._lib = engine._lib
        cfg = _PoolConfig()
        self._lib.q38_pool_config_default(C.byref(cfg))
        cfg.depth = depth if (engine.has_mtp or engine.has_dspine) else 0
        cfg.adaptive = adaptive  # with DSpine: only a negative value is used (the marginal rule's rate)
        cfg.max_streams = max_streams
        cfg.max_cache_bytes = max_cache_bytes
        if big_every is not None:
            # 64-row prefill passes: -1 never, 0 only while nothing decodes, N also one after
            # every N ordinary passes while streams decode (pool.h)
            cfg.big_every = big_every
        self.max_streams = max_streams
        self._p = self._lib.q38_pool_create(engine._h, C.byref(cfg))
        if not self._p:
            raise RuntimeError("q38_pool_create failed (check max_streams / depth)")
        self._ev = (_PoolEvent * 8)()

    def open(self, tokens: Sequence[int], *, temperature=1.0, top_k=20, top_p=0.95, seed=42,
             eos: Sequence[int] = (248046, 248044), max_tokens=1 << 40, snapshot_at: Sequence[int] = (),
             embeds=None, pos3=None) -> int:
        """Start a stream; returns its id, or Pool.BUSY when max_streams are running."""
        n = len(tokens)
        if n < 1 or n >= self._e.max_ctx:
            raise ValueError(f"prompt of {n} tokens does not fit the engine context {self._e.max_ctx}")
        r = _PoolRequest()
        arr = (C.c_int32 * n)(*tokens)
        r.tokens, r.n = arr, n
        n_emb = 0 if embeds is None else int(embeds.shape[0])
        if n_emb and (embeds.dtype.itemsize != 2 or embeds.shape[1] != self._e.hidden
                      or not embeds.flags["C_CONTIGUOUS"]):
            raise ValueError("embeds must be a C-contiguous [k, hidden] array of BF16 bits")
        if pos3 is not None and (pos3.shape != (n, 3) or pos3.dtype.itemsize != 4 or not pos3.flags["C_CONTIGUOUS"]):
            raise ValueError("pos3 must be a C-contiguous int32 [n, 3] array")
        r.embeds, r.n_emb = (embeds.ctypes.data if n_emb else None), n_emb
        r.pos3 = pos3.ctypes.data if pos3 is not None else None
        marks = sorted(m for m in set(snapshot_at) if 0 < m <= n)
        marr = (C.c_int * max(1, len(marks)))(*marks)
        r.snap_at, r.n_snap = marr, len(marks)
        r.sampling = _Sampling(float(temperature), int(top_k), float(top_p))
        r.seed = seed & 0xFFFFFFFFFFFFFFFF
        eos = list(eos)[:4]
        r.n_eos = len(eos)
        for k, t in enumerate(eos):
            r.eos[k] = t
        r.max_tokens = max(1, int(max_tokens))
        sid = self._lib.q38_pool_open(self._p, C.byref(r))
        if sid < 0 and sid != self.BUSY:
            raise RuntimeError(f"q38_pool_open failed: {sid}")
        return sid

    def close_stream(self, stream: int) -> None:
        self._lib.q38_pool_close(self._p, stream)

    def step(self) -> list[PoolEvent]:
        n = self._lib.q38_pool_step(self._p, self._ev, 8)
        if n < 0:
            raise RuntimeError(f"q38_pool_step failed: {n}")
        return [PoolEvent(e.stream, list(e.tokens[: e.n]), e.done, bool(e.prompt_done), e.reused)
                for e in self._ev[:n]]

    @property
    def active(self) -> int:
        return self._lib.q38_pool_active(self._p)

    def stats(self) -> dict:
        s = _PoolStats()
        self._lib.q38_pool_get_stats(self._p, C.byref(s))
        return {f: getattr(s, f) for f, _ in _PoolStats._fields_}

    def close(self) -> None:
        if getattr(self, "_p", None):
            self._lib.q38_pool_free(self._p)
            self._p = None

    def __del__(self):
        self.close()
