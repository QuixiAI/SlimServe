# SPDX-License-Identifier: Apache-2.0
"""Runtime low-rank adapters for the LTX-2.5 DiT.

An adapter is applied as y += strength * (x A^T) B^T next to the base GEMM
instead of being fused into the weights. The base checkpoint stays exactly
the official one, adapters attach and detach per stage or per request with no
reload, and the rank-450 distilled adapter costs 8.3 GiB instead of a second
39 GiB transformer. Files use `diffusion_model.<linear>.lora_{A,B}.weight`
(the ComfyUI layout upstream's LTXV_LORA_COMFY_RENAMING_MAP strips); as
upstream's fuse_loras, the delta is strength * B @ A with no alpha scaling
(the official files carry alpha == rank; any other alpha is ignored there too).
User LoRAs (upstream --lora PATH [STRENGTH]) load the same way from a path.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx

from slimserve.video.ltx25 import checkpoints
from slimserve.video.ltx25.dit import LTX25DiT

PREFIX = "diffusion_model."


class Lora:
    def __init__(self, component: str, root: Path | None = None):
        self.path = checkpoints.path_of(component, root)
        self.pairs: dict[str, tuple[mx.array, mx.array]] = {}
        self.reference_downscale = 1
        self.reference_temporal_scale = 1

    @classmethod
    def from_path(cls, path: str | Path) -> Lora:
        """A user adapter file (upstream --lora)."""
        self = cls.__new__(cls)
        self.path = Path(path).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(f"LoRA file not found: {self.path}")
        self.pairs = {}
        self.reference_downscale = 1
        self.reference_temporal_scale = 1
        return self

    def load(self) -> Lora:
        if self.pairs:
            return self
        header = checkpoints.read_header(self.path)
        self.reference_downscale = int(
            header.metadata.get("reference_downscale_factor", 1)
        )
        self.reference_temporal_scale = int(
            header.metadata.get("reference_temporal_scale_factor", 1)
        )
        raw = checkpoints.cast_operands(checkpoints.load_raw(self.path))
        for key in [k for k in raw if k.endswith(".lora_A.weight")]:
            name = key[len(PREFIX) :] if key.startswith(PREFIX) else key
            name = name[: -len(".lora_A.weight")]
            b_key = key.replace(".lora_A.", ".lora_B.")
            if b_key not in raw:
                raise KeyError(f"{self.path.name}: {key} has no lora_B partner")
            self.pairs[name] = (raw[key], raw[b_key])
        if not self.pairs:
            raise ValueError(
                f"{self.path.name}: no `<linear>.lora_A.weight` / `lora_B.weight` "
                "pairs (the ComfyUI LTXV layout upstream loads)"
            )
        return self

    def attach(self, dit: LTX25DiT, strength: float = 1.0) -> None:
        known = {k[: -len(".weight")] for k in dit.w if k.endswith(".weight")} | set(
            dit.split_k
        )
        missing = [n for n in self.pairs if n not in known]
        if missing:
            raise KeyError(
                f"{self.path.name}: {len(missing)} adapter targets not in the DiT, "
                f"e.g. {missing[:3]}"
            )
        for name, (a, b) in self.pairs.items():
            dit.lora.setdefault(name, []).append((a, b, strength))

    def detach(self, dit: LTX25DiT) -> None:
        """Undo one attach. Attaching an adapter twice adds its terms twice
        (upstream applies a LoRA given twice twice); each detach removes the
        most recent one, so attach/detach pairs nest like references."""
        for name, (a, _b) in self.pairs.items():
            terms = dit.lora.get(name, [])
            last = max((i for i, t in enumerate(terms) if t[0] is a), default=None)
            if last is None:
                continue
            terms = terms[:last] + terms[last + 1 :]
            if terms:
                dit.lora[name] = terms
            else:
                dit.lora.pop(name, None)


@contextmanager
def attached(dit: LTX25DiT, adapters: Iterable[tuple[Lora, float]]) -> Iterator[None]:
    """Attach (Lora, strength) pairs for the body of a `with`; whatever was
    attached is detached on the way out, including when an attach or the body
    raises, so a failed request never leaves an adapter on the resident DiT."""
    done: list[Lora] = []
    try:
        for lora, strength in adapters:
            lora.attach(dit, strength)
            done.append(lora)
        yield
    finally:
        for lora in reversed(done):
            lora.detach(dit)


# Loaded user adapters kept between requests. A user LoRA can be several GiB
# and every distinct path a client names would otherwise stay resident for
# the life of the server; SLIMSERVE_VIDEO_LORA_CACHE overrides the bound.
DEFAULT_USER_LORA_CACHE = 4


class LoraCache:
    """User adapters by resolved path, bounded, least recently used out first.

    Evicting drops the cache's reference, which releases the adapter's arrays
    as soon as no running request holds it: an adapter evicted while attached
    stays alive (and detachable) until that request detaches it. 0 entries
    disables reuse: every request loads its adapters afresh."""

    def __init__(self, max_entries: int | None = None):
        if max_entries is None:
            env = os.environ.get("SLIMSERVE_VIDEO_LORA_CACHE")
            max_entries = int(env) if env else DEFAULT_USER_LORA_CACHE
        if max_entries < 0:
            raise ValueError("the user LoRA cache size must be >= 0")
        self.max_entries = max_entries
        self._entries: OrderedDict[str, Lora] = OrderedDict()

    def get(
        self, path: str | Path, loader: Callable[[str], Lora] | None = None
    ) -> Lora:
        key = str(Path(path).expanduser().resolve())
        lora = self._entries.pop(key, None)
        if lora is None:
            lora = (loader or (lambda k: Lora.from_path(k).load()))(key)
        if self.max_entries:
            self._entries[key] = lora  # most recently used last
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return lora

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, path: object) -> bool:
        return str(Path(str(path)).expanduser().resolve()) in self._entries
