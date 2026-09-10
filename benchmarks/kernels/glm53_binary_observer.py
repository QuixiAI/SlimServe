# SPDX-License-Identifier: Apache-2.0
"""Startup-only proof of the binary passed to Torch's static CUDA loader.

Not installed in serving. Observe the actual driver-load input while it exists;
afterward retain strong object/handle provenance, never guess a cached image.
"""

import base64
import hashlib
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from benchmarks.kernels.glm53_rmsnorm_geometry import binary_sha
from slimserve.rmsnorm_diagnostic import sha


class StaticCudaBinaryObserver:
    def __init__(self, rank, cache_roots, *, expected_images=None, emit=None):
        if type(rank) is not int or rank not in range(4):
            raise ValueError("invalid binary-observer rank")
        self.rank = rank
        self.roots = {Path(root).resolve() for root in cache_roots}
        if not self.roots:
            raise ValueError("explicit private binary roots required")
        self.expected = {
            key: frozenset(value) for key, value in (expected_images or {}).items()
        }
        self.emit = emit or (lambda record: None)
        self.lock = threading.RLock()
        self.records = {}
        self.sealed = False
        self.installed = False

    @staticmethod
    def metadata(kernel):
        return {
            name: getattr(kernel, name)
            for name in ("name", "hash", "num_warps", "shared")
        }

    def image(self, kernel):
        """Read only the precise path about to be passed to the driver."""
        filename = getattr(kernel, "cubin_path", None)
        if not isinstance(filename, str):
            raise ValueError("pre-load cubin path required")
        path = Path(filename)
        if (
            not path.is_absolute()
            or path.resolve() != path
            or path.parent.parent not in self.roots
        ):
            raise ValueError("binary path outside exact private cache roots")
        key = base64.b32encode(bytes.fromhex(kernel.hash)).decode().rstrip("=")
        if path.parent.name != key or path.name != kernel.name + ".cubin":
            raise ValueError("binary path/key/name mismatch")
        digest = sha(path)
        raw = getattr(kernel, "cubin_raw", None)
        if raw is not None and (
            not isinstance(raw, bytes) or hashlib.sha256(raw).hexdigest() != digest
        ):
            raise ValueError("pre-load memory/disk binary disagreement")
        expected = self.expected.get((key, kernel.name))
        if expected is not None and digest not in expected:
            raise ValueError("unqualified target binary image")
        return path, digest

    def validate(self, kernel, record):
        if (
            record["kernel"] is not kernel
            or self.metadata(kernel) != record["metadata"]
            or (kernel.module, kernel.function) != record["handles"]
            or kernel.module is None
            or kernel.function is None
            or kernel.cubin_raw is not None
        ):
            raise ValueError("observed CUDA object, metadata or handles changed")

    def load(self, kernel, original_load, device):
        with self.lock:
            if type(device) is not int or device != self.rank:
                raise ValueError("binary loaded on wrong rank")
            known = self.records.get(id(kernel))
            if known is not None:
                self.validate(kernel, known)
                # make_launcher may restore a path before load_kernel's no-op.
                # If present, it must still name the same qualified bytes.
                if kernel.cubin_path is not None:
                    _, digest = self.image(kernel)
                    if digest != known["sha256"]:
                        raise ValueError("repeated-load binary changed")
                result = original_load(kernel, device)
                self.validate(kernel, known)
                self.emit(
                    dict(
                        event="binary_load_reuse", rank=self.rank, index=known["index"]
                    )
                )
                return result
            if self.sealed:
                raise ValueError("new CUDA binary load after seal")
            if kernel.module is not None or kernel.function is not None:
                raise ValueError("already-loaded CUDA object has no pre-load proof")
            metadata = self.metadata(kernel)
            path, digest = self.image(kernel)
            result = original_load(kernel, device)
            if (
                self.metadata(kernel) != metadata
                or kernel.module is None
                or kernel.function is None
                or kernel.cubin_raw is not None
                or sha(path) != digest
            ):
                raise ValueError("unexpected static CUDA load transition")
            # A concurrent make_launcher can restore the same path immediately
            # after load_kernel clears it. That is not another driver load.
            if kernel.cubin_path is not None and self.image(kernel)[1] != digest:
                raise ValueError("concurrent binary path changed")
            record = dict(
                kernel=kernel,
                metadata=metadata,
                handles=(kernel.module, kernel.function),
                sha256=digest,
                path=str(path),
                index=len(self.records) + 1,
            )
            self.records[id(kernel)] = record
            self.emit(
                dict(
                    event="binary_loaded",
                    rank=self.rank,
                    index=record["index"],
                    metadata=metadata,
                    handles=list(record["handles"]),
                    path=str(path),
                    cubin_sha256=digest,
                )
            )
            return result

    def digest(self, compiled):
        with self.lock:
            kernel = compiled.kernel
            known = self.records.get(id(kernel))
            if known is not None:
                self.validate(kernel, known)
                return known["sha256"]
            if (
                getattr(kernel, "module", None) is not None
                or getattr(kernel, "function", None) is not None
            ):
                raise ValueError("loaded binary lacks pre-load observation")
            # Replacement compilation may precede its first driver load. Only
            # actual retained in-memory bytes are admissible in this state.
            return binary_sha(compiled)

    def verify_launcher(self, compiled, launcher):
        with self.lock:
            kernel = compiled.kernel
            known = self.records.get(id(kernel))
            if known is None:
                raise ValueError("graph launcher lacks observed CUDA object")
            self.validate(kernel, known)
            runner = getattr(launcher, "__globals__", {}).get("runner")
            if (
                getattr(launcher, "_is_static", False) is not True
                or getattr(runner, "__self__", None) is not kernel
                or getattr(runner, "__func__", None) is not type(kernel).run
                or getattr(launcher, "cache_hash", None)
                != base64.b32encode(bytes.fromhex(kernel.hash)).decode().rstrip("=")
                or launcher.config.num_warps != kernel.num_warps
                or launcher.shared != kernel.shared
            ):
                raise ValueError("graph launcher is not bound to its observed binary")

    def seal(self):
        with self.lock:
            if not self.records:
                raise ValueError("cannot seal empty binary observer")
            self.sealed = True
            self.emit(
                dict(
                    event="binary_observer_sealed",
                    rank=self.rank,
                    objects=len(self.records),
                )
            )

    @contextmanager
    def intercept(self, kernel_class=None):
        """Install before AOT loading; restore the exact inherited/local API."""
        if kernel_class is None:
            from torch._inductor.runtime.static_triton_launcher import (
                StaticallyLaunchedCudaKernel,
            )

            kernel_class = StaticallyLaunchedCudaKernel
        original = kernel_class.load_kernel
        if self.installed or getattr(original, "_glm53_binary_observer", False):
            raise ValueError("static binary observer installed twice")
        local = kernel_class.__dict__.get("load_kernel")

        @wraps(original)
        def observed(kernel, device):
            return self.load(kernel, original, device)

        observed._glm53_binary_observer = True
        kernel_class.load_kernel = observed
        self.installed = True
        try:
            yield self
        finally:
            if kernel_class.load_kernel is not observed:
                self.installed = False
                raise ValueError("binary load hook changed while observer was active")
            if local is None:
                delattr(kernel_class, "load_kernel")
            else:
                kernel_class.load_kernel = local
            self.installed = False
