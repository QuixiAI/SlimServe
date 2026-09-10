# SPDX-License-Identifier: Apache-2.0
"""Scoped real-loader adapter for the qualified multi-source diagnostic.

Not installed in serving. Keep hooks active through loading, independent graph
inventory and target sealing. Target sealing does NOT seal the binary observer:
later non-target compilation is legitimate during serving graph capture.
"""

import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from benchmarks.kernels.check_glm53_attention_norms import (
    load_preserving_provenance,
    require,
)
from benchmarks.kernels.glm53_binary_observer import StaticCudaBinaryObserver
from benchmarks.kernels.glm53_rmsnorm_geometry import MultiIntervention
from slimserve.rmsnorm_diagnostic import sha


def expected_images(manifest, rank):
    images = {}
    for target in manifest["targets"][str(rank)]:
        for saved in target["configs"].values():
            key = (saved["triton_cache_hash"], target["kernel"])
            images.setdefault(key, set()).add(saved["cubin_sha256"])
    return images


class GeometryLoader:
    def __init__(self, rank, manifest, manifest_path, mode, *, emit=None):
        self.rank = rank
        self.manifest = manifest
        self.manifest_path = Path(manifest_path)
        self.private = Path(manifest["private_namespace"])
        self.cache = self.private / "inductor_cache"
        self.observer = StaticCudaBinaryObserver(
            rank,
            [self.cache / "triton" / str(rank)],
            expected_images=expected_images(manifest, rank),
            emit=emit,
        )
        self.controller = MultiIntervention(
            rank, manifest, manifest_path, mode, binary_observer=self.observer
        )
        self.installed = False
        self.templates = {}

    def compile_replacement(self, target, saved):
        """Use original debug provenance and the unchanged rank-local cache rule."""
        require(
            "TRITON_CACHE_DIR" not in os.environ
            and os.getenv("TORCHINDUCTOR_CACHE_DIR") == str(self.cache),
            "geometry compiler requires original per-device cache semantics",
        )
        import torch
        import triton
        from torch._inductor.runtime.cache_dir_utils import triton_cache_dir

        require(
            Path(triton_cache_dir(self.rank)) == self.cache / "triton" / str(self.rank),
            "resolved geometry cache is not rank-private",
        )
        original = Path(self.manifest["original_namespace"]) / target["relative"]
        copied = self.private / target["relative"]
        debug = Path(target["debug_source"])
        require(
            sha(original) == sha(copied) == target["source_sha256"]
            and sha(debug) == target["debug_source_sha256"],
            "replacement source/debug receipt changed",
        )
        template = self.templates.get(target["relative"])
        if template is None:
            name = (
                f"glm53_geometry_{sha(self.manifest_path)}_rank{self.rank}_"
                + copied.stem
            )
            template = load_preserving_provenance(
                copied, original, target["kernel"], name, debug_source=debug
            )
            self.templates[target["relative"]] = template
        with torch.cuda.device(self.rank):
            return template._precompile_config(
                triton.Config(
                    {key: saved[key] for key in ("XBLOCK", "R0_BLOCK")},
                    num_warps=saved["num_warps"],
                    num_stages=saved["num_stages"],
                )
            )

    @contextmanager
    def intercept(self, *, future_class=None, code_cache=None, kernel_class=None):
        if future_class is None or code_cache is None:
            from torch._inductor.codecache import PyCodeCache, StaticAutotunerFuture

            future_class = StaticAutotunerFuture
            code_cache = PyCodeCache
        original_result = future_class.result
        original_load = code_cache.__dict__["load_by_key_path"]
        require(
            isinstance(original_load, classmethod), "expected classmethod loader API"
        )
        require(
            not self.installed
            and not any(
                getattr(function, marker, False)
                for function in (original_result, original_load.__func__)
                for marker in ("_glm53_geometry_loader", "_glm53_rmsnorm_diagnostic")
            ),
            "conflicting geometry/legacy loader hook",
        )
        local_result = future_class.__dict__.get("result")

        @wraps(original_result)
        def result(future, timeout=None):
            return self.controller.resolve(
                future, original_result, self.compile_replacement, timeout
            )

        @wraps(original_load.__func__)
        def load_by_key_path(cls, *args, **kwargs):
            module = original_load.__func__(cls, *args, **kwargs)
            self.controller.bind_graph(module, self.compile_replacement)
            return module

        result._glm53_geometry_loader = True
        load_by_key_path._glm53_geometry_loader = True
        load_hook = classmethod(load_by_key_path)
        # Observer is installed FIRST: bundle loading can resolve binaries before
        # any graph-global callback has an opportunity to inspect them.
        with self.observer.intercept(kernel_class):
            future_class.result = result
            code_cache.load_by_key_path = load_hook
            self.installed = True
            try:
                yield self
            finally:
                changed = []
                if future_class.result is not result:
                    changed.append("static future")
                elif local_result is None:
                    delattr(future_class, "result")
                else:
                    future_class.result = local_result
                if code_cache.__dict__["load_by_key_path"] is not load_hook:
                    changed.append("code cache")
                else:
                    code_cache.load_by_key_path = original_load
                self.installed = False
                require(not changed, f"foreign loader hook change: {changed}")

    def close(self):
        require(not self.installed, "cannot close an active geometry loader")
        for child in self.controller.controllers.values():
            child.stream.close()
