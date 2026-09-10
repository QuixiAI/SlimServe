# SPDX-License-Identifier: Apache-2.0
"""Shared scoped hook lifecycle for source-qualified static-loader diagnostics."""

import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from benchmarks.analyze_glm53_quality_pair import require


class ScopedKernelLoader:
    """Subclasses supply controller, observer and compile_replacement.

    Controllers resolve futures atomically and bind finished graph modules.
    Installing this context alone neither loads a model nor launches a kernel.
    """

    hook_marker = ""

    def check_cache(self):
        from torch._inductor.runtime.cache_dir_utils import triton_cache_dir

        expected = self.cache / "triton" / str(self.rank)
        actual = os.getenv("TRITON_CACHE_DIR")
        require(
            actual in (None, str(expected))
            and os.getenv("TORCHINDUCTOR_CACHE_DIR") == str(self.cache),
            f"static compiler cache binding differs: triton={actual!r}, "
            f"expected={str(expected)!r}",
        )
        require(
            Path(triton_cache_dir(self.rank)) == expected
            and expected.resolve() == expected,
            "resolved compiler cache is not rank-private",
        )

    @contextmanager
    def intercept(self, *, future_class=None, code_cache=None, kernel_class=None):
        if future_class is None or code_cache is None:
            from torch._inductor.codecache import PyCodeCache, StaticAutotunerFuture

            future_class, code_cache = StaticAutotunerFuture, PyCodeCache
        original_result = future_class.result
        original_load = code_cache.__dict__["load_by_key_path"]
        require(
            isinstance(original_load, classmethod), "expected classmethod loader API"
        )
        markers = (
            "_glm53_geometry_loader",
            "_glm53_rmsnorm_diagnostic",
            "_glm53_kv_loader",
        )
        require(
            self.hook_marker in markers
            and not self.installed
            and not any(
                getattr(function, marker, False)
                for function in (original_result, original_load.__func__)
                for marker in markers
            ),
            "conflicting static loader hook",
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

        setattr(result, self.hook_marker, True)
        setattr(load_by_key_path, self.hook_marker, True)
        load_hook = classmethod(load_by_key_path)
        # Observe the actual driver-load input before any future/graph callback.
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
