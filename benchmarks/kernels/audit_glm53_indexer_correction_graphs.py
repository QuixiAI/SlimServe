# SPDX-License-Identifier: Apache-2.0
"""Inspect the actual AOT target dispatch, static image and selection arena."""

from pathlib import Path
from types import SimpleNamespace

from benchmarks.analyze_glm53_quality_pair import require
from benchmarks.kernels.audit_glm53_geometry_graphs import graph_inventory
from benchmarks.kernels.glm53_indexer_correction_loader import (
    SCHEMA,
    BoundIndexerCorrection,
    check_adapter,
)
from benchmarks.kernels.glm53_kv_loader import check_launcher, check_source
from slimserve.rmsnorm_diagnostic import single_launcher_receipt


def inventory(modules, manifest, rank, mode, observer):
    require(
        manifest["schema"] == SCHEMA and mode in ("control", "correction"),
        "invalid correction audit",
    )

    def check_target(tuner, target):
        private = Path(manifest["private_namespace"])
        check_source(target, private)
        check_source(target["correction"], private)
        combo = tuner.launchers[0]
        check_launcher(tuner.compile_results[0], combo, target, observer)
        run = vars(tuner).get("run")
        require(
            run is not None
            and getattr(tuner, "_cached_launcher", None) is None
            and tuner.save_cache_hook is None,
            "correction direct dispatch missing",
        )
        if mode == "control":
            require(run is combo, "control does not directly launch original")
            return dict(dispatch="direct_combo", appended=None)
        adapter = getattr(run, "__self__", None)
        require(
            getattr(run, "__func__", None) is BoundIndexerCorrection.run,
            "correction adapter bypassed",
        )
        extra = getattr(adapter, "correction", None)
        check_adapter(adapter, combo, extra, manifest["selection_capacity"])
        kernel = getattr(
            getattr(extra, "__globals__", {}).get("runner"), "__self__", None
        )
        require(kernel is not None, "correction has no actual static runner")
        compiled = SimpleNamespace(kernel=kernel)
        check_launcher(compiled, extra, target["correction"], observer)
        require(str(adapter.device) == f"cuda:{rank}", "selection arena on wrong rank")
        return dict(
            dispatch="combo_then_indexer_correction",
            appended=dict(
                source=target["correction"]["relative"],
                source_sha256=target["correction"]["source_sha256"],
                selected=single_launcher_receipt(extra),
                cubin_sha256=observer.digest(compiled),
                observed_binary_index=observer.records[id(kernel)]["index"],
                selection_capacity=adapter.capacity,
                selection_bytes=adapter.storage.numel(),
                selection_dtype="uint8",
                selection_device=str(adapter.device),
            ),
        )

    return graph_inventory(modules, manifest, rank, observer, check_target)
