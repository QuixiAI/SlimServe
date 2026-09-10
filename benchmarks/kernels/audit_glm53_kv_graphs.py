# SPDX-License-Identifier: Apache-2.0
"""Inspect actual AOT roots and both live launches, not controller attestations."""

from pathlib import Path
from types import SimpleNamespace

from benchmarks.analyze_glm53_quality_pair import require
from benchmarks.kernels.audit_glm53_geometry_graphs import graph_inventory
from benchmarks.kernels.glm53_attention_overwrite import KVOnlyOverwrite
from benchmarks.kernels.glm53_kv_loader import SCHEMA, check_launcher, check_source
from slimserve.rmsnorm_diagnostic import single_launcher_receipt


def inventory(modules, manifest, rank, mode, observer):
    require(
        manifest["schema"] == SCHEMA and mode in ("control", "kv"), "invalid KV audit"
    )

    def check_target(tuner, target):
        check_source(target, Path(manifest["private_namespace"]))
        check_source(target["kv"], Path(manifest["private_namespace"]))
        combo = tuner.launchers[0]
        check_launcher(tuner.compile_results[0], combo, target, observer)
        run = vars(tuner).get("run")
        require(
            run is not None
            and getattr(tuner, "_cached_launcher", None) is None
            and tuner.save_cache_hook is None,
            "KV direct dispatch missing or cached path changed",
        )
        if mode == "control":
            require(run is combo, "control does not directly launch original combo")
            return dict(dispatch="direct_combo", appended=None)
        adapter = getattr(run, "__self__", None)
        require(
            type(adapter) is KVOnlyOverwrite
            and getattr(run, "__func__", None) is KVOnlyOverwrite.run
            and "run" not in vars(adapter)
            and adapter.combo is combo,
            "graph call bypasses KV adapter or original combo",
        )
        extra = adapter.split_kv
        # Follow the live launcher's actual runner into the observed CUDA object.
        # Do not infer appended identity from the combo's compile_results.
        kernel = getattr(
            getattr(extra, "__globals__", {}).get("runner"), "__self__", None
        )
        require(kernel is not None, "KV appended launcher has no static runner")
        compiled = SimpleNamespace(kernel=kernel)
        check_launcher(compiled, extra, target["kv"], observer)
        return dict(
            dispatch="combo_then_kv",
            appended=dict(
                source=target["kv"]["relative"],
                source_sha256=target["kv"]["source_sha256"],
                selected=single_launcher_receipt(extra),
                cubin_sha256=observer.digest(compiled),
                observed_binary_index=observer.records[id(kernel)]["index"],
            ),
        )

    return graph_inventory(modules, manifest, rank, observer, check_target)
