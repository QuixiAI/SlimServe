# SPDX-License-Identifier: Apache-2.0
"""Serving lifecycle for the qualified diagnostic; no production policy.

Seal targets immediately after real AOT loading, before any model forward.
Keep static-load observation active through later non-target startup compilation.
Do not globally seal that observer at capture or put checks in the token loop.
"""

import json
import threading
from contextlib import ExitStack
from functools import wraps
from pathlib import Path

from benchmarks.kernels.audit_glm53_geometry_graphs import inventory
from benchmarks.kernels.check_glm53_attention_norms import require
from benchmarks.kernels.check_glm53_geometry_loader import checked_bundles
from benchmarks.kernels.check_glm53_rmsnorm_geometry import write_new
from benchmarks.kernels.glm53_artifact_roots import (
    ArtifactRootObserver,
    discover,
    module_inventory,
)
from benchmarks.kernels.glm53_geometry_loader import GeometryLoader
from slimserve.rmsnorm_diagnostic import sha


def stable_bindings(report):
    def stable(row):
        result = {
            k: v
            for k, v in row.items()
            if k not in ("module_index", "observed_binary_index")
        }
        # KV has a second independently observed launch. Ignore only its
        # process-local index, never its source, configuration or binary bytes.
        if isinstance(result.get("appended"), dict):
            result["appended"] = {
                k: v
                for k, v in result["appended"].items()
                if k != "observed_binary_index"
            }
        return result

    return sorted(
        [stable(r) for r in report["bindings"]],
        key=lambda r: (r["graph"], r["symbol"], r["source"]),
    )


def compare_qualified(actual, qualified):
    require(
        actual["graphs"] == qualified["graphs"]
        and actual["target_bindings"] == qualified["target_bindings"]
        and stable_bindings(actual) == stable_bindings(qualified),
        "serving root bindings differ from qualified AOT source/config/binary",
    )


class ServingGeometry:
    def __init__(
        self,
        rank,
        manifest,
        path,
        *,
        loader_type=None,
        graph_inventory=None,
        root_only=False,
    ):
        self.rank, self.manifest, self.path = rank, manifest, Path(path)
        self.inventory = graph_inventory or inventory
        self.root_only = root_only
        self.mode = manifest["mode"]
        self.private = Path(manifest["private_namespace"])
        self.folder = Path(manifest["worker_receipts"]) / f"rank-{rank}"
        receipt = manifest["qualified_graphs"][self.mode][str(rank)]
        require(
            sha(receipt["path"]) == receipt["sha256"], "qualified graph receipt changed"
        )
        self.qualified = json.loads(Path(receipt["path"]).read_text())
        require(not self.folder.exists(), "preserve prior worker attempt")
        self.folder.mkdir(parents=True)
        self.lock = threading.RLock()
        self.streams = {
            kind: (self.folder / f"{kind}.jsonl").open("x")
            for kind in ("binary-loads", "artifact-roots", "lifecycle", "loader-events")
        }
        self.stack = ExitStack()
        self.store = None
        self.loaded = self.installed = self.closed = False
        self.summary = dict(
            status="created",
            rank=rank,
            mode=self.mode,
            manifest_sha256=sha(path),
            source_sha256=sha(__file__),
            snapshots={},
            static_bundles=[],
            target_sealed=False,
            observer_globally_sealed=False,
        )
        self.save()
        self.loader = (loader_type or GeometryLoader)(
            rank,
            manifest,
            path,
            self.mode,
            emit=lambda row: self.emit(
                "loader-events" if row["event"].startswith("kv_") else "binary-loads",
                row,
            ),
        )
        self.roots = ArtifactRootObserver(
            manifest["artifact_roots"][str(rank)],
            self.private,
            emit=lambda row: self.emit("artifact-roots", row),
        )

    def emit(self, kind, row):
        with self.lock:
            self.streams[kind].write(json.dumps(row, sort_keys=True) + "\n")
            self.streams[kind].flush()

    def save(self):
        (self.folder / "summary.json").write_text(
            json.dumps(self.summary, indent=2) + "\n"
        )

    def raw_modules(self, phase):
        from torch._inductor.codecache import PyCodeCache

        path = self.folder / f"{phase}-modules.json"
        if not path.exists():
            write_new(path, module_inventory(PyCodeCache.modules))
        return path

    def snapshot(self, phase):
        from torch._inductor.codecache import PyCodeCache

        require(phase not in self.summary["snapshots"], "no repeated capture/snapshot")
        modules_path = self.raw_modules(phase)  # Retain evidence before validation.
        roots = self.roots.roots(self.store, PyCodeCache.modules)
        graph = self.inventory(
            roots, self.manifest, self.rank, self.mode, self.loader.observer
        )
        compare_qualified(graph, self.qualified)
        if self.loaded:
            self.loader.controller.verify_graphs(
                roots if self.root_only else PyCodeCache.modules
            )
        path = self.folder / f"{phase}-bindings.json"
        write_new(path, graph)
        self.summary["snapshots"][phase] = dict(
            modules=str(modules_path),
            modules_sha256=sha(modules_path),
            bindings=str(path),
            bindings_sha256=sha(path),
        )
        self.save()
        return graph

    def fail(self, phase, error):
        self.raw_modules(f"failed-{phase}")
        self.summary.update(status="failed", phase=phase, error=repr(error))
        self.save()
        self.emit("lifecycle", dict(event="failed", phase=phase, error=repr(error)))

    def install(self, runner):
        from torch._inductor.codecache import PyCodeCache
        from torch._inductor.triton_bundler import TritonBundler

        from vllm.compilation.caching import StandaloneCompiledArtifacts

        require(
            not self.installed and not self.closed, "geometry lifecycle installed twice"
        )
        original_load = StandaloneCompiledArtifacts.load_all
        original_capture = runner.capture_model
        try:
            require(
                not getattr(original_load, "_glm53_geometry_serving", False),
                "conflicting geometry store hook",
            )
            self.loader.check_cache()
        except BaseException as error:
            self.fail("install", error)
            self.close()
            raise

        @wraps(original_load)
        def load_all(store):
            phase = "aot-load"
            try:
                if self.store is not None:
                    require(
                        self.loaded and store is self.store,
                        "unexpected additional AOT store",
                    )
                    roots = self.roots.roots(store, PyCodeCache.modules)
                    compare_qualified(
                        self.inventory(
                            roots,
                            self.manifest,
                            self.rank,
                            self.mode,
                            self.loader.observer,
                        ),
                        self.qualified,
                    )
                    self.loader.controller.verify_graphs(
                        roots if self.root_only else PyCodeCache.modules
                    )
                    self.emit("lifecycle", dict(event="aot_store_reuse_verified"))
                    return
                self.store = store
                model = self.private / f"rank_{self.rank}_0/model"
                require(
                    sha(model)
                    == self.manifest["original_files"][f"rank_{self.rank}_0/model"],
                    "private cached model changed",
                )
                expected = self.manifest["artifact_roots"][str(self.rank)]
                require(
                    discover(
                        store,
                        self.private,
                        self.manifest["expected_graphs"][str(self.rank)],
                    )
                    == expected,
                    "serving serialized artifact roots changed",
                )
                self.summary.update(
                    status="loading",
                    artifacts=store.num_artifacts(),
                    submodules=store.num_entries(),
                    model_sha256=sha(model),
                )
                self.save()
                self.emit("lifecycle", dict(event="aot_load_begin"))
                with (
                    self.roots.intercept(),
                    checked_bundles(TritonBundler, self.summary["static_bundles"]),
                ):
                    original_load(store)
                require(
                    len(self.summary["static_bundles"]) == len(expected),
                    "serving static bundle coverage incomplete",
                )
                self.snapshot("before-forward")
                self.loader.controller.seal(
                    self.roots.roots(store, PyCodeCache.modules)
                    if self.root_only
                    else PyCodeCache.modules
                )
                self.loaded = True
                self.summary.update(
                    status="aot-qualified",
                    target_sealed=True,
                    loaded_artifacts=len(store.loaded_submodule_store),
                )
                require(
                    not self.loader.observer.sealed,
                    "serving observer sealed prematurely",
                )
                self.save()
                self.emit("lifecycle", dict(event="aot_qualified_before_forward"))
            except BaseException as error:
                self.fail(phase, error)
                self.close()
                raise

        @wraps(original_capture)
        def capture(*args, **kwargs):
            phase = "capture-before"
            try:
                require(
                    self.loaded and self.loader.controller.sealed,
                    "capture before qualified AOT target sealing",
                )
                self.snapshot(phase)
                self.emit("lifecycle", dict(event="capture_begin"))
                result = original_capture(*args, **kwargs)
                phase = "capture-after"
                self.snapshot(phase)
                require(
                    not self.loader.observer.sealed,
                    "serving observer sealed prematurely",
                )
                self.summary["status"] = "capture-qualified"
                self.save()
                self.emit("lifecycle", dict(event="capture_qualified"))
                return result
            except BaseException as error:
                self.fail(phase, error)
                self.close()
                raise

        load_all._glm53_geometry_serving = True

        def restore():
            changed = []
            if StandaloneCompiledArtifacts.load_all is load_all:
                StandaloneCompiledArtifacts.load_all = original_load
            else:
                changed.append("store")
            if runner.capture_model is capture:
                runner.capture_model = original_capture
            else:
                changed.append("capture")
            require(not changed, f"foreign serving geometry hook change: {changed}")

        try:
            self.stack.enter_context(self.loader.intercept())
            StandaloneCompiledArtifacts.load_all = load_all
            runner.capture_model = capture
            self.stack.callback(restore)
            self.installed = True
            self.summary["status"] = "installed"
            self.save()
            self.emit("lifecycle", dict(event="installed"))
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        try:
            self.stack.close()
        finally:
            self.installed, self.closed = False, True
            try:
                close = getattr(self.loader, "close", None)
                if close is not None:
                    close()
            finally:
                for stream in self.streams.values():
                    stream.close()
