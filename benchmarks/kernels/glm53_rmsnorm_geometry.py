# SPDX-License-Identifier: Apache-2.0
"""Multi-source RMSNorm intervention mechanics, not wired into serving.

Reuse the qualified single-source replacement/graph checks behind ONE atomic
resolver. A future must never pass through a chain of upstream cache rechecks.
The new manifest and arm names are separate from the historical legacy test.
GPU source/binary and actual AOT-loader qualification are still required.
"""

import copy
import hashlib
import threading
from pathlib import Path

from slimserve.rmsnorm_diagnostic import (
    Intervention,
    expected_receipt,
    launcher_receipt,
    sha,
)

SCHEMA = "glm53-rmsnorm-geometry-v1"
CONTROL = dict(XBLOCK=1, R0_BLOCK=4096, num_warps=16, num_stages=1)
GEOMETRY = dict(XBLOCK=1, R0_BLOCK=1024, num_warps=8, num_stages=1)


def binary_sha(compiled):
    return hashlib.sha256(compiled.kernel.asm["cubin"]).hexdigest()


class MultiIntervention:
    def __init__(self, rank, manifest, manifest_path, selected_mode):
        if manifest["schema"] != SCHEMA or selected_mode not in ("control", "geometry"):
            raise ValueError(
                "separate geometry manifest and control/geometry arm required"
            )
        if type(rank) is not int or rank not in range(4):
            raise ValueError("invalid geometry rank")
        self.manifest = copy.deepcopy(manifest)
        self.rank = rank
        self.mode = selected_mode
        self.lock = threading.RLock()
        self.sealed = False
        self.owners = {}
        self.controllers = {}
        self.paths = {}
        self.original = Path(manifest["original_namespace"]).resolve()
        self.private = Path(manifest["private_namespace"]).resolve()
        if self.private.is_relative_to(self.original) or self.original.is_relative_to(
            self.private
        ):
            raise ValueError("overlapping geometry caches")
        targets = self.manifest["targets"][str(rank)]
        if not targets:
            raise ValueError("empty geometry target set")
        # Validate the entire rank before opening any receipt stream.
        for target in targets:
            relative = Path(target["relative"])
            if (
                relative.is_absolute()
                or relative.parts
                != ("inductor_cache", relative.parent.name, relative.name)
                or ".." in relative.parts
                or relative.suffix != ".py"
                or relative.name in self.paths
            ):
                raise ValueError("invalid or duplicate geometry source")
            paths = tuple(root / relative for root in (self.original, self.private))
            if any(
                path.resolve() != path or sha(path) != target["source_sha256"]
                for path in paths
            ):
                raise ValueError("geometry source changed or aliased")
            if set(target["configs"]) != {"control", "geometry"}:
                raise ValueError("both geometry configurations required")
            for arm, config in (("control", CONTROL), ("geometry", GEOMETRY)):
                saved = target["configs"][arm]
                if {k: saved[k] for k in config} != config:
                    raise ValueError("unqualified geometry configuration")
                for key in ("triton_cache_hash", "cubin_sha256"):
                    if not isinstance(saved[key], str) or not saved[key]:
                        raise ValueError("qualified binary receipts required")
                digest = saved["cubin_sha256"]
                if len(digest) != 64 or any(
                    c not in "0123456789abcdef" for c in digest
                ):
                    raise ValueError("whole cubin SHA256 required")
            self.paths[relative.name] = paths
        folder = Path(manifest["receipts"])
        folder.mkdir(exist_ok=True)
        for index, target in enumerate(targets):
            name = Path(target["relative"]).name
            adapted = dict(
                target,
                filename=name,
                # The reused controller selects entry1 for control, entry0
                # otherwise. Its receipt still says "geometry", never "legacy".
                configs=[target["configs"]["geometry"], target["configs"]["control"]],
            )
            child_manifest = dict(
                manifest,
                receipts=str(folder / f"target-{index}"),
                targets={str(rank): adapted},
            )
            self.controllers[name] = Intervention(
                rank, child_manifest, manifest_path, selected_mode
            )
            self.controllers[name].emit(
                dict(
                    event="geometry_controller",
                    schema=SCHEMA,
                    source_sha256=sha(__file__),
                )
            )

    def owner(self, autotuner):
        filename = getattr(autotuner, "filename", None)
        path = Path(filename) if isinstance(filename, str) else None
        controller = self.controllers.get(path.name) if path is not None else None
        known = self.owners.get(id(autotuner))
        if known is not None and (
            known[0] is not autotuner
            or known[1] is not controller
            or path != self.paths[controller.target["filename"]][1]
        ):
            raise ValueError("geometry binding identity/source changed")
        if controller is not None and (
            path not in self.paths[path.name] or path.resolve() != path
        ):
            raise ValueError("geometry source outside exact cache paths")
        return controller

    def relocate(self, autotuner):
        # All static sources, not only targets, can retain old save-cache paths.
        next(iter(self.controllers.values())).relocate(autotuner)

    def replace(self, autotuner, compile_replacement, resolved_by="direct"):
        with self.lock:
            controller = self.owner(autotuner)
            if controller is None:
                return autotuner
            self.relocate(autotuner)
            before = launcher_receipt(autotuner)
            allowed = [
                c for c in controller.target["configs"] if expected_receipt(c) == before
            ]
            compiled = autotuner.compile_results
            if (
                len(allowed) != 1
                or not compiled
                or len(compiled) != 1
                or binary_sha(compiled[0]) != allowed[0]["cubin_sha256"]
            ):
                raise ValueError("geometry bound cubin/config differs")

            def checked_compile(saved):
                result = compile_replacement(controller.target, saved)
                if binary_sha(result) != saved["cubin_sha256"]:
                    raise ValueError("geometry replacement cubin differs")
                return result

            cached = controller.compiled_replacement
            selected = controller.target["configs"][int(self.mode == "control")]
            if cached is not None and binary_sha(cached) != selected["cubin_sha256"]:
                raise ValueError("geometry cached replacement cubin changed")
            controller.replace(autotuner, checked_compile, resolved_by)
            self.owners[id(autotuner)] = (autotuner, controller)
            return autotuner

    def resolve(self, future, original_result, compile_replacement, timeout=None):
        with self.lock:
            autotuner = future.static_autotuner
            controller = self.owner(autotuner)
            if (
                self.sealed
                and controller is not None
                and id(autotuner) not in self.owners
            ):
                raise ValueError("new geometry target binding after seal")
            self.relocate(autotuner)
            if id(autotuner) in self.owners:
                return self.replace(autotuner, compile_replacement, "reuse")
            # Exactly one upstream call, regardless of this rank's target count.
            resolved = original_result(future, timeout=timeout)
            if resolved is not autotuner:
                raise ValueError("static future changed autotuner identity")
            return self.replace(resolved, compile_replacement, "upstream")

    def bind_graph(self, module, compile_replacement):
        if not callable(getattr(module, "call", None)):
            return
        with self.lock:
            for name, autotuner in list(vars(module).items()):
                controller = self.owner(autotuner)
                if controller is None:
                    continue
                key = (id(module), name)
                if self.sealed and key not in controller.graph_bindings:
                    raise ValueError("new geometry graph binding after seal")
                self.relocate(autotuner)
                self.replace(autotuner, compile_replacement, "graph")
                controller.graph_bindings[key] = (module, autotuner)
                controller.emit(
                    dict(
                        event="graph_binding",
                        rank=self.rank,
                        symbol=name,
                        module=module.__file__,
                        module_sha256=sha(module.__file__),
                        filename=autotuner.filename,
                        binding_index=list(controller.replaced).index(id(autotuner))
                        + 1,
                        selected=launcher_receipt(autotuner),
                    )
                )

    def verify_graphs(self, modules):
        modules = tuple(modules)
        with self.lock:
            for controller in self.controllers.values():
                for module, autotuner in controller.graph_bindings.values():
                    if not any(module is other for other in modules):
                        raise ValueError("geometry graph missing from loader inventory")
                    self.owner(autotuner)
                    (compiled,) = autotuner.compile_results
                    selected = controller.target["configs"][int(self.mode == "control")]
                    if binary_sha(compiled) != selected["cubin_sha256"]:
                        raise ValueError("geometry graph cubin changed")
                controller.verify_graphs(modules)

    def seal(self, modules):
        """No partial seals: every prescribed source needs actual graph coverage."""
        with self.lock:
            self.verify_graphs(modules)
            for controller in self.controllers.values():
                controller.seal()
            self.sealed = True
