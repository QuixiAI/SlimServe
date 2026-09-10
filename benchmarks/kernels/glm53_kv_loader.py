# SPDX-License-Identifier: Apache-2.0
"""KV-only static graph diagnostic. Never enabled by a production profile.

Both arms bind the instance run entry directly, bypassing Inductor dispatch.
Original compile_results/launchers remain truthful: they describe the combo,
not the extra launch. Audit that extra launch separately from the live adapter.
"""

import copy
import threading
from pathlib import Path

from benchmarks.analyze_glm53_quality_pair import require
from benchmarks.kernels.check_glm53_attention_norms import load_preserving_provenance
from benchmarks.kernels.glm53_attention_overwrite import KVOnlyOverwrite
from benchmarks.kernels.glm53_binary_observer import StaticCudaBinaryObserver
from benchmarks.kernels.glm53_loader_hooks import ScopedKernelLoader
from slimserve.rmsnorm_diagnostic import Intervention, sha, single_launcher_receipt

SCHEMA = "glm53-kv-loader-v1"


def check_source(record, private):
    original, copied = Path(record["source"]), private / record["relative"]
    debug = Path(record["debug_source"])
    require(
        copied.is_relative_to(private)
        and all(p.is_absolute() and p.resolve() == p for p in (original, copied, debug))
        and original != copied
        and sha(original) == sha(copied) == record["source_sha256"]
        and sha(debug) == record["debug_source_sha256"],
        "KV diagnostic source/debug receipt changed or aliased",
    )


def check_launcher(compiled, launcher, saved, observer):
    require(
        compiled.kernel.name == saved["kernel"]
        and single_launcher_receipt(launcher) == saved["selected"]
        and observer.digest(compiled) == saved["cubin_sha256"],
        "KV diagnostic launcher config/binary differs",
    )
    observer.verify_launcher(compiled, launcher)


class KVIntervention:
    def __init__(self, rank, manifest, mode, observer, *, emit=None):
        require(
            manifest["schema"] == SCHEMA
            and mode in ("control", "kv")
            and type(rank) is int
            and rank in range(4),
            "separate KV manifest, valid rank and control/kv arm required",
        )
        self.manifest = copy.deepcopy(manifest)
        self.rank, self.mode, self.observer = rank, mode, observer
        self.emit = emit or (lambda record: None)
        self.original = Path(manifest["original_namespace"])
        self.private = Path(manifest["private_namespace"])
        require(
            all(
                p.is_absolute() and p.resolve() == p
                for p in (self.original, self.private)
            )
            and not self.private.is_relative_to(self.original)
            and not self.original.is_relative_to(self.private),
            "overlapping or aliased KV caches",
        )
        self.targets = {}
        for target in self.manifest["targets"][str(rank)]:
            relative = Path(target["relative"])
            require(
                len(relative.parts) == 3
                and relative.parts[0] == "inductor_cache"
                and ".." not in relative.parts
                and relative.suffix == ".py"
                and relative.name not in self.targets
                and self.original / relative == Path(target["source"]),
                "invalid or duplicate KV target source",
            )
            check_source(target, self.private)
            check_source(target["kv"], self.private)
            self.targets[relative.name] = target
        require(self.targets, "empty KV target set")
        self.lock = threading.RLock()
        self.owners, self.graph_bindings, self.appended = {}, {}, {}
        self.sealed = False

    def target(self, tuner):
        filename = getattr(tuner, "filename", None)
        path = Path(filename) if isinstance(filename, str) else None
        target = self.targets.get(path.name) if path is not None else None
        known = self.owners.get(id(tuner))
        if known is not None:
            require(
                known["tuner"] is tuner
                and known["target"] is target
                and path == self.private / target["relative"],
                "KV binding identity/source changed",
            )
        if target is not None:
            require(
                path
                in (
                    self.original / target["relative"],
                    self.private / target["relative"],
                )
                and path.resolve() == path,
                "KV target outside exact cache paths",
            )
        return target

    def verify(self, binding):
        tuner, target = binding["tuner"], binding["target"]
        self.target(tuner)
        check_source(target, self.private)
        check_source(target["kv"], self.private)
        require(
            len(tuner.compile_results) == len(tuner.launchers) == 1
            and tuner.compile_results[0] is binding["compiled"]
            and tuner.launchers[0] is binding["launcher"]
            and vars(tuner).get("run") is binding["run"]
            and getattr(tuner, "_cached_launcher", None) is None
            and tuner.save_cache_hook is None,
            "KV direct dispatch or original binding changed",
        )
        check_launcher(binding["compiled"], binding["launcher"], target, self.observer)
        if self.mode == "kv":
            compiled, launcher = binding["appended"]
            adapter = binding["adapter"]
            require(
                type(adapter) is KVOnlyOverwrite
                and adapter.combo is binding["launcher"]
                and adapter.split_kv is launcher
                and binding["run"].__self__ is adapter
                and binding["run"].__func__ is KVOnlyOverwrite.run,
                "KV adapter launch sequence changed",
            )
            check_launcher(compiled, launcher, target["kv"], self.observer)

    def replace(self, tuner, compile_replacement, resolved_by="direct"):
        with self.lock:
            target = self.target(tuner)
            if target is None:
                return tuner
            known = self.owners.get(id(tuner))
            if known is not None:
                self.verify(known)
                return tuner
            require(not self.sealed, "new KV target binding after seal")
            check_source(target, self.private)
            check_source(target["kv"], self.private)
            Intervention.relocate(self, tuner)
            require(
                len(tuner.compile_results) == len(tuner.launchers) == 1
                and "run" not in vars(tuner),
                "one original launcher and no foreign run override required: "
                f"results={len(tuner.compile_results)}, "
                f"launchers={len(tuner.launchers)}, "
                f"run_override={'run' in vars(tuner)}",
            )
            compiled, launcher = tuner.compile_results[0], tuner.launchers[0]
            check_launcher(compiled, launcher, target, self.observer)
            binding = dict(
                tuner=tuner, target=target, compiled=compiled, launcher=launcher
            )
            run = launcher
            if self.mode == "kv":
                pair = self.appended.get(target["relative"])
                if pair is None:
                    extra = compile_replacement(target["kv"])
                    require(
                        self.observer.digest(extra) == target["kv"]["cubin_sha256"],
                        "KV appended cubin differs before load",
                    )
                    pair = extra, extra.make_launcher()
                    check_launcher(*pair, target["kv"], self.observer)
                    self.appended[target["relative"]] = pair
                adapter = KVOnlyOverwrite(launcher, pair[1])
                binding.update(appended=pair, adapter=adapter)
                run = adapter.run
            tuner._cached_launcher = None
            tuner.save_cache_hook = None
            tuner.run = binding["run"] = run
            self.owners[id(tuner)] = binding
            self.verify(binding)
            self.emit(
                dict(
                    event="kv_binding",
                    rank=self.rank,
                    mode=self.mode,
                    source=target["relative"],
                    resolved_by=resolved_by,
                    binding_index=list(self.owners).index(id(tuner)) + 1,
                )
            )
            return tuner

    def resolve(self, future, original_result, compile_replacement, timeout=None):
        with self.lock:
            tuner = future.static_autotuner
            target = self.target(tuner)
            if id(tuner) in self.owners:
                return self.replace(tuner, compile_replacement, "reuse")
            require(
                not (self.sealed and target is not None), "new KV target after seal"
            )
            Intervention.relocate(self, tuner)
            resolved = original_result(future, timeout=timeout)
            require(resolved is tuner, "static future changed autotuner identity")
            return self.replace(tuner, compile_replacement, "upstream")

    def bind_graph(self, module, compile_replacement):
        if not callable(getattr(module, "call", None)):
            return
        with self.lock:
            bound = [
                (symbol, tuner, self.target(tuner))
                for symbol, tuner in list(vars(module).items())
                if self.target(tuner) is not None
            ]
            if not bound:
                return
            filename = getattr(module, "__file__", None)
            require(isinstance(filename, str), "target module lacks source provenance")
            path = Path(filename)
            require(
                path.resolve() == path and path.is_relative_to(self.private),
                "target module outside private cache",
            )
            relative = str(path.relative_to(self.private))
            expected = self.manifest["expected_graphs"][str(self.rank)]
            if relative not in expected:
                # AsyncCompile's synchronous path imports the standalone source
                # BEFORE precompile(). Its benchmark call export is not an AOT
                # root. Leave that template alone; bind its finished root later.
                require(len(bound) == 1, "unexpected non-root target module")
                symbol, _, target = bound[0]
                require(
                    relative == target["relative"] and symbol == target["kernel"],
                    "unexpected non-root target module",
                )
                check_source(target, self.private)
                return
            require(sha(path) == expected[relative], "target graph source changed")
            for symbol, tuner, _ in bound:
                key = id(module), symbol
                previous = self.graph_bindings.get(key)
                require(
                    not self.sealed or previous is not None, "new KV graph after seal"
                )
                require(
                    previous is None
                    or (previous[0] is module and previous[1] is tuner),
                    "KV graph rebound",
                )
                self.replace(tuner, compile_replacement, "graph")
                self.graph_bindings[key] = module, tuner
                self.emit(
                    dict(
                        event="kv_graph_binding",
                        rank=self.rank,
                        graph=getattr(module, "__file__", None),
                        symbol=symbol,
                        source=self.owners[id(tuner)]["target"]["relative"],
                        binding_index=list(self.owners).index(id(tuner)) + 1,
                    )
                )

    def verify_graphs(self, modules):
        from benchmarks.kernels.audit_glm53_kv_graphs import inventory

        with self.lock:
            modules = tuple(modules)
            for (module_id, symbol), (module, tuner) in self.graph_bindings.items():
                require(
                    module_id == id(module)
                    and any(module is m for m in modules)
                    and vars(module).get(symbol) is tuner,
                    "KV graph missing or rebound",
                )
            require(
                {id(t) for _, t in self.graph_bindings.values()} == set(self.owners),
                "KV target lacks actual graph coverage",
            )
            for binding in self.owners.values():
                self.verify(binding)
            return inventory(
                modules, self.manifest, self.rank, self.mode, self.observer
            )

    def seal(self, modules):
        with self.lock:
            report = self.verify_graphs(modules)
            if not self.sealed:
                self.emit(
                    dict(
                        event="kv_sealed",
                        rank=self.rank,
                        targets=len(self.owners),
                        graph_bindings=len(self.graph_bindings),
                    )
                )
            self.sealed = True
            return report


class KVLoader(ScopedKernelLoader):
    hook_marker = "_glm53_kv_loader"

    def __init__(self, rank, manifest, manifest_path, mode, *, emit=None):
        self.rank, self.manifest_path = rank, Path(manifest_path)
        self.private = Path(manifest["private_namespace"])
        self.cache = self.private / "inductor_cache"
        images = {}
        for target in manifest["targets"][str(rank)]:
            for record in (target, target["kv"]):
                images.setdefault(
                    (record["selected"]["hash"], record["kernel"]), set()
                ).add(record["cubin_sha256"])
        self.observer = StaticCudaBinaryObserver(
            rank, [self.cache / "triton" / str(rank)], expected_images=images, emit=emit
        )
        self.controller = KVIntervention(rank, manifest, mode, self.observer, emit=emit)
        self.installed, self.templates = False, {}
        self.controller.emit(
            dict(
                event="kv_begin",
                rank=rank,
                mode=mode,
                manifest_sha256=sha(self.manifest_path),
                source_sha256=sha(__file__),
            )
        )

    def compile_replacement(self, record):
        self.check_cache()
        check_source(record, self.private)
        import torch
        import triton

        template = self.templates.get(record["relative"])
        if template is None:
            template = load_preserving_provenance(
                self.private / record["relative"],
                Path(record["source"]),
                record["kernel"],
                f"glm53_kv_{sha(self.manifest_path)}_rank{self.rank}",
                debug_source=Path(record["debug_source"]),
            )
            self.templates[record["relative"]] = template
        self.check_cache()
        saved = record["selected"]["config"]
        with torch.cuda.device(self.rank):
            result = template._precompile_config(
                triton.Config(
                    {
                        k: v
                        for k, v in saved.items()
                        if k not in ("num_warps", "num_stages")
                    },
                    num_warps=saved["num_warps"],
                    num_stages=saved["num_stages"],
                )
            )
        self.check_cache()
        return result
