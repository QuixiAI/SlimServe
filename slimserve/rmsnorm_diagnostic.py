# SPDX-License-Identifier: Apache-2.0
"""Startup-only, source-bound RMSNorm intervention. Never a production policy.

The control reconstructs the recorded native-order launchers without changing
their binary hashes. The legacy arm substitutes only the four identified legacy
configurations. Both require cached AOT loading and private compilation caches.
"""

import hashlib
import importlib.util
import json
import os
import sys
import threading
from functools import wraps
from pathlib import Path

from slimserve import glm53_ordering

FLAG = "SLIMSERVE_GLM53_RMSNORM_DIAGNOSTIC"
MANIFEST = "SLIMSERVE_GLM53_RMSNORM_MANIFEST"
AUDIT_SHA = "249d26192e4a2ecc509f437e3560f982aa0b8eb82e2c53954c7212077bc636fb"
NAMESPACE = "c8b11c6e1ba1746d15afa3aa68486eb07d2a49251573b93de0cbddf72ad47a8f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mode():
    value = os.getenv(FLAG, "")
    if value not in ("", "control", "legacy"):
        raise ValueError(f"{FLAG} must be absent, control or legacy")
    return value


def validate_plan(plan):
    if not mode():
        return
    validate_environment()
    glm53_ordering.validate_plan(plan)
    read_manifest()


def validate_environment():
    if not glm53_ordering.enabled():
        raise ValueError("RMSNorm diagnostic requires native ordering")
    for suffix in ("MODEL_JOURNAL", "MOE_JOURNAL", "INDEX_JOURNAL", "SCORE_JOURNAL"):
        if os.getenv("SLIMSERVE_GLM53_" + suffix, "0") not in ("", "0"):
            raise ValueError("RMSNorm diagnostic forbids tensor journals")
    if (
        os.getenv("CUDA_LAUNCH_BLOCKING", "0") != "0"
        or os.getenv("VLLM_FORCE_AOT_LOAD") != "1"
        or os.getenv("VLLM_GLM5_MHC_PREFILL_TC", "0") != "0"
        or os.getenv("VLLM_GLM5_MHC_BF16_FN", "1") != "1"
    ):
        raise ValueError("RMSNorm diagnostic requires async, forced AOT, TC0, BF16fn1")


def read_manifest():
    if not os.environ.get(MANIFEST) or not os.environ.get("VLLM_CACHE_ROOT"):
        raise ValueError("RMSNorm diagnostic requires manifest and private cache paths")
    path = Path(os.environ[MANIFEST]).resolve()
    data = json.loads(path.read_text())
    if (
        data["schema"] != 1
        or data["audit_sha256"] != AUDIT_SHA
        or data["namespace"] != NAMESPACE
        or set(data["targets"]) != {str(i) for i in range(4)}
    ):
        raise ValueError("unexpected source-bound RMSNorm manifest")
    private = Path(data["cache_root"]).resolve()
    original = Path(data["original_namespace"]).resolve()
    if (
        private == original
        or original in private.parents
        or private in original.parents
    ):
        raise ValueError("private cache must not overlap original namespace")
    if Path(os.environ["VLLM_CACHE_ROOT"]).resolve() != private:
        raise ValueError("RMSNorm diagnostic requires its private VLLM cache")
    return data, path


def single_launcher_receipt(launcher):
    return dict(
        hash=launcher.cache_hash,
        config={
            **launcher.config.kwargs,
            "num_warps": launcher.config.num_warps,
            "num_stages": launcher.config.num_stages,
        },
    )


def launcher_receipt(autotuner):
    return [single_launcher_receipt(launcher) for launcher in autotuner.launchers]


def expected_receipt(saved):
    return [
        dict(
            hash=saved["triton_cache_hash"],
            config={
                k: saved[k] for k in ("XBLOCK", "R0_BLOCK", "num_warps", "num_stages")
            },
        )
    ]


class Intervention:
    def __init__(self, rank, manifest, manifest_path, selected_mode):
        self.rank = rank
        self.manifest = manifest
        self.mode = selected_mode
        self.target = manifest["targets"][str(rank)]
        self.lock = threading.RLock()
        # Keep the objects alive: bare id() values can be recycled by Python.
        self.replaced = {}
        self.resolutions = 0
        self.compiled_replacement = None
        self.sealed = False
        folder = Path(manifest["receipts"])
        folder.mkdir(exist_ok=True)
        self.path = folder / f"rank-{rank}.jsonl"
        self.stream = self.path.open("x")
        self.emit(
            dict(
                event="begin",
                rank=rank,
                mode=selected_mode,
                pid=os.getpid(),
                manifest_sha256=sha(manifest_path),
                source_sha256=sha(__file__),
            )
        )

    def emit(self, record):
        self.stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.stream.flush()

    def relocate(self, autotuner):
        """Static artifacts contain old filenames; avoid old cache writes."""
        if autotuner.filename is None:
            return
        path = Path(autotuner.filename)
        old = Path(self.manifest["original_namespace"])
        if path.is_relative_to(old):
            new = Path(self.manifest["private_namespace"]) / path.relative_to(old)
            if sha(path) != sha(new):
                raise ValueError("relocated generated source differs")
            autotuner.filename = str(new)
            # A serialized save hook can also retain the original local-cache
            # key. This diagnostic reads fixed choices and never persists tuning.
            autotuner.save_cache_hook = None

    def resolve(self, future, original_result, compile_replacement, timeout=None):
        """Resolve and transform atomically across concurrent AOT submodules.

        A cached future may be shared by concurrent source imports. Once its
        object has been selected, don't let another cache recheck undo that
        choice. Revalidate it and return the same object on subsequent calls.
        """
        with self.lock:
            autotuner = future.static_autotuner
            self.relocate(autotuner)
            if id(autotuner) in self.replaced:
                return self.replace(autotuner, compile_replacement, "reuse")
            autotuner = original_result(future, timeout=timeout)
            return self.replace(autotuner, compile_replacement, "upstream")

    def replace(self, autotuner, compile_replacement, resolved_by="direct"):
        """Hash-check every resolution; apply the selected config idempotently."""
        with self.lock:
            filename = Path(autotuner.filename or "")
            before = launcher_receipt(autotuner)
            target = filename.name == self.target["filename"]
            record = dict(
                event="launcher",
                rank=self.rank,
                filename=str(filename),
                target=target,
                before=before,
                resolved_by=resolved_by,
                sealed=self.sealed,
            )
            known = self.replaced.get(id(autotuner))
            if known is not None and (known is not autotuner or not target):
                raise ValueError("RMSNorm binding identity/source changed")
            if target:
                if self.sealed and known is None:
                    raise ValueError("new RMSNorm target binding after seal")
                if sha(filename) != self.target["source_sha256"]:
                    raise ValueError("RMSNorm target source changed")
                native = self.target["configs"][1]
                saved = self.target["configs"][int(self.mode == "control")]
                allowed = [expected_receipt(saved)] if known is not None else []
                if not self.sealed:
                    allowed.append(expected_receipt(native))
                if before not in allowed:
                    raise ValueError("RMSNorm native cached launcher does not match")
                if not self.sealed:
                    compiled = self.compiled_replacement or compile_replacement(saved)
                    launcher = compiled.make_launcher()
                    if [single_launcher_receipt(launcher)] != expected_receipt(saved):
                        raise ValueError(
                            "RMSNorm replacement binary/config does not match"
                        )
                    self.compiled_replacement = compiled
                    autotuner.compile_results = [compiled]
                    autotuner.launchers = [launcher]
                    autotuner.configs = None
                    autotuner._cached_launcher = None
                    autotuner.save_cache_hook = None
                # After seal, known exact-selected objects are read-only. New
                # objects or changed binaries fail before touching captured work.
                self.replaced[id(autotuner)] = autotuner
                self.resolutions += 1
                record.update(
                    binding_index=list(self.replaced).index(id(autotuner)) + 1,
                    resolution_index=self.resolutions,
                    repeated=known is not None,
                )
            record["after"] = launcher_receipt(autotuner)
            if not target and record["before"] != record["after"]:
                raise ValueError("unrelated launcher changed")
            self.emit(record)
            return autotuner

    def seal(self):
        with self.lock:
            # Separate AOT submodules can own distinct autotuner objects for
            # the same generated source. Every occurrence is hash-checked and
            # replaced above; source coverage is not object uniqueness.
            if not self.replaced:
                raise ValueError(
                    f"missing source-bound RMSNorm binding on rank{self.rank}"
                )
            self.sealed = True
            self.emit(
                dict(
                    event="sealed",
                    rank=self.rank,
                    targets=len(self.replaced),
                    resolutions=self.resolutions,
                    sources=1,
                )
            )


def install(runner):
    selected_mode = mode()
    if not selected_mode:
        return
    validate_environment()
    import torch
    import triton
    from torch._inductor.codecache import StaticAutotunerFuture

    from benchmarks.kernels.check_glm53_cached_rmsnorm import checked_config

    config = runner.model_config.hf_text_config
    parallel = runner.parallel_config
    if (
        config.hidden_size != 4096
        or config.num_hidden_layers != 45
        or parallel.tensor_parallel_size != 4
        or parallel.pipeline_parallel_size != 1
        or parallel.enable_expert_parallel
        or runner.speculative_config is not None
        or torch.cuda.get_device_capability() != (12, 0)
        or not glm53_ordering.enabled()
        or os.getenv("VLLM_FORCE_AOT_LOAD") != "1"
    ):
        raise ValueError(
            "RMSNorm diagnostic requires cached native-order GLM53 SM120 TP4"
        )
    manifest, path = read_manifest()
    diagnostic = Intervention(parallel.rank, manifest, path, selected_mode)
    original_result = StaticAutotunerFuture.result
    if getattr(original_result, "_glm53_rmsnorm_diagnostic", False):
        raise ValueError("RMSNorm diagnostic installed twice")

    def compile_replacement(saved):
        source = Path(diagnostic.target["injection_source"])
        if sha(source) != diagnostic.target["source_sha256"]:
            raise ValueError("RMSNorm copied source changed")
        name = f"glm53_rmsnorm_intervention_rank{parallel.rank}"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        template = getattr(module, diagnostic.target["kernel"])
        cfg = checked_config(saved)
        with torch.cuda.device(parallel.rank):
            return template._precompile_config(
                triton.Config(
                    {k: cfg[k] for k in ("XBLOCK", "R0_BLOCK")},
                    num_warps=cfg["num_warps"],
                    num_stages=cfg["num_stages"],
                )
            )

    @wraps(original_result)
    def result(future, timeout=None):
        return diagnostic.resolve(future, original_result, compile_replacement, timeout)

    result._glm53_rmsnorm_diagnostic = True
    StaticAutotunerFuture.result = result
    original_capture = runner.capture_model

    @wraps(original_capture)
    def capture(*args, **kwargs):
        diagnostic.seal()
        return original_capture(*args, **kwargs)

    runner.capture_model = capture
    runner._slimserve_rmsnorm_diagnostic = diagnostic
