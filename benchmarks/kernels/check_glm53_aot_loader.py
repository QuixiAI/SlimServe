#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Load real GLM53 cached submodules without model weights or forward calls.

Only use the trusted, locally produced source-bound intervention manifest.
Pickle is executable; this is not a tool for inspecting untrusted artifacts.
"""

import argparse
import io
import json
import os
import pickle
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

from slimserve import rmsnorm_diagnostic as diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=range(4), required=True)
    parser.add_argument(
        "--mode", choices=("observe", "control", "legacy"), required=True
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        parser.error("GPU workload already active")
    args.output.mkdir(parents=True)
    folder = args.output.resolve()
    manifest = json.loads(args.manifest.read_text())
    # Separate receipts for each process; never modify the supplied manifest.
    manifest["receipts"] = str(folder / "receipts")
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    os.environ[diagnostic.MANIFEST] = str(path)
    os.environ["VLLM_CACHE_ROOT"] = manifest["cache_root"]
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(
        Path(manifest["private_namespace"]) / "inductor_cache"
    )
    # Match forced-AOT serving, including its per-device Triton subdirectories.
    os.environ.pop("TRITON_CACHE_DIR", None)
    os.environ["SLIMSERVE_GLM53_NATIVE_ORDER"] = "1"
    os.environ["VLLM_FORCE_AOT_LOAD"] = "1"
    os.environ["VLLM_GLM5_MHC_PREFILL_TC"] = "0"
    os.environ["VLLM_GLM5_MHC_BF16_FN"] = "1"
    os.environ[diagnostic.FLAG] = "control" if args.mode == "observe" else args.mode
    diagnostic.validate_environment()
    diagnostic.read_manifest()
    import torch
    from torch._dynamo.aot_compile import AOTCompileUnpickler
    from torch._inductor.codecache import PyCodeCache, StaticAutotunerFuture
    from torch._inductor.triton_bundler import TritonBundler

    import vllm._custom_ops  # noqa: F401

    torch.cuda.set_device(args.rank)
    model = Path(manifest["private_namespace"]) / f"rank_{args.rank}_0/model"
    expected = manifest["original_files"][f"rank_{args.rank}_0/model"]
    if diagnostic.sha(model) != expected:
        raise ValueError("cached model differs from original snapshot")
    summary = dict(
        status="running",
        rank=args.rank,
        mode=args.mode,
        model_sha256=expected,
        source_sha256=diagnostic.sha(__file__),
        intervention_sha256=diagnostic.sha(diagnostic.__file__),
        model_forward_calls=0,
        weight_tensors_loaded=0,
    )
    output = folder / "summary.json"

    def save():
        output.write_text(json.dumps(summary, indent=2) + "\n")

    save()
    original_result = StaticAutotunerFuture.result
    original_py_load = PyCodeCache.__dict__["load_by_key_path"]
    original_load = TritonBundler.__dict__["load_autotuners"]
    bundle_counts = []
    bundle_lock = threading.RLock()

    def checked_load(cls, tuners):
        loaded = original_load.__func__(cls, tuners)
        expected = len(tuners or [])
        with bundle_lock:
            bundle_counts.append(dict(expected=expected, loaded=len(loaded)))
        if len(loaded) != expected:
            raise ValueError("static bundle fallback is not loader qualification")
        return loaded

    TritonBundler.load_autotuners = classmethod(checked_load)
    observer_stream = None
    try:
        if args.mode == "observe":
            observer = diagnostic.Intervention(args.rank, manifest, path, "observe")
            observer_stream = observer.stream
            objects, futures, counts = {}, {}, {}
            lock = threading.RLock()

            def observe(future, timeout=None):
                tuner = future.static_autotuner
                with lock:
                    objects.setdefault(id(tuner), tuner)  # strong identity, no reuse
                    futures.setdefault(id(future), future)
                    index = list(objects).index(id(tuner)) + 1
                    future_index = list(futures).index(id(future)) + 1
                    counts[index] = counts.get(index, 0) + 1
                    occurrence = counts[index]
                    observer.relocate(tuner)
                    observer.emit(
                        dict(
                            event="resolve_enter",
                            object_index=index,
                            future_index=future_index,
                            occurrence=occurrence,
                            thread=threading.get_ident(),
                            filename=tuner.filename,
                        )
                    )
                # Preserve the real loader's concurrency for the observation.
                result = original_result(future, timeout=timeout)
                with lock:
                    observer.emit(
                        dict(
                            event="resolve_exit",
                            object_index=index,
                            future_index=future_index,
                            occurrence=occurrence,
                            launchers=diagnostic.launcher_receipt(result),
                        )
                    )
                return result

            StaticAutotunerFuture.result = observe
        else:
            runner = SimpleNamespace(
                model_config=SimpleNamespace(
                    hf_text_config=SimpleNamespace(
                        hidden_size=4096, num_hidden_layers=45
                    )
                ),
                parallel_config=SimpleNamespace(
                    rank=args.rank,
                    tensor_parallel_size=4,
                    pipeline_parallel_size=1,
                    enable_expert_parallel=False,
                ),
                speculative_config=None,
                capture_model=lambda: None,
            )
            diagnostic.install(runner)
        # Stop short of the outer model deserializer: its seven actual cached
        # submodules exercise the same concurrent load_all path independently.
        outer = AOTCompileUnpickler({}, io.BytesIO(model.read_bytes())).load()
        inner = pickle.loads(outer["compiled_fn"][1])
        store = inner["standalone_compile_artifacts"]
        summary["artifacts"] = store.num_artifacts()
        summary["submodules"] = store.num_entries()
        with torch._functorch.config.patch(inner["aot_autograd_config"]):
            store.load_all()
        summary["loaded_artifacts"] = len(store.loaded_submodule_store)
        if args.mode == "observe":
            summary.update(
                objects=len(objects),
                futures=len(futures),
                resolutions=sum(counts.values()),
                repeated_objects={k: v for k, v in counts.items() if v > 1},
            )
        else:
            runner.capture_model()
        summary["status"] = "complete"
    except BaseException as error:
        summary.update(status="failed", error=repr(error))
        raise
    finally:
        StaticAutotunerFuture.result = original_result
        PyCodeCache.load_by_key_path = original_py_load
        TritonBundler.load_autotuners = original_load
        summary["static_bundles"] = bundle_counts
        if observer_stream is not None:
            observer_stream.close()
        original = Path(manifest["original_namespace"])
        snapshot = manifest["original_files"]
        unchanged = {
            str(p.relative_to(original)) for p in original.rglob("*") if p.is_file()
        } == set(snapshot) and all(
            diagnostic.sha(original / p) == h for p, h in snapshot.items()
        )
        summary["original_cache_unchanged"] = unchanged
        if not unchanged:
            summary.update(status="failed", cache_error="original cache changed")
        save()
        print(json.dumps(summary, indent=2), flush=True)
    if summary["status"] != "complete":
        raise RuntimeError("loader qualification failed")


if __name__ == "__main__":
    main()
