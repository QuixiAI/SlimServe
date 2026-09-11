# SPDX-License-Identifier: Apache-2.0
"""Read-only startup graph receipts for the opt-in deterministic compiler candidate."""

import hashlib
import json
import os
from functools import wraps
from pathlib import Path

from slimserve import glm53_ordering


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_snapshot(modules, cache_root):
    """Inspect actual globals, without resolving futures or touching autotuners.

    Keep module/object references alive throughout enumeration. Bare IDs are
    useful only inside this process; never compare them across independent runs.
    Missing/uninspectable named RMSNorm bindings are violations, not exclusions.
    """
    root = str(Path(cache_root).resolve())
    files, graphs, bindings, violations, references = {}, [], [], [], []

    def file_receipt(filename):
        if not isinstance(filename, str) or not Path(filename).is_file():
            raise ValueError(f"missing generated source: {filename}")
        if filename not in files:
            source = Path(filename).read_bytes()
            files[filename] = dict(
                sha256=hashlib.sha256(source).hexdigest(),
                normalized_sha256=hashlib.sha256(
                    source.replace(root.encode(), b"<CACHE>")
                ).hexdigest(),
            )
        return files[filename]

    seen = set()
    for module in list(modules):
        if id(module) in seen or not callable(getattr(module, "call", None)):
            continue
        seen.add(id(module))
        references.append(module)
        graph = dict(module_id=id(module), module=module.__file__)
        try:
            graph.update(file_receipt(module.__file__))
        except ValueError as error:
            violations.append(str(error))
        graphs.append(graph)
        for symbol, value in list(vars(module).items()):
            state = getattr(value, "__dict__", {})
            metadata = state.get("inductor_meta")
            named_norm = symbol.startswith("triton_") and "rms_norm" in symbol
            if not isinstance(metadata, dict):
                if named_norm:
                    violations.append(f"uninspectable RMSNorm global: {symbol}")
                continue
            references.append(value)
            kernel = metadata.get("kernel_name", symbol)
            norm = named_norm or "rms_norm" in kernel
            reduction = norm or bool(metadata.get("num_reduction", 0))
            row = dict(
                module_id=id(module),
                module=module.__file__,
                symbol=symbol,
                object_id=id(value),
                filename=state.get("filename"),
                kernel=kernel,
                rmsnorm=norm,
                reduction=reduction,
                heuristic=str(state.get("heuristic_type")),
                deterministic=metadata.get("deterministic"),
                batch_invariant=metadata.get("batch_invariant"),
                global_deterministic=metadata.get(
                    "are_deterministic_algorithms_enabled"
                ),
                runtime_deterministic=state.get("deterministic_mode"),
                coordinate_descent_configured=metadata.get(
                    "coordinate_descent_tuning", False
                ),
                dynamic_rblock_cached=state.get("_could_rblock_scale"),
                size_hints=state.get("size_hints"),
                selected=[],
            )
            try:
                row.update(file_receipt(row["filename"]))
                for launcher in state.get("launchers", []):
                    # Config.__dict__ includes kwargs and all backend options;
                    # don't silently discard num_ctas/maxnreg or future fields.
                    row["selected"].append(
                        dict(
                            hash=launcher.cache_hash,
                            config=dict(vars(launcher.config)),
                        )
                    )
            except (AttributeError, TypeError, ValueError) as error:
                violations.append(f"{symbol}: {error}")
            if reduction:
                if (
                    row["deterministic"] is not True
                    or row["runtime_deterministic"] is not True
                ):
                    violations.append(f"non-deterministic reduction: {symbol}")
                if row["batch_invariant"] or row["global_deterministic"]:
                    violations.append(f"unexpected global numerical mode: {symbol}")
                if len(row["selected"]) != 1:
                    violations.append(
                        f"reduction has no single selected binary: {symbol}"
                    )
                if row["dynamic_rblock_cached"] is True:
                    violations.append(
                        f"dynamic reduction scaling remains active: {symbol}"
                    )
            bindings.append(row)
    if not graphs or not any(row["rmsnorm"] for row in bindings):
        violations.append("missing graph-held RMSNorm coverage")
    return dict(graphs=graphs, bindings=bindings, violations=violations, files=files)


def install(runner):
    # No work or torch import on the default path or other profiles.
    if not glm53_ordering.enabled():
        return
    options = runner.compilation_config.inductor_compile_config
    if options.get("deterministic") is not True:
        return
    import torch
    from torch._inductor.codecache import PyCodeCache
    from torch._inductor.runtime import triton_heuristics

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
    ):
        raise ValueError("graph receipts require native-order GLM53 SM120 TP4")
    if not os.environ.get("VLLM_CACHE_ROOT"):
        raise ValueError("graph receipts require an explicit private cache root")
    root = Path(os.environ["VLLM_CACHE_ROOT"]).resolve()
    folder = root / "glm53-reduction-receipts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"rank-{parallel.rank}-pid-{os.getpid()}.json"
    if path.exists() or getattr(runner, "_glm53_reduction_receipts", False):
        raise ValueError("graph receipts cannot overwrite a prior capture")
    runner._glm53_reduction_receipts = True
    original = runner.capture_model

    @wraps(original)
    def capture(*args, **kwargs):
        result = dict(
            status="running",
            rank=parallel.rank,
            pid=os.getpid(),
            cache_root=str(root),
            source_sha256=sha(__file__),
            heuristic_source_sha256=sha(triton_heuristics.__file__),
            compiler_options=options,
            snapshots={},
        )
        # Exclusive creation preserves any failed capture as well as successes.
        with path.open("x") as stream:
            try:
                for phase in ("before", "after"):
                    snapshot = graph_snapshot(PyCodeCache.modules, root)
                    result["snapshots"][phase] = snapshot
                    if snapshot["violations"]:
                        raise ValueError(
                            f"{phase} capture graph receipt: {snapshot['violations']}"
                        )
                    if phase == "before":
                        captured = original(*args, **kwargs)
                result["status"] = "complete"
                return captured
            except BaseException as error:
                result.update(status="failed", error=repr(error))
                raise
            finally:
                json.dump(result, stream, indent=2, default=repr)
                stream.write("\n")

    runner.capture_model = capture
