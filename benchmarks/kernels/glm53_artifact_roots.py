# SPDX-License-Identifier: Apache-2.0
"""Trusted AOT artifact provenance, independent of kernel controller callbacks.

Discovery only unpickles the owned cache; it never post-compiles or executes it.
The observer joins each real deserialize invocation to its live CompiledFxGraph
and the returned AOT artifact. Imported benchmark helpers are not artifact roots.
"""

import hashlib
import io
import pickle
import threading
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

from benchmarks.kernels.audit_glm53_geometry_graphs import verify_call_export
from benchmarks.kernels.check_glm53_attention_norms import require
from slimserve.rmsnorm_diagnostic import sha


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_store(model):
    from torch._dynamo.aot_compile import AOTCompileUnpickler

    outer = AOTCompileUnpickler({}, io.BytesIO(model.read_bytes())).load()
    inner = pickle.loads(outer["compiled_fn"][1])
    return inner["standalone_compile_artifacts"], inner["aot_autograd_config"]


def discover(store, namespace, expected_graphs):
    from torch._functorch._aot_autograd.aot_autograd_result import (
        BundledAOTAutogradResult,
        BundledCompiledForward,
    )
    from torch._inductor.output_code import CompiledFxGraph

    require(not store.loaded_submodule_store, "discovery requires unloaded artifacts")
    require(
        set(store.submodule_bytes.values()) == set(store.submodule_bytes_store),
        "serialized submodule/artifact coverage differs",
    )
    rows, graphs = [], set()
    for artifact, blob in sorted(store.submodule_bytes_store.items()):
        require(digest_bytes(blob) == artifact, "serialized artifact digest changed")
        data = pickle.loads(blob)
        require(type(data) is bytes, "serialized artifact payload must be bytes")
        bundle = pickle.loads(data)
        require(type(bundle) is tuple and len(bundle) == 3, "unexpected AOT bundle")
        entry = bundle[0]
        require(
            isinstance(entry, BundledAOTAutogradResult)
            and isinstance(entry.compiled_fw, BundledCompiledForward)
            and entry.compiled_bw is None
            and type(entry.compiled_fw.result) is CompiledFxGraph,
            "expected one inference CompiledFxGraph root per artifact",
        )
        graph = entry.compiled_fw.result
        key = graph.cache_key
        require(
            isinstance(key, str)
            and key.isalnum()
            and isinstance(graph.source_code, str),
            "invalid serialized graph key/source",
        )
        relative = f"inductor_cache/{key[1:3]}/{key}.py"
        source_hash = digest_bytes(graph.source_code.encode())
        path = namespace / relative
        require(
            relative in expected_graphs
            and path.resolve() == path
            and source_hash == expected_graphs[relative] == sha(path),
            f"serialized root source differs from original graph: {relative}",
        )
        require(relative not in graphs, "duplicate serialized graph root")
        graphs.add(relative)
        rows.append(
            dict(
                artifact=artifact,
                payload_sha256=digest_bytes(data),
                cache_key=key,
                graph=relative,
                graph_sha256=source_hash,
                submodules=sorted(
                    k for k, v in store.submodule_bytes.items() if v == artifact
                ),
            )
        )
    require(graphs == set(expected_graphs), "serialized model graph roots incomplete")
    return rows


def module_inventory(modules):
    """Raw complete cache-module list; save this BEFORE any coverage validation."""
    rows, seen = [], set()
    for module in modules:
        if id(module) in seen:
            continue
        seen.add(id(module))
        path = getattr(module, "__file__", None)
        call = getattr(module, "call", None)
        function = getattr(call, "__func__", call)
        code = getattr(function, "__code__", None)
        row = dict(
            module_index=len(rows) + 1,
            path=path,
            name=getattr(module, "__name__", None),
            callable=callable(call),
            bound=getattr(call, "__self__", None) is not None,
            call_filename=getattr(code, "co_filename", None),
            call_line=getattr(code, "co_firstlineno", None),
        )
        try:
            row["source_sha256"] = sha(path)
        except (OSError, TypeError) as error:
            row["source_error"] = repr(error)
        rows.append(row)
    return rows


class ArtifactRootObserver:
    def __init__(self, expected, private, *, emit):
        self.expected = {r["payload_sha256"]: r for r in expected}
        require(len(self.expected) == len(expected), "duplicate artifact payload")
        self.private = Path(private)
        self.emit = emit
        self.local = threading.local()
        self.lock = threading.RLock()
        self.started, self.completed, self.bindings = set(), {}, {}

    def bind(self, graph, path, modules):
        row = getattr(self.local, "artifact", None)
        require(row is not None, "graph loaded outside observed artifact deserialize")
        require(
            graph.cache_key == row["cache_key"]
            and digest_bytes(graph.source_code.encode()) == row["graph_sha256"]
            and Path(path) == self.private / row["graph"]
            and Path(path).resolve() == Path(path)
            and sha(path) == row["graph_sha256"],
            f"live artifact root source/key/path differs: {path}",
        )
        matches = {
            id(m): m
            for m in modules
            if getattr(m, "__file__", None) == str(path)
            and getattr(m, "call", None) is graph.current_callable
        }
        require(len(matches) == 1, "artifact callable lacks unique live module binding")
        module = next(iter(matches.values()))
        verify_call_export(module, graph.source_code)
        require(
            graph.compiled_fn_runner is getattr(module, "runner", None),
            "artifact runner differs from exported module runner",
        )
        with self.lock:
            require(row["artifact"] not in self.bindings, "artifact root loaded twice")
            self.bindings[row["artifact"]] = (graph, module, graph.current_callable)
            indices = {id(m): i + 1 for i, m in enumerate(dict.fromkeys(modules))}
            self.emit(
                dict(
                    event="artifact_root_bound", module_index=indices[id(module)], **row
                )
            )

    @contextmanager
    def intercept(self, *, artifact_class=None, graph_class=None, code_cache=None):
        if artifact_class is None:
            from torch._inductor.standalone_compile import AOTCompiledArtifact

            artifact_class = AOTCompiledArtifact
        if graph_class is None:
            from torch._inductor.output_code import CompiledFxGraph

            graph_class = CompiledFxGraph
        if code_cache is None:
            from torch._inductor.codecache import PyCodeCache

            code_cache = PyCodeCache
        deserialize = artifact_class.__dict__["deserialize"]
        after = graph_class.after_deserialization
        require(
            isinstance(deserialize, staticmethod), "expected static AOT deserialize API"
        )
        require(
            not getattr(deserialize.__func__, "_glm53_artifact_roots", False)
            and not getattr(after, "_glm53_artifact_roots", False),
            "conflicting artifact observer hook",
        )

        @wraps(deserialize.__func__)
        def load(data):
            key = digest_bytes(data)
            require(key in self.expected, "unexpected serialized artifact payload")
            row = self.expected[key]
            require(
                getattr(self.local, "artifact", None) is None, "nested artifact load"
            )
            with self.lock:
                require(key not in self.started, "artifact deserialize repeated")
                self.started.add(key)
                self.emit(dict(event="artifact_deserialize_begin", **row))
            self.local.artifact = row
            try:
                result = deserialize.__func__(data)
                with self.lock:
                    require(
                        row["artifact"] in self.bindings,
                        "artifact has no observed root",
                    )
                    self.completed[row["artifact"]] = result
                    self.emit(dict(event="artifact_deserialize_complete", **row))
                return result
            finally:
                self.local.artifact = None

        @wraps(after)
        def loaded(graph, *args, **kwargs):
            path = after(graph, *args, **kwargs)
            self.bind(graph, path, code_cache.modules)
            return path

        load._glm53_artifact_roots = loaded._glm53_artifact_roots = True
        hook = staticmethod(load)
        artifact_class.deserialize = hook
        graph_class.after_deserialization = loaded
        try:
            yield self
        finally:
            changed = []
            if artifact_class.__dict__["deserialize"] is hook:
                artifact_class.deserialize = deserialize
            else:
                changed.append("artifact deserialize")
            if graph_class.after_deserialization is loaded:
                graph_class.after_deserialization = after
            else:
                changed.append("graph after_deserialization")
            require(not changed, f"foreign artifact observer hook change: {changed}")

    def roots(self, store, modules):
        """Check the actual returned artifacts and still-live callable bindings."""
        expected = {r["artifact"] for r in self.expected.values()}
        require(
            set(store.loaded_submodule_store)
            == set(self.completed)
            == set(self.bindings)
            == expected,
            "incomplete actual artifact/root coverage",
        )
        roots = []
        for artifact in sorted(expected):
            graph, module, call = self.bindings[artifact]
            require(
                store.loaded_submodule_store[artifact] is self.completed[artifact]
                and any(module is m for m in modules)
                and graph.current_callable is call is module.call
                and graph.compiled_fn_runner is getattr(module, "runner", None),
                "completed artifact/root callable binding changed",
            )
            roots.append(module)
        return roots
