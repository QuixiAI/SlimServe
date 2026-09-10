# SPDX-License-Identifier: Apache-2.0
"""Opt-in indexer correction loader policy; no production installation.

Reuse observed static CUDA loads and the extra-launch graph lifecycle. Compile
the unchanged qualified JIT, then adapt its already-compiled bytes to Torch's
static launcher; do not generate another kernel or inject a saved cubin.
"""

import base64
import hashlib
from pathlib import Path

from benchmarks.analyze_glm53_quality_pair import require
from benchmarks.kernels.glm53_indexer_correction import (
    OPTIONS,
    IndexerOnlyCorrection,
)
from benchmarks.kernels.glm53_kv_loader import (
    KVIntervention,
    KVLoader,
    check_source,
    load_preserving_provenance,
)
from slimserve.rmsnorm_diagnostic import sha

SCHEMA = "glm53-indexer-correction-loader-v1"
SIGNATURE = dict(
    Packed="*bf16", Gamma="*bf16", Bias="*bf16", Output="*bf16", Selected="*u8", N="i32"
)


def to_static(compiled, record, rank, binary_root):
    """Exact in-memory/disk image checks before any driver load."""
    import triton
    from torch._inductor.runtime.static_triton_launcher import (
        StaticallyLaunchedCudaKernel,
    )
    from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

    key = base64.b32encode(bytes.fromhex(compiled.hash)).decode().rstrip("=")
    path = Path(binary_root) / key / (record["kernel"] + ".cubin")
    require(
        compiled.src.fn.__name__ == record["kernel"] == "correct_indexer"
        and compiled.src.fn.arg_names == list(SIGNATURE)
        and compiled.src.signature == SIGNATURE
        and not compiled.src.constants
        and key == record["selected"]["hash"]
        and record["selected"]["config"] == {"num_warps": 1, "num_stages": 1}
        and {k: getattr(compiled.metadata, k) for k in OPTIONS} == OPTIONS,
        "correction JIT/signature/config/key differs",
    )
    image = compiled.asm["cubin"]
    require(
        isinstance(image, bytes)
        and path.resolve() == path
        and hashlib.sha256(image).hexdigest() == sha(path) == record["cubin_sha256"],
        "correction compiled image differs from qualification",
    )
    # Same conversion performed by StaticTritonCompileResult.can_statically_launch,
    # but no permissive fallback to another launcher and no recompile.
    compiled._cubin_path = str(path)
    static = StaticallyLaunchedCudaKernel(compiled)
    require(
        static.arg_tys == "OOOOOi" and not static.full_constexprs,
        "correction static ABI differs",
    )
    return StaticTritonCompileResult(
        static,
        triton.Config({}, num_warps=1, num_stages=1),
        dict(device=rank, device_type="cuda", constants={}, signature=SIGNATURE),
        dict(grid_type="FixedGrid", fixed_grid=["N", 1, 1]),
    )


class BoundIndexerCorrection:
    """Per-binding fixed scratch storage; no allocation or compilation in run.

    Only views and the qualified adapter are created on the host during graph
    capture. Each target binding owns its arena; CUDA graph replay uses its stable
    addresses. Serving must validate capacity against its scheduler before install.
    """

    def __init__(self, combo, correction, storage, capacity):
        self.combo, self.correction = combo, correction
        self.storage, self.capacity = storage, capacity
        self.address = storage.data_ptr()
        self.device = storage.device
        self.verify()

    def verify(self):
        import torch

        require(
            type(self.capacity) is int
            and 0 < self.capacity <= 65536
            and isinstance(self.storage, torch.Tensor)
            and self.storage.dtype == torch.uint8
            and self.storage.device == self.device
            and tuple(self.storage.shape) == (self.capacity + 2, 128)
            and self.storage.stride() == (128, 1)
            and self.storage.data_ptr() == self.address,
            "indexer selection arena rebound or malformed",
        )
        require(
            callable(self.combo) and callable(self.correction),
            "bound launchers required",
        )

    def run(self, *args, stream):
        require(len(args) == 11, "original combo ABI requires eleven arguments")
        rows = args[8]
        require(
            type(rows) is int
            and 0 < rows <= self.capacity
            and args[9:] == (rows, rows),
            "correction rows exceed fixed arena or disagree",
        )
        self.verify()
        selected = self.storage[1 : rows + 1]
        return IndexerOnlyCorrection(self.combo, self.correction, selected)(
            *args, stream=stream
        )


def check_adapter(adapter, combo, correction, capacity):
    require(
        type(adapter) is BoundIndexerCorrection
        and adapter.combo is combo
        and adapter.correction is correction
        and adapter.capacity == capacity
        and "run" not in vars(adapter)
        and "verify" not in vars(adapter),
        "indexer correction adapter dispatch changed",
    )
    adapter.verify()


class IndexerCorrectionIntervention(KVIntervention):
    schema = SCHEMA
    candidate_mode = "correction"
    extra_key = "correction"
    event_prefix = "indexer_correction"

    def make_adapter(self, launcher, pair):
        import torch

        capacity = self.manifest["selection_capacity"]
        require(
            type(capacity) is int and 0 < capacity <= 65536,
            "bounded selection capacity required",
        )
        storage = torch.full(
            (capacity + 2, 128), 171, dtype=torch.uint8, device=f"cuda:{self.rank}"
        )
        adapter = BoundIndexerCorrection(launcher, pair[1], storage, capacity)
        return adapter, adapter.run

    def verify_adapter(self, binding):
        adapter = binding["adapter"]
        require(
            getattr(binding["run"], "__self__", None) is adapter
            and getattr(binding["run"], "__func__", None) is BoundIndexerCorrection.run,
            "indexer correction run bypassed",
        )
        check_adapter(
            adapter,
            binding["launcher"],
            binding["appended"][1],
            self.manifest["selection_capacity"],
        )

    def graph_inventory(self, modules):
        from benchmarks.kernels.audit_glm53_indexer_correction_graphs import inventory

        return inventory(modules, self.manifest, self.rank, self.mode, self.observer)


class IndexerCorrectionLoader(KVLoader):
    hook_marker = "_glm53_indexer_correction_loader"
    controller_class = IndexerCorrectionIntervention
    source_file = __file__

    def compile_replacement(self, record):
        import torch
        import triton
        from triton.compiler import ASTSource

        self.check_cache()
        check_source(record, self.private)
        jit = self.templates.get(record["relative"])
        if jit is None:
            jit = load_preserving_provenance(
                self.private / record["relative"],
                Path(record["source"]),
                record["kernel"],
                f"glm53_indexer_{sha(self.manifest_path)}_rank{self.rank}",
                debug_source=Path(record["debug_source"]),
            )
            self.templates[record["relative"]] = jit
        root = self.cache / "triton" / str(self.rank)
        with torch.cuda.device(self.rank), triton.knobs.cache.scope():
            triton.knobs.cache.dir = str(root)
            compiled = triton.compile(
                ASTSource(jit, signature=SIGNATURE), options=OPTIONS
            )
            result = to_static(compiled, record, self.rank, root)
        self.check_cache()
        require(
            self.observer.digest(result) == record["cubin_sha256"],
            "static conversion changed image",
        )
        return result
