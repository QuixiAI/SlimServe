# SPDX-License-Identifier: Apache-2.0
"""One source catalog for GLM53 campaign clients and prepared source freezes."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATHS = (
    *(
        f"benchmarks/{name}.py"
        for name in (
            "benchmark_glm53_campaign",
            "benchmark_glm53_server",
            "benchmark_glm53_b12x",
            "benchmark_dsv4_exact",
            "benchmark_glm53_quality",
            "benchmark_glm53_prefill",
        )
    ),
    "slimserve/campaign_sources.py",
    "slimserve/smoke.py",
    "slimserve/stream.py",
    "slimserve/cli.py",
    "slimserve/server.py",
    "vllm/utils/jit_monitor.py",
    "slimserve/glm53_ordering.py",
    "slimserve/deterministic_reductions.py",
    "slimserve/reduction_receipts.py",
    "slimserve/rmsnorm_diagnostic.py",
    "slimserve/rmsnorm_geometry.py",
    "benchmarks/kernels/glm53_geometry_serving.py",
    "benchmarks/kernels/glm53_geometry_loader.py",
    "benchmarks/kernels/glm53_artifact_roots.py",
    "benchmarks/kernels/glm53_binary_observer.py",
    "benchmarks/kernels/glm53_rmsnorm_geometry.py",
    "benchmarks/kernels/audit_glm53_geometry_graphs.py",
    "benchmarks/kernels/check_glm53_cached_rmsnorm.py",
    "vllm/v1/worker/gpu_model_runner.py",
    "slimserve/canonical_moe.py",
    "slimserve/canonical_indexer.py",
    "vllm/model_executor/layers/fused_moe/router/glm_route_align.py",
    "vllm/model_executor/layers/fused_moe/router/glm_stable_align.py",
    "vllm/model_executor/layers/fused_moe/experts/marlin_moe.py",
    "vllm/model_executor/layers/glm5_next_indexer.py",
    "vllm/quixicore/ops.py",
)


def snapshot():
    return {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PATHS
    }
