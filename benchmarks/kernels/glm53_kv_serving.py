# SPDX-License-Identifier: Apache-2.0
"""KV policy for the shared before-forward/capture serving lifecycle."""

from benchmarks.kernels.audit_glm53_kv_graphs import inventory
from benchmarks.kernels.glm53_geometry_serving import ServingGeometry
from benchmarks.kernels.glm53_kv_loader import KVLoader


class ServingKV(ServingGeometry):
    def __init__(self, rank, manifest, path):
        super().__init__(
            rank,
            manifest,
            path,
            loader_type=KVLoader,
            graph_inventory=inventory,
            root_only=True,
        )
