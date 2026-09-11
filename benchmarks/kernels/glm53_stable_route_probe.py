# SPDX-License-Identifier: Apache-2.0
"""Isolated SM120 compilation of both small-M route/alignment policies."""

import os
from pathlib import Path

from benchmarks.kernels.benchmark_mhc_output_parallel import build as build_probe


def build():
    directory = os.environ.get("SLIMSERVE_GLM53_STABLE_ROUTE_PROBE")
    if not directory:
        raise RuntimeError(
            "set SLIMSERVE_GLM53_STABLE_ROUTE_PROBE to an isolated build directory"
        )
    return build_probe(Path(directory), name="glm53_stable_route_probe")
