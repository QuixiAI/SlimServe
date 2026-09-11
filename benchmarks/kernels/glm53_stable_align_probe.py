# SPDX-License-Identifier: Apache-2.0
"""Isolated SM120 stable large-M alignment candidate, not used by serving."""

import os
from pathlib import Path

from benchmarks.kernels.benchmark_mhc_output_parallel import build as build_probe


def build():
    directory = os.environ.get("SLIMSERVE_GLM53_STABLE_ALIGN_PROBE")
    if not directory:
        raise RuntimeError(
            "set SLIMSERVE_GLM53_STABLE_ALIGN_PROBE to an isolated build directory"
        )
    return build_probe(Path(directory), name="glm53_stable_align_probe")
