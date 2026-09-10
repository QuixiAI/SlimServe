# SPDX-License-Identifier: Apache-2.0
"""Explicit schema dispatch for the shared diagnostic serving harness."""

from slimserve import indexer_correction_diagnostic, kv_diagnostic, rmsnorm_geometry


def policy(manifest):
    for candidate in (rmsnorm_geometry, kv_diagnostic, indexer_correction_diagnostic):
        if manifest["serving_schema"] == candidate.SERVING_SCHEMA:
            return candidate
    raise ValueError("unknown diagnostic serving schema")


def cases(manifest):
    selected = policy(manifest)
    if selected is not rmsnorm_geometry:
        return selected.CASES
    from benchmarks.kernels.prepare_glm53_geometry_serving import CASES

    return CASES
