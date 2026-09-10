# SPDX-License-Identifier: Apache-2.0
"""Explicit schema dispatch for the shared diagnostic serving harness."""

from slimserve import kv_diagnostic, rmsnorm_geometry


def policy(manifest):
    for candidate in (rmsnorm_geometry, kv_diagnostic):
        if manifest["serving_schema"] == candidate.SERVING_SCHEMA:
            return candidate
    raise ValueError("unknown diagnostic serving schema")


def cases(manifest):
    if policy(manifest) is kv_diagnostic:
        return kv_diagnostic.CASES
    from benchmarks.kernels.prepare_glm53_geometry_serving import CASES

    return CASES
