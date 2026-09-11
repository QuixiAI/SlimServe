# SPDX-License-Identifier: Apache-2.0
"""Explicit schema dispatch for the shared diagnostic serving harness."""

from slimserve import (
    indexer_correction_diagnostic,
    kv_diagnostic,
    prompt_score_diagnostic,
    rmsnorm_geometry,
)


def policy(manifest):
    for candidate in (
        rmsnorm_geometry,
        kv_diagnostic,
        indexer_correction_diagnostic,
        prompt_score_diagnostic,
    ):
        if manifest["serving_schema"] == candidate.SERVING_SCHEMA:
            return candidate
    raise ValueError("unknown diagnostic serving schema")


def cases(manifest):
    return policy(manifest).CASES
