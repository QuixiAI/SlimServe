# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable capture location shared by the proxy and read-only analyzers."""

import os
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def results_dir(run_id: str) -> Path:
    """Default raw-output location: perf/results/<UTC date>/<run-id>/ (git-ignored)."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return REPO_ROOT / "perf" / "results" / day / run_id


_override = os.environ.get("PROXY_LOG_DIR")
DEFAULT_LOG_DIR = (
    Path(_override).expanduser() if _override else results_dir("traffic-proxy")
)
DEFAULT_LOG_PATH = DEFAULT_LOG_DIR / "traffic.jsonl"
