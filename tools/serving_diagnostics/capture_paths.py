# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable capture location shared by the proxy and read-only analyzers."""

import os
from pathlib import Path

DEFAULT_LOG_DIR = Path(
    os.environ.get("PROXY_LOG_DIR", str(Path.home() / ".cache/slimserve/traffic-proxy"))
).expanduser()
DEFAULT_LOG_PATH = DEFAULT_LOG_DIR / "traffic.jsonl"
