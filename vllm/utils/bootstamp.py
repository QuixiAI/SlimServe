# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wall-clock boot instrumentation shared across the server process tree.

The first process to call :func:`bootstamp` pins ``VLLM_BOOT_T0`` in the
environment; children (spawned or forked) inherit it, so every stamp is an
offset from the same origin. An external launcher may pre-set ``VLLM_BOOT_T0``
(epoch seconds) to anchor t0 at the true process-spawn time.
"""

import os
import time

from vllm.logger import init_logger

logger = init_logger(__name__)

_T0: float | None = None


def _t0() -> float:
    global _T0
    if _T0 is None:
        env = os.environ.get("VLLM_BOOT_T0")
        if env is None:
            env = str(time.time())
            os.environ["VLLM_BOOT_T0"] = env
        _T0 = float(env)
    return _T0


def bootstamp(tag: str) -> None:
    """Log ``tag`` with seconds elapsed since the boot origin."""
    logger.info("[boot +%7.2fs pid=%d] %s", time.time() - _t0(), os.getpid(), tag)
