# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer NaN birth probe (VLLM_NAN_WATCH_LAYERS=1), diagnostic only.

The DSV4 degeneration incident surfaces as NaN logits rows; the logits
tripwire says *that* a step went bad but not *where*. This probe keeps a
sticky per-layer counter on the GPU: each decoder layer ORs "any NaN in my
output" into its slot. No host syncs on the hot path — the accumulate ops
are capture-safe (the buffer is allocated eagerly during warmup, so its
address is stable for graph replay). When the logits tripwire fires, it
calls snapshot() (one D2H copy, off the hot path) and the minimum nonzero
layer index is the NaN's birth layer.

The counters are cumulative for the process lifetime: the first snapshot
after the first event is the meaningful one (birth layer = min index; all
later layers go NaN too once poisoned hidden states flow through them).
"""

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_MAX_SLOTS = 256
_BUF: torch.Tensor | None = None
_ENABLED: bool | None = None


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.getenv("VLLM_NAN_WATCH_LAYERS", "0").lower() in (
            "1", "true", "on", "yes",
        )
        if _ENABLED:
            logger.warning(
                "NAN_WATCH_LAYERS enabled: per-layer NaN counters on every "
                "decoder layer output (diagnostic mode)")
    return _ENABLED


def probe(slot: int, *tensors: torch.Tensor | None) -> None:
    """Accumulate NaN presence for this layer slot. Capture-safe after the
    first eager call has allocated the buffer."""
    global _BUF
    if not enabled():
        return
    if _BUF is None:
        if torch.cuda.is_current_stream_capturing():
            return
        _BUF = torch.zeros(_MAX_SLOTS, dtype=torch.int32, device="cuda")
    flag = None
    for t in tensors:
        if t is None:
            continue
        part = torch.isnan(t).any()
        flag = part if flag is None else (flag | part)
    if flag is not None:
        _BUF[slot] += flag.to(torch.int32)


def snapshot() -> list[tuple[int, int]] | None:
    """(slot, count) pairs for every slot that has seen NaN. Syncs; call
    only from an error handler. Logs the never-allocated case loudly: it
    means no probe() ran in this process, i.e. the probed model code did
    not execute here — itself a diagnostic result."""
    if _BUF is None:
        logger.error(
            "NAN_PROBE snapshot: buffer never allocated in this process "
            "(enabled=%s) — the probed model-forward code did not run here",
            _ENABLED)
        return None
    host = _BUF.cpu()
    return [(i, int(c)) for i, c in enumerate(host.tolist()) if c]
