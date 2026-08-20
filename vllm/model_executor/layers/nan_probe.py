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


# ---------------------------------------------------------------------------
# Determinism probe (VLLM_DET_PROBE=/path, diagnostic only). Appends one line
# per (step, slot) with an fp64 checksum of the probed tensors to that file
# from every rank (rank in the line). Host-syncs on every probe: never on in
# production; used to bisect run-to-run nondeterminism (XPU bring-up
# 2026-08-18) by diffing two identical requests layer by layer.
_DET_PATH = os.getenv("VLLM_DET_PROBE", "")
_DET_STEP = 0


def det_step() -> None:
    global _DET_STEP
    _DET_STEP += 1


def det_probe(tag: str, *tensors: torch.Tensor | None) -> None:
    if not _DET_PATH:
        return
    if torch.xpu.is_available() and torch.xpu.is_current_stream_capturing():
        return  # host sync inside a graph capture is illegal
    dump = os.getenv("VLLM_DET_DUMP", "")
    if dump and tag.startswith(dump):
        try:
            import torch.distributed as dist

            r = dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            r = 0
        torch.save(
            [None if t is None else t.detach().cpu() for t in tensors],
            f"{_DET_PATH}.dump.{tag}.step{_DET_STEP}.rank{r}.pt",
        )
    parts = []
    for t in tensors:
        if t is None:
            continue
        tf = t.detach().float()
        # fp32: Arc has no fp64 (UR_RESULT_ERROR_UNSUPPORTED_FEATURE).
        flat = tf.reshape(-1)
        w = torch.arange(1, flat.numel() + 1, device=flat.device, dtype=torch.float32)
        w = (w % 977) / 977.0  # position hash: catches permutations/mislocated rows
        parts.append(f"{flat.sum().item():.4f}/{flat.abs().sum().item():.4f}/{(flat * w).sum().item():.4f}")
    try:
        import torch.distributed as dist

        rank = dist.get_rank() if dist.is_initialized() else 0
    except Exception:
        rank = 0
    with open(f"{_DET_PATH}.rank{rank}", "a") as f:
        f.write(f"{_DET_STEP} {tag} {' '.join(parts)}\n")
