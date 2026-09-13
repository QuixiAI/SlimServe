# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand-off of a batch's shared-expert output to the routed experts' combine.

At decode the MoE runner launches the shared-expert MLP before the routed
experts (on the aux stream when the shared-expert overlap is on) and
publishes its output here. A routed-experts implementation whose combine can
fold an extra addend (Marlin's ``moe_sum`` through QuixiCore ``moe_sum_add``)
consumes it, joins the aux stream itself, and marks the entry folded; the
runner then skips its own ``shared + fused`` add. Entries are matched on the
identity of the batch's ``topk_ids`` tensor so a stale publication can never
be folded into another batch, and the runner clears the slot after every
forward whether or not it was consumed."""

from dataclasses import dataclass

import torch

MAX_TOKENS = 16

@dataclass
class SharedOutput:
    topk_ids: torch.Tensor
    output: torch.Tensor
    # The stream the shared-expert MLP was launched on when it differs from
    # the current stream; the consumer must wait on it before reading output.
    stream: torch.cuda.Stream | None
    folded: bool = False


_published: SharedOutput | None = None


def publish(entry: SharedOutput) -> None:
    global _published
    _published = entry


def consume(topk_ids: torch.Tensor) -> SharedOutput | None:
    """The pending entry for this batch, or None. The caller sets
    ``folded`` once it has actually added the output into its combine."""
    entry = _published
    if entry is not None and entry.topk_ids is topk_ids:
        return entry
    return None


def clear() -> None:
    global _published
    _published = None
