# SPDX-License-Identifier: Apache-2.0
"""Model-owned scratch for sequential GLM pooled indexer layers.

This is not KV state. A layer finishes copying its selected indices before
the following layer reuses the logits. Separate model/PP-stage instances
must own separate workspaces; never share through a process-global cache.
Allocation is lazy so stages without a pooled indexer pay nothing.
"""

import torch


class Glm5NextIndexerWorkspace:
    def __init__(self):
        self._decode_logits: torch.Tensor | None = None

    @classmethod
    def from_config(cls, extra):
        enabled = (
            extra.get("glm5_next_shared_indexer_scratch", False)
            if isinstance(extra, dict)
            else False
        )
        if not isinstance(enabled, bool):
            raise ValueError("glm5_next_shared_indexer_scratch must be a boolean")
        return cls() if enabled else None

    def get_decode_logits(
        self, rows: int, pools: int, device: torch.device
    ) -> torch.Tensor:
        if rows <= 0 or pools <= 0:
            raise ValueError("Indexer scratch dimensions must be positive")
        device = torch.device(device)
        if self._decode_logits is None:
            self._decode_logits = torch.empty(
                (rows, pools), dtype=torch.float32, device=device
            )
        elif (
            self._decode_logits.shape != (rows, pools)
            or self._decode_logits.device != device
        ):
            raise ValueError("A shared indexer workspace cannot change shape or device")
        return self._decode_logits
