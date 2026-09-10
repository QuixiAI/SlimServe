# SPDX-License-Identifier: Apache-2.0
"""Production-shaped packed indexer pages for isolated GLM TP8 benchmarks."""

import torch


def packed_indexer_cache(layers, pages, block_size, device):
    """Eleven columns at BS4608 give the serving 6,488,064-byte page stride."""
    return torch.randn(pages, layers, block_size, 64, device=device,
                       dtype=torch.bfloat16).transpose(0, 1)
