# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic KV tier roundtrip benchmark; this is not serving throughput."""

import argparse
import importlib.util
import json
import statistics
import sys
import tempfile
import time

import torch

from vllm.v1.worker.gpu.kv_tier_dma import TierOpBatch
from vllm.v1.worker.gpu.kv_tier_nvme import KVTierNVMe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", help="saved backend module for A/B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--stride", type=int, default=65536)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    cls = KVTierNVMe
    if args.implementation:
        spec = importlib.util.spec_from_file_location(
            "tier_baseline", args.implementation
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        cls = module.KVTierNVMe
    size = args.rows * args.stride
    device = torch.device(args.device)
    source = torch.randint(-128, 127, (args.rows, args.stride), dtype=torch.int8)
    backing = torch.zeros(2 * size, dtype=torch.int8, device=device)
    backing[:size].copy_(source.flatten())
    result = {"kind": "synthetic", "args": vars(args), "bytes_each_way": size}
    with tempfile.TemporaryDirectory(dir=args.directory) as directory:
        tier = cls(backing, args.stride, args.rows, device, directory)
        try:
            measurements = []
            for iteration in range(args.repeats + 1):
                start = time.perf_counter()
                tier.issue(
                    TierOpBatch(iteration * 2, [(i, i) for i in range(args.rows)], [])
                )
                tier.flush()
                saved = time.perf_counter()
                tier.issue(
                    TierOpBatch(
                        iteration * 2 + 1,
                        [],
                        [(i, i + args.rows) for i in range(args.rows)],
                    )
                )
                tier.flush()
                restored = time.perf_counter()
                assert torch.equal(backing[size:].cpu(), source.flatten())
                if iteration:
                    measurements.append(
                        {"offload_s": saved - start, "restore_s": restored - saved}
                    )
            result["measurements"] = measurements
            result["correct"] = True
            result["median_offload_GB_s"] = (
                size / statistics.median(m["offload_s"] for m in measurements) / 1e9
            )
            result["median_restore_GB_s"] = (
                size / statistics.median(m["restore_s"] for m in measurements) / 1e9
            )
        finally:
            tier.shutdown()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
