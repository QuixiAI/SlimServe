#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Bounded installed top-k repeatability probe; synthetic inputs, no timing claim."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def selection_validity(logits, indices, k=512):
    """Independent CPU membership/score check; tie indices need not be canonical."""
    logits, indices = logits.cpu(), indices.cpu().long()
    n = logits.shape[1]
    count = min(n, k)
    valid = bool(((indices[:, :count] >= 0) & (indices[:, :count] < n)).all())
    valid &= bool((indices[:, count:] == -1).all())
    chosen = indices[:, :count].sort(dim=1).values
    valid &= bool((chosen[:, 1:] != chosen[:, :-1]).all())
    if not valid:
        return False
    values = logits.gather(1, chosen).sort(dim=1, descending=True).values
    best = logits.sort(dim=1, descending=True).values[:, :count]
    return torch.equal(values, best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if active:
        parser.error(f"GPU compute processes already active: {active}")
    args.output.mkdir(parents=True, exist_ok=False)
    from vllm import _custom_ops as ops

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    native = root / "vllm/_C_stable_libtorch.abi3.so"
    result = {
        "status": "running",
        "diagnostic_only": True,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": sha(Path(__file__)),
        "native_sha256": sha(native),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "protocol": "6 sizes x3 families;10 eager and10 graph calls each,8 rows/top512",
        "cases": [],
    }

    def save():
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    for columns in (160, 512, 513, 1904, 2048, 8192):
        generator = torch.Generator().manual_seed(752 + columns)
        unique = (
            torch.stack(
                [torch.randperm(columns, generator=generator) for _ in range(8)]
            ).float()
            / columns
        )
        tied = torch.zeros_like(unique)
        tied[:, :31] = torch.arange(1, 32).float()
        random = torch.randn(8, columns, generator=generator)
        inputs = {"unique": unique, "tied": tied, "random": random}
        logits = unique.cuda()
        starts = torch.zeros(8, dtype=torch.int32, device="cuda")
        ends = torch.full((8,), columns, dtype=torch.int32, device="cuda")
        output = torch.empty(8, 512, dtype=torch.int32, device="cuda")

        def call(
            logits=logits, starts=starts, ends=ends, output=output, columns=columns
        ):
            ops.top_k_per_row_prefill(logits, starts, ends, output, 8, columns, 1, 512)

        call()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=logits.device)
        stream.wait_stream(torch.cuda.current_stream(logits.device))
        with torch.cuda.graph(graph, stream=stream):
            call()
        for family, host in inputs.items():
            logits.copy_(host)
            observations = {"eager": [], "graph": []}
            for mode in observations:
                for _ in range(10):
                    output.fill_(-777)
                    if mode == "eager":
                        call()
                    else:
                        graph.replay()
                    observations[mode].append(output.cpu())
            row = {"columns": columns, "family": family, "modes": {}}
            for mode, outputs in observations.items():
                ref = outputs[0]
                ref_set = ref.sort(dim=1).values
                row["modes"][mode] = {
                    "all_selections_valid": all(
                        selection_validity(host, x) for x in outputs
                    ),
                    "changed_positions_vs_first": [
                        int((x != ref).sum()) for x in outputs
                    ],
                    "changed_set_positions_vs_first": [
                        int((x.sort(dim=1).values != ref_set).sum()) for x in outputs
                    ],
                }
            artifact = args.output / f"selection-{columns}-{family}.pt"
            torch.save(
                {
                    "logits": host,
                    "row_starts": starts.cpu(),
                    "row_ends": ends.cpu(),
                    "outputs": {k: torch.stack(v) for k, v in observations.items()},
                },
                artifact,
            )
            row["artifact"] = {"path": str(artifact), "sha256": sha(artifact)}
            result["cases"].append(row)
            save()
            print(json.dumps(row), flush=True)
        del graph, call
    assert sha(native) == result["native_sha256"]
    assert sha(Path(__file__)) == result["source_sha256"]
    result["status"] = "complete"
    result["all_selections_valid"] = all(
        m["all_selections_valid"] for r in result["cases"] for m in r["modes"].values()
    )
    save()


if __name__ == "__main__":
    main()
