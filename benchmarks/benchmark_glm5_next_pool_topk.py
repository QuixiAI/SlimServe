# SPDX-License-Identifier: Apache-2.0
"""Quarantined selection-only diagnostic; never imported by serving.

GLM decode currently calls prefill top-k, whose insertion/radix dispatch
depends on row number, not visible columns. Compare existing native kernels
before writing a replacement. decode_capacity keeps the real 1M-context
workspace shape under changing lengths. decode_bound is ONLY a diagnostic:
its graph has a smaller fixed upper bound and cannot serve longer contexts.
Synthetic logits and kernel timings are not model-quality or serving gains.
"""

import argparse
import json
import statistics

import torch

from vllm import _custom_ops as ops


def validate_selection(logits, lengths, selected):
    """Exact value oracle, including ties, unique IDs and -1 padding."""
    k = selected.shape[1]
    for row, count in enumerate(lengths.tolist()):
        indices = selected[row]
        valid = indices[indices >= 0].long()
        assert valid.numel() == min(count, k)
        assert (indices[indices < 0] == -1).all()
        assert (valid < count).all()
        assert valid.unique().numel() == valid.numel()
        expected = logits[row, :count].topk(min(count, k)).values
        actual = logits[row, valid].sort(descending=True).values
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@torch.inference_mode()
def measure(rows, context, *, distribution, layers, repeats, steps):
    torch.manual_seed(23100 + rows + context)
    capacity, k, visible = 262144, 512, context // 4
    logits = torch.empty(layers, rows, capacity, device="cuda")
    lengths = torch.full((rows,), visible, device="cuda", dtype=torch.int32)
    zeros = torch.zeros_like(lengths)
    outputs = {
        name: torch.empty(layers, rows, k, device="cuda", dtype=torch.int32)
        for name in ("prefill", "decode_capacity", "decode_bound_diagnostic")
    }
    workspace = torch.empty(
        ops.top_k_per_row_decode_workspace_size(rows, capacity, k),
        dtype=torch.uint8,
        device="cuda",
    )
    bounds = [logits[layer, :, :visible] for layer in range(layers)]

    def run(name):
        for layer in range(layers):
            if name == "prefill":
                ops.top_k_per_row_prefill(
                    logits[layer],
                    zeros,
                    lengths,
                    outputs[name][layer],
                    rows,
                    logits.stride(1),
                    1,
                    k,
                )
            else:
                source = (
                    bounds[layer]
                    if name == "decode_bound_diagnostic"
                    else logits[layer]
                )
                ops.top_k_per_row_decode(
                    source,
                    1,
                    lengths,
                    outputs[name][layer],
                    workspace,
                    rows,
                    logits.stride(1),
                    1,
                    k,
                )

    def refill():
        logits.normal_()
        if distribution == "narrow":
            logits.mul_(1e-5).add_(1.0)

    refill()
    for name in outputs:
        for _ in range(3):
            run(name)
    torch.cuda.synchronize()
    graphs = {}
    for name in outputs:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(name)
        graphs[name] = graph
    set_differences = dict.fromkeys(outputs, 0)
    for replay in range(4):
        refill()
        lengths.fill_(visible)
        if replay % 2:
            lengths[::2] = visible // 2
        if replay == 3:
            lengths[-1] = 0
        for name, graph in graphs.items():
            graph.replay()
            for layer in range(layers):
                validate_selection(logits[layer], lengths, outputs[name][layer])
            if name != "prefill":
                different = (
                    (
                        outputs[name].sort(dim=-1).values
                        != outputs["prefill"].sort(dim=-1).values
                    )
                    .any(dim=-1)
                    .count_nonzero()
                    .item()
                )
                set_differences[name] += different
    lengths.fill_(visible)
    refill()
    times = {name: [] for name in outputs}
    for repeat in range(repeats):
        names = list(graphs) if repeat % 2 == 0 else list(reversed(graphs))
        for name in names:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(steps):
                graphs[name].replay()
            end.record()
            end.synchronize()
            times[name].append(begin.elapsed_time(end) * 1000 / steps / layers)
    return dict(
        rows=rows,
        context=context,
        distribution=distribution,
        layers=layers,
        max_pools=capacity,
        bound_pools=visible,
        scratch_bytes=workspace.numel(),
        exact_selected_values=True,
        changed_length_replays=4,
        selected_set_differences=set_differences,
        us=times,
        median_us={n: statistics.median(v) for n, v in times.items()},
        serving_speedup_claim=False,
        caveat="Synthetic scores; smaller-bound arm is not a 1M serving path.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 131072])
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["normal", "narrow"],
        default=["normal", "narrow"],
    )
    parser.add_argument("--layers", type=int, default=11)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    if min(args.rows + [args.layers, args.repeats, args.steps]) < 1:
        parser.error("positive sizes and iteration counts required")
    if any(c < 4 or c > 1048576 for c in args.contexts):
        parser.error("contexts must fit the retained 1M workspace")
    for distribution in args.distributions:
        for context in args.contexts:
            for rows in args.rows:
                print(
                    json.dumps(
                        measure(
                            rows,
                            context,
                            distribution=distribution,
                            layers=args.layers,
                            repeats=args.repeats,
                            steps=args.steps,
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
