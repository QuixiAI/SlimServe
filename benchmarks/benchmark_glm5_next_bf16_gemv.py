# SPDX-License-Identifier: Apache-2.0
"""C1 projection comparison with16 disjoint matrices, not an L2-only loop.

Numerical gate uses the existing merged-KDA BF16 tolerance (0.008/0.008),
and separately reports bit-exactness. Passing this is not permission to
weaken the mHC transition/projection exact gate or claim model quality.
"""

import argparse
import json

import torch
from triton.testing import do_bench_cudagraph

from benchmarks.glm5_next_bf16_gemv_candidate import gemv
from vllm.quixicore import quixicore_ops


@torch.no_grad()
def measure(width):
    torch.manual_seed(9700 + width)
    x = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    weights = [
        torch.randn(width, 4096, device="cuda", dtype=torch.bfloat16) * 0.01
        for _ in range(16)
    ]
    expected = [
        torch.empty(1, width, device="cuda", dtype=torch.bfloat16) for _ in weights
    ]
    outputs = [torch.empty_like(t) for t in expected]

    def baseline():
        for w, out in zip(weights, expected):
            torch.mm(x, w.T, out=out)
        return expected

    def native():
        return [quixicore_ops.dsv4_projection_gemv(x, w, True) for w in weights]

    arms = {"cublas": baseline, "owned_dsv4": native}
    for bn in (2, 4, 8):
        for bk in (256, 512, 1024):

            def candidate(bn=bn, bk=bk):
                for w, out in zip(weights, outputs):
                    gemv(x, w, out, bn=bn, bk=bk)
                return outputs

            arms[f"grouped_n{bn}_k{bk}"] = candidate
    records = {}
    for name, arm in arms.items():
        baseline()
        arm()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = arm()
        maximum, changed, count = 0.0, 0, 0
        for _ in range(8):
            x.normal_()
            graph.replay()
            baseline()
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0.008, rtol=0.008)
                maximum = max(maximum, (a.float() - b.float()).abs().max().item())
                changed += (a != b).count_nonzero().item()
                count += a.numel()
        records[name] = dict(
            max_abs_error=maximum,
            changed_values=changed,
            compared_values=count,
            bit_exact=changed == 0,
            changed_input_replays=8,
            us=[],
        )
        del graph
    for repeat in range(5):
        order = list(arms) if repeat % 2 == 0 else list(reversed(arms))
        for name in order:
            records[name]["us"].append(
                1000 * do_bench_cudagraph(arms[name], rep=100) / len(weights)
            )
    return dict(
        width=width,
        k=4096,
        tokens=1,
        disjoint_matrices=len(weights),
        weight_bytes=sum(w.numel() * w.element_size() for w in weights),
        arms=records,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=[3336, 6416])
    args = parser.parse_args()
    for width in args.widths:
        print(json.dumps(measure(width)), flush=True)
