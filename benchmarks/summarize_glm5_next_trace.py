# SPDX-License-Identifier: Apache-2.0
"""CPU-only, stream-aware census of one complete GLM decode trace region.

Kernel sums on different streams overlap; never add them as wall latency.
The mHC markers identify sites on the main stream. This is profiling
attribution, not an exact-token throughput benchmark or speedup prediction.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(trace, config, tokens=1):
    events = json.loads(trace.read_text())["traceEvents"]
    cfg = json.loads(config.read_text())
    return summarize_events(events, cfg, str(trace), tokens=tokens)


def summarize_events(events, cfg, trace_label="synthetic", tokens=1):
    if tokens < 1:
        raise ValueError("tokens must be positive")
    cfg = cfg.get("text_config", cfg)
    annotations = defaultdict(list)
    for event in events:
        if event.get("cat") == "gpu_user_annotation" and event["name"].startswith(
            f"execute_{tokens}_context"
        ):
            annotations[event["name"]].append(event)
    # Some regions have nested annotations with the same name. Keep the
    # outer span, then choose the latest region with the requested token count.
    regions = [max(group, key=lambda e: e["dur"]) for group in annotations.values()]
    if not regions:
        raise ValueError(f"No {tokens}-token decode GPU annotation found")
    region = max(regions, key=lambda e: e["ts"])
    start, end = region["ts"], region["ts"] + region["dur"]
    kernels = sorted(
        (
            e
            for e in events
            if e.get("cat") == "kernel"
            and e["pid"] == region["pid"]
            and start <= e["ts"] < end
        ),
        key=lambda e: e["ts"],
    )
    main = [e for e in kernels if e["args"]["stream"] == region["tid"]]
    marks = [i for i, e in enumerate(main) if e["name"] == "_mhc_partials"]
    layer_types = cfg["mlp_layer_types"]
    expected_marks = 2 * len(layer_types)
    total_marks = sum(e["name"] == "_mhc_partials" for e in kernels)
    if total_marks != expected_marks:
        raise ValueError(f"Expected {expected_marks} mHC sites, got {total_marks}")
    # CUDA graph replay may schedule even dependent main-stream nodes on
    # internal streams. A time-ordered list on one stream no longer maps to
    # model layers. Keep the complete multi-stream census, but decline site
    # attribution until graph dependencies/correlation can establish it.
    single_stream_sites = len(marks) == expected_marks
    if not single_stream_sites:
        marks = []
    full = set(cfg["linear_attn_config"]["full_attn_layers"])
    sites = defaultdict(
        lambda: dict(count=0, main_kernel_us=0.0, mhc_us=0.0, kernels=0)
    )
    examples = {}
    for site, offset in enumerate(marks):
        segment = main[offset : marks[site + 1] if site + 1 < len(marks) else len(main)]
        layer = site // 2
        kind = (
            ("MLA" if layer in full else "KDA")
            if site % 2 == 0
            else ("MoE" if layer_types[layer] == "sparse" else "dense_MLP")
        )
        record = sites[kind]
        record["count"] += 1
        record["main_kernel_us"] += sum(e["dur"] for e in segment)
        record["mhc_us"] += sum(
            e["dur"] for e in segment if e["name"] in ("_mhc_partials", "_mhc_finalize")
        )
        record["kernels"] += len(segment)
        if kind not in examples:
            examples[kind] = [
                dict(name=e["name"], us=e["dur"], grid=e["args"].get("grid"))
                for e in segment
            ]
    streams = defaultdict(lambda: dict(kernels=0, kernel_us=0.0))
    totals = Counter()
    counts = Counter()
    all_totals = Counter()
    all_counts = Counter()
    busy_union = 0.0
    busy_end = start
    for event in kernels:
        event_end = min(end, event["ts"] + event["dur"])
        busy_union += max(0.0, event_end - max(busy_end, event["ts"]))
        busy_end = max(busy_end, event_end)
        stream = streams[event["args"]["stream"]]
        stream["kernels"] += 1
        stream["kernel_us"] += event["dur"]
        all_totals[event["name"]] += event["dur"]
        all_counts[event["name"]] += 1
        if event["args"]["stream"] == region["tid"]:
            totals[event["name"]] += event["dur"]
            counts[event["name"]] += 1
    return dict(
        trace=trace_label,
        tokens=tokens,
        region=region["name"],
        region_us=region["dur"],
        main_stream=region["tid"],
        streams=dict(streams),
        total_mhc_markers=total_marks,
        site_attribution=(
            "single_stream"
            if single_stream_sites
            else "unavailable: graph nodes span multiple streams"
        ),
        kernel_busy_union_us=busy_union,
        kernel_sum_us=sum(all_totals.values()),
        sites=dict(sites),
        examples=examples,
        main_kernels=[
            dict(name=name, us=us, calls=counts[name])
            for name, us in totals.most_common()
        ],
        all_stream_kernels=[
            dict(name=name, us=us, calls=all_counts[name])
            for name, us in all_totals.most_common()
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1)
    args = parser.parse_args()
    result = summarize(args.trace, args.config, tokens=args.tokens)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("region_us", "main_stream", "streams", "sites")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
