"""Render a concurrency sweep produced by concurrent_gate.sh as one table."""

import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
runs = sorted(out.glob("c*.json"), key=lambda p: int(p.stem[1:]))

base = None
print(
    f"{'c':>3} {'rows':>5} {'exact':>6} {'agg tok/s':>10} {'per-req':>8} "
    f"{'x c=1':>6} {'wall s':>7}  sha"
)
for path in runs:
    try:
        d = json.loads(path.read_text())
    except json.JSONDecodeError:
        print(f"{path.stem[1:]:>3}  (no json - see {path.with_suffix('.err').name})")
        continue
    c = d["concurrency"]
    agg = d["aggregate_output_tps"]
    per = d["requested_output_tokens"] / d["request_latency_mean_seconds"]
    if base is None:
        base = agg
    shas = d.get("response_sha256") or []
    # K=1 MTP: every request contributes a draft row and a verify row.
    print(
        f"{c:>3} {2 * c:>5} {str(d['exact']):>6} {agg:>10.2f} {per:>8.2f} "
        f"{agg / base:>5.2f}x {d['wall_seconds']:>7.1f}  {shas[0][:12] if shas else '?'}"
    )
