"""Concurrent gate check for glm53f-q2-1.

Runs the exact-token harness at the pinned concurrencies (1000-token prompts,
1000 output tokens - decode-dominated) against a live server and compares
with pins.json in the same directory:
  - `exact` must be true at every concurrency (every request returned the
    requested prompt and completion counts);
  - aggregate tok/s must be >= (1 - tolerance) x the pinned value;
  - per-request shas are RECORDED, not pinned: at c >= 4 the per-step row
    count, hence the dense route, depends on arrival timing, so the same
    prompt legitimately produces different tokens run to run. The c=1 pins
    in perf/results/2026-09-11/glm53f-q2-baseline/gate.sh remain the
    bit-exact contract.
Usage: concurrent_gate_pin.py <outdir> [--pin]   (--pin rewrites pins.json)
"""

import json
import pathlib
import subprocess
import sys

WT = pathlib.Path("/Users/seangherardi/Code/slimserve/SlimServe-glm53f")
PY = "/Users/seangherardi/Code/slimserve/SlimServe/.venv/bin/python"
GGUF = (
    pathlib.Path.home() / "models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf"
)
HERE = pathlib.Path(__file__).resolve().parent
CONC = (4, 8, 16)
TOL = 0.05

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
pin = "--pin" in sys.argv[2:]
pins_path = HERE / "pins.json"
pins = json.loads(pins_path.read_text()) if pins_path.exists() else {}

results = {}
for c in CONC:
    args = [
        PY, "benchmarks/benchmark_dsv4_exact.py", "--model", str(GGUF),
        "--served-model-name", "GLM-5.3-Flash", "--source",
        "perf/results/harness_assets/m2_source.txt", "--url",
        "http://127.0.0.1:8000/v1/completions", "--metrics-url", "none",
        "--allow-no-spec", "--temperature", "0", "--warmup-output-tokens", "1",
        "--concurrency", str(c), "--input-tokens", "1000", "--output-tokens",
        "1000", "--prompt-offset", "1",
    ]
    proc = subprocess.run(
        args,
        cwd=WT,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(WT), "PATH": "/usr/bin:/bin"},
    )
    (out / f"c{c}.err").write_text(proc.stderr)
    try:
        d = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f"c={c}: harness produced no json (see c{c}.err)")
        results[c] = None
        continue
    (out / f"c{c}.json").write_text(json.dumps(d, indent=2, sort_keys=True))
    per = d["requested_output_tokens"] / d["request_latency_mean_seconds"]
    results[c] = {
        "exact": d["exact"],
        "aggregate_output_tps": d["aggregate_output_tps"],
        "per_request_tps": per,
        "shas": [s[:12] for s in (d.get("response_sha256") or [])],
    }
    print(
        f"c={c:2d}: exact={d['exact']} agg {d['aggregate_output_tps']:.2f} "
        f"per-req {per:.2f}"
    )

ok = True
for c, r in results.items():
    if r is None or not r["exact"]:
        ok = False
        print(f"c={c}: FAIL (exact)")
        continue
    p = pins.get(str(c))
    if p and r["aggregate_output_tps"] < (1 - TOL) * p["aggregate_output_tps"]:
        ok = False
        print(
            f"c={c}: FAIL throughput {r['aggregate_output_tps']:.2f} < "
            f"{(1 - TOL) * p['aggregate_output_tps']:.2f} "
            f"(pin {p['aggregate_output_tps']:.2f})"
        )
    elif p:
        print(f"c={c}: ok vs pin {p['aggregate_output_tps']:.2f}")
(out / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True))
if pin and ok:
    pins_path.write_text(
        json.dumps({str(c): r for c, r in results.items()}, indent=2, sort_keys=True)
    )
    print(f"pins written to {pins_path}")
print("CONCURRENT GATE", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
