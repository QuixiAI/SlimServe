# Serving diagnostics

Optional streaming capture proxy and read-only protocol analyzers for SlimServe.
The tools are independent of the model profile. They require Python; the proxy
also uses `httpx`, `starlette`, and `uvicorn` from the serving environment.

## Capture proxy in front of SlimServe

Start the registered SlimServe profile on a loopback backend port, then launch
this proxy in another terminal. `PROFILE`, `BACKEND_PORT`, and `PROXY_PORT` are
yours to choose; nothing here assumes a host, port, or service layout:

```bash
slimserve PROFILE --host 127.0.0.1 --port BACKEND_PORT -y
PROXY_UPSTREAM=http://127.0.0.1:BACKEND_PORT \
  python -m uvicorn slimserve_traffic_proxy:app \
  --app-dir tools/serving_diagnostics --host 127.0.0.1 --port PROXY_PORT
```

`PROXY_UPSTREAM` is required; the proxy refuses to start without it.
It records `/v1/responses` and `/v1/chat/completions` in rotating JSONL files.
By default captures go to the git-ignored notebook area,
`perf/results/<UTC date>/traffic-proxy/traffic.jsonl`; set `PROXY_LOG_DIR` to use
another directory (for example `perf/results/<date>/<run-id>/`). The analyzers
honor the same variable and also accept explicit log paths as positional
arguments. Note that the default directory is dated, so analyze a previous day's
capture by passing its path or `PROXY_LOG_DIR`.

The proxy is passive: it forwards request bodies unchanged and never sets or
rewrites sampling parameters. Any traffic you drive through it for diagnostics
must follow the repository's sampling policy (temperature 1.0, top_p 0.95,
top_k 20, seeded; thinking enabled; never temperature 0).

Capture controls: `PROXY_CAPTURE_LIMIT_BYTES` defaults to 32 MiB per response,
`PROXY_LOG_MAX_BYTES` to 256 MiB per file, and `PROXY_LOG_BACKUPS` to 8. Reaching
the capture limit does not stop forwarding. This directory contains no service
installation, credentials, captures, or machine-specific configuration, and
none may be added here: deployment configuration stays on operator machines.

## Compact analysis

Run from the repository root:

```bash
# Aggregate HTTP, JSON, completion, and stream-consistency counters.
python tools/serving_diagnostics/audit_protocol.py --since '2026-09-19T22:13:24Z'

# Only assistant history and response metadata; never display message/code text.
python tools/serving_diagnostics/inspect_assistant_protocol.py --errors-only --limit 5
python tools/serving_diagnostics/inspect_assistant_protocol.py --request-id REQUEST_ID

# HTTP 200 can still end without a terminal response event.
python tools/serving_diagnostics/inspect_assistant_protocol.py --unterminated-only --limit 5

# Compare streamed item.done calls against the terminal response's function calls.
python tools/serving_diagnostics/audit_stream_final_identity.py
```

The inspector reports IDs, item positions, text/argument lengths and SHA256
fingerprints, JSON validity/error offsets, statuses, and differences between
streamed argument deltas, done events, and final calls. It excludes user/tool
messages from displayed diagnostics. JSON validation checks syntax, not schemas,
and does not repair arguments. The identity auditor pairs function calls by
output order only when counts and function-name order agree; it distinguishes
changed IDs from changed argument strings. Detailed stream comparisons target
the Responses API; the aggregate auditor has limited Chat Completions coverage.

`--since` accepts Unix seconds or ISO-8601; naive ISO values mean UTC. The first
two analyzers also support an exclusive `--until`. The inspector's `--limit`
bounds displayed finished records. Pass rotated log files explicitly when
needed; default commands read only the current file.

## Capture limits and interpretation

The proxy forwards raw upstream chunks without parsing or rewriting SSE. It
requests identity encoding and preserves upstream content headers. HTTP status
is sent before generation finishes, so HTTP 200 does not prove a completed
model response. A background capture record may follow a downstream disconnect;
missing terminal events can therefore mean cancellation or a transport/model
failure. The existing records do not distinguish all of those causes.

New captures record upstream header, first/last chunk, observed downstream
disconnect, and send-error timings as monotonic milliseconds from request start.
They also record upstream received and downstream successfully sent chunk/byte
counts, upstream state, and a termination reason. Successful ASGI sends do not
prove client receipt. First transport bytes are not necessarily first model tokens.
Older captures lack these timings; null means unavailable. Aggregate elapsed time
sums request durations and is not benchmark wall time. Captures can be truncated
at the configured byte limit; analyzers flag this separately.

Capture finalization runs once even after stream errors or cancellation. Cleanup
is shielded with a bounded 10-second explicit upstream close; close failures are
recorded. This lifecycle uses asyncio, matching Uvicorn. The proxy
preserves raw response bytes and duplicate response headers; it adds no heartbeat
or synthetic model events. A disconnected, unfinished request body is recorded
as client-abandoned input (499), not a model-generation failure.

Capture files contain request/response bodies, including user messages, tool
results, and generated code. Header and selected JSON-key redaction are not a
complete secret scrubber, and raw SSE is retained. Keep captures private and out
of version control. The compact analyzers display fingerprints and structure
instead of those payloads.

## CPU validation

```bash
python -m unittest discover -s tools/serving_diagnostics -p 'test_*.py' -q
```

The 19 analyzer fixtures cover JSON syntax, empty assistant history, payload-free
display, truncation, missing terminal events, and streamed/final call mismatches.

Eleven additional ASGI lifecycle fixtures cover EOF, byte/header fidelity, upstream
errors, downstream disconnect/send failure, cancellation, and cleanup.

## Token-progress watchdog

`watch_progress.py` observes loopback metrics every five seconds. With requests
running, 45 seconds without prompt or generation counter progress triggers one
read-only snapshot per stall. Recovery, idle periods, and counter/creation-epoch
resets rearm detection. Missing metrics are reported separately.

```bash
python tools/serving_diagnostics/watch_progress.py \
  --metrics-url http://127.0.0.1:BACKEND_PORT/metrics \
  --pid SLIMSERVE_PID --duration 3600 \
  --py-spy "$(command -v py-spy)"
```

`--metrics-url` is required and must be plain HTTP on loopback. Output defaults
to `perf/results/<UTC date>/progress-watchdog/progress.jsonl` (mode 0600); pass
`--output` to choose a run directory. If the server runs under a systemd user
unit, `--service-unit UNIT` and `--journal-unit UNIT` can replace `--pid`; the
unit itself is operator configuration and is not part of this repository.

With `--service-unit`, the user service MainPID is discovered anew at each
stall. Alternatively, use
`--pid PID` for a fixed process-tree root. Optional py-spy captures native/Python
frames without locals, prioritizing workers before engine/API helpers, up to 12
processes. Explicit `--sudo-py-spy` enables `sudo -n` only for stack capture when
ptrace permissions require it. Process arguments, environment, journal payloads,
and model text are excluded. Output files have mode 0600. No restart or GPU reset
is performed.

Capturing stacks can briefly pause each target. Snapshots run inline, so sampling
pauses during bounded diagnostic subprocesses; each stack attempt has an eight
second limit, subject to the overall one-hour run limit. Start under a service
manager to survive the invoking terminal. The detector is diagnostic, not proof
of a deadlock: unusually long prefill can also cross the threshold.

## Worker phase records

For multiprocess (`MultiprocExecutor`) workers, setting
`VLLM_WORKER_PROGRESS_DIR=DIR` before the server starts makes each worker keep a
fixed-size, payload-free ring of host phase entry/exit records (RPC execute /
sample / draft-ids, sampling, prior-output and prior-draft waits, MTP drafting,
draft copy, async output) in `DIR/worker-<pid>.bin`. It never queries the
device, stores no RPC names, arguments, or model data, and is fully inert when
the variable is unset. Read a snapshot while the server is stalled with:

```bash
python vllm/utils/worker_progress.py --directory DIR --pids WORKER_PID [...]
```

Records mark host entry/exit, not GPU completion.
