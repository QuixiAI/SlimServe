# Serving diagnostics

Optional streaming capture proxy and read-only protocol analyzers for SlimServe.
The tools are independent of the model profile. They require Python; the proxy
also uses `httpx`, `starlette`, and `uvicorn` from the serving environment.

## Proxy on port 8000, SlimServe on port 8001

Start the registered SlimServe profile on a loopback backend port, then launch
this proxy in another terminal. Substitute the profile for your hardware:

```bash
python -m slimserve.cli qwen38-abliterated-b70-4 --serve --host 127.0.0.1 --port 8001 -y
python -m uvicorn slimserve_traffic_proxy:app \
  --app-dir tools/serving_diagnostics --host 0.0.0.0 --port 8000
```

The proxy forwards to `http://127.0.0.1:8001`. Override this with `PROXY_UPSTREAM`.
It records `/v1/responses` and `/v1/chat/completions` in rotating JSONL files.
Defaults are portable: `~/.cache/slimserve/traffic-proxy/traffic.jsonl`; set
`PROXY_LOG_DIR` to use another directory. The analyzers honor the same variable
and also accept explicit log paths as positional arguments.

Capture controls: `PROXY_CAPTURE_LIMIT_BYTES` defaults to 32 MiB per response,
`PROXY_LOG_MAX_BYTES` to 256 MiB per file, and `PROXY_LOG_BACKUPS` to 8. Reaching
the capture limit does not stop forwarding. This directory contains no service
installation, credentials, captures, or machine-specific configuration.

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

Only total request elapsed time is recorded. Per-chunk arrival times and true
first-token latency cannot be reconstructed from event counts or sequence
numbers. `first_chunk_ms: null` means unavailable. Aggregate elapsed time sums
request durations and is not benchmark wall time. Captures can be truncated at
the configured byte limit; the analyzers flag this separately.

Capture files contain request/response bodies, including user messages, tool
results, and generated code. Header and selected JSON-key redaction are not a
complete secret scrubber, and raw SSE is retained. Keep captures private and out
of version control. The compact analyzers display fingerprints and structure
instead of those payloads.

## CPU validation

```bash
python -m unittest discover -s tools/serving_diagnostics -p 'test_*.py' -q
```

The 19 benign fixtures cover JSON syntax, empty assistant history, payload-free
display, truncation, missing terminal events, and streamed/final call mismatches.
