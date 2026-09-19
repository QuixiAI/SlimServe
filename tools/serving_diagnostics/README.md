# Serving capture proxy

This optional proxy forwards OpenAI-compatible HTTP and SSE traffic unchanged
and captures `/v1/responses` and `/v1/chat/completions` for protocol debugging.
Start your registered SlimServe profile on `127.0.0.1:8001`, then run:

```bash
python -m uvicorn slimserve_traffic_proxy:app \
  --app-dir tools/serving_diagnostics --host 127.0.0.1 --port 8000
```

The proxy uses `httpx`, `anyio`, `starlette`, and `uvicorn` from the serving
environment. `PROXY_UPSTREAM` overrides `http://127.0.0.1:8001`.
`PROXY_LOG_DIR` overrides `~/.cache/slimserve/traffic-proxy`.

Response capture is limited to 32 MiB by `PROXY_CAPTURE_LIMIT_BYTES`; forwarding
continues after that limit. `PROXY_LOG_MAX_BYTES` defaults to 256 MiB per file
and `PROXY_LOG_BACKUPS` to 8. Request bodies are buffered in full. JSONL records
carry HTTP status, total elapsed time, headers, and request/response bodies.
There are no per-chunk timestamps; HTTP 200 does not prove successful generation.
A missing terminal event can reflect cancellation or a transport/model failure.

Captures include user messages, tool results, and generated code. Selected
header/JSON-key redaction is not a complete secret scrubber, and raw SSE is
retained. Keep logs private and out of version control. No credentials, service
installation, model files, or traffic captures are included here.

Run the CPU tests with:

```bash
python -m unittest discover -s tools/serving_diagnostics -p 'test_*.py' -q
```
