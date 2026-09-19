# B70 Responses correctness follow-up — 2026-09-19

Public port 8000 again belongs to the streaming diagnostic proxy. SlimServe runs
at 127.0.0.1:8001 from the original dirty checkout. Both health endpoints passed
after restart at approximately 22:13 UTC. FP8, TP4, MTP k=3, and the profile's
2,000-token thinking default were preserved.

## Fix boundaries and deployment state

- `aae9efe33`: empty assistant Responses output history no longer indexes
  content[0]. Empty, multipart text/refusal, and merged history have CPU coverage.
  Loaded in the live service. 59 CPU tests passed (20 new, 39 existing).
- `12856ae2a`: generation failures emit typed response.failed SSE with the
  original error and response ID, including background storage. Loaded live.
  47 CPU tests passed (8 new, the same 39 existing).
- `88ebacda1`: Qwen argument conversion waits for tool-call closure. Repeated
  XML names accepted by the non-strict grammar previously overwrote an already
  streamed value; the final-prefix guard silently discarded the suffix, leaving
  invalid JSON. This fix retains last-value semantics and incremental reasoning,
  content, and headers, but defers partial argument previews. 158 CPU tests passed.
  Applied to live source for the restart requested at 22:33:13 UTC.
  Duplicate XML remains a synthetic reproduction; the captured inconsistency
  below is independently established.

The live benign regression replayed an assistant message with content=[] and
requested a required tool call: HTTP200, completed, one JSON-object argument.
This is a regression check, not a throughput qualification. No JSON repair loop
or additional strict-schema enforcement was introduced.

## Captured workload and limits

Existing Evalhub coding run run_1a0bb79a5f8b59406 resumed once into attempt 2 at
22:13:24 UTC, same 125 samples and concurrency 8. Current registry uses code_agent
and code_agent_harness, distinct from the earlier artifacts. The first attempt
was interrupted with infrastructure/model errors and is not a passing baseline.
At the latest snapshot, attempt 2 had 21 successful calls, 12 failed calls, and
no scored tasks. It continues running; consult the local aggregate report.

Eight initial HTTP200 SSE streams ended within 5.13 ms of one another after
approximately 59 seconds. Each contained 111–117 reasoning deltas, with no
visible/tool output or terminal event. A later unterminated stream lasted
226 seconds. This suggests cancellation/deadline behavior, but does not prove
its source. Saved connection/run metadata exposes no per-request timeout.
HTTP200 alone does not imply a successful stream. Later complete responses
have valid JSON and matching delta/done/final arguments in the observed window.
No new HTTP400/500 was observed at this snapshot. This is not full correctness
or throughput qualification. Existing captures lack first-chunk timestamps.

## Local handoff locations

Operational scripts and raw captures remain outside Git:

- ~/qwen38-deploy/traffic-proxy/slimserve_traffic_proxy.py
- ~/qwen38-deploy/traffic-proxy/inspect_assistant_protocol.py
- ~/qwen38-deploy/traffic-proxy/audit_protocol.py
- ~/qwen38-deploy/traffic-proxy/logs/traffic.jsonl
- ~/qwen38-deploy/evalhub-code-agent-monitor/run_1a0bb79a5f8b59406-attempt2-REPORT.md
- ~/qwen38-deploy/evalhub-code-agent-monitor/proxy-attempt2-metrics.jsonl
- ~/qwen38-deploy/evalhub-code-agent-monitor/ATTEMPT2-CONNECTION-NOTES.md
- ~/qwen38-deploy/code-agent-errors-20260919/live-empty-history-regression.json

Analyze assistant history and responses only. The compact inspector emits IDs,
lengths, fingerprints, JSON validity/error offsets, stream event counts, and
argument delta/done/final mismatches; it omits payload text and user/tool bodies.
Use --errors-only for invalid JSON and --unterminated-only for missing terminals.
16 combined inspector/auditor fixtures passed. Proxy forwards unchanged traffic.

```bash
python3 ~/qwen38-deploy/traffic-proxy/inspect_assistant_protocol.py \
  --since 2026-09-19T22:13:24Z --unterminated-only --limit 10
python3 ~/qwen38-deploy/traffic-proxy/audit_protocol.py \
  --since 2026-09-19T22:13:24Z \
  ~/qwen38-deploy/traffic-proxy/logs/traffic.jsonl
```

Monitor candidates: HTTP error count; absent/multiple terminal events; incomplete
and failed responses; invalid final tool JSON; delta/done mismatch; empty assistant
history; request duration; running/waiting requests; generation/prompt token counter
deltas; KV use; preemptions; prefix-cache hits/queries; speculative acceptance;
queue/prefill/decode durations. Treat server interval TPS as diagnostic only.
Separate resumed-attempt counters from inherited usage/sample errors.

## Confirmed live failure and follow-up fix

Later capture inspection proved the HTTP400 chain. A streamed call carried invalid
20-character arguments, with output_item.done correctly marked incomplete and no
arguments.done. The terminal response nevertheless claimed completed, rebuilding
that call with different item/call IDs and valid empty-object arguments. The next
request replayed the original invalid arguments, labeled completed, and received
HTTP400. Thus the stream/final server mismatch is proven independently of the
synthetic repeated-XML test.

All 85 paired tool calls in the initial attempt-2 identity audit regenerated IDs;
three additionally changed incomplete arguments/status into completed calls.
`811839379` makes finalized streamed items authoritative for SimpleContext terminal
output, preserving IDs, arguments, order, status, and stored retrieval. It removes
the second full parse. Invalid/incomplete calls fail on normal stop; token exhaustion
remains incomplete and cancellation remains cancelled. 221 CPU tests passed across
the relevant parser/Responses suite, including real parser-to-SSE parity, multiple
calls, truncation, storage, and cancellation. These counts overlap earlier suites.

Both this fix and Qwen argument buffering were applied to live source. Backend
restart requested 2026-09-19T22:33:13.650769Z; proxy remained on8000. Treat the
restart as a hard measurement boundary, excluding downtime from comparisons and
handling the generation-token counter reset. The initial failed-attempt snapshots
above are historical; consult the updated report for current counts.

No concatenated `{...}{...}` argument bodies were found in the audited window:
98 assembled streams, 90 item-done arguments, and 193 replayed assistant argument
strings (overlapping representations). This is bounded absence evidence, not proof
that the earlier duplication bug can never recur.

Additional local evidence:

- ~/qwen38-deploy/code-agent-errors-20260919/STREAM-FINAL-REPLAY-FINDINGS.md
- ~/qwen38-deploy/code-agent-errors-20260919/replay-json400-correlation.json
- ~/qwen38-deploy/code-agent-errors-20260919/stream-final-identity-attempt2.json
- ~/qwen38-deploy/code-agent-errors-20260919/concatenated-json-audit.json
- ~/qwen38-deploy/code-agent-errors-20260919/canonical-stream-restart.json
- ~/qwen38-deploy/code-agent-errors-20260919/check_stream_replay.py
- ~/qwen38-deploy/traffic-proxy/audit_stream_final_identity.py

The identity auditor and compact inspectors have 19 passing fixture tests. They
print protocol metadata and fingerprints, not payload text.

Next: validate after startup through the same coding load, compare malformed JSON,
identity/status consistency, terminal failures, and exact-token performance. Trace
the shared cancellation boundary if it recurs. Do not restart/relaunch a run merely
because the local agent turn ends. Credentials and payloads stay out of Git.

## Live deployment validation

Backend API PID1156825 reached proxy health200 at 2026-09-19T22:36:35.504300Z.
At22:36:52.590944Z the bounded stream/replay regression passed: exact equality of
all streamed item-done objects and terminal output, one valid completed tool call,
and HTTP200/completed when replaying those same items into the next request.

Attempt2's circuit breaker interrupted it23.23s after the restart request, with
84calls/44successful/40failed and no scored tasks. Those maintenance failures are
not post-fix correctness measurements. Resume was requested only after health and
the live replay regression passed, retaining the existing coding run/artifacts.
The full workload is still required to qualify the fixes and any throughput claim.
