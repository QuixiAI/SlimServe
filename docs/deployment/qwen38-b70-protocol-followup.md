# B70 Responses protocol findings

These findings were recorded on 2026-09-19 against the saved B70 integration
checkout. They motivate the focused forward ports in
[the review map](qwen38-b70-review-stack.md); they are not current-main live results.

Empty assistant output history (`content=[]`) previously caused indexing errors.
The history fix preserves empty, multipart text and refusal messages, including
custom-tool history supported by current main. Typed generation-error events
retain the original error and response identity in both SSE and stored responses.

The Qwen parser could emit an argument prefix before a repeated XML parameter
replaced its value. That synthetic reproduction explains why arguments now wait
until the call closes, while reasoning, ordinary content and tool headers remain
incremental. It does not prove every observed malformed argument had that cause.

A separate capture established a stream/final mismatch: `output_item.done` carried
invalid 20-character arguments and an incomplete status without an arguments-done
event, while the terminal response rebuilt the call with different IDs, `{}`
arguments and completed status. Replaying the streamed malformed arguments then
received HTTP 400. All 85 paired calls in the initial audit regenerated IDs;
three also changed incomplete arguments/status into completed calls.

The fix retains finalized streamed items as terminal output, including their IDs,
call IDs, arguments, order and status. Normal-stop invalid calls fail the response;
token exhaustion remains incomplete and cancellation remains cancelled. It does
not repair malformed JSON or silently enforce non-strict parameter schemas.

A historical bounded replay after restart passed exact item-done/final equality
and successfully replayed the same call into the next request. A later snapshot
contained 34 completed Responses and 63 paired calls with no identity, argument
or status mismatch and no HTTP errors. Four streams lacked terminal events and
one failed. Full coding evaluation and sustained stability were still unqualified.

Eight earlier HTTP 200 streams ended within 5.13 ms of one another, after about
59 seconds, without visible/tool output or a terminal event. Another lasted 226
seconds. This suggests a shared cancellation/deadline boundary; the available
captures did not establish its source. Captures lacked first-chunk timestamps,
so event counts cannot reconstruct TTFT. Maintenance/restart failures must be
separated from post-fix correctness measurements.

Use the optional [protocol analyzers](../../tools/serving_diagnostics/README.md)
to compare fresh captures. They report metadata, lengths, fingerprints, statuses
and JSON error offsets without displaying message/code payloads. Captures remain
private and outside Git. Missing terminals, failed/incomplete calls and exact
streamed/final equality remain acceptance checks for the next sustained run.
