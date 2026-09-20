# B70 current-main PR integration

Base: main d79030d5ac79564f4d99b0b161daf8cc20d162b6.
Branch: integration/b70-pr-validation-20260920.

Selected PRs:35,43–54,59–61,63–66, plus a selective current-main
forward port of the XPU foundation from21. The original21 branch history is
not merged. Q8, BF16, other-hardware campaigns and semantic-cache changes are
not included. Each selected change has a separate provenance commit.

Local follow-ups preserve grammar masks at the thinking cap, coalesce trailing
empty assistant history, verify SSE routing, allow explicit qualification of
in-progress profiles, and avoid eager unrelated DeepSeek device-model imports
during Qwen FP8 configuration. Current-main natural reasoning-close and Qwen
vision/Mamba implementations are retained.

## Verification before sustained evaluation

- Native build:81 SYCL translation units and the pybind,spinloop,fs_io modules.
- Integrated CPU suite:426 passed; separate native-foundation/shutdown/staging
  and lazy-import tests passed during forward port.
- Actual engine configuration passed:TP4,native FP8,FP8 KV,FULL_DECODE_ONLY,
  MTP k=3,262144 context,32 sequences,16384 batched tokens,2K reasoning default.
- Cold boot ready2026-09-20 00:58:35UTC.
- Three live Responses checks passed:valid tool JSON,stream/final equality,
  budget boundary and empty-assistant history.
- Vision check passed:solid red image answered Red with finish_reason=stop.
- Real coding Evalhub run run_1a0bb79a5f8b59406 is the sustained qualification gate.

These checks are not a declaration that all selected PRs are ready to merge.
The recipe remains in-progress; --allow-in-progress explicitly enables this
qualification deployment while default gating and hardware checks remain intact.

## Runtime and rollback

Service slimserve-abliterated uses this checkout and its isolated Python entry
point. Dependencies are shared read-only from the prior environment; old editable
import finders are not loaded. Fresh native binaries are built for Torch
2.15.0.dev20260815+xpu; paths and SHA256 receipts are kept privately on the host.
The public capture proxy remains on8000; the backend remains127.0.0.1:8001.
Remove only the90-tested-main-stack.conf user-service drop-in,daemon-reload and
restart to return to the preserved prior checkout. No weights were changed.

Attempt7 on the old stack failed with HTTP200 body-read errors. Cloudflare QUIC
connection timeouts preceded the supplied error and match local cancellation
timing. This identifies the failed transport; it does not establish whether
the root cause is host connectivity, the network path, or Cloudflare. A proposed
HTTP2 transport change is not applied at the start of this evaluation. Older
whole-engine sample_tokens stalls are separate and remain under watchdog capture.

## Commit order

```text
660919274f Send SSE keepalive comments during idle streams
6a38e3b560 Preserve empty and multipart Responses assistant history
06f3fb8779 Count Qwen reasoning opened by the chat template
abe0f07f5a Emit stable Qwen tool arguments when each call closes
65c2739800 Stop speculative grammar validation at the stop token
96e791cea3 Stop advancing rejected drafts across reasoning boundaries
7fbfa6d073 Constrain non-strict tool calls to valid JSON objects
1308f8afce Reject conflicting assistant schemas before installing tool grammar
48958f9f36 Report interrupted Responses tool calls as incomplete
c8e6bbd07b Emit typed Responses failure events for generation errors
bbfe24ddde Prevent required tool requests from stopping during reasoning
54c995b20c Preserve streamed item identity in terminal Responses output
5ba202e989 Honor thinking budgets in Responses sampling parameters
1ca1a42083 Add a Qwen tool-policy chat template with native thinking tokens
47a99d2e1f Respect grammar masks and speculative reasoning boundaries at budget limits
b83cf06732 Keep empty Responses items attached to the assistant tool turn
8113703292 Verify existing SSE keepalive integration with load tracking disabled
22b5cd20d5 Allow XPU deployments to tune Triton softmax segment count
4bff91a721 Authenticate registered Hugging Face downloads without losing resume headers
b9b7bebb45 Detect Intel Arc Pro B70 without initializing a GPU context
010f2e2277 Register a gated Qwen3.8 FP8 recipe for four B70 GPUs
326549f061 Restore Intel XPU serving foundation on current main
2826b26e62 Quiesce XPU workers before coordinated process teardown
1676043c1d Limit HSA host-staging probes to ROCm workers
0c8469f2d0 Allow explicit qualification runs for in-progress profiles
cf149880cd Load DeepSeek model implementations only when their classes are requested
```

## Sustained-load status: 2026-09-20

Attempt 8 remains an unqualified run. Early intervals with eight active
requests were close to attempt 7 (158.51 versus 161.54 aggregate generation
tok/s), but those intervals do not contain identical prompts. By 03:29 UTC,
32 requests were running, 11 queued, KV use was 93.23%, and fresh draft
acceptance was 10.26%. That 30-second interval generated 86.80 tokens/s.

Proxy metadata showed long overlapping copies of identical request bodies;
requests without max_output_tokens received budgets near 250,000 tokens.
The thinking limit bounds reasoning only. Establish which retry owner retains
earlier connections and set an intentional completion budget before using
this saturated run to compare kernels. A token ceiling is a resource bound,
not a fix for pathological generation or a guarantee of complete tool JSON.
The existing override_generation_config.max_new_tokens also limits explicit
client budgets and must not be presented as an overridable default.

Disconnect cleanup has separately failed isolated ASGI tests. Iterator
ownership must reach the engine generation iterator, with bounded shielding
for abort cleanup, while background Responses remain independently owned.
This defect is not needed to explain the unfinished-request backlog.

No output cap, speculative setting, or live process was changed during this
slowdown investigation. See the performance notebook and private host
receipts under qwen38-deploy/pr-integration-20260920/throughput-review/;
retry/cancellation API findings are in attempt8-RETRY-CANCELLATION-BUDGET.md.

Cancellation follow-up `d7517c031b` is committed but not deployed in this
measurement. It adds response-owned, shielded and bounded cleanup plus
explicit nested Responses generator ownership. The route-to-AsyncLLM tests
use a mocked engine transport, retain generators to exclude GC cleanup,
and verify awaited abort, task cleanup, and preserved background ownership.
The protocol regression suite passed 344 tests; Ruff and diff checks passed.
This is CPU correctness evidence, not sustained live validation.
