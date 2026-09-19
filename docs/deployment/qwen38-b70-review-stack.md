# B70 forward-port review map

Source: `feat/b70-qwen38-serving` at `db179333c`, with saved base `c6d985940`.
The 17 source commits have been split into focused single-issue PR branches.
Each branch below adds one signed commit relative to its listed base.
Independent changes use main at `d79030d5a`; stacks show only their own issue.
The historical integration branch is preserved and should not be merged wholesale.

## Focused branches

| PR | Change | Base | Branch |
| --- | --- | --- | --- |
| [#43](https://github.com/QuixiAI/SlimServe/pull/43) | Preserve empty and multipart Responses assistant history | `main` | `fix/responses-assistant-history` |
| [#44](https://github.com/QuixiAI/SlimServe/pull/44) | Count Qwen reasoning opened by the chat template | `main` | `fix/qwen-reasoning-usage` |
| [#45](https://github.com/QuixiAI/SlimServe/pull/45) | Emit stable Qwen tool arguments when each call closes | `main` | `fix/qwen-streamed-arguments` |
| [#46](https://github.com/QuixiAI/SlimServe/pull/46) | Stop speculative grammar validation at the stop token | `main` | `fix/xgrammar-terminated-drafts` |
| [#47](https://github.com/QuixiAI/SlimServe/pull/47) | Stop advancing rejected drafts across reasoning boundaries | `main` | `fix/structured-reasoning-draft-path` |
| [#48](https://github.com/QuixiAI/SlimServe/pull/48) | Constrain non-strict tool calls to valid JSON objects | `main` | `fix/tool-json-grammar` |
| [#49](https://github.com/QuixiAI/SlimServe/pull/49) | Reject conflicting assistant schemas before installing tool grammar | [#48](https://github.com/QuixiAI/SlimServe/pull/48) | `fix/tool-output-schema-conflicts` |
| [#50](https://github.com/QuixiAI/SlimServe/pull/50) | Report interrupted Responses tool calls as incomplete | `main` | `fix/responses-incomplete-tools` |
| [#51](https://github.com/QuixiAI/SlimServe/pull/51) | Emit typed Responses failure events for generation errors | [#50](https://github.com/QuixiAI/SlimServe/pull/50) | `fix/responses-generation-errors` |
| [#52](https://github.com/QuixiAI/SlimServe/pull/52) | Prevent required tool requests from stopping during reasoning | [#51](https://github.com/QuixiAI/SlimServe/pull/51) | `fix/required-tool-reasoning-stop` |
| [#53](https://github.com/QuixiAI/SlimServe/pull/53) | Preserve streamed item identity in terminal Responses output | [#52](https://github.com/QuixiAI/SlimServe/pull/52) | `fix/responses-stream-output-identity` |
| [#54](https://github.com/QuixiAI/SlimServe/pull/54) | Honor thinking budgets in Responses sampling parameters | `main` | `fix/responses-thinking-budget` |
| [#55](https://github.com/QuixiAI/SlimServe/pull/55) | Restore grouped GDN head layout when loading Qwen GGUF | `main` | `fix/qwen35-gguf-gdn-layout` |
| [#56](https://github.com/QuixiAI/SlimServe/pull/56) | Load embedded Qwen GGUF MTP heads without mutating the target | [#55](https://github.com/QuixiAI/SlimServe/pull/55) | `fix/qwen35-gguf-mtp` |
| [#57](https://github.com/QuixiAI/SlimServe/pull/57) | Bound semantic checkpoint copies before rendering history | [#34](https://github.com/QuixiAI/SlimServe/pull/34) | `fix/semantic-boundary-copy-cost` |
| [#58](https://github.com/QuixiAI/SlimServe/pull/58) | Mask unwritten tensor-descriptor tails before attention accumulation | `main` | `fix/triton-attention-ragged-tail` |
| [#59](https://github.com/QuixiAI/SlimServe/pull/59) | Allow XPU deployments to tune Triton softmax segment count | `main` | `feat/xpu-attention-softmax-segments` |
| [#60](https://github.com/QuixiAI/SlimServe/pull/60) | Authenticate registered Hugging Face downloads without losing resume headers | `main` | `fix/huggingface-download-auth` |
| [#61](https://github.com/QuixiAI/SlimServe/pull/61) | Add a Qwen tool-policy chat template with native thinking tokens | `main` | `feat/qwen38-tool-template` |
| [#62](https://github.com/QuixiAI/SlimServe/pull/62) | Reuse split Q8 weights in XPU decode kernels | [#21](https://github.com/QuixiAI/SlimServe/pull/21) | `perf/xpu-split-q8-decode` |
| [#63](https://github.com/QuixiAI/SlimServe/pull/63) | Quiesce XPU workers before coordinated process teardown | [#21](https://github.com/QuixiAI/SlimServe/pull/21) | `fix/xpu-coordinated-shutdown` |
| [#64](https://github.com/QuixiAI/SlimServe/pull/64) | Limit HSA host-staging probes to ROCm workers | `main` | `fix/rocm-only-host-staging` |
| [#65](https://github.com/QuixiAI/SlimServe/pull/65) | Detect Intel Arc Pro B70 without initializing a GPU context | `main` | `feat/intel-b70-detection` |
| [#66](https://github.com/QuixiAI/SlimServe/pull/66) | Register a gated Qwen3.8 FP8 recipe for four B70 GPUs | [#65](https://github.com/QuixiAI/SlimServe/pull/65) | `feat/qwen38-b70-fp8-profile` |
| [#67](https://github.com/QuixiAI/SlimServe/pull/67) | Register a gated Qwen3.8-27B Q8_K_P recipe for B70 | [#66](https://github.com/QuixiAI/SlimServe/pull/66) | `feat/qwen38-b70-q8-profile` |
| [#68](https://github.com/QuixiAI/SlimServe/pull/68) | Register a gated Qwen3.8-27B BF16 recipe for B70 | [#67](https://github.com/QuixiAI/SlimServe/pull/67) | `feat/qwen38-b70-bf16-profile` |
| [#69](https://github.com/QuixiAI/SlimServe/pull/69) | Add an optional streaming traffic capture proxy | `main` | `feat/serving-capture-proxy` |
| [#70](https://github.com/QuixiAI/SlimServe/pull/70) | Add read-only auditors for captured serving protocol traffic | [#69](https://github.com/QuixiAI/SlimServe/pull/69) | `feat/serving-protocol-auditors` |
| [#71](https://github.com/QuixiAI/SlimServe/pull/71) | Add a multi-turn Responses tool protocol benchmark | `main` | `bench/qwen-tool-protocol` |

## Dependencies and qualification

- The GGUF MTP fix follows the grouped-GDN loader fix. Both retain main's vision
  and GPTQ support; CPU layout tests do not establish live model parity.
- Native Q8 and XPU shutdown are sibling changes above existing PR #21. That
  historical foundation needs integration with current main before either can
  merge; preserve main's current reasoning-budget and multimodal fixes.
- Semantic boundary copying follows PR #34, which carries the complete
  renderer/IPC/scheduler checkpoint-hint pipeline.
- Responses completion, failure events, required-tool reasoning, and streamed
  item identity form one stack. Schema-conflict handling follows tool grammar.
  Assistant history, reasoning accounting and Qwen argument buffering are
  separate fixes that also belong in deployment qualification.
- Recipes form a registry stack after B70 hardware detection. FP8 also needs
  the Qwen template, authenticated downloads, attention settings and serving
  protocol fixes; Q8 needs the GGUF/MTP and native kernel changes. All recipes
  need the integrated XPU foundation and coordinated teardown.
- The capture proxy and its offline analyzers form their own stack. The protocol
  benchmark is independent. Neither includes traffic captures or credentials.

All recipes stay gated `in-progress`. Native/GPU and profile PRs remain draft
pending hardware validation. The available host has an Apple GPU, no B70/XPU or
SYCL build environment, and no replacement deployment was requested. Historical
throughput and failures are documented in [the deployment guide](qwen38-b70.md).

## Source coverage and current-main differences

- Main already includes template asset packaging, registry selection and
  authoritative request metadata forwarding; those duplicate source hunks are
  omitted. Safetensors use main's source `format`, replacing the old per-quant
  `directory` flag; profiles use per-platform variants and current profile IDs.
- Main's thinking-budget resolver supersedes the source Chat Completions helper.
  Responses now uses that resolver too. Explicit `-1` disables the profile default;
  `null` inherits it. Both sampler regressions for an MTP proposal with a natural
  in-budget closing marker pass; the older duplicate-`</think>` behavior is not
  imported. The full CPU budget suite hits a baseline Torch pinned-allocation
  failure on the available host, so those two targeted tests are the verified scope.
- The semantic transport/request/scheduler hunks already belong to #34. Only
  bounded boundary selection/copying is new here.
- Source GGUF tests are split into loader-layout and embedded-MTP coverage.
  The obsolete text-only model assertion and proposer fallback are omitted:
  current main supports the vision model interface.
- The source's graph-replayer comment-only correction is omitted from this
  functional series. The separate UR/XPTI tracing workaround is in the saved
  base, outside the 17-commit source range; it is an explicit deployment
  prerequisite to review separately, not silently included in a recipe.
- Historical deployment notes are consolidated into this guide and the protocol
  findings. Per-issue test evidence is recorded in the PRs and performance notebook;
  old overlapping test counts are not presented as fresh results.

## Local validation limits

All 26 current-main branches applied together in an isolated checkout: 295
regression tests passed and eight XPU numerical cases skipped. The proxy/auditor
suite passed another 22 fixtures. The #21 native siblings and #34 cache refinement
were validated separately against their stated bases.

The forward port used Python 3.14.7, Torch 2.14.0, XGrammar 0.2.7 and
compressed-tensors 0.18.0 (the repository pins 0.17.0). Four existing GLM custom-tool
cases fail identically on untouched main with this XGrammar environment. Native
SYCL compilation, numerical GPU parity, actual model restore, text/image canaries
and sustained serving remain required on the deployment machine. Mocked kernel
and worker tests verify layout/dispatch and ordering contracts only.
