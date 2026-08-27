# Reasoning token budgets

SlimServe can force a reasoning model to emit its native thinking-end token
after a configured number of generated thinking tokens. The server setting is
`thinking_token_budget` inside the generation-config override. It accepts
either a scalar for backward compatibility or a map keyed by
`reasoning_effort`.

For a Qwen3.8 deployment, whose template supports `low`, `medium`, and `xhigh`,
an initial production configuration is:

```bash
--override-generation-config '{
  "thinking_token_budget": {
    "low": 1024,
    "medium": 4096,
    "xhigh": 8192
  }
}'
```

The generic chat request schema recognizes `none`, `minimal`, `low`, `medium`,
`high`, `xhigh`, and `max`, but a model's chat template may support only a
subset. Budget-map keys may be any thinking level from that generic schema;
`none` is not a map key because it disables thinking.

Budget selection follows these rules:

1. A request's explicit numeric `thinking_token_budget` overrides the server
   scalar or map.
2. A request value of `-1` opts out of the server cutoff.
3. Otherwise a map selects the request's `reasoning_effort` entry. If the
   request omits `reasoning_effort`, or the map has no entry for it, SlimServe
   uses `medium`. Every map must therefore define `medium`.
4. A scalar server value applies to every reasoning effort.

The omitted-level `medium` fallback is budget-selection policy; it does not
rewrite the model template's own default. Clients that need the prompt-level
effort instruction and the cutoff label to match exactly should send
`reasoning_effort` explicitly.

For example:

```json
{
  "model": "Qwen3.8-27B",
  "messages": [{"role": "user", "content": "Analyze this failure."}],
  "reasoning_effort": "low",
  "max_tokens": 12000
}
```

Here the selected thinking cutoff is 1024. A request can instead set
`"thinking_token_budget": 6000`, or set it to `-1` for no cutoff.

`max_tokens` remains the total completion ceiling, covering both thinking and
the final answer. The effective cutoff is
`min(selected_thinking_budget, max_tokens - 1)`; the one-token margin is only
for an injected native thinking-end marker. There is no fixed final-answer
reserve: answer capacity is `max_tokens` minus the thinking and transition
tokens actually emitted. Clients that need longer answers should raise their
total `max_tokens` accordingly.

## Model runner support

Both model runners enforce the budget. The V2 runner (the default for
every SlimServe profile) enforces it GPU-side at the same logits seam as
structured-output grammars, so it is async-safe and exact under
speculative decoding; it requires the model's reasoning start/end markers
to each tokenize to a single token (true for the qwen3 family). Models
with multi-token markers are rejected per request on V2 and need
`VLLM_USE_V2_MODEL_RUNNER=0`.
