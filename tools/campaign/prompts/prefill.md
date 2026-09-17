## Phase: prefill

Make prefill as fast as this hardware allows, without touching the decode
pins. Start from the plan's prefill waves and a prefill attribution at
1000 / 2048 / 2500 tokens (walls at 512 / 1000 / 2048 / 3000 through
disjoint prompt windows; phase split and the platform's GPU timeline
before fusing).

Levers, in the order earlier campaigns found them: the tile / grouped
GEMM route actually engaging for this model's activation dtype and expert
count; prefill-shaped attention (tensor-core rows over the selection or
dense under the model's dense limit); the recurrence's prefill scan
parallel over tokens or chunks instead of serial; indexer / pooling once
per request; single-chunk scheduling (`max_num_batched_tokens` sized to
the block and the speculative width); host-sync removal on the prefill
path; TTFT for multi-chunk requests. Prefix caching is on for every
profile; measure warm TTFT too.

Gate every change: prefill throughput at the three lengths, TTFT cold and
warm at 32k and the deepest context the record serves, the decode pins
unchanged (bit-exact expected), needles at the deep tiers once per
numerics change, and concurrent prefill (c=4 / c=8 at 1000-token prompts)
so a prefill that blocks decode is seen.

Exit: prefill beats the bar at short, mid and deep context, TTFT recorded
cold and warm, concurrent prefill measured, every change journaled and
committed, status log updated.

Marker evidence keys: `prefill` (list of {tokens, ours, bar}), `ttft`
({context, cold_s, warm_s} list), `concurrent_prefill` (c, aggregate),
`decode_pins_held` (true / rolled with reason), `waves`.
