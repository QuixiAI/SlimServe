# GLM-5.2-Vision — image fidelity investigation

## SOLVED (2026-07-27). Root cause + fix at the top; evidence trail below.

**Root cause:** vLLM loaded the vision tower from `mmproj-GLM-5.2-Vision-f16.gguf`,
whose QKV is stored in llama.cpp's **clip layout** — an element permutation of the
HF layout (identical value multiset, `sorted()` matches to 3e-8).
`glm_dsa.py::_vision_weights_gguf` maps `v.blk.N.attn_qkv.weight -> wqkv.weight`
with **no layout conversion**, so the tower ran correct math on wrongly arranged
weights. Coarse image structure survived; text and fine detail were destroyed.

**THE FIX — convert the layout, keep loading the GGUF mmproj.**
`glm_dsa.py::_qk_split_to_interleaved` un-permutes Q and K (V has no rotary) when
reading `v.blk.N.attn_qkv.{weight,bias}`.

llama.cpp's converter permutes Q/K into "split" 2D-RoPE format — its own comment
in `tools/mtmd/models/kimik25.cpp` says so:

> "Kimi-K2.5 uses interleaved 2D RoPE pattern natively, but Q / K are permuted
> during conversion to use split format."

`clip::build_rope_2d` then treats each head as `[x half | y half]` (dims 0..35 =
width, 36..71 = height, each rotated as adjacent pairs). vLLM's
`Rope2DPosEmbRepeated` keeps the native interleaved form — freqs packed
`[x0,y0,x1,y1,...]`, `apply_rope` pairing adjacent elements — so dims (4j, 4j+1)
are x_j and (4j+2, 4j+3) are y_j. Per head, with H = D//2:

```
interleaved[4j+0] <- split[2j]           interleaved[4j+2] <- split[H+2j]
interleaved[4j+1] <- split[2j+1]         interleaved[4j+3] <- split[H+2j+1]
```

Verified against `vision_tower.safetensors` for blocks 0/1/5/13/26:
**max abs diff 2.98e-08 for weights (float32 rounding), exactly 0.0 for biases**,
versus up to 8.3e-01 (weights) and 9.28 (bias) unconverted.

The mmproj GGUF stays the vision source — that is the point of the deployment.
An earlier attempt added a `VLLM_GGUF_MMPROJ=none` escape hatch to fall back to
`vision_tower.safetensors`; that was **removed** as unnecessary. It only worked
when a 933 MB HF vision checkpoint happened to be on disk, which defeats a
GGUF-only deployment, and it papered over the layout bug instead of fixing it.

Note `_find_mmproj` searches the text GGUF's dir AND its parent — so deleting or
symlinking mmproj files is not a reliable way to control the source (an earlier
"fix" that removed a symlink silently changed nothing for that reason).

**Result — 4/4 correct loading FROM THE GGUF MMPROJ** (`gguf_fixed.log`, no flags,
no HF vision checkpoint involved):

| img | vLLM output | ok |
|---|---|---|
| 1 | "river flowing through a valley surrounded by tall, rocky cliffs ... smooth, rounded rocks and boulders" | yes |
| 2 | "majestic snow-capped mountain peak ... sharp, jagged peak" | yes |
| 3 | "stone **bench** with a metal frame ... a dark metal **plaque** ... raised **anchor** emblem and the text **ESPERANCA** ... **patterned paving stones**" | yes |
| 4 | "screenshot from a **Pokemon** game, specifically a **battle scene** ... **Pidgey** with a level of **17** ... small, brown and tan bird-like Pokemon ... **Pikachu**" | yes |

Matches llama.cpp on img4 and beats it on img3 (llama.cpp misreads that plaque as
"SEBASTIAN VACA"; vLLM reads ESPERANCA).

Earlier result via the now-removed safetensors fallback (`sf_vision.log`), kept
only as corroboration that the two sources agree once the layout is fixed:

| img | vLLM output | ok |
|---|---|---|
| 1 | "granite peaks ... calm dark river with smooth rounded stones ... fallen logs" | yes |
| 2 | "majestic snow-capped mountain peak ... sharp jagged peak" | yes |
| 3 | "plaque ... the word **ESPERANCA** (Portuguese for 'hope') and an **anchor** symbol ... a dark metal **bench** with a wooden seat ... **patterned paving stones**" | yes |
| 4 | "screenshot from a **Pokemon** video game, specifically a **battle scene** ... **Pikachu** ... black-tipped ears, red cheeks, lightning bolt-shaped tail" | yes |

img3 is MORE accurate than llama.cpp, which misread that plaque as
"SEBASTIAN VACA"; vLLM reads ESPERANCA correctly.

**Housekeeping done:** mmproj restored to its original path (llama.cpp needs it);
`QF_MM_DUMP` instrumentation reverted out of `kimi_k25.py`.


Goal: the model correctly discerns the content of four test images.
Text generation and the TP2 correctness bug are DONE — see `worklog.md`.

## Ground truth (verified by direct inspection, not assumed)

| img | size | content |
|---|---|---|
| img1 | 1436x962 | Yosemite Valley. El Capitan left, Cathedral Rocks right, Merced River in foreground with rocks + grass tufts, pine forest, blue sky with clouds. |
| img2 | 200x300 | Snowy mountain peak at sunrise/sunset, pink-orange clouds, smooth snow slope foreground. Minimalist alpine. |
| img3 | 512x512 | Weathered bench against a peeling whitewashed wall. Sign reads **ESPERANCA** with an anchor emblem. Black/white patterned tile floor. |
| img4 | 512x384 | Pokemon battle screenshot. **Pidgey (f) Lv.17** top-left with HP bar; **Pikachu (m) Lv.42, HP 83/83** right; prompt **"What will Pikachu do?"**; menu **FIGHT / BAG / POKEMON / RUN**. Pikachu yellow in left foreground, Pidgey on a grass platform right. |

Local copies: `<SCRATCH>/imgs/img{1..4}_rgb.png`.

## Scoring rule

"Correct" = names the actual subject, not a plausible-sounding scene. img4 is the
strictest test because it contains readable UI text; img3 has a single readable
word. A description that gets colour and vague shape right but invents the subject
counts as FAILED, not partial.

## Dead hypotheses — do not retry

1. **Non-square patch grid.** `aspect1.log`: 512x384 (grid 28x38), 448x448
   (32x32), 512x512 (38x38), 384x512 (38x28) ALL produce fluent text. Geometry is
   not the trigger.
2. **`get_rope_shape` h/w transposition under `torch.compile`.** Directly tested
   against eager `F.interpolate` for (32,32),(38,38),(28,38),(38,28),(14,19),
   (19,14) after the square (64,64) warmup: max err 4.8e-7 everywhere. Correct.
3. **`grid_thws` vs `image_grid_thw` key mismatch** in
   `KimiK25MultiModalProcessor._get_mm_fields_config`. The HF processor renames
   the key ONLY on the standard-HF `images=` path (`kimi_k25_processor.py:134`);
   vLLM uses the native `medias=` path and gets `grid_thws`. No mismatch.
4. **Placeholder/patch count misalignment.** Verified for 448x448, 512x384,
   448x448, 512x512, 392x392: placeholders always == patches/4 exactly.

5. **Multimodal batching / shared PIL object.** `batch1.log`: A (single request),
   B (batch, 4 prompts sharing ONE image object), C (batch, `img.copy()` each)
   all produce essentially IDENTICAL fluent output. Neither batching nor object
   identity matters. DEAD.

## The actual symptom (not a crash — low fidelity)

`batch1.log`, img4, fluent but wrong:

| question | model answer | truth |
|---|---|---|
| what kind of game | "2D platformer" | Pokemon battle |
| menu options | "PLAY, OPTIONS, QUIT" | FIGHT / BAG / POKEMON / RUN |
| read all text | "Pikachu", "HP", rest confabulated | Pidgey, Pikachu, HP, 83/83, Lv.17, Lv.42 |
| describe | "character selection screen" | battle screen |

It DOES extract `Pikachu` and `HP` — real strings from the image — then invents
the rest. So the tower passes real signal at low resolution/precision; the model
is not blind and not crashing.

**The `pokemon1.log` numeric garbage did NOT reproduce** in `batch1.log` with the
same image, same 4 questions, same max_tokens. Difference: `pokemon1` ran on GPUs
0-1 with all 8 visible while another vLLM instance was loading; `batch1` ran with
`HIP_VISIBLE_DEVICES=2,3`. Treat that garbage as unexplained-and-not-reproduced,
NOT as a known bug. Re-check if it recurs.

6. **Q2_K quantization ceiling.** DEAD — see the llama.cpp reference below.
   llama.cpp reads the image correctly from the IDENTICAL GGUF + mmproj, so 2-bit
   weights are not the limit and vLLM is losing information llama.cpp keeps.

## REFERENCE: llama.cpp is correct on the same weights (`llama_img4.log`)

`llama-mtmd-cli`, branch `mtmd-glm-5.2-vision`, same GGUF + same mmproj, img4:

> "This is a screenshot from a **Pokemon** game, specifically Pokemon Yellow. ...
> the player's name, **"Pidgey"**, and their current level, **"Lv. 17"**. ... the
> opponent's name, **"Pikachu"**, and their current level, **"Lv. 42"**. ... the
> player's options: **"FIGHT"**, **"BAG"**, **"POKEMON"**, and [RUN]"

Essentially ground truth. vLLM on the same weights: "2D platformer",
"PLAY/OPTIONS/QUIT", "character selection screen".

**This is a vLLM vision-path defect.** Same shape as the text bug that was already
fixed: llama.cpp right, vLLM wrong, identical weights. Mmproj image encode took
80 ms in llama.cpp, so it is genuinely running the tower.

## Baseline scores, all four images (`four1.log`)

| img | vLLM says | verdict |
|---|---|---|
| 1 Yosemite | "panoramic landscape, green valley, winding river, rocky mountains" | PASS (scene-level) |
| 2 snowy peak | "winter landscape at dusk, pastel pink sky, snow-covered mountains" | PASS |
| 3 bench/ESPERANCA | "white stone wall with a dark arched doorway, stone slabs" | **FAIL** |
| 4 Pokemon | "character creation screen, yellow cartoonish character" | **FAIL** |

Phase 1 (one per generate) == Phase 2 (batched). Global scene gist survives;
objects and text do not. Signature of under-sampled vision features.

## Dead hypotheses, round 2

7. **AITER Flash Attention as the ViT backend.** `vit_sdpa.log` with
   `mm_encoder_attn_backend=TORCH_SDPA` gives essentially IDENTICAL output to the
   AITER default on all four images. Not the ViT attention kernel.
8. **Fused numba CPU preprocessing.** Proven offline, no GPU: the fused processor
   is **bit-identical** to the HF reference — `maxabs=0.000000` on img1/img3/img4,
   `grid_thws` equal ([[1,70,104]], [[1,38,38]], [[1,28,38]]). Not preprocessing.
9. **FP8 vision weights / fnuz-vs-OCP.** `vision_tower.safetensors` is 329 tensors
   ALL BF16, and `mm_projector.safetensors` is BF16. Nothing fp8 in the vision
   path, so the gfx942 fnuz trap does not apply.
10. **2x2 patch-merge spatial ordering.** Read the code
    (`kimi_k25_vit.py:583`): `view(t, nh, kh, nw, kw, d)` then
    `permute(0,2,1,3,4)` correctly groups each 2x2 block. Correct as written.

## LIVE: vision-token resolution

The mmproj GGUF metadata says llama.cpp may use up to **4096 image tokens**
(`clip.vision.image_max_pixels=3211264`, patch 14, merge 2 -> /784), and
`clip.vision.image_size=896`.

vLLM feeds img4 at its native 512x384 -> grid 28x38 -> **266 tokens**, i.e. each
token covers ~28x28 px, which is larger than the UI font. Measured token counts
through vLLM's own preprocessor:

| upscale | input | grid | patches | tokens |
|---|---|---|---|---|
| 1x | 512x384 | 28x38 | 1064 | **266** |
| 2x | 1024x768 | 56x74 | 4144 | 1036 |
| 3x | 1536x1152 | 84x110 | 9240 | 2310 |
| 4x | 2048x1536 | 110x148 | 16280 | 4070 |

vLLM's `in_patch_limit` is 16384 patches = 4096 tokens — the same ceiling — but
nothing upscales a small image toward it, so a 512x384 screenshot is encoded at
6% of the available token budget. llama.cpp's `image_min_pixels`/`image_max_pixels`
drive a resize; vLLM appears to only downscale.

**RESULT (`res1.log`): hypothesis DEAD, and INVERTED — more tokens is worse.**

```
img4 1x (266 tok)  'character creation screen... cartoonish figure'   wrong but coherent
img4 2x (1036 tok) 'chaotic, abstract composition of overlapping, distorted,
                    fragmented shapes ... the overall effect is visual noise'
img4 3x (2310 tok) 'chaotic and abstract ... heavily manipulated or GLITCHED
                    image ... the text is illegible'
img3 1x  'weathered white stone wall with arched doorway'
img3 2x  'close-up of a weathered textured surface, wood or stone'
img3 3x  'close-up of a textured surface, likely fabric, woven or knitted'
```

The model reports "glitched", "visual noise", "fragmented" — the features it is
handed genuinely ARE noise. **Degradation scales with patch count.** Do not raise
resolution as a fix; it makes things worse.

11. **Patch ordering out of the preprocessor.** Verified with a synthetic image
    whose every 14x14 patch encodes its own row-major index: recovered ids are
    exactly `[0..23]` = ROW-MAJOR, matching the pos-emb convention
    (`get_rope_shape` -> `reshape(-1, dim)` over (h, w)). Not column-major, not
    merge-block order. Preprocessing order is correct.

## VERIFIED: llama.cpp is correct on all four images (`llama_all4.log`)

Same GGUF + same mmproj, `llama-mtmd-cli`, identical prompt.

| img | llama.cpp | ok |
|---|---|---|
| 1 | "river through a valley surrounded by towering GRANITE cliffs, dome-shaped peak on the RIGHT, sheer cliff face on the LEFT, rounded boulders" | yes |
| 2 | "snow-covered peak rising above a sea of clouds, sharp pointed summit, warm golden light from the setting sun" | yes |
| 3 | "weathered stone BENCH against a white textured wall, dark stone PLAQUE with a carved ANCHOR, PATTERNED PAVING stones, peeling wall" | yes* |
| 4 | "Pokemon ... 'Pidgey' ... 'Lv. 17' ... 'Pikachu' ... 'Lv. 42' ... 'FIGHT', 'BAG', 'POKEMON'" | yes |

*Two honest imperfections: img3's plaque text is read as "SEBASTIAN VACA" but
actually says "ESPERANCA" (it finds and parses the plaque, but the OCR is wrong),
and img4 inverts player/opponent roles. Neither affects the conclusion.

**Score: llama.cpp 4/4, vLLM 0/4.** The reference is solid; the defect is in vLLM.

## Reframing: vision features are degraded at ALL resolutions

Earlier scoring of img1/img2 as "PASS" was too generous. Against llama.cpp on
identical weights (`llama_all4.log`):

| img | llama.cpp | vLLM |
|---|---|---|
| 1 | "river through a valley surrounded by towering GRANITE cliffs, dome-shaped peak on the RIGHT, sheer cliff face on the LEFT, rounded boulders" — spatially correct | "green valley, rocky mountains **with patches of snow on their peaks**" — snow fabricated, no spatial detail |
| 2 | "snow-covered peak rising above a sea of clouds, sharp pointed summit, warm golden light from the setting sun" | "winter landscape at dusk, pink sky, **small bushes poking through the snow**" — bushes fabricated |
| 4 | Pidgey Lv.17 / Pikachu Lv.42 / FIGHT / BAG / POKEMON | "character creation screen" |

vLLM is degraded on EVERY image, not only text-heavy ones. At low token counts the
LM confabulates a plausible scene from weak signal; at high token counts the noise
dominates and it says so. So the vision tower output is wrong everywhere and the
1x results were never "working".

## *** ROOT CAUSE LOCATED: the ViT's wqkv weights are mis-sharded at TP2 ***

`<SCRATCH>/cmp_vit_weights.py` diffs the runtime's rank-0 ViT weight shards
against `vision_tower.safetensors`:

```
norm0 / patch_proj / pos_emb   IDENTICAL      (replicated, correct)
wo   (1152,576)                MATCH cols[0:576]      (row-parallel, correct)
fc0  (2152,1152)               MATCH rows[0:2152]     (column-parallel, correct)
wqkv (1728,1152)               matches NEITHER the correct interleaved
                               q|k|v slice NOR a naive contiguous slice
```

### RETRACTED: the "2-of-4 permutation" claim was a measurement artifact

An earlier pass fingerprinted rows by `row.sum()` and reported a clean
`dst[i] = 4*(i//2) + (i%2)` permutation. **That was wrong** — row sums collide, and
the "regular pattern" was an artifact of `argmin` over near-equal sums.

Redone with exact bf16 row hashing: only **918/1728** runtime rows appear anywhere
in the checkpoint, with scattered `src-dst` offsets (most common 1152, 239 times).
It is NOT a row permutation of `vision_tower.safetensors`.

**Rule: fingerprint by exact content (hash the raw bytes), never by a reduction
like sum(). A lossy fingerprint will manufacture a convincing false pattern.**

What is still solid (all independently measured):
- runtime TP2 vs standalone TP1 embeddings: **cosine 0.648**
- standalone TP1 == llama.cpp at every stage (patch_embed / block0 / block26 /
  proj_out, all <1%)
- ViT attention backend irrelevant (TP1 SDPA vs AITER cosine 0.9996)
- `mm_encoder_tp_mode` irrelevant (weights vs data cosine 0.9993)
- `wo`, `fc0`, `norm0`, `patch_proj`, `pos_emb` match the checkpoint EXACTLY
- `wqkv` matches NEITHER the correct interleaved slice NOR a naive contiguous one

### ROOT CAUSE (proven): vision weights were being loaded from the mmproj GGUF,
### whose QKV layout is a PERMUTATION of the HF layout

Per-rank dump (`w_tp.r0.pt` / `w_tp.r1.pt`): neither rank's `wqkv` shard is any
slice of `vision_tower.safetensors`, and reconstructing r0+r1 does not equal the
checkpoint (2850/3456 rows differ, maxabs 0.83). The runtime simply had
**different values** — i.e. a different source.

Source identified: `mmproj-GLM-5.2-Vision-f16.gguf`.

```
mmproj  v.blk.0.attn_qkv.weight  (3456,1152)  sum = 42.3514
safeten vision_tower...wqkv.weight (3456,1152) sum = 42.3514   <- IDENTICAL sum
elementwise maxabs = 0.832
sorted(gguf) == sorted(safetensors)  ->  True (max diff 3e-8)
```

Same multiset of values, different arrangement: **the mmproj GGUF stores QKV in
llama.cpp's clip layout, a permutation of the HF layout.** llama.cpp expects that
and reads it correctly. vLLM loads it into `QKVParallelLinear`, which expects the
HF layout — so the tower is mathematically correct but runs on wrongly-arranged
weights. Exactly "structurally plausible, semantically scrambled".

**Self-inflicted:** I created the symlink
`antirez-routed/mmproj-GLM-5.2-Vision-f16.gguf -> ../` earlier in this
investigation so `detect_gguf_multimodal` would find the mmproj. That is what
switched the vision weight source from the HF safetensors to the GGUF.

**CONFIRMED by direct weight comparison** (`w_tp.r0.pt`, `w_fix.r0.pt`):

```
runtime wqkv == correct TP slice of MMPROJ GGUF : True
runtime wqkv == correct TP slice of SAFETENSORS : False
```

So **the TP sharding is CORRECT** — rank 0 holds exactly the right slice. (This
also retracts the earlier "mis-sharded at TP2" framing: the shard is right, the
SOURCE is wrong.) The tower runs correct math on clip-layout weights.

**Removing the symlink did nothing** — `glm_dsa.py::_find_mmproj` searches the text
GGUF's directory **and its parent**, so it picked up
`GLM-5.2-Vision-GGUF/mmproj-*.gguf` anyway (`weights changed by removing symlink:
False`). `VLLM_GGUF_MMPROJ` overrides the path but there is no "disable" value.

`_vision_weights_gguf` maps `v.blk.N.attn_qkv.weight -> ...blocks.N.wqkv.weight`
(`_VISION_BLK_RENAMES`, `glm_dsa.py:121`) with **no layout conversion**.

The exact permutation is NOT a simple row shuffle: exhaustive reshape/permute over
(3,H,D,C) and (H,3,D,C), plus rotary half-interleave variants, all fail; exact row
hashing locates only 1933/3456 rows. But `sorted(gguf) == sorted(safetensors)` to
3e-8, so it is a genuine element permutation that also moves elements WITHIN rows.
(BF16->F16 is lossless, so exact comparison is valid here.)

**Test in flight:** move the mmproj out of both search dirs
(`GLM-5.2-Vision-GGUF/_mmproj_held/`) so `_find_mmproj` returns None and vision
falls back to `vision_tower.safetensors`. Run: `sf_vision.log`.
**RESTORE THE FILE AFTERWARDS** — llama.cpp needs it.

If confirmed, the durable fix is one of:
1. do not ship the mmproj next to the GGUF when an HF vision checkpoint is present
   (simplest, what is being tested), or
2. teach the plugin's mmproj loader to un-permute clip-layout QKV into HF layout
   (needed for GGUF-only deployments with no safetensors).

Notes narrowing the mechanism:
- The ViT's `wqkv` holds a plain float `.weight` (not a GGUF `qweight`), so it is
  **unquantized**: the GGUF plugin loader is NOT involved. This is vLLM's stock
  `QKVParallelLinear` fused-weight path.
- `kimi_k25.py` has no `packed_modules_mapping` and no `wqkv` entry in
  `hf_to_vllm_mapper`.
- `wo` and `fc0` (row- and column-parallel) shard correctly, so only the QKV
  fused path is wrong.
- Both `mm_encoder_tp_mode` values give cosine 0.9993 with each other and 0.648
  with TP1 — consistent with a single common cause at load time.

**Next:** read `QKVParallelLinear.weight_loader`'s `loaded_shard_id is None`
(fused-on-disk) branch for how it slices a pre-fused wqkv at TP>1. Fix, then
re-verify with `cmp_vit_weights.py` (fast) before spending an engine boot.

## Superseded framing: "corrupted by tensor parallelism"

Controlled measurement, `<SCRATCH>/mm_runtime.pt` vs the standalone pipeline, using
the **exact runtime `pixel_values`** (so preprocessing is eliminated as a variable):

| | runtime (TP2) | standalone (TP1, == llama.cpp) |
|---|---|---|
| sum | -3155.95 | -3460.80 |
| std | 0.31156 | 0.34146 |
| row0[:3] | 0.5664, 0.1162, 0.4512 | 0.1084, -0.1553, -0.0713 |

**cosine = 0.6477, mean rel diff 84.9%.** Same input, same weights, same grid
`[[1,28,38]]`, same 266 output embeddings — different values.

The only difference is HOW the tower is executed:
- standalone: `tower(pv, grid)` directly, TP1, weights unsplit.
- runtime: `vision_tower_forward` under TP2 with `mm_encoder_tp_mode="weights"`,
  i.e. the ViT's 16 attention heads split 8/8 across ranks
  (`kimi_k25_vit.py:704`).

**This is why every component check passed** — all of them exercised the
UNSHARDED path. The tower math is right; its tensor-parallel execution is wrong.

Corrects an earlier call: the `mm_encoder_tp_mode="data"` test was scored on
generated TEXT, which is far too coarse to distinguish "wrong" from "differently
wrong". It should have been scored on the embedding tensor. Re-running that way
-> `mm_runtime_dp.pt` / `dump2.log`.

**Fix direction:** make the ViT run unsharded (replicated) when the LM is TP2. If
`"data"` mode yields embeddings matching standalone, that is both the diagnosis
and an immediate workaround.

Reusable harness: `QF_MM_DUMP=<path>` (patched into
`kimi_k25.py::_process_media_input`) dumps runtime features + pixel_values + grid;
diff against the standalone tower+projector. NOTE it fires during profiling too —
the dummy is `grid=[[4,130,130]]`; the real image is the LAST write.

## Supporting: vLLM's vision tower AND projector are numerically CORRECT (standalone/TP1)

Method: `llama-mtmd-debug -p encode -n 896 --image cb` runs clip on a synthetic
checkerboard and dumps every intermediate tensor (1033 of them) — the vision
equivalent of the text eval-callback. Note it feeds RAW 0.0/1.0 values straight to
clip as `clip_image_f32` (no mean/std), so the vLLM side must match that exactly.

`<SCRATCH>/vit_ref.py` runs vLLM's `MoonViT3dPretrainedModel` +
`KimiK25MultiModalProjector` standalone (loads only the 833 MB
`vision_tower.safetensors`, no 262 GB LLM) on the identical input:

| stage | llama.cpp sum | vLLM sum | rel |
|---|---|---|---|
| `patch_bias` / patch_embed | 2308268.50 | 2290685.13 | 0.76% |
| `layer_out-0` / block0 | 2868307.25 | 2863131.58 | 0.18% |
| `layer_out-26` / block26 | 2686676.50 | 2678554.20 | 0.30% |
| `proj_out` | -9539.96 | -9459.28 | 0.85% |

proj_out row0 llama `[-0.5402, 1.8917, 1.3184 ... -1.6192, 0.3156, 0.3756]`
vs vLLM `[-0.5859, 1.9062, 1.2422 ... -1.5781, 0.2969, 0.3477]`.

All within bf16-vs-f32 noise. **Patch embed, all 27 blocks, 2D RoPE, attention,
the 2x2 merge and the projector are all correct.** This retroactively explains why
reading that code found nothing.

Reusable: `vit_ref.py` takes a size argument and needs no LLM — use it to check any
vision-tower change in ~30 s instead of a 280 s engine boot.

## Next action

12. **llama.cpp vs vLLM preprocessing grid.** DEAD, settled by code
    (`clip.cpp:3397`): for KIMIK25 llama.cpp computes
    `CLIP_ALIGN(nx,28)/28 * CLIP_ALIGN(ny,28)/28`; for 512x384 that is
    `19 * 14 = 266` merged tokens — **exactly vLLM's 266**. Its `pos_h = i/38`,
    `pos_w = i%38` also confirms row-major with width 38, matching vLLM's
    `grid_thw=[1,28,38]`. (An earlier hand-computed guess of 252 was wrong; do
    not trust that estimate.)

## REFRAME: this may not be a vision bug at all — test LENGTH, not vision

Everything in the vision path is now verified correct:
preprocessing (bit-identical to HF), patch embed, all 27 blocks, 2D RoPE,
attention, 2x2 merge, projector (all within bf16 noise of llama.cpp), the patch
grid (266 tokens, matching llama.cpp exactly), row-major ordering, AND both ViT
TP modes. Yet quality degrades as MORE vision tokens are injected
(266 wrong-but-coherent -> 1036 "visual noise" -> 2310 "glitched").

**That is a sequence-length signature, not a vision one.**

The text path was only ever verified on a FIVE-token prompt. This model uses DSA
sparse attention (`index_topk=2048`) through `rocm_aiter_mla_sparse.py`, which
already needed a metadata fix earlier in this project. Vision fails simply because
it injects hundreds of tokens; short text prompts never exercised that path.

**RESULT (`len2.log`): text PASSES at every length. Reframe DEAD.**

```
   38 tok  PASS -> 'BLUE-HERRING-7742'
  407 tok  PASS
 1879 tok  PASS
 5559 tok  PASS
14759 tok  PASS
```

Well past the 2048 DSA threshold. Long-context text attention, the sparse MLA
path and the DSA indexer are all fine. (A 6th case errored only because the test
overran `max_model_len` — test bug, not model.)

Config gotcha found on the way: `limit_mm_per_prompt={"vision_chunk": 0}` on this
multimodal model crashes engine init at 16k context with
`'NoneType' object has no attribute 'size'` (`deepseek_v2.py` forward, input_ids
None during profiling). Use `{"vision_chunk": 1}` even for text-only runs.

**This clears the ENTIRE text stack and leaves exactly one candidate:
embedding injection.**

## DEAD: ViT tensor-parallel weight sharding

`mm_encoder_tp_mode` defaults to **"weights"**, and `is_vit_use_data_parallel`
returns False here (16 heads % 2 == 0), so **at TP2 the vision tower's weights are
SPLIT across both GPUs**.

Every numeric verification above ran the tower standalone at **TP1**, where no
splitting happens. A bug in the ViT's TP weight sharding is therefore invisible to
all of it, and would corrupt runtime output exactly as observed — including getting
worse with more patches.

**RESULT (`mmdp.log`): does NOT fix it.** With `mm_encoder_tp_mode="data"`
(tower replicated per rank, no weight splitting) the output is unchanged —
img3 still "a stone statue of a seated figure, likely a Buddha", img4 still
"character selection screen". ViT weight-TP sharding is NOT the bug.
Knob remains available as `QF_MM_TP_MODE` in `<SCRATCH>/serve_four_var.py`.

## Superseded hypotheses (both ROCm/vLLM-specific, neither exists in llama.cpp)

From the vLLM load log:

1. **`rocm.py:661` — AITER Flash Attention is the ViT attention backend** on gfx9.
   The vision tower runs a ROCm-specific FA kernel with packed variable-length
   sequences. A subtly wrong varlen/cu_seqlens path corrupts fine detail while
   coarse layout survives — exactly the observed symptom. AITER has already
   produced one silent corruption in this project (the warpSize/blockDim bug).
   **Test in flight:** `vit_sdpa.log`, `mm_encoder_attn_backend=TORCH_SDPA`,
   which swaps ONLY the ViT backend and leaves the text path alone.
2. **`kimi_k25.py:122` — "fused CPU image preprocessing"**
   (`KimiK25FusedVisionProcessor`, used whenever numba is importable). A resize,
   normalization, or patch-ordering difference vs the reference HF/clip
   preprocessing would degrade detail the same way.
   **Test knob:** `QF_NO_FUSED_PREPROC=1` in `<SCRATCH>/serve_four_var.py`
   monkeypatches `is_numba_available` to False -> "remote HF" path.

Harness: `<SCRATCH>/serve_four_var.py`, env `QF_VIT_BACKEND` and
`QF_NO_FUSED_PREPROC`. Run variants concurrently on separate GPU pairs.

`four1.log` (GPUs 0-1) is the unmodified baseline across all four images,
one-per-generate vs batched.

## Known-good baseline

Fluent, partially-grounded output IS achievable: img4 at several geometries gave
"yellow spiky creature ... grassy field", which is a low-fidelity but real read of
Pikachu on grass. So the tower passes usable signal; the open question is whether
the failures are batching corruption, resolution/detail loss, or 2-bit quantization
of the language model.

## Notes

- Chat template is mandatory (`enable_thinking=False`); raw completion prompts make
  this instruct model echo the instruction forever at temp 0.
- `vision_chunk` item must be `VisionChunkImage(type="image", image=..., uuid=None)`.
- 8x MI300X available: TP2 uses 2, so run experiments concurrently with
  `HIP_VISIBLE_DEVICES`. Do not serialize.
