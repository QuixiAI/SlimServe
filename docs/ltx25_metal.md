# LTX-2.5 video generation on Apple Silicon

SlimServe serves Lightricks' LTX-2.5 (22B audio+video diffusion transformer)
on Apple Silicon through its own engine, `slimserve/video/ltx25/`. It loads
the official Lightricks safetensors directly, runs on MLX, and produces an
H.264 + AAC mp4. One profile per upstream pipeline:

| profile | pipeline | what it is for |
| --- | --- | --- |
| `ltx25-distilled` | DistilledPipeline: 8 + 3 steps, no guidance | fast iteration |
| `ltx25-dev` | TI2VidTwoStages: 30 guided steps (CFG 3 / STG 1 / modality 3), then 3 with the distilled LoRA | quality |
| `ltx25-dfr` | DFRPipeline: distilled flow with generated keyframe slots and the detailing IC-LoRA | Lightricks' production path |
| `ltx25-hq` | TI2VidTwoStagesHQ: 15 res_2s steps x 2 evaluations, distilled LoRA in both stages | Lightricks' HQ preset |
| `ltx25-keyframes` | KeyframeInterpolation: the dev flow with every still appended as a keyframe at its frame | a clip through given stills |
| `ltx25-one-stage` | TI2VidOneStage: 30 guided steps at the output size, no upsampler | prototyping (upstream: educational) |
| `ltx25-retake` | RetakePipeline: regenerate [start, end) seconds of a source clip, the rest kept | editing an existing clip |
| `ltx25-ic-lora` | ICLoraPipeline: a reference video drives the clip through an IC-LoRA adapter (colorize, deblur, restore, relight, matte, ...) | video-to-video |
| `ltx25-a2vid` | A2VidPipelineTwoStage: the dev flow driven by an audio file, which rides frozen and ships | audio-to-video |
| `ltx25-dubit` | DubItPipeline: re-voice a clip under the Dub-It IC-LoRA, the clip's video and audio as reference tokens | dubbing |
| `ltx25-t2a` | T2AOneStagePipeline: audio only, the dev transformer's audio half, a WAV | text-to-audio |
| `ltx25-alpha` | AlphaGenPipeline: a video-only stage under the Alpha-Gen IC-LoRA, the clip to matte as reference | alpha mattes |
| `ltx25-hdr-ic-lora` | HDRICLoraPipeline: SDR video to HDR under the SDR-To-HDR IC-LoRA, EXR + HLG out | SDR to HDR |

Distilled versus dev is a quality-versus-time choice for the person asking,
not something the server picks.

## Use

```
slimserve ltx25-distilled --dry-run                 # show the resolved plan
slimserve ltx25-distilled -y                        # fetch weights if needed, then serve on :8000
slimserve ltx25-dev -p "A red fox trotting through a snowy pine forest at dawn" --output fox.mp4
slimserve ltx25-dfr -p "..." --size 768x512 --seconds 5 --seed 7
slimserve ltx25-distilled -p "the fox turns and runs" --image fox.png     # image-to-video
slimserve ltx25-distilled -p "..." --image a.png --image b.png 48 0.8      # a second still pinned at frame 48
slimserve ltx25-keyframes -p "..." --image a.png 0 --image b.png 96        # interpolate between stills
slimserve ltx25-retake -p "..." --video-path clip.mp4 --start-time 2 --end-time 3.5   # regenerate a region
slimserve ltx25-ic-lora -p "..." --video-conditioning gray.mp4 --lora colorization.safetensors   # video-to-video
slimserve ltx25-a2vid -p "..." --audio-path speech.m4a --size 768x512          # audio-to-video (length from the audio)
slimserve ltx25-dubit -p "..." --reference-video talk.mp4 --lora dubit.safetensors   # dubbing
slimserve ltx25-t2a -p "rain on a tin roof" --seconds 5 --output rain.wav   # text-to-audio
slimserve ltx25-alpha -p "..." --video-conditioning clip.mp4 --lora alpha-gen.safetensors   # alpha matte
slimserve ltx25-hdr-ic-lora --video-path clip.mp4 --lora sdr-to-hdr.safetensors --text-embeddings scene-emb.safetensors   # SDR to HDR
slimserve ltx25-distilled-fast -p "..."                                   # the fast tier (see below)
slimserve ltx25-hq -p "..."                                               # Lightricks' HQ preset (res_2s)
slimserve ltx25-dev -p "a cat watches rain" --enhance-prompt                # Gemma rewrites the prompt first
```

The weights are gated and licensed: LTX-2.5 and its adapters are released
under Lightricks' LTX-2 Community License Agreement, which you accept on
Hugging Face before the download works (read its terms, including its
commercial-use conditions, before using the outputs). The Hugging Face token
on the machine must have accepted `Lightricks/LTX-2.5` and, for `ltx25-dfr`,
`Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler` (the other IC-LoRA
adapters are gated the same way). The prompt enhancer
(`google/gemma-4-E2B-it`, 10.2 GB, under Google's Gemma terms) is optional:
it is fetched only when a run asks for it (`--enhance-prompt`, or
`--prompt-enhancer` when serving so requests may set `enhance_prompt`); a
server without it answers such requests with a 400. Install
`requirements/video-metal.txt` (MLX, numpy, safetensors, Pillow, openexr);
`ffmpeg` and `ffprobe` must be on PATH (with libx265 for HDR masters); the
`openexr` package is needed for EXR input and output.

Video options go after the profile id and are listed by
`slimserve <video profile> --help`; the LLM-only options (`--spec`,
`--no-spec`, `--ctx`, `--thinking`, `--model`, `--engine-log`,
`--torch-profile-dir`, `--enable-return-routed-experts`) are refused for
video profiles.

When serving, request fields that name files on the server (`video_path`,
`audio_path`, `reference_video`, `video_conditioning[].path`,
`attention_mask`, `loras[].path`, `images[].path`, `text_embeddings`) are
refused unless the operator gives an input root (`--input-root DIR` or
`SLIMSERVE_VIDEO_INPUT_ROOT`): with one, paths resolve inside it (relative
paths against it, symlinks followed) and anything outside is refused. The
IC-LoRA, Dub-It, Alpha-Gen and HDR IC-LoRA pipelines take their adapters by
path, so they need an input root to be served. Base64 uploads (`video`,
`audio`, `image`, `images[].image`) work either way and are deleted when the
job has run. Request bodies above 256 MiB (`--max-body-mb` or
`SLIMSERVE_VIDEO_MAX_BODY_MB`) are answered with a 413; a body sent with
`Transfer-Encoding` instead of a `Content-Length` gets a 411, and a field of
the wrong type (a non-numeric `width`, `seconds` or `seed`, a body that is not
UTF-8) a 400.

The input root must be a directory the operator owns and nobody else can
write to. The server checks a requested path when the request arrives (it
resolves symlinks and confirms the result is inside the root) but the engine
opens the file later, when the job runs; anyone who can rename or replace
entries under the root in between can point an accepted request at a file
outside it. Do not use a shared, world-writable or client-writable directory
(an upload drop folder, `/tmp`) as the input root.

At most 4 requests are prepared at once (decoding and writing their base64
sources, probing them); more are answered 429 until one finishes
(`SLIMSERVE_VIDEO_MAX_PREPARING`). User LoRAs stay loaded between requests,
the 4 most recently used (`SLIMSERVE_VIDEO_LORA_CACHE`; 0 loads them per
request); older ones are released. Once the server is stopping it refuses new
requests with a 503 and removes the uploads of the queued and running jobs.

Serving API (a clip takes minutes, so it is job-shaped):

```
POST   /v1/videos               {"prompt": "...", "size": "1536x1024", "seconds": 5, "seed": 42,
                                 "enhance_prompt": false,
                                 "image": "<base64 or data: URL>", "image_strength": 1.0,
                                 "images": [{"image": "<base64>", "frame": 48, "strength": 1.0, "crf": 18}]}
GET    /v1/videos/<id>          status (queued | in_progress | completed | failed), progress, timings
GET    /v1/videos/<id>/content  the mp4
DELETE /v1/videos/<id>
GET    /health, /v1/models
```

`ltx25-retake` takes a source clip instead of a size and length:
`video_path` (a file on the server) or `video` (base64 mp4), `start_time` and
`end_time` in seconds, optional `regenerate_video` / `regenerate_audio`
(default true; false freezes that modality). The clip must have 8k + 1
frames and sides that are multiples of 32 and fit the token envelope; its
size, frame count and rate are the output's. The clip is decoded with ffmpeg
exactly as upstream's PyAV path does, video-VAE-encoded with upstream's
default tiling (frames 80/24, 768/64 px), the audio through the audio VAE
encoder (torchaudio's sinc resample to 16 kHz, Slaney log-mel), and only
the tokens whose time span overlaps the region are regenerated (upstream's
TemporalRegionMask), on the distilled transformer with its 8 sigmas and
plain Euler, as upstream's CLI runs it. A source without an audio track
gets audio generated over the whole clip.

`ltx25-ic-lora` (upstream's ICLoraPipeline, its CLI recipe) takes
`video_conditioning` (`[{"path", "strength"}]`, reference clips on the
server; CLI `--video-conditioning PATH [STRENGTH]`, repeatable) and the
IC-LoRA adapter through `loras` / `--lora` (Lightricks publishes thirteen
for 2.5, all gated behind their own terms: Colorization, Deblur, Restore,
Day-To-Night, Clean-Plate, Alpha-Gen, SDR-To-HDR, Refine-Details,
Decompression, Layout-To-Render, Water-Simulation, Ingredients,
Pixel-Spatial-Upscaler). Stage 1 (half resolution, the 8 distilled sigmas,
plain Euler) runs under the adapter with each reference encoded at the
stage's size (over the adapter's `reference_downscale_factor`, frame 0 then
every `reference_temporal_scale_factor`th frame) and appended as clean
reference tokens whose positions are scaled onto the target grid; stage 2
is the bare 3-sigma refinement (`stage_2_ic_lora` keeps the adapter and
references, `skip_stage_2` ships the half-size stage 1). `attention_strength`
(0-1) and `attention_mask` (a grayscale mask video; CLI
`--conditioning-attention-mask MASK STRENGTH`) scale how strongly reference
and target tokens attend to each other (upstream's log-space attention
bias; it materializes an (N + M)^2 fp16 mask, 1.8 GB at the 1536x1024x121
envelope). `tile` / `--tile` (`tile_height`, `tile_width`, default 1024x1536,
half overlap) runs each transformer call over pinned equal windows blended
with trapezoids (upstream FixedSizeSpatialTiling); a tiled full-resolution
stage needs `stage_2_ic_lora`. Stills condition as elsewhere. A tiled stage
encodes its references with our default source tiling, not upstream's
machine-dependent decode tiling (the one documented deviation).

`ltx25-a2vid` (upstream's A2VidPipelineTwoStage) takes `audio_path` (a file
with an audio stream on the server) or `audio` (base64), `audio_start_time`
and `audio_max_duration` in seconds (CLI flags of the same names). The
samples are resampled to 16 kHz, encoded by the audio VAE and ride frozen
through both dev stages (video guidance only, the audio guider upstream's
default), so the picture is generated to the sound; without `seconds` /
`num_frames` the clip is int(audio seconds x fps) snapped to 8k + 1 (capped
at the envelope; `audio_max_duration` and a length are exclusive, as
upstream); the source waveform ships, cut to the clip.

`ltx25-dubit` (upstream's DubItPipeline) takes `reference_video` (or
`video`), exactly one adapter in `loras` (Lightricks/LTX-2.3-22b-IC-LoRA-
DubIt, gated), `reference_strength`, and a size; the reference's frame count
(snapped to 8k + 1) and rate are the output's. Both distilled stages carry
the adapter; the reference video is tiled-encoded at each stage's size and
appended as reference tokens; its audio is encoded and appended after the
target audio as frozen reference tokens at negative times (stage 1), and
stage 2 freezes the stage-1 audio and appends it as its own reference. The
stage-1 audio ships.

`ltx25-t2a` (upstream's T2AOneStagePipeline) takes `seconds` / `num_frames`
at `fps` (the duration head decides otherwise), `seed`, `negative_prompt`,
`steps` and `guidance.audio`; no size, stills or decoder. The transformer
runs with no video modality (each block's audio half only, as upstream's
`video=None`), the audio guider as the guided profiles, and the job's
content is a 48 kHz stereo WAV (`audio/wav`).

`generated_keyframes` (upstream's `--num-generated-keyframes`; CLI flag of the
same name) on the distilled, dev, hq, one-stage, a2vid, ic-lora and dubit
profiles asks for that many generated keyframe slots at evenly spaced
interior frames (or a list of frames): extra single-pixel-frame tokens the
model denoises with the clip in both stages (stage 2 seeded from the
upscaled stage-1 slots), the DFR mechanism on the plain flows;
`decode_with_keyframes` (`--decode-with-keyframes`) then anchors the
diffusion decode on them (the conv decoder ignores them, as upstream).

Long clips (upstream's `--chunked`, `--chunk-pixel-frames`,
`--chunk-carry-frames`, `--chunk-blend-frames`; API `chunked`,
`chunk_pixel_frames`, `chunk_carry_frames`, `chunk_blend_frames`) on the
distilled, dev, a2vid, ic-lora and dubit profiles generate in overlapping
temporal windows
(default 97 frames with a 25-frame carry): each window is denoised with
the previous window's last carry latent frames pinned at its start (video
and audio), both stages, stills and generated keyframes landing in each
window as upstream plans them, then decoded per window with the carried
frames dropped and the seam crossfaded (linear over `chunk_blend_frames`,
default the carry; audio over 40 ms equal-power). The clip may then be up
to 1024 frames; the window must fit the profile's envelope. On the IC-LoRA
and Dub-It profiles each window encodes its own stretch of the reference
video (by frame index) and drops the reference latents the incoming carry
already covers, as upstream; Dub-It keeps the clip's own length.

`ltx25-alpha` (upstream's AlphaGenPipeline) takes `video_conditioning` (the
clip to matte) and the Alpha-Gen adapter in `loras`
(Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen, gated), a size (snapped to 32),
stills, `seconds`, `steps`, `guidance.video` and `negative_prompt` (default
upstream's "worst quality, inconsistent motion, blurry, jittery,
distorted"); one video-only stage (no audio modality at all: the
transformer's audio half and both cross-attentions are skipped) at CFG 1,
no STG, rescale 0.7 by default; the matte ships as a silent clip.

HDR (upstream's `--hdr {srgb_linear, acescg, acescct}`; API `hdr`, CLI
`--hdr`) on every profile: EXR stills (`images: [{"path": "a.exr", ...}]`,
`--image a.exr`) and EXR frame folders as sources and references
(`--frame-rate` is required for a folder, which carries none) are read
scene-linear and compressed into the model's ACEScct working space in the
declared colour space (Rec.709 or ACEScg primaries, or ACEScct codes as
they are), without the H.264 round trip; the decode stays in fp32 and the
job writes an EXR frame folder `<stem>_<hdr>_exr/` (half floats, ZIP, the
primaries' chromaticities and a colorSpace tag; log codes for acescct,
scene-linear otherwise) beside an HLG master at the output path (HEVC
Main10, Rec.2020 / BT.2100 HLG / bt2020nc limited, `hvc1`, upstream's x265
parameters, diffuse white mapped to signal 0.75 with highlights rolled
toward 1). The colour math matches upstream to float precision and the HLG
10-bit planes bit for bit. Real HDR output needs the SDR-To-HDR IC-LoRA;
the plain model reproduces SDR content in the log space.

`ltx25-hdr-ic-lora` (upstream's HDRICLoraPipeline) takes no prompt:
`video_path` (an mp4, `input_colorspace` srgb_gamma for display video or
srgb for linear; or an EXR folder in srgb / acescg / acescct with `fps`),
the SDR-To-HDR adapter in `loras` and its scene embedding file in
`text_embeddings` (both in Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR,
gated), `exr_colorspace` (default acescg), `high_quality`, `keyframes`
(default true: a generated slot and a 1-frame SDR guide at
`keyframe_strength` 0.95 on every DFR seam), `conditioning_strength`. One
video-only distilled stage with the clip's ACEScct codes as reference
tokens; the frame is reflect-padded to multiples of 32 and cropped back;
the outputs are the HDR ones above.

`ltx25-dfr` takes two more options (upstream's `--temporal-upscalings` and
`--spatial-upscalings`; CLI flags of the same names): `temporal_upscalings`
1 or 2 doubles the frame rate per round (the canvas is x2 temporally
upsampled and re-denoised in keyframe-seam windows; a 5 s clip at 24 fps
ships at 48 or 96 fps) and `spatial_upscalings` 2 runs the first two stages
at a quarter and half of the output size and adds the tiled full-resolution
detailing epilogue (sizes must be multiples of 128; the way to a sharp
2048x1024 from the same base as a 1024x512 clip).

`"wait": true` holds the POST open until the clip is done. A request with
neither `seconds` nor `num_frames` gets its length from the model's duration
head (Lightricks' auto-duration: the clip the prompt implies, 1-20 s, snapped
to the 8k + 1 frame grid, capped at the profile's envelope for that size); the
job reports the pick as `num_frames` and `predicted_seconds`. `negative_prompt`
is accepted by the guided pipelines (`ltx25-dev`, `ltx25-hq`, `ltx25-keyframes`,
`ltx25-one-stage`), as are `steps` and `guidance` (per modality, upstream's
MultiModalGuiderParams: `cfg`, `stg`, `stg_blocks`, `rescale`, `modality`
(a2v on video, v2a on audio), `skip_step`; CLI flags with upstream's names,
`--num-inference-steps`, `--video-cfg-guidance-scale`, `--video-skip-step`,
`--a2v-guidance-scale`, ...). `skip_step` N runs that modality's stream only
on every (N + 1)th step, reusing its last x0 on the others while the other
modality keeps cross-attending to its untouched state (a skipped step of both
costs no forward). `loras` / `--lora PATH [STRENGTH]` attach user adapters
(ComfyUI LTXV layout, `diffusion_model.<linear>.lora_{A,B}.weight`, no alpha
scaling, as upstream) to every stage of every profile; `lora_strengths` /
`--distilled-lora-strength-stage-1/-2` are hq's two strengths. `decoder` is `diffusion` (default, Lightricks'
recommended decoder: sharper faces, textures and text) or `conv` (about 4x
faster decode); the CLI flag is `--decoder`. Width and height are multiples of 64, frame
counts are 8k + 1. Requests larger than the profile's validated clip
(1536x1024x121, 24,576 latent tokens) are refused with a 400: that envelope is
what was measured to fit in memory. There is one GPU, so requests run one at
a time in arrival order with the weights resident; up to 16 may wait (then
429). Finished clips are kept under `$SLIMSERVE_VIDEO_DIR`
(default `~/.cache/slimserve/videos`), the most recent 32.

### Prompt enhancement

`enhance_prompt: true` (CLI `--enhance-prompt`; off by default, as upstream's
`--enhance-prompt`) rewrites the request into the long caption style the
model was trained on before anything else runs. LTX-2.5's own text encoder is
a fine-tune that cannot generate, so Lightricks' pipelines do this with a
separate generative instruct Gemma (`--prompt-enhancer-gemma-root`); the
profiles fetch (on demand, see above) the one their README names, `google/gemma-4-E2B-it`, and run
its language model in MLX (`enhancer.py`: 35 layers with per-layer input
embeddings, a shared-K/V tail, 512-token sliding and global attention). It is
loaded for the rewrite and dropped again (4.3 GiB in fp16, 0.8 s to load; the
4.7 GiB per-layer embedding table is never loaded, its rows are read from the
file), so the measured resident set is unchanged. The recipe is
upstream's exactly: its T2V system prompt (`prompts/`), `user prompt: ...`
through Gemma's chat template, greedy decoding with no repeated 5-gram, at
most 600 new tokens, curly quotes and a leading non-letter run cleaned off. A
rewrite takes about 5 s (a 160-220 token caption) and is reported as the job's
`enhanced_prompt` and `enhance` span; the CLI prints it.

Checked against transformers' fp32 Gemma-4 on the CPU
(`perf/ltx25_harness/n11_enhancer_parity.py`): same prompt tokens, first-step
logits within 2.6e-4 (relative), and teacher-forced logits at every one of 218
generated positions with the same argmax (median 3.2e-4, max 2.9e-3). Free
running, one of two test prompts reproduced the reference caption token for
token; the other followed it for 118 tokens and then took the other side of a
0.01-logit tie (`,` 17.879 vs ` with` 17.869 in the reference), which is below
the fp16-vs-fp32 error and gives an equally valid caption. Greedy decoding is
deterministic on a given machine.

An image request is enhanced with the still, as upstream's `enhance_i2v`:
Gemma gets the I2V system prompt and sees the image before `User Raw Input
Prompt: ...`. The still is decoded like the conditioning frame and scaled to a
896 long side with torch's uint8 bilinear arithmetic (`image.py
resize_bilinear_uint8`, bit-exact: the vision tower turns the one-level
rounding differences of a float resize into a 10% change of the soft tokens),
fitted by Gemma's PIL image processor onto a 48-pixel grid of at most 2520
patches (bicubic), and encoded by the vision tower (16 layers, 2-D RoPE,
clamped linears) into at most 280 soft tokens that take the place of the
`<|image|>` placeholders. The tower runs with fp32 operands (0.2 s per still;
in fp16 its output is off by 5e-3, in fp32 by 4e-6). On the I2V reference
(a snowy forest with a fox, a prompt about a woman and a window) ours is
identical to transformers for all 191 generated tokens: Gemma describes what
is in the image, then folds the request in.

### Image-to-video

`image` makes a still the clip's first frame, on every pipeline. The engine
follows Lightricks' `ltx_pipelines` conditioning path exactly:

- The still is decoded to sRGB (EXIF orientation applied), then re-compressed
  once as a single H.264 frame (libx264, preset veryfast, yuv420p) at CRF 18
  and decoded back (upstream `media_io/decode.py preprocess`; CRF 18 is
  `LTX_2_4_IMAGE_CRF`, the value for 2.4+ checkpoints, not the 2.0-era 33).
  The model was trained on video frames that carry H.264 compression
  statistics; a pristine PNG conditions it off-distribution and the first
  frame then "pops". Odd edges are trimmed to even sizes first, as the codec
  requires. The round trip runs through the `ffmpeg` binary (as the muxer
  does), with bilinear chroma scaling on both legs like PyAV's default.
- It is resized to fill the target (scale = max(H/src_h, W/src_w), ceil'd
  sizes, bilinear, `align_corners=False`, no antialias), center-cropped and
  mapped to [-1, 1] by `x / 127.5 - 1` (`resize.py resize_and_center_crop`,
  `range_map.py normalize_images`).
- Both stages are conditioned (upstream
  `chunks/conditionings.py image_conditionings_for_chunk` encodes at each
  chunk's own pixel size): in stage 1 the still is encoded at the half
  resolution through the conv VAE encoder, in stage 2 at the full
  resolution. Each encoding is written into the latent state's clean tokens
  at latent frame 0 with denoise mask `1 - image_strength`
  (`ltx_core latent_cond.py _apply_condition_by_latent_index`), and the
  noised state is `lerp(clean, noised, mask)` so a strength-1 frame starts
  clean and stays clean through the sampler; the transformer sees per-token
  timesteps `mask * sigma`. In DFR this happens before the keyframe slots
  and the reference tokens are appended.

CLI: `--image PATH` and `--image-strength` (0-1, default 1.0). API: `image`
is the encoded still (PNG, JPEG, ...) as base64, optionally a
`data:image/png;base64,...` URL, with `image_strength`; undecodable payloads
are a 400. The encode is timed as the `image` span (both stages summed).

Stills at other frames (upstream's repeatable `--image PATH FRAME_IDX
STRENGTH [CRF]`): `--image b.png 48 0.8` on the CLI, `images: [{"image":
..., "frame": 48, "strength": 0.8, "crf": 18}]` in the API, any number, on
every profile. A frame-0 still replaces latent frame 0 as above; a still at
any other pixel frame is encoded as a one-frame latent and appended to the
token sequence as a clean keyframe block whose RoPE position is that single
pixel frame (`VideoConditionByKeyframeIndex`; `combined_image_conditionings`),
so the clip passes through the image at that moment while the frames around
it are generated. `crf` overrides the H.264 round trip for that still (0:
none). In DFR the stills ride into every temporal-round window and epilogue
window their (x2-per-round) frame falls in. `ltx25-keyframes` is the
interpolation pipeline proper: every still, frame 0 included, is an appended
keyframe, and the audio is refined in stage 2 rather than frozen.

### The fast tier

`ltx25-distilled-fast`, `ltx25-dev-fast` and `ltx25-dfr-fast` are the same
engine with output-changing settings stacked on: they are for iterating, and
for anyone who judges a clip by eye rather than against a reference. Reviewed
at 1536x1024x121: no meaningful difference from the exact clips in any mode;
dev (exact or fast) is clearly the best of the three in quality and in prompt
adherence, so `ltx25-dev-fast` is the profile to reach for first. The
exact profiles stay the reference the fast ones are measured against. At
1536x1024x121 on the M1 Ultra (ledger section 24):

| profile | exact | fast | what is on |
| --- | ---: | ---: | --- |
| distilled | 413 s | **278 s** (229 s with the opt-in 5-step stage 1) | conv decoder; 2-step stage 2 |
| dev | 1513 s | **911 s** | 20 steps; first-block step cache 0.10; conv decoder (all four guidance passes kept) |
| DFR | 681 s | **421 s** | 2-step stage 2; 2x2-tiled stage-2 attention; conv decoder |

The editing and guided profiles carry the same levers (`-fast` records,
ledger sections 31-35, 43): keyframes, one-stage and a2vid take dev's (20
steps, step cache, conv decoder); ic-lora and dubit take distilled's
(2-step stage 2, conv decoder); retake adds 2x2-tiled attention (745 s ->
363 s at 1536x1024x121); alpha (406 s -> 191 s at 768x512x65) and
hdr-ic-lora (161 s -> 123 s) take the step cache and the conv decoder on
their single stage. `ltx25-t2a` has none: its 22 s is text encoding and
model load around an 8 s audio stage.

`ltx25-hq` (Lightricks' HQ preset, res_2s) takes 1902 s at the same size: half
the steps of dev but two model evaluations per step and the distilled LoRA
in both stages.

Every lever was measured alone first, with a PSNR against the exact clip and a
look at the frames. Kept: fewer refinement steps (30-33 dB, the same shot),
the conv decoder (35 dB), on dev fewer steps and the step cache
(21-23 dB each, the same shot at sheet scale), on DFR tiled attention (28 dB,
no seam: the reference tokens anchor every tile). Rejected: dropping dev's
modality guidance pass (its footsteps come out 4 dB softer on top of the 3 dB
that fewer steps already cost; a less-work-less-punch tradeoff, as with fine
detail, so the profile keeps all four passes), tiled attention
on distilled and dev (a seam through faces at the frame centre), the step
cache on the ancestral schedules (the block-0 residual moves 22-70% per step,
so it never fires), and 5-step distilled stage 1 is left out of the default
because it changes the composition (a different sample, not a worse one).
The settings are the profile's `engine.fast` block (`pipeline.Fast`); a
profile's request envelope and API are otherwise identical.

## Measured (M1 Ultra 64-core GPU, 128 GiB, macOS 15.7.2, seed 42)

Baseline is dgrauet/ltx-2-mlx v0.15.12, the fastest LTX-2.5 path on a Mac
before this work, in its recommended q8 pack and its bf16 pack. Wall time is
a cold process, prompt to mp4, including model loads.

| clip | pipeline | baseline q8 | baseline bf16 | SlimServe | vs q8 |
| --- | --- | ---: | ---: | ---: | ---: |
| 768x512x121 | distilled, conv decoder | 116.9 s | 202.4 s | **86 s** cold, **71 s** resident | 1.36x |
| 768x512x121 | dev, 30 steps | 606.9 s | | **343 s** | 1.77x |
| 768x512x121 | DFR | 170.9 s | | **125 s** | 1.37x |
| 1536x1024x121 | distilled | 504.9 s (tiled attention and decode; untiled dies in decode) | | **343 s**, untiled | 1.47x |

The baseline rows use its conv decoder; with the diffusion decoder (now the
default, ledger section 18) add about 23 s to a 768x512 clip and 127 s to an
HD one. SlimServe runs the unquantized official weights; the q8 baseline is an 8-bit
pack. Per-forward accuracy against an fp32 run of the same weights on
identical inputs: SlimServe rel-L2 0.0069, the baseline as shipped 0.042.
Evidence, per-op profiles and every intermediate number are in
`perf/ltx25_metal_campaign.md`.

## How it is fast

1. **Precision path.** On M1-M4 the only matrix-multiply-accumulate the GPU
   has is fp16. The baseline's fp32 AdaLN tables silently promote its whole
   residual stream to fp32, so every GEMM runs at a third of the fp16 rate.
   The engine keeps the residual stream, norms, AdaLN, RoPE tables, x0
   recovery and the model boundary in fp32 and casts to fp16 exactly at each
   GEMM and attention operand. The shipped bf16 weights convert to fp16
   exactly. This is 2.4x on the transformer and more accurate than the
   baseline. Nothing is quantized.
2. **Split-K GEMM.** MLX's fp16 GEMM falls from 19 to 10 TF/s when K reaches
   16384 (the FFN's second linear), and to 6 TF/s at 24k rows. Summing eight
   K=2048 GEMMs stays at 18-19 TF/s: 1.8 s per forward at 6k tokens, about
   17 s per forward at HD.
3. **Fused glue.** AdaLN modulate + cast, gated residual, GELU and the
   per-head gate are compiled chains, one kernel pass each. Bit-identical.
4. **Slab VAE.** MLX's conv3d scratch scales with the whole clip. Every
   convolution runs in temporal slabs, with the activation and padding done
   inside the slab. Exact output; the 121-frame decode went from about 60 GiB
   to 10 GiB and HD decodes untiled in 34 s at 27 GiB.
5. **Batched, de-duplicated guidance.** The dev pipeline's four passes per
   step run as one batch; the STG pass is forked from the conditional pass at
   block 28 instead of recomputed, and the three conditional-text passes
   share their text key/value projections. 10.9 to 9.1 s per step, exact.
6. **Runtime adapters.** The rank-450 distilled LoRA and the detailing
   IC-LoRA are applied as low-rank terms next to the base GEMM. No second
   39 GiB transformer, no reload between stages, exact detach.
7. **Diffusion decoder with a Metal neighborhood-attention kernel.** The
   recommended decoder's 3-D neighborhood attention has no fast MLX op; the
   baseline's formulation took 69 s for 49 frames. One Metal kernel (eight
   lanes per query, fp32 online softmax) decodes the same in 9 s, exact to
   1e-7 against a reference, and the 121-frame clip in 30 s.
8. **Shape-aware convolutions.** MLX's conv3d runs 1024-channel layers on
   small grids at 1.5 TF/s; those run as per-tap split-K GEMMs instead (the
   upscaler 2.5 to 0.45 s), chosen per shape by a persisted measurement.
9. **Compiled filters.** The vocoder's 12-tap anti-aliasing filters are
   compiled multiply-add chains instead of 200 dispatch-bound convolutions
   (audio 2.5 to 0.36 s).
10. **Text path.** Gemma-4 12B runs on the prompt's real tokens only (padding
   is masked and replaced by registers in the connector either way): 0.85 s
   per prompt.

Where it stands against the hardware: the M1 Ultra's measured fp16 MMA
ceiling is 19.4 TF/s (register-resident, section 17 of the ledger). MLX's
GEMM runs at 95-99% of it and its attention at 78%, so what is left on this chip is
small; the larger remaining wins are opt-in (step caching, sparse attention)
and the M5's int8 path. See the ledger's kernel sections for what was tried.

## Memory, and why the engine is strict about it

Metal memory is wired: it cannot be compressed or swapped. Overshooting does
not fail an allocation, it starves the OS (on 2026-10-02 four concurrent
model-loading jobs panicked this machine). The engine therefore:

- caps MLX active memory at RAM - 24 GiB - 16 GiB cache and the buffer cache
  at 16 GiB;
- loads each checkpoint in 4 GiB chunks, dropping the bf16 source as it casts;
- plans the VAE decode from a measured model (5 GiB + 135 B per output
  pixel-frame in fp32) against what the resident models leave, releases the
  text encoder first if that is what makes an untiled decode fit, and tiles
  only after that;
- refuses requests beyond the validated envelope.

Resident set while serving: distilled about 71 GiB (transformer 40, Gemma 28,
VAE/audio 3), dev about 80 GiB (plus the 8 GiB LoRA). The profiles are gated
to 128 GiB machines because that is what was validated.

Development rule (HANDOFF-LTX25.md): one model-loading process at a time, through
`perf/ltx25_harness/gpu_run.py`.

## Layout

| file | contents |
| --- | --- |
| `checkpoints.py` | official file set, safetensors headers, load-time dtype policy |
| `dit.py` | the transformer forward (upstream weight names), split-K, fused glue, LoRA hook, STG masks, first-block step cache, tiled self-attention |
| `text.py` | Gemma-4 tower, multi-layer feature projection, the two connectors |
| `sampling.py` | latent state, positions, noise, Euler / ancestral Euler, dev schedule and guider, DFR canvas and slots |
| `pipeline.py` | `LTX25Engine`: distilled, dev, dfr, render; memory budgeting; `Fast` (the fast tier's settings) |
| `lora.py` | runtime low-rank adapters |
| `vae.py`, `upscaler.py` | conv VAE (slab conv3d, tiling planner), latent upscalers |
| `image.py` | image-to-video still: decode, CRF-18 round trip, upstream resize/crop/normalize |
| `duration.py` | the duration head (auto clip length from the prompt) |
| `enhancer.py`, `prompts/` | the prompt enhancer: Gemma-4 E2B-it language model and vision tower, greedy decoding, upstream's system prompts |
| `audio.py`, `mux.py` | audio VAE (encoder and decoder) + vocoder + bandwidth extension; ffmpeg mux |
| `media.py` | source clips for the editing pipelines: ffmpeg probe / frame / audio decode as PyAV does, torchaudio's sinc resampler; EXR frame folders |
| `hdr.py` | the ACEScct working space, primaries, EXR read / write (OpenEXR), the HLG master (BT.2100 OETF, BT.2020 10-bit planes, libx265 through ffmpeg) |
| `chunks.py` | long clips in overlapping temporal windows: layouts, carry, keyframe plans, seam blends |
| `../server.py`, `../cli.py` | the job queue and HTTP API; `slimserve` integration |

## Not verified yet

Every upstream pipeline and option (ledger section 30) is ported; what
remains is evidence, not code:

- The I2V first-frame path is wired but its end-to-end output has not been
  compared against upstream on this machine.
- Dub-It ran its mechanics with a stand-in adapter: Lightricks has not
  published a Dub-It IC-LoRA for 2.5. Alpha-Gen, Colorization (through
  `ltx25-ic-lora`) and SDR-To-HDR are checked end to end with their real,
  gated adapters (ledger section 43; accept the terms on Hugging Face, then
  download the adapter files and pass them with `--lora`).
- M3+/M5 variants: native bf16 and the M5 int8 path are unverified on
  hardware and are separate profile records when they exist.
