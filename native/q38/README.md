# q38: native Metal engine for Qwen3.8-27B (Apple Silicon)

A purpose-built engine for one model on one platform: Qwen/Qwen3.8-27B (64
layers = 16 x (3 gated-DeltaNet + 1 gated full attention), 27.78B weights) on
an Apple GPU. One Metal command buffer per step over mmap'd, no-copy weights.
A step carries rows of up to 8 sequences, so concurrent requests share every
pass over the weights.

## Layout

- `include/q38/engine.h` - C ABI: `q38_step(tokens, m)` / `q38_commit(n)` over
  one sequence (decode, prefill chunks and speculative verify are the same
  call), host sampling, the MTP draft head, sequence snapshots.
  The same header has the multi-sequence calls (`q38_seq_*`,
  `q38_step_groups`, `q38_mtp_feed_groups`) and image rows (`q38_step_mm`:
  embedding rows and three-axis rotary positions).
- `include/q38/generate.h` - single-stream generator: prompt handling with
  prefix snapshots, plain or speculative (MTP, lossless rejection sampling)
  decode cycles.
- `include/q38/pool.h` - stream pool: several generations in shared passes
  (decode rows, drafts, prompt chunks), prefix reuse shared by all streams.
  This is what the server calls.
- `src/engine.mm`, `src/generate.cpp`, `src/pool.cpp` - implementation.
- `kernels/*.metal` - the engine's own kernels (NVFP4 and u4/u5 GEMV
  families, the many-row GEMM for prompt chunks, GDN core, residual norm,
  grouped-query attention). Other kernels come from QuixiCore
  (`csrc/quixicore/metal`).
- `tools/` - `q38_lab` (bench and self-tests), packers (`pack_hf.py`,
  `pack_mtp.py`), CPU oracle (`q38ref.py`), KL harness (`kld.py`),
  quantizers (`quantize.py`, `gptq.py`, `quantize_fp8.py`), `mixpack.py`
  (per-tensor mixes of packs), `mix_eval.sh`, `bench.sh`.
- Python binding, vision tower and OpenAI server: `slimserve/native/q38.py`,
  `slimserve/native/vision.py`, `slimserve/native/server.py`.

## Build

    ./build.sh        # build/q38_lab, build/libq38.dylib, build/q38_native.metallib

The QuixiCore kernels must be compiled for this OS's Metal language version
(the metallib shipped in `vllm/` targets Metal 4.0 and does not load on
macOS 15):

    cd ../../csrc/quixicore/metal
    xcrun metal -std=metal3.1 -O2 -I include/metal -I kernels/common \
      $(find kernels -name '*.metal') -o ../../../native/q38/build/q38_kernels.metallib

## Run

    ML=build/q38_kernels.metallib:build/q38_native.metallib
    ./build/q38_lab --pack W.q38w --metallib $ML --tokens prompt.tokens --gen 300 --sample
    ./build/q38_lab --pack W.q38w --metallib $ML --tokens prompt.tokens --gen 300 \
        --mtp MTP.q38w --depth 2 --adaptive 0.5            # speculative
    tools/bench.sh W.q38w MTP.q38w -- --depth 2 --adaptive 0.5

DSpine block drafter (tencent/Qwen3.8-27B-DSpine-b16) instead of the MTP head:
one pass proposes a block of up to 7 tokens, the generator and the pool use it
when it is loaded. Same aggregate speed as MTP on the bench set (60.1 against
60.0 tok/s), faster where text is predictable (math 76 against 68.5), slower on
free prose (essay 52 against 56); with two streams decoding the MTP head is
ahead (79 against 74 tok/s aggregate), from three on neither drafts.

    python3 tools/draft_vocab.py rank.json                 # token frequency ranking (once)
    python3 tools/pack_dspine.py --src DSPINE_HF_DIR --target TARGET_HF_DIR \
        --vocab-rank rank.json --out D-bf16.q38w
    python3 tools/quantize.py --src D-bf16.q38w --out D.q38w
    ./build/q38_lab ... --dspine D.q38w --depth 7 --adaptive -0.06 --gen 600 --gen-once
    DSPINE=$HOME/.../D.q38w tools/bench.sh W.q38w -- --depth 7 --adaptive -0.06
    ./build/q38_lab ... --tokens TRAJ.tokens --plen P --dspine D.q38w --dspine-accept OUT.q   # teacher-forced q(true)
    server: --q38-dspine-pack D.q38w

Self-tests (exit code 0 = pass):

    ./build/q38_lab ... --snap-test       # snapshot restore is bit-exact
    ./build/q38_lab ... --mtp MTP.q38w --gen 150 --gen-api   # generator API, snapshot reuse
    ./build/q38_lab ... --multi-test 4    # 4 sequences in shared steps: isolation is bit-exact
    ./build/q38_lab ... --mtp MTP.q38w --gen 300 --depth 2 --adaptive 0.5 --pool-bench 1
                                          # one pool stream == the generator API, token for token

Prompt prefill: a step of one sequence may carry up to 64 rows
(`q38_big_rows`); it goes through `kernels/q38_gemm.metal` at about twice the
rows per second of 8-row steps and leaves only its last rows addressable. The
pool uses such steps for long prompts (`big_every` in `q38_pool_config`).

    ./build/q38_lab ... --big-test 64     # 8-row vs 64-row prefill: timing, then decode rows compared
    Q38_BT_HID=1 ./build/q38_lab ... --cap 4 --big-test 64   # same on the final hidden rows (works under --cap)
    python tools/kld.py engine ... --big-head 192            # KL of the rows after a big prefill
    python tools/kld.py engine ... --score-from 192          # the same rows after an 8-row prefill

Concurrency benchmark (N streams of the prompt, different seeds):

    ./build/q38_lab ... --mtp MTP.q38w --gen 300 --depth 2 --adaptive 0.5 --pool-bench 4 [--stagger 20]

Quality (teacher-forced KL against BF16 reference logits):

    python tools/kld.py engine --ref ref_eval.npz --set eval_set.json --pack W.q38w --chunk 4

Server (text and images; see the module docstring for what is rejected):

    PYTHONPATH=../.. python -m slimserve.native.server --model <hf dir> \
        --q38-pack W.q38w --q38-mtp-pack MTP.q38w --enable-auto-tool-choice \
        --tool-call-parser qwen3_xml --reasoning-parser qwen3 [--q38-max-streams 8]

The vision tower runs through PyTorch in the server process (needs
torchvision for the HF image processor).

While the tree is being changed, tools take the engine binaries from
`$Q38_BUILD` (a frozen copy such as `build.stable/`) instead of `build/`.

## Rules of the house

- One GPU process at a time on the M1 Ultra: stop the server before running
  the lab or the KL harness.
- Benchmarks sample at the model defaults (temperature 1.0, top-p 0.95, top-k
  20, seeded), thinking on.
- Metal counts whole buffers in the process footprint: per-sequence K/V is
  allocated as a sequence grows, never for the full context up front.
- A struct with 64-bit members read from a `device` buffer came out wrong in
  the GDN kernel (the same layout is fine in a `setBytes` constant buffer);
  kernel-side tables use 32-bit fields.
- Environment switches (`Q38_FOLD`, `Q38_XH`, `Q38_ATTN_OFF`, `Q38_GDN_OFF`,
  `Q38_NORM_OFF`, `Q38_GU_OFF`, `Q38_NV`, `Q38_SKIP`, `Q38_KLOG`) are
  diagnostics; defaults are the measured-best paths. Results and rejected
  experiments are in `perf/optimization_status.md`.
