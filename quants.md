## DeepSeek v4 Flash 0731 antirez 

```
   Variant         Routed-expert quantization                                 Size    Practical positioning
  ━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   IQ2XXS + Q2K    Gate/up: IQ2_XXS; down/W2: Q2_K in every layer       80.764 GiB    Smallest, most aggressive
  ──────────────  ──────────────────────────────────────────────────  ─────────────  ─────────────────────────────────
   Hybrid          Layers 0–36 as above; layers 37–42 entirely Q4_K     90.889 GiB    Better-quality compact option
  ──────────────  ──────────────────────────────────────────────────  ─────────────  ─────────────────────────────────
   MXFP4           Gate/up/down all MXFP4                              145.264 GiB    FP4-oriented performance option
  ──────────────  ──────────────────────────────────────────────────  ─────────────  ─────────────────────────────────
   Q4K             Gate/up/down all Q4_K                               153.327 GiB    Quality-first quantization
```

### 1. IQ2XXS-w2Q2K — 80.8 GiB

  The routed experts use:

  - ffn_gate_exps and ffn_up_exps: IQ2_XXS, effectively 2.0625 bits/weight
  - ffn_down_exps, also called W2: Q2_K, effectively 2.625 bits/weight

  IQ2_XXS is an extremely compact importance-aware quantization. Q2_K retains slightly more information for the down projection, where
  quantization damage can be especially visible.

  This is the memory-saving version. Expect the largest reduction in reasoning, coding accuracy, and difficult long-context behavior compared with
  the 4-bit variants.

### 2. Layers37–42Q4K hybrid — 90.9 GiB

  This is the same layout as the 80.8 GiB model for layers 0–36, but the final six zero-indexed layers—37 through 42—use Q4_K for all routed-
  expert matrices.

  So:

  - Layers 0–36: gate/up IQ2_XXS, down Q2_K
  - Layers 37–42: gate/up/down Q4_K

  It spends about 10.1 GiB more to preserve the model’s final transformation stages. It should recover some quality versus the full IQ2/Q2 version
  without approaching the 153 GiB footprint of full Q4_K.

  fixed is only present in the filename; the GGUF metadata does not describe what was fixed. The file itself parses correctly.

### 3. MXFP4 experts — 145.3 GiB

  Every routed-expert matrix uses MXFP4:

  - 4-bit E2M1 floating-point values
  - One shared E8M0 scale per block of 32 values
  - Effective storage: 4.25 bits/weight

  MXFP4 is designed around efficient FP4 execution. It can be the throughput-oriented choice when the serving hardware and kernel backend have a
  strong MXFP4 path. On hardware without efficient FP4 support, its theoretical advantage can disappear because of conversion or emulation
  overhead.

  Unlike the other three, this file contains no recorded importance-matrix metadata.

### 4. Q4_K experts — 153.3 GiB

  Every routed-expert gate, up, and down tensor uses Q4_K, effectively 4.5 bits/weight.

  Q4_K uses sub-block scales and offsets and this file was created with an importance matrix. Of these four, it is the safest quality-first
  choice, assuming sufficient memory.

  It is about:

  - 8.1 GiB larger than MXFP4
  - 62.4 GiB larger than the hybrid
  - 72.6 GiB larger than IQ2/Q2
