# GLM53 SM120 tensor-core mHC: independent accuracy qualification v1

Prescribed 2026-09-09, before running this new qualification. The operator
approved evaluating independent accuracy and model quality after the original
parity failure. This document does **not** change or supersede the original
strict parity gate: `mhc-tc-census/` remains FAILED at case2638/2700. The probe
stays outside serving until the independent and subsequent gates are complete.

## Scope and hypothesis

Same recipe `glm53-redhatai-nvfp4-fp8-kda-tp4-v1`: losslessly stored BF16 fn
and activations, FP32 accumulation/base/scale, BF16 KV/lm_head. Only dot
summation order changes. A parity comparator can reject a more accurately
rounded BF16 output. We will evaluate both installed and candidate outputs
against the mathematical equations in CPU FP64, without choosing either
kernel as the accuracy oracle. This is a bounded regression qualification,
not a general proof of accuracy or model capability.

The fused residual is outside the changed arithmetic: it must remain bit-exact
against the installed operator, including signed zero. The independent oracle
starts with that verified BF16 residual. It computes all24 dot products, RMS,
pre/post sigmoids, the full20-step Sinkhorn matrix, and the four-stream weighted
layer input. It does not reuse CUDA partial sums, reduction trees or kernels.

## Fixed numerical gates

- All inputs/outputs finite; exact shapes/dtypes; exact fused residual bits.
- FP32 post/comb against FP64: the original FP32 contract, rtol2e-5/atol2e-6.
- BF16 layer input: error against the unrounded FP64 ideal may not exceed
  the best possible BF16 rounding error plus an FP32 arithmetic budget.
  For each hidden coordinate, that budget is
  `(2e-5 + gamma4) * sum(abs(pre[s] * residual[s]))
   + 2e-6 * sum(abs(residual[s]))`, with
  `gamma4 = 4*2^-24 / (1 - 4*2^-24)`.
  This propagates the existing FP32 coefficient tolerance through the
  four-stream reduction and includes FP32 summation roundoff. It is an
  explicit engineering accuracy budget, not a derived universal error proof.
  Only8 FP64 eps times abs(ideal) is allowed for comparison roundoff.
- Direct FP64-to-BF16 round-to-nearest-even is independently tested at exact
  midpoints and their FP64 neighbors, for both signs, subnormals and signed
  zero. Torch's double-to-BF16 conversion alone is not the oracle because it
  can double-round through FP32. Overflow/nonfinite inputs fail closed.
- Both kernels must pass those same pointwise gates. Within every checked
  case, candidate normalized RMS error must be <=1.001 times baseline error
  +1e-7. This additional noninferiority check prevents accumulating individually
  admissible errors into an unnoticed aggregate degradation.
- The original sampled FP64 partial gate is unchanged: dotNRMS<=2e-6,
  row-peak<=2e-5, square relative error<=1e-5. Candidate graph outputs must be
  bit-exact to candidate eager outputs after three input changes; the installed
  graph must likewise match installed eager. Preserve strict baseline/candidate
  parity as a reported diagnostic, not as a renamed passing test.

## Prescribed census and timing

All90 actual parameter sites, batches64/65/128/129/7616, magnitudes.01/1/100,
pre/fused:2700 eager cases, all rows and all hidden columns in the FP64 gate.
Use original eager seeds301+site (pre),2003+3*site (fused). This includes the
exact failed site79/T7616/magnitude1/fused seed2240, not a replacement random
input. For each case, three held-out graph input seeds12001+3*site+replay;
bit checks cover every output, FP64 checks sample the predefined tile-boundary
rows and last row (never describe those sampled checks as all-row coverage).
All seeds, row coverage, numerical metrics, failures and code/library hashes
are recorded. Stop on a new accuracy failure and preserve the unfinished run;
do not silently retry, exclude it or widen this contract.

One bounded first check at site79/T7616 precedes the full prescribed census;
it includes all three magnitudes and both modes, not only the favorable point.
No tuning or source changes during a census. CPU memory is row-chunk bounded.
This accuracy run does not time kernels. Only after the full accuracy census
passes: five A/B/A rounds x four replays at all five batches, six banks of
all90 fn matrices (>3x128MiB L2), three warmups, both complete operators.
Preserve all timings; no serving-speed claim from these isolated graphs.

## Gates before serving promotion

1. Full independent census above; exact-input failure retained separately.
2. Memcheck and synccheck: sites0/89, batches64/65/129/7616, all three
   magnitudes, pre/fused. Targeted racecheck for64/65/129; retain exact command,
   kernel coverage and untested limits. Original strict parity runs may still
   fail: sanitizer correctness is not proof of old output parity.
3. Only then a gated SM120/BF16 implementation, with fallback for other dtypes,
   devices/alignment; native regression tests and unchanged selected recipe.
4. A fixed one-control/three-candidate/one-return unprofiled serving series,
   three repetitions c1/c8/c16 exact cold1000/300; text/image,4096 scored text
   tokens, all six needle contrasts, and cold32K/128K prefill on every start.
   No discarded/extra starts. Use matching prompt IDs and per-token scores.
5. Candidate mean continuation logprob must be no worse than the lower of
   the two controls by more than0.01 nat/token, both aggregate and per window.
   All needle margins remain positive; inspect changed per-token scores and
   generated outputs for failure, not merely the aggregate number. Report
   any control variation explicitly. This bounded quality gate is not general
   quality certification. Any failure prevents promotion pending diagnosis.
6. Retain only if serving prefill improves without a decode regression beyond
   the prescribed controls' measured spread. A microbenchmark-only gain is
   insufficient. Update the notebook, hashes, profile and commit at that point.
