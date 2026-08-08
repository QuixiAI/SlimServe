# SlimServe repository guidance

## Scope and ownership

- We own the complete serving stack, including Python orchestration, vLLM,
  ROCm/HIP code, communication paths, generated kernels, and handwritten GPU
  kernels.
- Do not stop at an upstream-library boundary or classify a kernel failure as
  external. Reproduce it, isolate the first failing operation, and fix the
  responsible layer in this repository or the owned dependency tree.
- Long-running and concurrent workloads are correctness requirements. A short
  smoke test does not replace the exact workload that exposed a failure.

## GPU implementation scope

- AITER is a supported dependency. Do not remove or replace a working AITER
  path solely to eliminate that dependency.
- AITER and its kernels are part of the stack we debug and fix when a supported
  workload exposes a failure there.
- Kernel work is in scope. Use serialized launches, device assertions,
  sanitizers, reduced reproducers, and targeted instrumentation as needed.

## Reference implementations

Use these local trees when checking algorithms, layouts, bounds, and kernel
behavior:

- `~/llama.cpp`
- `~/ds4`
- `~/QuixiCore/QuixiCore-ROCm`
- `~/QuixiCore/QuixiCore-Metal`

## Profile validation

- Live validation must discover every registry profile compatible with the
  current machine. Do not substitute one tensor-parallel size for another.
- Every supported profile must run with its registered DSpark drafter and
  TurboQuant draft KV configuration.
- For vision profiles, test both text and image requests. For text-only
  profiles, test text requests.

## Commit authorship

- Eric Hartford is the sole author. Do not add co-author or assistance
  trailers, and do not discuss automated assistance in commit messages.
