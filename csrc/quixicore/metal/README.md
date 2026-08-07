# QuixiCore-Metal vendored kernels

Apple Silicon serving kernels vendored from
[QuixiAI/QuixiCore-Metal](https://github.com/QuixiAI/QuixiCore-Metal)
(ThunderKittens-derived, Metal Shading Language), providing the Apple
counterpart of the gfx942 and sm80 kernel work in the sibling directories.

- Vendored from commit: `71a08cd4cbcdc622ce31b3fc91e1f505e144b516` (2026-07-27),
  branch `main`.
- Source layout mirrors `QuixiCore-Metal/` so files diff cleanly against
  upstream; resync by copying the same paths.
- Everything here is an unmodified copy. SlimServe-specific glue lives in
  `csrc/quixicore/tm_metal/` (the binding) and `vllm/quixicore/` (the Python
  surface), never in these files.

## What was and was not taken

Taken:

| Path | What |
| --- | --- |
| `kernels/**/*.metal` | 79 MSL kernel files, the whole op surface |
| `include/metal/**` | The tile substrate `tk.metal` pulls in |
| `kernels/common/tk_launch.h` | Host-side launch ABI — pure C++, no MLX, no Metal |
| `kernels/common/base_q_descriptor.h` | BaseQN packed-quant descriptor |

Deliberately not taken:

- The per-kernel `.cpp`/`.h` pairs beside each `.metal`. Those are MLX
  primitives (`#include "mlx/ops.h"`); this fork drives the kernels from
  PyTorch MPS instead, through `tk_launch.h`, which is why that header is the
  only host-side file worth carrying.
- `bindings/python/` and `bindings/mlx/` — the MLX extension and a full
  vendored MLX source tree, both irrelevant here.
- `bindings/pytorch_mps/` — its `torch_kernels.mm` is the structural model for
  `csrc/quixicore/tm_metal/`, but it JIT-builds the metallib and the extension
  at *import* (~22 s cold on an M5 Max). A served fork compiles both ahead of
  time, so the binding is written here rather than copied.

## Building the metallib

The kernels compile to a single `.metallib` as a build step (see the `METAL`
block in the top-level `CMakeLists.txt`). The recipe upstream uses, and the one
to keep in sync with:

```bash
xcrun metal -std=metal3.1 -O2 -I include/metal \
  $(find kernels -name '*.metal') -o quixicore_metal.metallib
```

This needs the Xcode Metal toolchain component, which is a separate download
from Xcode itself:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Two `-Wunused-variable` warnings come out of `include/metal/ops/warp/shared/tile/maps.metal`.
They are upstream's and are left alone.
