# AITER dependency patches

SlimServe builds AITER from the exact upstream commit recorded by
`AITER_COMMIT` in `docker/Dockerfile.rocm_base`, then applies every patch in
this directory before compiling the wheel.

`custom-all-reduce-graph-buffer-order.patch` keeps CUDA/HIP graph input and
output buffers in capture order. AITER previously stored inputs and outputs in
separate growing lists. The second all-reduce input therefore reused the graph
slot already assigned to the first output, which could produce an illegal peer
memory access during graph replay. The patch also extends AITER's multi-GPU
test to capture two consecutive all-reduces.

To reproduce the dependency outside the container build:

```bash
git -C /path/to/aiter checkout --detach \
  17f24ec6e93f48722a6b4ec8e54738434194f3d6
git -C /path/to/aiter apply --unidiff-zero \
  /path/to/SlimServe/docker/patches/aiter/custom-all-reduce-graph-buffer-order.patch
```

The zero-context form keeps the vendored patch free of trailing whitespace.
It must continue to pass `git apply --unidiff-zero --check` when `AITER_COMMIT`
is updated. The container build enforces that invariant before compilation.
