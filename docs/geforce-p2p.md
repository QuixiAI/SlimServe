# GeForce multi-GPU serving: use the P2P driver

SlimServe's tensor-parallel profiles for consumer NVIDIA cards (RTX 3090 /
4090 / 5090, e.g. `qwen38fn-fp8-8` on 8x 3090) run fine on the stock
NVIDIA driver — but if you have **two or more consumer cards and want the
throughput they can actually deliver, install the P2P driver**. Stock
drivers refuse P2P on GeForce boards, which forces every inter-GPU byte
through host RAM; on a typical EPYC/Threadripper host the concurrent
device-to-host writes collapse to a few GB/s aggregate, and TP decode is
bounded by collective latency instead of kernel speed. Measured on
`qwen38fn-fp8-8`, the difference is +36% single-stream and +58% at
concurrency 8 (table below).

The driver is **QuixiAI's patched fork**:

> https://github.com/QuixiAI/open-gpu-kernel-modules

It is NVIDIA's open GPU kernel module tree (610.57.04) with a small set of
force-enables — inspired by George Hotz / tinygrad's original patch — that
turn on the driver's native BAR1-P2P path for GA102 / AD102 / GB202. The
repo's README is the authoritative install guide, including the two parts
people miss:

1. **BAR1 must be resized to cover the framebuffer** (32 GiB on a 3090).
   With the stock 256 MiB BAR1 the P2P matrix reports OK and bandwidth
   tests pass, but NCCL hangs at its first collective. Desktop boards: turn
   on Resizable BAR in BIOS. Server boards without that switch: use
   `tools/resize-bar1.sh` from the driver repo (and its systemd unit — the
   resize must be redone every boot).
2. **Kernel command line needs `iommu=pt`** (with `pci=realloc` on server
   boards). IOMMU translated mode collapses peer bandwidth to ~1 GB/s.

## What SlimServe expects

GeForce profiles set `NCCL_P2P_LEVEL=SYS` in their profile environment —
NCCL's default refuses P2P for GPU pairs whose path crosses the CPU root
complex, silently falling back to SHM. The setting is harmless on a stock
driver (it only widens which pairs *may* use P2P; with no P2P capability
NCCL falls back to SHM as usual), so the same profile serves with or
without the patched driver. The profile env is applied by
`slimserve <profile> --serve`; nothing to do manually, but if you
benchmark collectives outside SlimServe, set it yourself.

Verify the fabric before blaming a profile for low throughput:

```bash
nvidia-smi -q -i 0 | grep -A1 "BAR1 Memory"   # Total: 32768 MiB, not 256
nvidia-smi topo -p2p r                         # all OK
```

The acid test is a real NCCL collective (hangs if BAR1 is small); the
driver repo README carries a two-rank snippet.

## Measured impact (8x RTX 3090, EPYC Rome, PCIe 4.0)

| Metric | Stock (SHM) | P2P driver + 32 GiB BAR1 |
| --- | ---: | ---: |
| NCCL all-reduce busbw (84 MB) | 2.7 GB/s | 24.7 GB/s |
| `qwen38fn-fp8-8` c1 tok/s | 109.61 | 148.81 (+36%) |
| `qwen38fn-fp8-8` c8 tok/s | 409.66 | 647.88 (+58%) |

Full history and raw logs: `perf/baseline_status.md`,
`perf/results/2026-08-27/qwen38fn-3090-p2p/`.
