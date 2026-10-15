# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 audio+video generation on Apple Silicon (MLX host, Metal kernels).

Design, dtype policy and evidence: perf/ltx25_metal_campaign.md and
docs/ltx25_metal.md. The official Lightricks bf16 safetensors are loaded
directly; there is no converted pack.
"""
