# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for the GPU-gated owned sparse MLA kernel."""

from vllm.quixicore.sparse_mla_tc import compile_only, sparse_tc_nope

__all__ = ["sparse_tc_nope"]


if __name__ == "__main__":
    compile_only()
