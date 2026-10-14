# SPDX-License-Identifier: Apache-2.0
"""CPU interpreter witness for the actual prefill metadata kernel's row maps.

Run with CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 OMP_NUM_THREADS=1.
This executes the repository Triton metadata source, not a reimplemented map.
It is not GPU validation and does not modify serving code or global defaults
outside this isolated process.
"""

import json
import os

import torch

from vllm.v1.attention.backends.mla import indexer


def main():
    assert os.environ.get("TRITON_INTERPRET") == "1"
    assert not torch.cuda.is_available(), "This witness must remain CPU-only"
    indexer._use_native_indexer_metadata = lambda: False
    cases = [
        ("fresh", [4, 8], [4, 8], slice(0, 12)),
        ("cached_prefix", [4096, 8192], [4, 8], slice(0, 12)),
        ("query_slice", [4, 8], [4, 8], slice(5, 9)),
        ("cached_query_slice", [4096, 8192], [4, 8], slice(5, 9)),
    ]
    for name, lengths, queries, query_slice in cases:
        seq_lens = torch.tensor(lengths, dtype=torch.int32)
        query_lens = torch.tensor(queries, dtype=torch.int32)
        locs = torch.cat((torch.zeros(1, dtype=torch.int32),
                          query_lens.cumsum(0).int()))
        chunk = indexer.build_prefill_chunk_metadata(
            0, len(lengths), locs, locs, seq_lens, seq_lens, seq_lens,
            torch.zeros(len(lengths), 128, dtype=torch.int32), 1,
            query_slice=query_slice,
        )
        rows = chunk.token_end - chunk.token_start
        legacy = chunk.token_to_seq[:rows]
        expected = torch.repeat_interleave(torch.arange(len(lengths)), query_lens)
        expected = expected[query_slice]
        # DCP=1 and compression=1: query row starts identify block-table rows.
        proposed = torch.index_select(chunk.token_to_seq, 0, chunk.cu_seqlen_ks)
        assert torch.equal(proposed, expected)
        mismatch = not torch.equal(legacy, expected)
        assert mismatch == (name != "fresh")
        print(json.dumps(dict(case=name, legacy=legacy.tolist(),
                              expected=expected.tolist(), proposed=proposed.tolist(),
                              mismatch_detected=mismatch,
                              scope="Actual Triton kernel, CPU interpreter only")),
              flush=True)


if __name__ == "__main__":
    main()
