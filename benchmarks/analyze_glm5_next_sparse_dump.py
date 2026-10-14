# SPDX-License-Identifier: Apache-2.0
"""CPU-only analysis of the opt-in GLM sparse-attention failure snapshot."""

import argparse
import json

import torch


def analyze(data):
    q, out = data["q"], data["out"]
    idx, bt = data["idx"].long(), data["bt"].long()
    pages, ids = data["cache_pages"], data["page_ids"]
    size = data["block_size"]
    valid = idx >= 0
    cols = (idx.clamp_min(0) // size).clamp_max(bt.shape[1] - 1)
    physical = bt.gather(1, cols)
    local = torch.searchsorted(ids, physical).clamp_max(len(ids) - 1)
    row_finite = torch.isfinite(pages).all(-1)
    referenced_bad = valid & ~row_finite[local, idx.clamp_min(0) % size]
    bad_out = ~torch.isfinite(out).flatten(1).all(1)
    bad_q = ~torch.isfinite(q).flatten(1).all(1)
    record = dict(
        layer=data["layer_name"], q_shape=list(q.shape),
        cache_shape=data["cache_shape"], cache_stride=data["cache_stride"],
        q_bad_rows=bad_q.nonzero().flatten().tolist(),
        output_bad_rows=bad_out.nonzero().flatten().tolist(),
        selected_bad_counts=referenced_bad.sum(1).tolist(),
        selected_min=torch.where(valid, idx, torch.iinfo(idx.dtype).max)
        .min(1).values.tolist(),
        selected_max=idx.max(1).values.tolist(),
        page_ids=ids.tolist(), bad_cache_rows_per_page=(~row_finite).sum(1).tolist(),
        block_table_distinct_rows=torch.unique(bt, dim=0).shape[0],
        logical_index_oob=int((valid & (idx // size >= bt.shape[1])).sum()),
    )
    insert = data.get("insert")
    if insert is not None:
        record["insert"] = {key: insert[key] for key in (
            "equal", "source_bad", "written_bad", "cache_shape", "cache_stride")}
        record["insert"]["slots"] = insert["slots"].tolist()
    examples = [0]
    if bad_out.any():
        examples.append(int(bad_out.nonzero()[0]))
    record["examples"] = []
    for row in sorted(set(examples)):
        bad_cols = referenced_bad[row].nonzero().flatten()
        selected = valid[row]
        kv = pages[local[row, selected], idx[row, selected] % size].float()
        # Scale is not required to diagnose finiteness. Report Q and KV facts
        # separately rather than claiming a numerical oracle for unknown scale.
        scores = q[row].float() @ kv.T
        record["examples"].append(dict(
            query=row, selected_bad_tokens=idx[row, bad_cols].tolist(),
            selected_bad_slots=(physical[row, bad_cols] * size
                                + idx[row, bad_cols] % size).tolist(),
            unscaled_scores_finite=bool(torch.isfinite(scores).all()),
            selected_kv_max_finite_abs=float(torch.nan_to_num(kv).abs().max()),
        ))
    history = data.get("write_history", [])
    record["write_steps"] = [dict(
        step=step, max_seq_len=write["max_seq_len"],
        tokens=write["slots"].numel(),
        pages=torch.unique(write["slots"][write["slots"] >= 0] // size).tolist(),
    ) for step, write in enumerate(history)]
    record["bad_page_history"] = []
    for page_id in ids[(~row_finite).any(1)].tolist():
        expected = torch.zeros_like(pages[0])
        known = torch.zeros(size, dtype=torch.bool)
        writers = []
        for step, write in enumerate(history):
            slots = write["slots"]
            slots = slots[slots >= 0]
            take = slots // size == page_id
            if take.any():
                offsets = slots[take] % size
                expected[offsets] = write["source"][take]
                known[offsets] = True
                writers.append(step)
        current = pages[int((ids == page_id).nonzero()[0])]
        same = (current == expected).all(-1) & known
        record["bad_page_history"].append(dict(
            page=page_id, writer_steps=writers, known_rows=int(known.sum()),
            matching_rows=int(same.sum()),
            differing_known_rows=(known & ~same).nonzero().flatten().tolist(),
        ))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    args = parser.parse_args()
    data = torch.load(args.dump, map_location="cpu", weights_only=True)
    print(json.dumps(analyze(data)))


if __name__ == "__main__":
    main()
