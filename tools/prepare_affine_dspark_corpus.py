"""Filter and interleave on-policy Affine King responses for DSpark training."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--magpie", type=Path, required=True)
    parser.add_argument("--ultrachat", type=Path, required=True)
    parser.add_argument("--extra", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-loss-tokens", type=int, default=256)
    args = parser.parse_args()

    selected = []
    summary = {}
    seen_ids = set()
    sources = [("magpie", args.magpie), ("ultrachat", args.ultrachat)]
    sources.extend((f"extra_{i}", path) for i, path in enumerate(args.extra))
    for name, path in sources:
        rows = []
        counts = {
            "read": 0,
            "duplicates": 0,
            "truncated": 0,
            "too_long": 0,
            "too_short": 0,
        }
        with path.open() as source:
            for line in source:
                row = json.loads(line)
                counts["read"] += 1
                row_id = row["primary_id"]
                if row_id in seen_ids:
                    counts["duplicates"] += 1
                    continue
                if row["metadata"].get("finish_reason") != "stop":
                    counts["truncated"] += 1
                elif len(row["input_ids"]) > args.max_length:
                    counts["too_long"] += 1
                elif sum(row["loss_mask"]) < args.min_loss_tokens:
                    counts["too_short"] += 1
                else:
                    rows.append(row)
                    seen_ids.add(row_id)
        counts["selected"] = len(rows)
        counts["selected_tokens"] = sum(len(row["input_ids"]) for row in rows)
        summary[name] = counts
        selected.append(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as destination:
        main = selected[:2]
        for pair in zip(*main, strict=False):
            for row in pair:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        for rows in main:
            for row in rows[min(map(len, main)) :]:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        for rows in selected[2:]:
            for row in rows:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["total_selected"] = sum(len(rows) for rows in selected)
    summary["max_length"] = args.max_length
    summary["min_loss_tokens"] = args.min_loss_tokens
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
