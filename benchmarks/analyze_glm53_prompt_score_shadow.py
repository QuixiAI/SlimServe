# SPDX-License-Identifier: Apache-2.0
"""Join every live score chunk to original HTTP responses, without GPU imports."""

import array
import hashlib
import json
import sys
from pathlib import Path

from benchmarks.analyze_glm53_quality_pair import require, verified_quality
from slimserve.prompt_score_shadow import prompt_digest


def digest(values, code):
    require(sys.byteorder == "little", "recorded CUDA bytes require little-endian host")
    return hashlib.sha256(array.array(code, values).tobytes()).hexdigest()


def request_inventory(quality):
    verified_quality(quality)
    requests = [(row["prompt_ids"], row["response"]) for row in quality["text"]]
    requests += [
        (row["prefix_ids"] + item["suffix_ids"], item["response"])
        for row in quality["needles"]
        for item in row["candidates"]
    ]
    result = {}
    for ids, response in requests:
        key = prompt_digest(ids)
        require(key not in result, "ambiguous repeated quality request")
        require(
            response["usage"]["prompt_tokens"] == len(ids), "HTTP prompt count differs"
        )
        cached = response["usage"].get("prompt_tokens_details", {}).get("cached_tokens")
        require(cached == 0, "shadow quality requests must be uncached")
        rows = response["choices"][0]["prompt_logprobs"]
        require(len(rows) == len(ids) and rows[0] is None, "HTTP score extent differs")
        target_scores = [
            row[str(token)]["logprob"]
            for token, row in zip(ids[1:], rows[1:], strict=True)
        ]
        ranks = [
            row[str(token)]["rank"]
            for token, row in zip(ids[1:], rows[1:], strict=True)
        ]
        result[key] = dict(ids=ids, scores=target_scores, ranks=ranks)
    require(len(result) == 56, "expected all 32 text and 24 needle requests")
    return result


def audit(directory, quality, expected_sources):
    requests = request_inventory(quality)
    paths = sorted(Path(directory).glob("shadow-rank*-pid*.jsonl"))
    require(len(paths) == 4, "all four rank journals required")
    ranks, reports = set(), []
    for path in paths:
        events = [json.loads(line) for line in path.read_text().splitlines()]
        header = events[0]
        rank = header["rank"]
        require(
            header["kind"] == "header"
            and header["schema"] == 1
            and header["diagnostic_only"] is True
            and rank in range(4)
            and rank not in ranks
            and header["vocab"] == 154880
            and header["max_rows"] == 8192
            and header["copy_bytes"] == 32 * 1024**2
            and header["sources"]
            == {name: expected_sources[name] for name in header["sources"]}
            and set(header["sources"])
            == {
                "slimserve/prompt_score_shadow.py",
                "vllm/v1/worker/gpu_model_runner.py",
                "vllm/v1/sample/prompt_logprobs.py",
                "vllm/v1/sample/sampler.py",
                "vllm/v1/sample/ops/logprobs.py",
            },
            "wrong/duplicate rank or implementation header",
        )
        ranks.add(rank)
        require(
            len(events) % 2 == 1 and 1 < len(events) <= 1025,
            "incomplete/unbounded journal",
        )
        positions, seen_ids, paired_rows = {}, {}, []
        for call, index in enumerate(range(1, len(events), 2), 1):
            begin, complete = events[index : index + 2]
            key = begin["prompt_sha256"]
            require(
                begin["kind"] == "begin"
                and complete["kind"] == "complete"
                and begin["call"] == complete["call"] == call
                and key in requests,
                "missing/failed/out-of-order shadow call",
            )
            request = requests[key]
            rows, start = begin["rows"], begin["start_idx"]
            require(
                type(rows) is int
                and 1 <= rows <= 8192
                and start == positions.get(key, 0)
                and begin["prompt_tokens"] == len(request["ids"])
                and start + rows <= len(request["ids"]) - 1,
                "chunk interval has gap, overlap or wrong prompt",
            )
            require(
                key not in seen_ids or seen_ids[key] == begin["request_id"],
                "request identity changed",
            )
            seen_ids[key] = begin["request_id"]
            require(
                complete["count"] == 0
                and complete["mode"] == "raw_logprobs"
                and complete["paired"] is (rows > 1024),
                "incorrect scorer coverage",
            )
            inputs = complete["input_sha256"]
            require(
                (
                    isinstance(inputs, list)
                    and len(inputs) == 2
                    and all(
                        isinstance(h, str)
                        and len(h) == 64
                        and set(h) <= set("0123456789abcdef")
                        for h in inputs
                    )
                )
                if rows > 1024
                else inputs is None,
                "missing input preservation hashes",
            )
            require(
                complete["token_ids"] == request["ids"][start + 1 : start + 1 + rows]
                and complete["logprobs"] == request["scores"][start : start + rows]
                and complete["ranks"] == request["ranks"][start : start + rows],
                "worker target scores/ranks differ from HTTP",
            )
            outputs = complete["outputs"]
            require(len(outputs) == 3, "missing output fields")
            for field, values, code, dtype, shape in zip(
                outputs,
                (complete["token_ids"], complete["logprobs"], complete["ranks"]),
                ("i", "f", "q"),
                ("torch.int32", "torch.float32", "torch.int64"),
                ([rows, 1], [rows, 1], [rows]),
                strict=True,
            ):
                require(
                    field["shape"] == shape
                    and field["dtype"] == dtype
                    and field["reference_sha256"]
                    == field["chunked_sha256"]
                    == digest(values, code),
                    "output bit receipt differs from recorded/HTTP values",
                )
            if rows > 1024:
                require(
                    inputs[1] == digest(complete["token_ids"], "q"),
                    "target input hash differs",
                )
                paired_rows.append(rows)
            positions[key] = start + rows
        require(
            positions == {k: len(v["ids"]) - 1 for k, v in requests.items()},
            "not every HTTP prompt row was covered",
        )
        require(
            paired_rows and max(paired_rows) >= 7616, "no realistic long-chunk coverage"
        )
        reports.append(
            dict(
                rank=rank,
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                calls=(len(events) - 1) // 2,
                requests=len(positions),
                paired_calls=len(paired_rows),
                paired_rows=sum(paired_rows),
                paired_shapes=sorted(set(paired_rows)),
            )
        )
    require(ranks == set(range(4)), "rank coverage incomplete")
    return dict(
        status="complete",
        exact_live_scorer_parity=True,
        ranks=reports,
        scope=(
            "Same-input raw-logprob k0 parity and HTTP join; "
            "no model-quality, memory or TPS promotion"
        ),
    )
