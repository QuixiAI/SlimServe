# SPDX-License-Identifier: Apache-2.0
import json

import pytest

from benchmarks.kernels.profile_glm53_marlin import route_cases


def make_journal(tmp_path, *, invalid=False):
    header = dict(
        kind="header", num_layers=45, first_moe_layer=3, num_experts=288, top_k=8
    )
    rows = [header]
    for step in range(9):
        for batch in [1, 8, 16]:
            rows.append(dict(kind="decode", step=step, request_ids=list(range(batch))))
    if invalid:
        rows.append(dict(kind="invalid"))
    path = tmp_path / "journal.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_selection_is_predetermined_and_keeps_actual_steps(tmp_path):
    path = make_journal(tmp_path)
    selected = route_cases(path, [1, 8, 16], 3)
    assert [(batch, row["step"]) for batch, row in selected] == [
        (batch, step) for batch in (1, 8, 16) for step in (0, 4, 8)
    ]
    assert route_cases(path, [8], 1)[0][1]["step"] == 4


def test_reject_failed_or_insufficient_capture(tmp_path):
    with pytest.raises(ValueError, match="invalid capture"):
        route_cases(make_journal(tmp_path, invalid=True), [8], 1)
    with pytest.raises(ValueError, match="insufficient"):
        route_cases(make_journal(tmp_path), [8], 10)


def test_reject_other_model_journal(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text(
        json.dumps(
            dict(
                kind="header",
                num_layers=46,
                first_moe_layer=3,
                num_experts=288,
                top_k=8,
            )
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="GLM53"):
        route_cases(path, [8], 1)
