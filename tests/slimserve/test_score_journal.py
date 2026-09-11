# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
import torch

from slimserve.score_journal import STAGES, ScoreJournal, private_open


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.json"
    value = {
        "schema": 1,
        "prompt_ids": list(range(640)),
        "max_matches": 3,
        "output_directory": str(tmp_path / "journal"),
    }
    path.write_text(json.dumps(value))
    return path, value


def test_disabled_does_not_read_configuration(monkeypatch):
    monkeypatch.delenv("SLIMSERVE_GLM53_SCORE_JOURNAL", raising=False)
    assert ScoreJournal.from_env("unrelated") is None


def test_journal_permissions_ignore_permissive_umask(config):
    from slimserve.model_journal import ModelJournal

    previous = os.umask(0)
    try:
        score = ScoreJournal(config[0])
        model = ModelJournal(score)
    finally:
        os.umask(previous)
    try:
        assert stat.S_IMODE(score.path.parent.stat().st_mode) == 0o700
        for path in (score.path, model.path):
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        model.close()
        score.close()


@pytest.mark.parametrize("kind", ["public", "symlink"])
def test_nonprivate_journal_directory_rejected_without_chmod(config, kind):
    path, value = config
    directory = Path(value["output_directory"])
    if kind == "public":
        directory.mkdir(mode=0o755)
        directory.chmod(0o755)
    else:
        target = path.parent / "target"
        target.mkdir(mode=0o700)
        directory.symlink_to(target, target_is_directory=True)
    before = directory.lstat().st_mode
    with pytest.raises(ValueError, match="must be private"):
        ScoreJournal(path)
    assert directory.lstat().st_mode == before
    assert list(directory.iterdir()) == []


def test_private_open_preserves_existing_file_and_symlink_target(tmp_path):
    target = tmp_path / "existing"
    target.write_text("preserve")
    link = tmp_path / "link"
    link.symlink_to(target)
    for path in (target, link):
        with pytest.raises(FileExistsError):
            private_open(path)
    assert target.read_text() == "preserve"


def test_enabled_is_model_scoped(config, monkeypatch):
    path, _ = config
    monkeypatch.setenv("SLIMSERVE_GLM53_SCORE_JOURNAL", str(path))
    with pytest.raises(ValueError, match="scoped"):
        ScoreJournal.from_env("unrelated")
    journal = ScoreJournal.from_env("glm5_next")
    journal.close()
    with pytest.raises(FileExistsError):
        ScoreJournal.from_env("glm5_next_text")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "list",
        "schema",
        "schema_bool",
        "short_ids",
        "negative_id",
        "bool_id",
        "limit_zero",
        "limit_large",
        "limit_bool",
        "directory",
    ],
)
def test_invalid_config_creates_no_journal(config, mutation):
    path, value = config
    if mutation == "missing":
        del value["schema"]
    elif mutation == "extra":
        value["other"] = 1
    elif mutation == "list":
        value = []
    elif mutation == "schema":
        value["schema"] = 2
    elif mutation == "schema_bool":
        value["schema"] = True
    elif mutation == "short_ids":
        value["prompt_ids"].pop()
    elif mutation == "negative_id":
        value["prompt_ids"][0] = -1
    elif mutation == "bool_id":
        value["prompt_ids"][0] = True
    elif mutation == "limit_zero":
        value["max_matches"] = 0
    elif mutation == "limit_large":
        value["max_matches"] = 9
    elif mutation == "limit_bool":
        value["max_matches"] = True
    else:
        value["output_directory"] = ""
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        ScoreJournal(path)
    assert not (path.parent / "journal").exists()


def begin(journal, **changes):
    kwargs = dict(
        prompt_ids=list(range(640)),
        start_idx=0,
        num_logits=639,
        num_logprobs=0,
        request_id="test",
    )
    return journal.begin(**(kwargs | changes))


@pytest.mark.parametrize(
    "changes",
    [
        {"start_idx": 1},
        {"num_logits": 128},
        {"num_logprobs": 1},
    ],
)
def test_chunked_cached_or_topk_match_refused(config, changes):
    journal = ScoreJournal(config[0])
    with pytest.raises(ValueError, match="uncached"):
        begin(journal, **changes)
    assert journal.matches == 0
    journal.close()


def tensors():
    hidden = torch.tensor([0.0, -0.0, 1.5], dtype=torch.bfloat16).repeat(639, 1)
    return [
        hidden,
        hidden.float(),
        hidden.float(),
        torch.arange(1, 640),
        torch.zeros(639, 1),
    ]


def test_complete_repeats_hash_raw_bits_without_mutation(config):
    journal = ScoreJournal(config[0])
    assert begin(journal, prompt_ids=[9] * 640) is None
    assert journal.matches == 0
    values = tensors()
    expected = [
        hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
        for t in values
    ]
    for match in range(1, 4):
        assert begin(journal) == match
        for stage, tensor in zip(STAGES, values):
            journal.record(match, stage, tensor)
        journal.finish(match)
    assert begin(journal, prompt_ids=[9] * 640) is None
    with pytest.raises(ValueError, match="extra"):
        begin(journal)
    journal.close()
    rows = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert rows[0]["diagnostic_only"]
    assert (
        rows[0]["config_sha256"] == hashlib.sha256(config[0].read_bytes()).hexdigest()
    )
    assert len(rows) == 22
    for match in range(1, 4):
        records = [r for r in rows if r["kind"] == "tensor" and r["match"] == match]
        assert [r["sha256"] for r in records] == expected
        assert [r["dtype"] for r in records] == [str(t.dtype) for t in values]
        assert records[-2]["values"] == list(range(1, 640))
        assert records[-1]["values"] == [[0.0]] * 639
        assert records[0]["values"] is None
    assert expected == [
        hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
        for t in values
    ]


def test_order_overlap_and_wrong_ticket_refused(config):
    journal = ScoreJournal(config[0])
    with pytest.raises(ValueError, match="stage order"):
        journal.record(0, STAGES[0], tensors()[0])
    begin(journal)
    with pytest.raises(ValueError, match="overlapping"):
        begin(journal)
    with pytest.raises(ValueError, match="incomplete"):
        journal.finish(1)
    for match, stage in [(2, STAGES[0]), (1, STAGES[1])]:
        with pytest.raises(ValueError, match="stage order"):
            journal.record(match, stage, tensors()[0])
    for stage, tensor in zip(STAGES, tensors()):
        journal.record(1, stage, tensor)
    with pytest.raises(ValueError, match="stage order"):
        journal.record(1, STAGES[-1], tensors()[-1])
    journal.finish(1)
    journal.close()


@pytest.mark.parametrize("shape", [(), (638, 1), (639, 500000)])
def test_tensor_bounds_before_copy(config, shape):
    journal = ScoreJournal(config[0])
    begin(journal)
    # Meta tensors allocate no data; attempting CPU copy would fail differently.
    tensor = torch.empty(shape, device="meta")
    with pytest.raises(ValueError, match="byte bound"):
        journal.record(1, STAGES[0], tensor)
    journal.close()


def test_noncontiguous_logical_bytes_and_final_vector_bound(config):
    journal = ScoreJournal(config[0])
    begin(journal)
    tensor = torch.arange(639 * 4).reshape(639, 4)[:, ::2]
    assert not tensor.is_contiguous()
    for stage in STAGES[:3]:
        journal.record(1, stage, tensor)
    with pytest.raises(ValueError, match="one target"):
        journal.record(1, STAGES[3], tensor)
    journal.close()
    row = json.loads(journal.path.read_text().splitlines()[2])
    assert row["sha256"] == hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
