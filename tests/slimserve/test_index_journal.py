# SPDX-License-Identifier: Apache-2.0
import hashlib
import inspect
import json
from types import SimpleNamespace

import pytest
import torch

from slimserve import index_journal as ij


def config(tmp_path, **changes):
    values = dict(
        schema=1,
        prompt_ids=list(range(8199)),
        max_matches=3,
        output_directory=str(tmp_path / "trace"),
    )
    values.update(changes)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    return path


@pytest.fixture
def journal(tmp_path):
    value = ij.IndexJournal(config(tmp_path))
    yield value
    value.close()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("SLIMSERVE_GLM53_INDEX_JOURNAL", "config.json")
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "1")
    monkeypatch.setenv("SLIMSERVE_GLM53_CANONICAL_MOE", "1")


def native(
    logits,
    cu_seqlen_ks,
    cu_seqlen_ke,
    raw_topk_indices,
    num_rows,
    stride0,
    stride1,
    topk_tokens,
):
    raw_topk_indices.fill_(-1)
    return logits


def run_topk(function, rows=4, columns=12):
    logits = torch.randn(rows, columns)
    starts = torch.zeros(rows, dtype=torch.int32)
    ends = torch.full_like(starts, columns)
    indices = torch.empty(rows, 512, dtype=torch.int32)
    assert function(logits, starts, ends, indices, rows, columns, 1, 512) is logits
    return indices


def test_disabled_identity_and_runner(monkeypatch):
    monkeypatch.delenv("SLIMSERVE_GLM53_INDEX_JOURNAL", raising=False)
    assert ij.instrument_topk(native) is native
    assert ij.instrument_pooled_indexer(native) is native
    runner = SimpleNamespace(_model_forward=native)
    ij.install_index_journal(runner)
    assert vars(runner) == {"_model_forward": native}


@pytest.mark.parametrize("missing", ["MODEL_JOURNAL", "CANONICAL_MOE"])
def test_prerequisites_fail_closed(enabled, monkeypatch, missing):
    monkeypatch.delenv("SLIMSERVE_GLM53_" + missing)
    with pytest.raises(ValueError, match="requires"):
        ij.enabled()


@pytest.mark.parametrize(
    "change",
    [
        {"schema": True},
        {"schema": 2},
        {"prompt_ids": [1]},
        {"prompt_ids": [-1] * 8199},
        {"prompt_ids": [True] * 8199},
        {"max_matches": 4},
        {"max_matches": 0},
        {"max_matches": True},
        {"output_directory": ""},
        {"output_directory": 12},
        {"extra": 1},
        {"capture_layer": None},
        {"capture_layer": True},
        {"capture_layer": "23"},
        {"capture_layer": 23.0},
        {"capture_layer": 0},
        {"capture_layer": 1},
        {"capture_layer": 45},
    ],
)
def test_config_strict(tmp_path, change):
    with pytest.raises(ValueError, match="requires"):
        ij.IndexJournal(config(tmp_path, **change))


def test_three_requests_multiple_chunks_and_extra_match(journal):
    for match in range(3):
        for computed, rows in ((0, 7616), (7616, 583)):
            journal.begin(str(match), computed, rows, "cpu")
            journal.layers = list(ij.LAYERS)
            journal.finish()
    with pytest.raises(ValueError, match="extra"):
        journal.begin("four", 0, 4, "cpu")
    events = [json.loads(line) for line in journal.path.read_text().splitlines()]
    assert len([e for e in events if e["kind"] == "request_complete"]) == 3
    assert len([e for e in events if e["kind"] == "forward_complete"]) == 6
    assert len(events[0]["implementation_sha256"]) == 9
    assert events[0]["selection_order"] == "native"
    assert events[0]["selection_ties"] == "native"
    assert events[0]["selection_order_implementation"] == "native"
    assert events[0]["capture_layer"] == 3


def test_overlap_missing_layers_and_wrong_continuation(journal):
    with pytest.raises(ValueError, match="incomplete"):
        journal.finish()
    journal.begin("one", 0, 4, "cpu")
    with pytest.raises(ValueError, match="overlapping"):
        journal.begin("two", 0, 4, "cpu")
    with pytest.raises(ValueError, match="incomplete"):
        journal.finish()
    journal.layers = list(ij.LAYERS)
    journal.finish()
    for request, cursor, device in (
        ("other", 4, "cpu"),
        ("one", 3, "cpu"),
        ("one", 4, "cuda:0"),
    ):
        with pytest.raises(ValueError, match="contract"):
            journal.begin(request, cursor, 4, device)


def test_chunk_bound(journal):
    for cursor in range(8):
        journal.begin("one", cursor, 1, "cpu")
        journal.layers = list(ij.LAYERS)
        journal.finish()
    with pytest.raises(ValueError, match="contract"):
        journal.begin("one", 8, 1, "cpu")


def test_cpu_mask_is_private_and_preserves_defined_bits():
    logits = torch.tensor(
        [
            [float("nan"), -0.0, 1.0, float("nan")],
            [float("nan"), 2.0, float("nan"), -3.0],
        ]
    )
    before = logits.view(torch.uint8).clone()
    starts, ends = (
        torch.tensor([1, 1], dtype=torch.int32),
        torch.tensor([3, 2], dtype=torch.int32),
    )
    host, lo, hi = ij.cpu_logits(logits, starts, ends)
    assert torch.equal(logits.view(torch.uint8), before)
    assert host.data_ptr() != logits.data_ptr()
    assert torch.equal(
        host, torch.tensor([[0.0, -0.0, 1.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    )
    assert bool(torch.signbit(host[0, 1]))
    assert torch.equal(lo, starts) and torch.equal(hi, ends)


@pytest.mark.parametrize("starts,ends", [([-1], [2]), ([2], [1]), ([0], [4])])
def test_invalid_ranges(starts, ends):
    with pytest.raises(ValueError, match="ranges"):
        ij.cpu_logits(
            torch.zeros(1, 3),
            torch.tensor(starts, dtype=torch.int32),
            torch.tensor(ends, dtype=torch.int32),
        )


def test_snapshot_archive_compacts_storage_and_hashes_raw_bits(journal, monkeypatch):
    journal.begin("one", 0, 4, "cpu")
    large = torch.zeros(400, dtype=torch.bfloat16)
    value = large[10:14]
    value[0] = -0.0
    journal.snapshot("sample", value, save=True)
    row = json.loads(journal.path.read_text().splitlines()[-1])
    path = journal.archive / "match-1-chunk-1-sample.pt"
    actual = torch.load(path, weights_only=True)
    assert actual.untyped_storage().nbytes() == value.numel() * value.element_size()
    assert (
        row["sha256"]
        == hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    )
    assert row["archive"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="duplicate"):
        journal.snapshot("sample", value)
    monkeypatch.setattr(ij, "MAX_TENSOR_BYTES", 1)
    with pytest.raises(ValueError, match="byte bound"):
        journal.snapshot("large", value)
    monkeypatch.setattr(ij, "MAX_TENSOR_BYTES", 128 * 1024**2)
    monkeypatch.setattr(ij, "MAX_FILE_BYTES", journal.saved_bytes + 1)
    with pytest.raises(ValueError, match="worker byte bound"):
        journal.snapshot("archive", value, save=True)


def test_wrappers_forward_without_active_context(enabled):
    wrapped = ij.instrument_topk(native)
    assert inspect.signature(wrapped) == inspect.signature(native)
    assert (run_topk(wrapped) == -1).all()
    pooled = ij.instrument_pooled_indexer(lambda k_cache_prefix: k_cache_prefix)
    assert pooled("arbitrary-untraced-layer") == "arbitrary-untraced-layer"


def test_complete_layer_coverage_forwards_arguments_and_snapshots(journal, enabled):
    journal.begin("one", 0, 4, "cpu")
    topk = ij.instrument_topk(native)

    @ij.instrument_pooled_indexer
    def pooled(k_cache_prefix):
        return run_topk(topk)

    token = ij.ACTIVE.set(journal)
    try:
        for layer in ij.LAYERS:
            assert (
                pooled(f"model.layers.{layer}.self_attn.indexer.k_cache") == -1
            ).all()
            assert ij.LAYER.get() is None
    finally:
        ij.ACTIVE.reset(token)
    journal.finish()
    assert len(list(journal.archive.glob("*.pt"))) == 4
    tensors = [
        json.loads(line)
        for line in journal.path.read_text().splitlines()
        if json.loads(line)["kind"] == "tensor"
    ]
    assert len(tensors) == 44


def test_missing_layer_and_selector_rejected_with_context_reset(journal, enabled):
    journal.begin("one", 0, 4, "cpu")
    token = ij.ACTIVE.set(journal)
    try:
        with pytest.raises(ValueError, match="missing"):
            run_topk(ij.instrument_topk(native))
        pooled = ij.instrument_pooled_indexer(lambda k_cache_prefix: None)
        with pytest.raises(ValueError, match="order"):
            pooled("model.layers.7.self_attn.k_cache")
        with pytest.raises(ValueError, match="unrecognized"):
            pooled("bad-prefix")
        with pytest.raises(ValueError, match="coverage"):
            pooled("model.layers.3.self_attn.k_cache")
        assert ij.LAYER.get() is None
    finally:
        ij.ACTIVE.reset(token)


@pytest.mark.parametrize("layer", ij.LAYERS)
def test_configured_layer_is_the_only_archived_layer(tmp_path, enabled, layer):
    value = ij.IndexJournal(config(tmp_path, capture_layer=layer))
    value.begin("selected-layer", 0, 4, "cpu")
    topk = ij.instrument_topk(native)

    @ij.instrument_pooled_indexer
    def pooled(k_cache_prefix):
        return run_topk(topk)

    token = ij.ACTIVE.set(value)
    try:
        for current in ij.LAYERS:
            pooled(f"model.layers.{current}.self_attn.indexer.k_cache")
        value.finish()
    finally:
        ij.ACTIVE.reset(token)
        value.close()
    paths = list(value.archive.glob("*.pt"))
    assert len(paths) == 4
    assert all(f"layer-{layer:02d}.call-0." in p.name for p in paths)
    header = json.loads(value.path.read_text().splitlines()[0])
    assert header["capture_layer"] == layer


def test_runner_configuration_and_unmatched_request(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("SLIMSERVE_GLM53_INDEX_JOURNAL", str(config(tmp_path)))
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=4096, num_hidden_layers=45)
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2, pipeline_parallel_size=1
        ),
        speculative_config=None,
        _model_forward=lambda **kw: kw,
        num_prompt_logprobs={},
    )
    with pytest.raises(ValueError, match="TP4"):
        ij.install_index_journal(runner)
    runner.parallel_config.tensor_parallel_size = 4
    ij.install_index_journal(runner)
    assert runner._model_forward(test="unmatched") == {"test": "unmatched"}
    runner._slimserve_index_journal.close()
