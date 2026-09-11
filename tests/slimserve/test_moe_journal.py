# SPDX-License-Identifier: Apache-2.0
import hashlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from slimserve import moe_journal as mj
from slimserve.model_journal import ACTIVE


class FakeJournal:
    def __init__(self, path):
        self.path = path
        self.matches = 1
        self.operations = 8
        self.pending = None
        self.moe_capture = None
        self.rows = []

    def write(self, row):
        self.rows.append(row)

    def record(self, stage, tensor):
        host = tensor.detach().cpu().contiguous()
        self.write(
            {
                "kind": "tensor",
                "stage": stage,
                "shape": list(tensor.shape),
                "sha256": hashlib.sha256(
                    host.reshape(-1).view(torch.uint8).numpy()
                ).hexdigest(),
            }
        )
        return host


@pytest.fixture
def journal(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "1")
    monkeypatch.setenv("SLIMSERVE_GLM53_MOE_JOURNAL", "1")
    return FakeJournal(tmp_path / "model.jsonl")


@pytest.mark.parametrize(
    "decorator",
    [mj.instrument_router, mj.instrument_marlin_gemm, mj.instrument_moe_sum],
)
def test_disabled_decorators_are_identity(monkeypatch, decorator):
    monkeypatch.delenv("SLIMSERVE_GLM53_MOE_JOURNAL", raising=False)
    function = lambda: None
    assert decorator(function) is function


@pytest.mark.parametrize("moe,model", [("yes", "1"), ("1", "0")])
def test_bad_flags_fail(monkeypatch, moe, model):
    monkeypatch.setenv("SLIMSERVE_GLM53_MOE_JOURNAL", moe)
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", model)
    with pytest.raises(ValueError):
        mj.enabled()


def test_snapshot_values_metadata_and_parameter_dedup(journal):
    capture = mj.MoECapture(journal)
    original = torch.arange(80, dtype=torch.float32)
    tensor = original[4:12]
    for match in (1, 2, 3):
        journal.matches = match
        capture.snapshot("up.b_scales", tensor, parameter=True)
    files = list(capture.directory.glob("*.pt"))
    assert len(files) == 1
    saved = torch.load(files[0], weights_only=True)
    assert torch.equal(saved, tensor)
    assert saved.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    assert torch.equal(original, torch.arange(80, dtype=torch.float32))
    row = next(r for r in journal.rows if r["kind"] == "moe_snapshot")
    assert row["file_bytes"] == capture.saved_bytes == files[0].stat().st_size
    assert row["file_sha256"] == hashlib.sha256(files[0].read_bytes()).hexdigest()
    assert len([r for r in journal.rows if r["kind"] == "tensor"]) == 3


def test_duplicate_stage_and_unsafe_path_fail(journal):
    capture = mj.MoECapture(journal)
    capture.snapshot("up.input", torch.ones(1))
    with pytest.raises(ValueError, match="duplicate"):
        capture.snapshot("up.input", torch.ones(1))
    with pytest.raises(ValueError, match="stage"):
        capture.snapshot("../../bad", torch.ones(1))


def test_dump_bound_fails_before_file_creation(journal, monkeypatch):
    capture = mj.MoECapture(journal)
    monkeypatch.setattr(mj, "MAX_DUMP_BYTES", 16)
    with pytest.raises(ValueError, match="byte bound"):
        capture.snapshot("up.input", torch.ones(1))
    assert not list(capture.directory.iterdir())


def test_router_wrapper_forwards_same_objects_and_only_at_site8(journal):
    weights = torch.ones(640, 8)
    ids = torch.arange(8).repeat(640, 1)
    returned = (weights, ids)

    def select_experts(hidden_states, router_logits, marker=None):
        assert marker is returned
        return returned

    wrapped = mj.instrument_router(select_experts)
    assert inspect.signature(wrapped) == inspect.signature(select_experts)
    hidden = torch.zeros(640, 4096, dtype=torch.bfloat16)
    logits = torch.zeros(640, 288)
    assert wrapped(hidden, logits, returned) is returned
    assert journal.moe_capture is None
    token = ACTIVE.set(journal)
    try:
        journal.operations = 7
        assert wrapped(hidden, logits, returned) is returned
        assert journal.moe_capture is None
        journal.operations = 8
        assert wrapped(hidden, logits, returned) is returned
    finally:
        ACTIVE.reset(token)
    assert len([r for r in journal.rows if r["kind"] == "tensor"]) == 4


def marlin_args():
    return dict(
        size_m=640,
        size_n=1024,
        size_k=4096,
        top_k=8,
        moe_block_size=32,
        input=torch.zeros(640, 4096, dtype=torch.bfloat16),
        topk_weights=torch.ones(640, 8),
        b_qweight=torch.ones(8, dtype=torch.int32),
        b_scales=torch.ones(8, dtype=torch.bfloat16),
        global_scale=torch.tensor(1.0),
        workspace=torch.zeros(8, dtype=torch.int32),
        sorted_token_ids=torch.arange(64, dtype=torch.int32),
        expert_ids=torch.arange(2, dtype=torch.int32),
        num_tokens_past_padded=torch.tensor([32], dtype=torch.int32),
        b_q_type=SimpleNamespace(id=1),
        mul_topk_weights=False,
        is_k_full=True,
        use_atomic_add=False,
        use_fp32_reduce=True,
        is_zp_float=False,
        thread_k=-1,
        thread_n=-1,
        blocks_per_sm=-1,
        b_bias=None,
        a_scales=None,
        b_qzeros=None,
        g_idx=torch.empty(0),
        perm=None,
    )


def test_marlin_captures_only_defined_alignment_and_forwards_result(journal):
    capture = mj.MoECapture(journal)
    args = marlin_args()
    result = torch.zeros(2, dtype=torch.bfloat16)
    assert mj._marlin(capture, args, lambda: result) is result
    rows = {r["stage"]: r for r in journal.rows if r["kind"] == "tensor"}
    assert rows["moe3.up.sorted_ids"]["shape"] == [32]
    assert rows["moe3.up.expert_ids"]["shape"] == [1]
    assert rows["moe3.up.global_scale"]["shape"] == []
    assert capture.gemm_calls == {1: 1}


@pytest.mark.parametrize("failure", ["recipe", "optional", "alignment", "extra"])
def test_bad_marlin_scope_fails(journal, failure):
    capture = mj.MoECapture(journal)
    args = marlin_args()
    if failure == "recipe":
        args["use_atomic_add"] = True
    elif failure == "optional":
        args["b_bias"] = torch.ones(1)
    elif failure == "alignment":
        args["num_tokens_past_padded"].fill_(65)
    else:
        capture.gemm_calls[1] = 2
    with pytest.raises(ValueError):
        mj._marlin(capture, args, lambda: pytest.fail("must fail before GEMM"))


def test_sum_wrapper_preserves_inplace_result_and_completion(journal):
    capture = mj.MoECapture(journal)
    journal.moe_capture = capture
    capture.gemm_calls[1] = 2
    x = torch.zeros(640, 8, 4096, dtype=torch.bfloat16)
    shared = torch.ones(640, 4096, dtype=torch.bfloat16)
    out = torch.empty_like(shared)

    def sum_add(x, shared, out):
        out.copy_(shared)

    wrapped = mj.instrument_moe_sum(sum_add)
    token = ACTIVE.set(journal)
    try:
        assert wrapped(x, shared, out) is None
    finally:
        ACTIVE.reset(token)
    assert torch.equal(out, shared)
    assert capture.completed == {1}
    assert journal.rows[-1]["kind"] == "moe_complete"
