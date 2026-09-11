# SPDX-License-Identifier: Apache-2.0
"""Synthetic GPU observer qualification, not full-model repeatability evidence."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from slimserve import index_journal as ij
from slimserve.canonical_indexer import maybe_ordered_topk
from vllm import _custom_ops as ops
from vllm.model_executor.layers import glm5_next_indexer as gi
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
    DeepseekV32IndexerPrefillMetadata,
    build_prefill_chunk_metadata,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)),
    reason="requires SM120",
)


@pytest.fixture
def config(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            dict(
                schema=1,
                prompt_ids=list(range(8199)),
                max_matches=3,
                output_directory=str(tmp_path / "trace"),
            )
        )
    )
    monkeypatch.setenv("SLIMSERVE_GLM53_INDEX_JOURNAL", str(path))
    monkeypatch.setenv("SLIMSERVE_GLM53_MODEL_JOURNAL", "1")
    monkeypatch.setenv("SLIMSERVE_GLM53_CANONICAL_MOE", "1")
    return path


@pytest.mark.parametrize("family", ["unique", "tied", "random"])
def test_native_observer_keeps_inputs_and_records_actual_output(config, family):
    rows, columns = 8, 2049
    g = torch.Generator(device="cuda").manual_seed(927)
    logits = torch.randn(rows, columns, device="cuda", generator=g)
    if family == "unique":
        logits = torch.stack(
            [torch.randperm(columns, device="cuda", generator=g) for _ in range(rows)]
        ).float()
    elif family == "tied":
        logits.zero_()
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.tensor(
        [0, 1, 160, 512, 513, 1904, 2048, 2049], dtype=torch.int32, device="cuda"
    )
    undefined = torch.arange(columns, device="cuda")[None, :] >= ends[:, None]
    logits.masked_fill_(undefined, float("nan"))
    original = logits.view(torch.uint8).clone()
    indices = torch.full((rows, 512), -777, dtype=torch.int32, device="cuda")
    wrapped = ij.instrument_topk(ops.top_k_per_row_prefill)
    journal = ij.IndexJournal(config)
    journal.begin("synthetic", 0, rows, logits.device)
    journal.layer_rows = journal.layer_calls = 0
    token, layer_token = ij.ACTIVE.set(journal), ij.LAYER.set(3)
    try:
        wrapped(logits, starts, ends, indices, rows, columns, 1, 512)
    finally:
        ij.LAYER.reset(layer_token)
        ij.ACTIVE.reset(token)
        journal.close()
    assert torch.equal(logits.view(torch.uint8), original)
    captured = torch.load(
        journal.archive / "match-1-chunk-1-layer-03.call-0.indices.pt",
        weights_only=True,
    )
    assert torch.equal(captured, indices.cpu())
    masked = torch.load(
        journal.archive / "match-1-chunk-1-layer-03.call-0.logits.pt", weights_only=True
    )
    assert (masked[undefined.cpu()] == 0).all()
    for row, end in enumerate(ends.tolist()):
        count = min(end, 512)
        selected = captured[row, :count].long()
        assert ((selected >= 0) & (selected < end)).all()
        assert selected.unique().numel() == count
        assert (captured[row, count:] == -1).all()
        want = logits[row, :end].cpu().sort(descending=True).values[:count]
        got = masked[row, selected].sort(descending=True).values
        assert torch.equal(want, got)
        if end <= 512:
            assert torch.equal(selected, torch.arange(count))


@pytest.mark.parametrize("tiles", [(8, 64), (2, 128)])
@pytest.mark.parametrize(
    "canonical,ties,fused",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
@pytest.mark.parametrize("capture_layer", [3, 23])
def test_compiled_real_indexer_journal_and_runner_chunks(
    config, monkeypatch, tiles, canonical, ties, fused, capture_layer
):
    """The registered real opaque op, real CUDA selector and synthetic paged KV.

    Input/return buffers are unchanged by observing. Selection order above512
    pools is already variable, so do not claim raw repeated-index parity there.
    """
    # Each scenario registers a fresh synthetic op in the same namespace.
    # Do not accumulate guards for destroyed op instances across test cases.
    torch._dynamo.reset()
    monkeypatch.setattr(gi, "_ROW_TILE", tiles[0])
    monkeypatch.setattr(gi, "_POOL_TILE", tiles[1])
    monkeypatch.setenv("SLIMSERVE_GLM53_CANONICAL_INDEX_ORDER", str(int(canonical)))
    monkeypatch.setenv("SLIMSERVE_GLM53_CANONICAL_INDEX_TIES", str(int(ties)))
    monkeypatch.setenv("SLIMSERVE_GLM53_CANONICAL_INDEX_FUSED", str(int(fused)))
    settings = json.loads(config.read_text())
    settings["capture_layer"] = capture_layer
    config.write_text(json.dumps(settings))
    monkeypatch.setattr(
        gi,
        "top_k_per_row_prefill",
        ij.instrument_topk(maybe_ordered_topk(ops.top_k_per_row_prefill)),
    )
    library = torch.library.Library("slimserve_index_journal_test", "FRAGMENT")
    direct_register_custom_op(
        op_name="pooled",
        op_func=ij.instrument_pooled_indexer(gi.glm5_next_pooled_indexer),
        fake_impl=gi.glm5_next_pooled_indexer_fake,
        mutates_args=["kv_cache", "topk_indices_buffer", "decode_logits"],
        target_lib=library,
    )
    g = torch.Generator(device="cuda").manual_seed(238)
    q = torch.randn(8199, 32, 128, device="cuda", dtype=torch.bfloat16, generator=g)
    packed = torch.randn(8199, 256, device="cuda", dtype=torch.bfloat16, generator=g)
    weights = torch.randn(8199, 32, device="cuda", generator=g)
    ape = torch.randn(4, 128, device="cuda", generator=g)
    cache = torch.zeros(8, 1088, 256, device="cuda", dtype=torch.bfloat16)
    out = torch.full((8199, 2080), -1, device="cuda", dtype=torch.int32)
    decode_logits = torch.zeros(1, 2049, device="cuda")
    embeds = torch.randn(8199, 4096, device="cuda", dtype=torch.bfloat16, generator=g)
    prefixes = [f"model.layers.{i}.self_attn.indexer.k_cache" for i in ij.LAYERS]
    context = SimpleNamespace(attn_metadata={})
    monkeypatch.setattr(gi, "get_forward_context", lambda: context)

    def forward(q, packed, weights, embeds, out):
        for prefix in prefixes:
            torch.ops.slimserve_index_journal_test.pooled(
                q,
                packed,
                weights,
                ape,
                prefix,
                cache,
                out,
                decode_logits,
                2049,
                512,
                4,
                128**-0.5,
            )
        return embeds

    compiled = torch.compile(forward, fullgraph=True, backend="inductor")
    state = SimpleNamespace(prompt_token_ids=list(range(8199)), num_computed_tokens=0)
    runner = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(hidden_size=4096, num_hidden_layers=45)
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=4, pipeline_parallel_size=1
        ),
        speculative_config=None,
        num_prompt_logprobs={"synthetic": 0},
        requests={"synthetic": state},
        input_batch=SimpleNamespace(num_reqs=1),
    )
    current_data = None

    def model_forward(**kwargs):
        return compiled(*current_data)

    runner._model_forward = model_forward
    ij.install_index_journal(runner)
    for begin, rows in ((0, 8192), (8192, 7)):
        state.num_computed_tokens = begin
        end = begin + rows
        sl_cpu = torch.tensor([end], dtype=torch.int32)
        qsl_cpu = torch.tensor([0, rows], dtype=torch.int32)
        sl = sl_cpu.cuda()
        bt = torch.arange(8, device="cuda", dtype=torch.int32)[None, :]
        chunk = build_prefill_chunk_metadata(
            0, 1, qsl_cpu.cuda(), qsl_cpu, sl, sl, sl_cpu, bt, 1
        )
        md = DeepseekV32IndexerMetadata(
            seq_lens=sl,
            max_seq_len=end,
            slot_mapping=torch.arange(begin, end, device="cuda"),
            num_decodes=0,
            num_decode_tokens=0,
            num_prefills=1,
            num_prefill_tokens=rows,
            prefill=DeepseekV32IndexerPrefillMetadata(chunks=[chunk]),
        )
        context.attn_metadata = dict.fromkeys(prefixes, md)
        data = (
            q[begin:end],
            packed[begin:end],
            weights[begin:end],
            embeds[begin:end],
            out[begin:end],
        )
        expected = compiled(*data).clone()
        expected_indices = data[-1].clone()
        current_data = data
        actual = runner._model_forward(
            input_ids=torch.arange(begin, end, device="cuda"),
            inputs_embeds=data[3],
            positions=torch.arange(begin, end, device="cuda"),
        )
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
        if canonical:
            assert torch.equal(data[-1], expected_indices)
        assert ij.ACTIVE.get() is None and ij.LAYER.get() is None
    events = [
        json.loads(line)
        for line in next((config.parent / "trace").glob("index-*.jsonl"))
        .read_text()
        .splitlines()
    ]
    assert events[-1]["kind"] == "request_complete" and events[-1]["chunks"] == 2
    assert events[0]["selection_order_implementation"] == (
        "native-bitonic" if fused else "post-sort" if canonical else "native"
    )
    runner._slimserve_index_journal.close()
    assert len([e for e in events if e["kind"] == "tensor"]) == 2 * (11 * 4 + 4)
    for item in (e for e in events if "archive" in e):
        assert item["stage"].startswith(f"layer-{capture_layer:02d}.call-0.")
        path = Path(item["archive"]["path"])
        expected = (
            config.parent
            / "trace"
            / f"index-{events[0]['pid']}"
            / (item["archive"]["path"].split("/")[-1])
        )
        assert path == expected
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == item["archive"]["sha256"]
        )
