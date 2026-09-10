# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance must reject empty answers and unproven tier restores."""

from types import SimpleNamespace

import pytest

from benchmarks.benchmark_kv_tier_serving import Probe, acceptance


@pytest.mark.parametrize(
    ("mamba", "tokens", "passes"),
    [
        (True, 12000, True),
        (True, 11200, False),
        (True, 800, False),
        (False, 12000, False),
    ],
)
def test_restore_depth_respects_speculative_mamba_boundary(
    tmp_path, mamba, tokens, passes
):
    from benchmarks.benchmark_kv_tier_serving import restore_evidence

    header = "host-tier: 1000 slots, block 800 tokens\n"
    if mamba:
        header += "host-tier: group 0: MambaSpec bs=800 layers=5 members=[]\n"
    log = tmp_path / "server.log"
    log.write_text(
        header + f"host-tier: hit for req1: resume at block {tokens // 800} "
        f"({tokens} tokens)\n" + "host-tier: worker done recv=['req1']\n"
    )
    args = SimpleNamespace(tier="on", sessions=1)
    # Align-mode speculative prefill preserves the 12,000-token state and
    # computes the remaining 863 tokens; 12,800 need not have a state snapshot.
    if passes:
        evidence = restore_evidence(log, len(header), args, 12863)
        assert evidence["minimum_restore_tokens"] == 12000
    else:
        with pytest.raises(AssertionError, match="restore depth"):
            restore_evidence(log, len(header), args, 12863)


@pytest.mark.parametrize("changed", [False, True])
def test_replay_requires_generation_equality_after_verified_restore(tmp_path, changed):
    from benchmarks.benchmark_kv_tier_serving import replay_acceptance

    log = tmp_path / "server.log"
    log.write_text("host-tier: 22 slots, block 64 tokens\n")
    source = tmp_path / "source.txt"
    source.write_text("A useful reference document.")
    generated = 0

    def complete(prompt, count):
        nonlocal generated
        text = "A clear explanation."
        if count == 128:
            generated += 1
            if generated > 1:
                with log.open("a") as stream:
                    stream.write(
                        "host-tier: hit for abc123: resume at block 4 (256 tokens)\n"
                        "host-tier: worker done recv=['abc123']\n"
                    )
                if changed:
                    text = "Different output."
        return {
            "seconds": 0.1,
            "response": {
                "usage": {"prompt_tokens": 300, "completion_tokens": count},
                "choices": [{"text": text}],
            },
        }

    probe = SimpleNamespace(
        output=tmp_path,
        tokens=lambda _: list(range(300)),
        complete=complete,
        post=lambda *args: {"success": True},
    )
    args = SimpleNamespace(
        sessions=1,
        rounds=1,
        context_words=200,
        source=source,
        tier="on",
        replay_output_tokens=128,
    )
    checkpoint: list[dict[str, object]] = []
    if changed:
        with pytest.raises(AssertionError, match="differs from GPU-cache control"):
            replay_acceptance(probe, log, args, checkpoint)
        assert not checkpoint
    else:
        replay_acceptance(probe, log, args, checkpoint)
        assert checkpoint[0]["generation_exact"]
    assert (tmp_path / "replay-round-0.json").exists()


def test_benchmark_rejects_sleep_but_keeps_raw_sample(tmp_path, monkeypatch):
    import json

    from benchmarks.benchmark_kv_tier_serving import benchmark

    source = tmp_path / "source.txt"
    source.write_text("benchmark context")
    clocks = iter([100.0, 101.0])
    wall = iter([1000.0, 1031.0])
    drafts = iter([0, 64])
    monkeypatch.setattr(
        "benchmarks.benchmark_kv_tier_serving.time.perf_counter", lambda: next(clocks)
    )
    monkeypatch.setattr(
        "benchmarks.benchmark_kv_tier_serving.time.time", lambda: next(wall)
    )
    probe = SimpleNamespace(
        output=tmp_path,
        tokens=lambda _: list(range(32)),
        draft_tokens=lambda: next(drafts),
        complete=lambda prompt, count: {
            "response": {
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": count}
            }
        },
    )
    args = SimpleNamespace(
        source=source, input_tokens=8, output_tokens=32, concurrency=[1], repeats=1
    )
    with pytest.raises(AssertionError, match="clock discontinuity or machine sleep"):
        benchmark(probe, args)
    sample = json.loads((tmp_path / "bench-c1-r0.json").read_text())
    assert sample["clock_gap_seconds"] == 30
    assert len(sample["responses"]) == 1


def test_benchmark_preserves_successful_peers_after_timeout(tmp_path):
    import json

    from benchmarks.benchmark_kv_tier_serving import benchmark

    source = tmp_path / "source.txt"
    source.write_text("benchmark context")

    def complete(prompt, count):
        if count == 32 and prompt[0] == 0:
            raise TimeoutError("timed out")
        return {
            "response": {
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": count},
                "choices": [{"text": "The other request completed."}],
            }
        }

    probe = SimpleNamespace(
        output=tmp_path,
        tokens=lambda _: list(range(32)),
        draft_tokens=lambda: 0,
        complete=complete,
    )
    args = SimpleNamespace(
        source=source, input_tokens=8, output_tokens=32, concurrency=[2], repeats=1
    )
    with pytest.raises(AssertionError, match="TimeoutError: timed out"):
        benchmark(probe, args)
    sample = json.loads((tmp_path / "failed-bench-c2-r0.json").read_text())
    assert sample["responses"][0] is None
    assert sample["responses"][1]["response"]["usage"]["completion_tokens"] == 32
    assert sample["errors"] == [
        {"request_index": 0, "error": "TimeoutError: timed out"}
    ]
    assert "tokens_per_second" not in sample
    assert not (tmp_path / "bench-c2-r0.json").exists()


def test_eviction_requires_successful_cache_reset(monkeypatch):
    from benchmarks.benchmark_kv_tier_serving import reset_prefix_cache

    drains = []
    probe = SimpleNamespace(
        post=lambda *args: {"success": False},
        tokens=lambda text: [1],
        complete=lambda *args: drains.append(args),
    )
    monkeypatch.setattr(
        "benchmarks.benchmark_kv_tier_serving.time.sleep", lambda _: None
    )
    with pytest.raises(AssertionError, match="reset did not succeed"):
        reset_prefix_cache(probe)
    assert len(drains) == 10


def test_smoke_uses_platform_modalities():
    from slimserve.registry import resolve
    from slimserve.smoke import profile_modalities

    for profile in ("qwen38-nvfp4-1", "qwen38-nvfp4-1-tq"):
        assert profile_modalities(resolve(profile, "metal", 1, None, 128 << 30)) == [
            "text"
        ]
    vision = resolve("qwen38-q2kxl-1", "metal", 1, None, 128 << 30)
    assert profile_modalities(vision) == ["text", "image"]


def test_completion_rejects_degenerate_exact_token_output(tmp_path):
    server = SimpleNamespace(plan=SimpleNamespace(engine={"served_model_name": "test"}))
    probe = Probe(server, tmp_path, 1)
    probe.post = lambda *args: {
        "usage": {"completion_tokens": 32},
        "choices": [{"text": ""}],
    }
    with pytest.raises(AssertionError, match="degenerate completion"):
        probe.complete([1], 32)


@pytest.mark.parametrize("text", [" to" * 200 + " while" + " to" * 200, "!" * 400])
def test_completion_rejects_repeated_output_despite_exact_count(tmp_path, text):
    server = SimpleNamespace(plan=SimpleNamespace(engine={"served_model_name": "test"}))
    probe = Probe(server, tmp_path, 1)
    probe.post = lambda *args: {
        "usage": {"completion_tokens": 400},
        "choices": [{"text": text}],
    }
    with pytest.raises(AssertionError, match="repetitive output"):
        probe.complete([1], 400)


def test_chat_stream_propagates_engine_error(monkeypatch):
    import requests

    from slimserve.stream import chat_completion

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_lines(self, **kwargs):
            yield 'data: {"error": {"message": "EngineCore failed"}}'
            yield "data: [DONE]"

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Response())
    with pytest.raises(RuntimeError, match="EngineCore failed"):
        list(chat_completion("http://unused", "test", [], max_tokens=32))


def test_smoke_rejects_correct_reasoning_without_final_answer(monkeypatch):
    import requests

    from slimserve.smoke import _request

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {"reasoning": "The answer is 4.", "content": None},
                        "finish_reason": "length",
                    }
                ]
            }

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: Response())
    plan = SimpleNamespace(engine={}, chat_template_kwargs={})
    with pytest.raises(RuntimeError, match="empty or truncated"):
        _request(plan, "http://unused", "2+2", image_url=None, max_tokens=32, timeout=1)


@pytest.mark.parametrize(
    "content,finish", [("", "stop"), (None, "stop"), ("opal", "length")]
)
def test_chat_rejects_empty_or_truncated_answers(tmp_path, content, finish):
    server = SimpleNamespace(plan=SimpleNamespace(engine={"served_model_name": "test"}))
    probe = Probe(server, tmp_path, 1)
    probe.post = lambda *args: {
        "choices": [{"message": {"content": content}, "finish_reason": finish}]
    }
    with pytest.raises(AssertionError, match="empty or truncated"):
        probe.chat([], "bad")
    assert (tmp_path / "bad.json").exists()


@pytest.mark.parametrize(
    "trail,passes",
    [
        ("", False),
        ("host-tier: issuing restore seq=1", False),
        (
            (
                "host-tier: hit for abcdef12: resume at block 8 (128 tokens)\n"
                "host-tier: worker done recv=['abcdef12']\n"
            ),
            True,
        ),
        (
            (
                "host-tier: hit for abcdef12: resume at block 3 (48 tokens)\n"
                "host-tier: worker done recv=['abcdef12']\n"
            ),
            False,
        ),
        ("host-tier: worker done recv=['abcdef12']\nVERIFY MISMATCH", False),
    ],
)
def test_recall_requires_completed_restore_without_error(tmp_path, trail, passes):
    log = tmp_path / "server.log"
    log.write_text("")

    def chat(messages, label):
        if label.startswith("recall"):
            log.write_text(trail)
        return {"role": "assistant", "content": "opal harbor lantern"}, 0.1

    probe = SimpleNamespace(
        output=tmp_path,
        prompt_counts={"plant-0": 100},
        chat=chat,
        post=lambda *args: {"success": True},
        tokens=lambda *args: [1],
        complete=lambda *args: None,
    )
    args = SimpleNamespace(
        sessions=1, rounds=1, context_words=1, pressure_tokens=0, tier="on"
    )
    checkpoints: list[dict[str, object]] = []
    if passes:
        acceptance(probe, log, args, checkpoints)
        assert checkpoints[0]["restore_completions"] == 1
    else:
        with pytest.raises(AssertionError):
            acceptance(probe, log, args, checkpoints)
        assert not checkpoints
