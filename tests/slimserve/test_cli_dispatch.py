# SPDX-License-Identifier: Apache-2.0
"""What a bare `slimserve <profile>` does.

Serving is the tool's purpose, so it is what the bare command does; a
conversation is the mode you ask for. `--serve` stays accepted because units,
benchmark harnesses and documented commands pass it, and it must keep meaning
exactly what it always did.
"""
import pytest

from slimserve import cli


@pytest.fixture
def calls(monkeypatch):
    """Record which mode main() dispatched to, without starting an engine."""
    seen = {}

    def fake_exec_server(plan, host, port):
        seen.update(mode="serve", profile=plan.profile_id, host=host, port=port)
        return 0

    def fake_chat(plan, prompt, log_path):
        seen.update(mode="chat", profile=plan.profile_id, prompt=prompt)
        return 0

    import slimserve.server

    monkeypatch.setattr(slimserve.server, "exec_server", fake_exec_server)
    monkeypatch.setattr(cli, "_chat", fake_chat)
    # Resolve against the machine the record targets, whatever runs the tests.
    monkeypatch.setattr(
        cli.hardware,
        "detect",
        lambda: cli.hardware.Machine(
            platform="a100",
            device_name="NVIDIA A100-SXM4-80GB",
            count=8,
            memory_bytes=0,
            host_ram_bytes=787 << 30,
        ),
    )
    monkeypatch.setattr(cli.fetch, "ensure", lambda plan, assume_yes=False: None)
    return seen


PROFILE = "glm53f-nvfp4-8"


def test_a_bare_profile_serves(calls):
    assert cli.main([PROFILE, "-y"]) == 0
    assert calls["mode"] == "serve"
    assert (calls["host"], calls["port"]) == ("127.0.0.1", 8000)


def test_serve_flag_still_serves_and_still_takes_host_and_port(calls):
    assert cli.main([PROFILE, "--serve", "--host", "0.0.0.0", "--port", "27830", "-y"]) == 0
    assert calls["mode"] == "serve"
    assert (calls["host"], calls["port"]) == ("0.0.0.0", 27830)


def test_chat_asks_for_a_conversation(calls):
    assert cli.main([PROFILE, "--chat", "-y"]) == 0
    assert calls["mode"] == "chat" and calls["prompt"] is None


def test_a_prompt_is_a_conversation_without_asking(calls):
    assert cli.main([PROFILE, "-p", "What is 2 + 2?", "-y"]) == 0
    assert calls["mode"] == "chat" and calls["prompt"] == "What is 2 + 2?"


def test_serve_and_chat_are_mutually_exclusive(calls):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([PROFILE, "--serve", "--chat", "-y"])
    assert excinfo.value.code == 2
    assert not calls


def test_a_prompt_cannot_be_combined_with_serve(calls):
    # Silently dropping one of them is how a script ends up serving nothing
    # or answering nobody.
    with pytest.raises(SystemExit) as excinfo:
        cli.main([PROFILE, "--serve", "-p", "hello", "-y"])
    assert excinfo.value.code == 2
    assert not calls


def test_dry_run_still_stops_before_either_mode(calls, capsys):
    assert cli.main([PROFILE, "--dry-run"]) == 0
    assert not calls
    assert PROFILE in capsys.readouterr().out
