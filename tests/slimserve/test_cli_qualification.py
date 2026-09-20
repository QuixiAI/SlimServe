# SPDX-License-Identifier: Apache-2.0
"""Explicit qualification may bypass one recipe gate, never hardware gates."""

from copy import deepcopy

import pytest

from slimserve import cli, registry

PROFILE = "qwen38-uncensored-fp8-4"


@pytest.fixture
def qualification(monkeypatch):
    data = deepcopy(registry._registry())
    monkeypatch.setattr(registry, "_registry", lambda: data)
    seen = {}

    def fake_exec_server(plan, host, port):
        seen.update(profile=plan.profile_id, host=host, port=port)
        return 0

    import slimserve.server

    monkeypatch.setattr(slimserve.server, "exec_server", fake_exec_server)
    monkeypatch.setattr(cli.fetch, "ensure", lambda plan, assume_yes=False: None)
    monkeypatch.setattr(
        cli.hardware,
        "detect",
        lambda: cli.hardware.Machine(
            platform="b70", device_name="Intel Arc Pro B70", count=4
        ),
    )
    return data, seen


def test_in_progress_profile_is_refused_by_default(qualification):
    _, seen = qualification
    with pytest.raises(SystemExit) as excinfo:
        cli.main([PROFILE, "-y"])
    assert excinfo.value.code == 2
    assert not seen


@pytest.mark.parametrize("dry_run", [False, True])
def test_explicit_qualification_allows_serve_and_dry_run(
    qualification, capsys, dry_run
):
    _, seen = qualification
    args = [PROFILE, "--allow-in-progress", "-y"]
    if dry_run:
        args.append("--dry-run")
    assert cli.main(args) == 0
    assert bool(seen) is not dry_run
    assert registry.profile_blocked(PROFILE, "b70")
    assert registry.variant(PROFILE, "b70")["status"] == "in-progress"
    warning = capsys.readouterr().err
    assert "qualification override" in warning
    assert "hardware qualification is not complete" in warning


def test_qualification_preserves_platform_gate(qualification):
    data, seen = qualification
    data["platforms"]["b70"]["status"] = "in-progress"
    with pytest.raises(SystemExit) as excinfo:
        cli.main([PROFILE, "--allow-in-progress", "-y"])
    assert excinfo.value.code == 2
    assert not seen


@pytest.mark.parametrize("status", ["unsupported", "disabled", "broken"])
def test_qualification_does_not_override_other_profile_statuses(qualification, status):
    data, seen = qualification
    data["profiles"][PROFILE]["variants"]["b70"]["status"] = status
    with pytest.raises(SystemExit) as excinfo:
        cli.main([PROFILE, "--allow-in-progress", "-y"])
    assert excinfo.value.code == 2
    assert not seen


def test_qualification_preserves_gpu_count_check(qualification, monkeypatch):
    _, seen = qualification
    monkeypatch.setattr(
        cli.hardware,
        "detect",
        lambda: cli.hardware.Machine(
            platform="b70", device_name="Intel Arc Pro B70", count=1
        ),
    )
    assert cli.main([PROFILE, "--allow-in-progress", "-y"]) == 2
    assert not seen


def test_qualification_preserves_quantization_check(qualification):
    _, seen = qualification
    assert cli.main([PROFILE, "--allow-in-progress", "--quant", "invalid", "-y"]) == 2
    assert not seen
