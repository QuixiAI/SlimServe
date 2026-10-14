# SPDX-License-Identifier: Apache-2.0
"""Options belong to an engine family.

The main parser holds what every profile shares plus the LLM engine's options;
a video profile parses its own options from what is left. An LLM profile must
not silently accept a video option, and a video profile must refuse the
options that configure the LLM engine instead of ignoring or half-applying
them."""

import pytest

from slimserve import cli
from slimserve.video import cli as video_cli

LLM = "glm53f-nvfp4-8"
VIDEO = "ltx25-distilled"


@pytest.fixture
def machine(monkeypatch):
    seen = {}

    def detect():
        platform = seen.get("platform", "a100")
        if platform == "metal":
            return cli.hardware.Machine(
                platform="metal",
                device_name="Apple M1 Ultra",
                count=1,
                memory_bytes=128 << 30,
                host_ram_bytes=128 << 30,
            )
        return cli.hardware.Machine(
            platform="a100",
            device_name="NVIDIA A100-SXM4-80GB",
            count=8,
            memory_bytes=0,
            host_ram_bytes=787 << 30,
        )

    def ensure(plan, assume_yes=False):
        seen["fetched"] = plan

    def run(plan, args):
        seen.update(run=plan, args=args)
        return 0

    import slimserve.server

    monkeypatch.setattr(cli.hardware, "detect", detect)
    monkeypatch.setattr(cli.fetch, "ensure", ensure)
    monkeypatch.setattr(video_cli, "run", run)
    monkeypatch.setattr(slimserve.server, "exec_server", lambda plan, host, port: 0)
    return seen


def test_an_llm_profile_refuses_video_options(machine, capsys):
    with pytest.raises(SystemExit) as info:
        cli.main([LLM, "--size", "768x512", "-y"])
    assert info.value.code == 2
    assert "unrecognized arguments: --size 768x512" in capsys.readouterr().err


def test_video_options_before_the_profile_are_refused(machine, capsys):
    with pytest.raises(SystemExit):
        cli.main(["--seed"])
    # `--seed 7`: argparse takes 7 for the profile id; the error says why
    assert cli.main(["--seed", "7", VIDEO]) == 2
    assert "after the profile id" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag",
    [
        ["--spec"],
        ["--no-spec"],
        ["--ctx", "4096"],
        ["--thinking"],
        ["--model", "some/repo"],
        ["--engine-log", "x.log"],
        ["--torch-profile-dir", "traces"],
        ["--enable-return-routed-experts"],
    ],
)
def test_a_video_profile_refuses_llm_options(machine, capsys, flag):
    machine["platform"] = "metal"
    with pytest.raises(SystemExit) as info:
        cli.main([VIDEO, *flag, "-y"])
    assert info.value.code == 2
    err = capsys.readouterr().err
    assert flag[0] in err and "video profile" in err
    assert "run" not in machine


def test_a_video_profile_parses_its_own_options(machine):
    machine["platform"] = "metal"
    argv = [VIDEO, "-p", "a fox", "--size", "768x512", "--seed", "7", "-y"]
    argv += ["--image", "a.png", "0", "0.8", "--lora", "x.safetensors", "0.5"]
    assert cli.main(argv) == 0
    args = machine["args"]
    assert (args.prompt, args.size, args.seed) == ("a fox", "768x512", 7)
    assert args.image == [["a.png", "0", "0.8"]]
    assert args.lora == [["x.safetensors", "0.5"]]
    assert machine["run"].engine["pipeline"] == "distilled"
    assert "max_model_len" not in machine["run"].engine


def test_a_video_profile_rejects_unknown_options(machine):
    machine["platform"] = "metal"
    with pytest.raises(SystemExit):
        cli.main([VIDEO, "--bogus", "-y"])


def test_the_enhancer_is_fetched_only_with_enhancement(machine):
    machine["platform"] = "metal"
    assert cli.main([VIDEO, "-p", "a fox", "-y"]) == 0
    assert machine["fetched"].optional == ()
    assert cli.main([VIDEO, "-p", "a fox", "--enhance-prompt", "-y"]) == 0
    assert machine["fetched"].optional == ("prompt_enhancer",)
    assert cli.main([VIDEO, "--prompt-enhancer", "-y"]) == 0
    assert machine["fetched"].optional == ("prompt_enhancer",)


def test_a_video_profile_help_lists_the_video_options(machine, capsys):
    assert cli.main([VIDEO, "--help"]) == 0
    out = capsys.readouterr().out
    assert "--negative-prompt" in out and "--input-root" in out


def test_the_llm_help_does_not_list_video_options(machine, capsys):
    with pytest.raises(SystemExit):  # argparse's own help is the closest check
        cli._parser().parse_args(["--size", "1"])
    assert cli.main(["--help"]) == 0
    assert "--negative-prompt" not in capsys.readouterr().out
