# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""slimserve — run one of the tested configurations, and nothing else.

    slimserve                 pick a profile, then chat
    slimserve glm52-2         chat on that profile
    slimserve k3-6 --serve    OpenAI-compatible endpoint instead of the prompt

Every legal configuration lives in profiles.json. The CLI's job is to refuse
anything that is not in there, before a 244 GiB load discovers it the hard way.
"""

from __future__ import annotations

import argparse
import os
import sys

from slimserve import chat, fetch, hardware, registry, term
from slimserve.registry import Plan, ProfileError

USAGE = "slimserve [PROFILE] [options]"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slimserve",
        usage=USAGE,
        description="Serve GLM-5.2-Vision or Kimi K3 from a tested profile.",
        add_help=False,
    )
    parser.add_argument("profile", nargs="?", help="profile id, e.g. glm52-2")
    parser.add_argument("-h", "--help", action="store_true", help="show this help")
    parser.add_argument("--list", action="store_true", help="list every profile")
    parser.add_argument("--quant", help="quant to serve; profile default otherwise")
    parser.add_argument("-p", "--prompt", help="run one prompt and exit")
    parser.add_argument("--serve", action="store_true", help="open an HTTP endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cache", help="model directory (default $SLIMSERVE_CACHE or ~/models)"
    )
    parser.add_argument(
        "--download-only", action="store_true", help="fetch weights, do not run"
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="do not ask before downloading"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the resolved plan and stop"
    )
    parser.add_argument(
        "--engine-log",
        help="write the engine's own log here instead of discarding it",
    )
    return parser


def _help() -> None:
    out = sys.stdout
    print(term.paint("slimserve", term.BOLD, out))
    print("Run GLM-5.2-Vision or Kimi K3 from a tested profile.\n")
    print(f"Usage: {USAGE}\n")
    machine = hardware.detect()
    _print_profiles(machine)
    print("\nOptions:")
    for flag, description in (
        ("--quant NAME", "Quant to serve. Profile default otherwise."),
        ("-p, --prompt TEXT", "Run one prompt and exit."),
        ("--serve", "OpenAI-compatible endpoint instead of the prompt."),
        ("--host HOST", "Bind address for --serve. Default: 127.0.0.1"),
        ("--port N", "Bind port for --serve. Default: 8000"),
        ("--cache DIR", "Model directory. Default: $SLIMSERVE_CACHE or ~/models"),
        ("--download-only", "Fetch the weights and stop."),
        ("-y, --yes", "Do not ask before downloading."),
        ("--dry-run", "Print the resolved plan and stop."),
        ("--engine-log FILE", "Keep the engine's own log instead of discarding it."),
        ("--list", "List every profile, including ones this machine cannot run."),
    ):
        print(f"  {term.paint(flag, term.CYAN, out):<38} {description}")
    print("\nExamples:")
    for label, command in (
        ("pick a profile", "slimserve"),
        ("chat", "slimserve glm52-2"),
        ("one shot", 'slimserve k3-6 -p "What is 2 + 2?"'),
        ("serve", "slimserve glm52-4 --serve --port 8000"),
        ("higher quality", "slimserve glm52-4 --quant Q4_K"),
    ):
        print(f"  {label:<16} {term.paint(command, term.CYAN, out)}")


def _print_profiles(machine: hardware.Machine, everything: bool = False) -> None:
    print(f"Profiles ({machine.device_name}, {machine.count} visible):")
    out = sys.stdout
    for profile_id in registry.profile_ids():
        entry = registry.describe(profile_id)
        runnable, why = _runnable(profile_id, machine)
        if not runnable and not everything and machine.known:
            continue
        mark = "  " if runnable else "! "
        colour = term.CYAN if runnable else term.GREY
        label = f"{mark}{profile_id:<10}"
        detail = entry["title"]
        if not runnable:
            detail = f"{detail} — {why}"
        print(f"  {term.paint(label, colour, out)} {detail}")


def _runnable(profile_id: str, machine: hardware.Machine) -> tuple[bool, str]:
    entry = registry.describe(profile_id)
    if not machine.known:
        return False, "unrecognized hardware"
    if machine.platform not in entry["platforms"]:
        return False, f"not supported on {registry.platform_title(machine.platform)}"
    if machine.count < entry["gpus"]:
        return False, f"needs {entry['gpus']} GPUs, this machine shows {machine.count}"
    return True, ""


def _pick(machine: hardware.Machine) -> str | None:
    """Ask which profile to run. Only offers ones that would actually start."""
    choices = [
        profile_id
        for profile_id in registry.profile_ids()
        if _runnable(profile_id, machine)[0]
    ]
    if not choices:
        term.fail(
            f"no profile runs on this machine ({machine.device_name}, "
            f"{machine.count} GPUs)"
        )
        _print_profiles(machine, everything=True)
        return None

    out = sys.stdout
    print(f"{machine.device_name} x{machine.count}\n")
    for index, profile_id in enumerate(choices, start=1):
        entry = registry.describe(profile_id)
        print(
            f"  {term.paint(str(index), term.CYAN, out)}. "
            f"{term.paint(profile_id, term.BOLD, out)}  {entry['title']}"
        )
        print(f"     {term.paint(entry['summary'], term.GREY, out)}")
    print()
    return _choose(choices, "profile")


def _pick_quant(profile_id: str, platform: str) -> str | None:
    """Ask which quant, showing what the choice costs and buys."""
    quants = registry.quants_for(profile_id, platform)
    if len(quants) <= 1:
        return quants[0].name if quants else None

    default = registry.describe(profile_id)["default_quant"]
    out = sys.stdout
    print("\nQuant:")
    for index, quant in enumerate(quants, start=1):
        suffix = "  (default)" if quant.name == default else ""
        print(
            f"  {term.paint(str(index), term.CYAN, out)}. "
            f"{term.paint(quant.name, term.BOLD, out)}  "
            f"{term.human_bytes(quant.bytes)}{suffix}"
        )
        print(f"     {term.paint(quant.summary, term.GREY, out)}")
    print()
    names = [quant.name for quant in quants]
    return _choose(names, "quant", default=default)


def _choose(options: list[str], what: str, default: str | None = None) -> str | None:
    hint = f" [{default}]" if default else ""
    while True:
        try:
            answer = input(f"{what}{hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not answer and default:
            return default
        if answer in options:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        term.fail(f"pick 1-{len(options)} or a name")


def _show(plan: Plan) -> None:
    out = sys.stdout
    print(f"{term.paint(plan.profile_id, term.BOLD, out)}  {plan.title}")
    print(f"  quant     {plan.quant.title}  ({term.human_bytes(plan.quant.bytes)})")
    print(f"  platform  {registry.platform_title(plan.platform)} x{plan.gpus}")
    print(f"  model     {plan.entry_file}")
    for key, value in sorted(plan.engine.items()):
        print(f"  {key:<9} {value}")
    if plan.speculative:
        print("  spec      DSpark k=3")
    for key, value in sorted(plan.env.items()):
        print(f"  env       {key}={value}")
    for note in plan.notes:
        print(f"  note      {note}")


def _chat(plan: Plan, prompt: str | None, log_path: str | None) -> int:
    """Start a private engine, then talk to it over the same API `--serve` opens.

    Going through HTTP is what gives the prompt token-by-token streaming, and it
    means an interactive answer and a served answer come from one code path.
    """
    from slimserve.server import Server

    chat.banner(plan)
    with Server(plan) as server:
        server.start(log_path=log_path)
        try:
            server.wait_until_ready()
        except RuntimeError as error:
            term.fail(str(error))
            if log_path:
                term.fail(f"the engine's own log is at {log_path}")
            return 1
        return chat.run(plan, server.base_url, prompt)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)

    if args.help:
        _help()
        return 0

    if args.cache:
        os.environ["SLIMSERVE_CACHE"] = args.cache

    machine = hardware.detect()

    if args.list:
        _print_profiles(machine, everything=True)
        return 0

    profile_id = args.profile
    interactive = profile_id is None
    if interactive:
        profile_id = _pick(machine)
        if profile_id is None:
            return 1

    try:
        registry.describe(profile_id)
    except ProfileError as error:
        term.fail(str(error))
        return 2

    if not machine.known:
        term.die(
            f"unrecognized hardware ({machine.device_name}); "
            "slimserve runs on MI300X and A100"
        )

    quant = args.quant
    if quant is None and interactive:
        quant = _pick_quant(profile_id, machine.platform)
        if quant is None:
            return 1

    try:
        plan = registry.resolve(profile_id, machine.platform, machine.count, quant)
    except ProfileError as error:
        term.fail(str(error))
        return 2

    if args.dry_run:
        _show(plan)
        return 0

    try:
        fetch.ensure(plan, assume_yes=args.yes or args.download_only)
    except Exception as error:
        term.fail(str(error))
        return 1
    if args.download_only:
        term.ok(f"ready: {plan.entry_file}")
        return 0

    if args.serve:
        from slimserve.server import exec_server

        return exec_server(plan, args.host, args.port)

    return _chat(plan, args.prompt, args.engine_log)


if __name__ == "__main__":
    raise SystemExit(main())
