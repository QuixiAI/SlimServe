# SPDX-License-Identifier: Apache-2.0
"""The LTX-2.5 harness and handoff carry no host paths or personal identity:
machine-specific locations come from perf/ltx25_harness/paths.py (environment
variables with documented defaults), never from literals in each script."""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HARNESS = REPO / "perf" / "ltx25_harness"

# a literal home-relative or absolute host location in a script
HOST_PATH = re.compile(
    r"~/models|~/\.local/scratch|~/Code|/Users/|/home/\w|Path\.home\(\)|miniconda"
)


def test_harness_scripts_take_host_paths_from_paths_py():
    offenders = []
    for script in sorted(HARNESS.iterdir()):
        if script.name == "paths.py" or script.suffix not in (".py", ".sh"):
            continue
        for number, line in enumerate(script.read_text().splitlines(), 1):
            if HOST_PATH.search(line):
                offenders.append(f"{script.name}:{number}: {line.strip()}")
    assert offenders == []


def test_handoff_names_no_personal_account_or_host_path():
    text = (REPO / "HANDOFF-LTX25.md").read_text()
    assert "@users.noreply.github.com>" not in text.replace(
        "<id+handle@users.noreply.github.com>", ""
    )
    assert "~/Code/" not in text
    assert not re.search(r"HF token on this machine is \w", text)
