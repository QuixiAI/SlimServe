# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from slimserve import cli
from slimserve.registry import resolve


def test_show_uses_resolved_speculator(capsys):
    plan = resolve("glm53f-nvfp4-8", "a100", 8, "NVFP4")
    plan = replace(
        plan,
        speculative=True,
        variant_speculator={
            "engine": {"method": "mtp", "num_speculative_tokens": 3}
        },
    )

    cli._show(plan)

    assert "spec      mtp k=3" in capsys.readouterr().out
