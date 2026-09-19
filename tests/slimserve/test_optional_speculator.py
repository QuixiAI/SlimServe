# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from slimserve import cli
from slimserve.engine import engine_kwargs
from slimserve.registry import resolve
from slimserve.smoke import validate_acceleration


def test_non_speculative_plan_does_not_require_registered_drafter():
    plan = resolve("glm53f-nvfp4-8", "a100", 8, "NVFP4")
    plan = replace(
        plan,
        speculative=False,
        variant_speculator=None,
        source={**plan.source, "speculator": None},
    )

    assert validate_acceleration(plan) == {}
    assert "speculative_config" not in engine_kwargs(plan)


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
