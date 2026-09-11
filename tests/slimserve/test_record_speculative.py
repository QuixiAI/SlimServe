"""A platform record may opt into or out of speculation on its own (glm53-nvfp4-4)."""

from dataclasses import replace

from slimserve import registry
from slimserve.engine import _speculative_config


def _plan(platform):
    return registry.resolve("glm53-nvfp4-4", platform, 4, None, 0)


def test_glm53_records_serve_spec_off_by_default():
    # rtx6000: measured off on the Foundry fan-out (perf entry 2026-09-04);
    # a100: never validated with the drafter.
    for platform in ("rtx6000", "a100"):
        plan = _plan(platform)
        assert plan.speculative is False
        assert _speculative_config(plan) is None


def test_rtx6000_opt_in_keeps_the_measured_depth(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIMSERVE_CACHE", str(tmp_path))
    # The supported --spec CLI applies this exact override to the resolved plan.
    plan = replace(_plan("rtx6000"), speculative=True)
    assert plan.speculative_overrides == {"num_speculative_tokens": 3}
    spec = plan.speculator
    assert spec["engine"]["method"] == "mtp"
    assert spec["engine"]["moe_backend"] == "triton"
    assert spec["engine"]["attention_backend"] == "QUIXICORE_MLA_SPARSE"
    assert spec["engine"]["index_share_for_mtp_iteration"] is True
    assert _speculative_config(plan) == {
        "model": "RedHatAI/GLM-5.3-Flash-NVFP4",
        "revision": "36c184c6cda000a481711306df5adde42f63321a",
        "method": "mtp",
        "num_speculative_tokens": 3,
        "index_share_for_mtp_iteration": True,
        "moe_backend": "triton",
        "attention_backend": "QUIXICORE_MLA_SPARSE",
    }


def test_record_level_speculative_overrides_the_profile_flag():
    profile = {
        "speculative": False,
        "variants": {
            "x": {
                "platform": "x",
                "engine": {"a": 1},
                "default_quant": "Q",
                "speculative": True,
            }
        },
    }
    merged = registry._merge_platform(profile, "x")
    assert merged["speculative"] is True
    profile["variants"]["x"].pop("speculative")
    assert registry._merge_platform(profile, "x")["speculative"] is None
