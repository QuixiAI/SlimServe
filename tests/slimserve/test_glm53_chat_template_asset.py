# SPDX-License-Identifier: Apache-2.0

import pytest

from slimserve import registry
from slimserve.registry import ProfileError, resolve


def test_glm53_profile_uses_checked_in_tool_template():
    plan = resolve("glm53f-nvfp4-8", "a100", 8, "NVFP4")

    assert plan.chat_template_file is not None
    assert plan.chat_template_file.is_file()
    assert plan.engine["chat_template"] == str(plan.chat_template_file)

    template = plan.chat_template_file.read_text(encoding="utf-8")
    assert "enable_thinking is defined and not enable_thinking" in template
    assert "effective_tool_choice == 'required'" in template
    assert "effective_tool_choice == 'none'" in template
    assert "effective_tool_choice.type == 'allowed_tools'" in template
    assert "effective_tool_choice.type in ['function', 'custom']" in template


def test_every_named_chat_template_asset_is_checked_in():
    for source in registry._registry()["sources"].values():
        if asset := source.get("chat_template_asset"):
            assert (registry.CHAT_TEMPLATE_DIR / asset).is_file(), asset


def test_resolve_rejects_a_missing_chat_template_asset(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "CHAT_TEMPLATE_DIR", tmp_path)
    with pytest.raises(ProfileError, match="not checked in"):
        resolve("glm53f-nvfp4-8", "a100", 8, "NVFP4")
