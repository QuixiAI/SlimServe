# SPDX-License-Identifier: Apache-2.0
"""Routing diagnosis must preserve the registered engine and stay opt-in."""
from unittest.mock import Mock

import pytest

from slimserve import cli
from slimserve.engine import engine_kwargs, serve_argv
from slimserve.registry import resolve


@pytest.mark.parametrize("enabled", [False, True])
def test_routing_capture_preserves_registered_plan(monkeypatch, enabled):
    original = resolve("glm53f-nvfp4-8", "a100", 8, "NVFP4")
    monkeypatch.setattr(cli.hardware, "detect", Mock(return_value=Mock(
        known=True, platform="a100", count=8, memory_bytes=0,
        host_ram_bytes=0, device_name="A100",
    )))
    monkeypatch.setattr(cli.registry, "resolve", Mock(return_value=original))
    ensure = Mock()
    monkeypatch.setattr(cli.fetch, "ensure", ensure)
    seen = []
    monkeypatch.setattr(cli, "_show", seen.append)
    args = ["glm53f-nvfp4-8", "--quant", "NVFP4", "--dry-run"]
    if enabled:
        args.append("--enable-return-routed-experts")
    assert cli.main(args) == 0
    ensure.assert_not_called()
    assert len(seen) == 1
    captured = seen[0]
    expected = dict(original.engine)
    if enabled:
        expected["enable_return_routed_experts"] = True
    assert captured.engine == expected
    assert captured.env == original.env
    assert captured.speculative == original.speculative
    assert "enable_return_routed_experts" not in original.engine
    assert engine_kwargs(captured).get("enable_return_routed_experts", False) is enabled
    argv = serve_argv(captured, "127.0.0.1", 8400)
    assert ("--enable-return-routed-experts" in argv) is enabled
