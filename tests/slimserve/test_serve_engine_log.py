# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

from slimserve import cli
from slimserve.registry import resolve


def test_serve_cli_forwards_engine_log(monkeypatch, tmp_path):
    plan = resolve("glm53f-nvfp4-8", "a100", 8, "NVFP4")
    monkeypatch.setattr(
        cli.hardware,
        "detect",
        Mock(
            return_value=Mock(
                known=True,
                platform="a100",
                count=8,
                memory_bytes=0,
                device_name="A100",
            )
        ),
    )
    monkeypatch.setattr(cli.registry, "resolve", Mock(return_value=plan))
    monkeypatch.setattr(cli.fetch, "ensure", Mock())

    from slimserve import server

    launch = Mock(return_value=0)
    monkeypatch.setattr(server, "exec_server", launch)
    log_path = tmp_path / "engine.log"

    assert (
        cli.main(
            [
                "glm53f-nvfp4-8",
                "--quant",
                "NVFP4",
                "--serve",
                "--host",
                "127.0.0.1",
                "--port",
                "8123",
                "--engine-log",
                str(log_path),
            ]
        )
        == 0
    )
    launch.assert_called_once_with(plan, "127.0.0.1", 8123, str(log_path))
