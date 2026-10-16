# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Repository-policy defaults: no site ports or host paths, results in perf/."""

import asyncio
import contextlib
import importlib
import io
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import capture_paths
import watch_progress


class PolicyDefaultsTest(unittest.TestCase):
    def test_results_dir_is_dated_perf_results(self):
        path = capture_paths.results_dir("run-x")
        self.assertEqual(path.parent.parent, capture_paths.REPO_ROOT / "perf/results")
        self.assertEqual(path.name, "run-x")
        self.assertRegex(path.parent.name, r"^\d{4}-\d{2}-\d{2}$")

    def test_log_dir_default_and_override(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROXY_LOG_DIR", None)
            module = importlib.reload(capture_paths)
            self.assertEqual(module.DEFAULT_LOG_DIR.name, "traffic-proxy")
            self.assertIn("perf/results", module.DEFAULT_LOG_DIR.as_posix())
        with patch.dict(os.environ, {"PROXY_LOG_DIR": "/srv/x"}):
            module = importlib.reload(capture_paths)
            self.assertEqual(module.DEFAULT_LOG_DIR, Path("/srv/x"))
        importlib.reload(capture_paths)

    def test_watchdog_requires_metrics_url(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            watch_progress.main(["--duration", "1"])

    def test_proxy_requires_upstream(self):
        with TemporaryDirectory() as log_dir:
            env = {"PROXY_LOG_DIR": log_dir}
            with patch.dict(os.environ, env):
                os.environ.pop("PROXY_UPSTREAM", None)
                importlib.reload(capture_paths)
                proxy = importlib.import_module("slimserve_traffic_proxy")
                try:
                    with (
                        patch.object(proxy, "UPSTREAM", ""),
                        self.assertRaises(RuntimeError),
                    ):
                        asyncio.run(proxy.startup())
                finally:
                    for handler in list(proxy.LOGGER.handlers):
                        handler.close()
                        proxy.LOGGER.removeHandler(handler)
            importlib.reload(capture_paths)


if __name__ == "__main__":
    unittest.main()
