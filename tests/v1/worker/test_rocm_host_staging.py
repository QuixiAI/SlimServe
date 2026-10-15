# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

from vllm.v1.worker import gpu_worker


def test_hsa_staging_probe_is_rocm_only():
    with patch.object(gpu_worker.current_platform, "is_rocm", return_value=True):
        assert gpu_worker._should_warm_hsa_host_staging()
    with patch.object(gpu_worker.current_platform, "is_rocm", return_value=False):
        assert not gpu_worker._should_warm_hsa_host_staging()
