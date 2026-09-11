"""VLLM_CUSTOM_AR_PCIE_MAX_BYTES: on a PCIe-only (not fully connected) topology
the custom one-shot all-reduce beats NCCL only at decode-size payloads
(4x RTX 3090: 24-25 us vs 40 at <=16 KiB, 50+ us from 40 KiB), so payloads
above the cap must fall back to NCCL."""

import torch

import vllm.envs as envs
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def _ar(world_size: int, fully_connected: bool) -> CustomAllreduce:
    ar = CustomAllreduce.__new__(CustomAllreduce)
    ar.disabled = False
    ar.world_size = world_size
    ar.fully_connected = fully_connected
    ar.max_size = 8192 * 1024
    return ar


def test_pcie_cap_sends_large_payloads_back_to_nccl(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_CUSTOM_AR_ALLOW_PCIE", True)
    monkeypatch.setattr(envs, "VLLM_CUSTOM_AR_PCIE_MAX_BYTES", 16 * 1024)
    ar = _ar(4, fully_connected=False)
    assert ar.should_custom_ar(torch.empty(3, 2560, dtype=torch.bfloat16))  # 15,360 B
    assert not ar.should_custom_ar(torch.empty(8, 2560, dtype=torch.bfloat16))  # 40,960 B


def test_cap_is_inert_without_the_pcie_opt_in_and_on_nvlink(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_CUSTOM_AR_PCIE_MAX_BYTES", 16 * 1024)
    monkeypatch.setattr(envs, "VLLM_CUSTOM_AR_ALLOW_PCIE", False)
    assert not _ar(4, fully_connected=False).should_custom_ar(torch.empty(3, 2560, dtype=torch.bfloat16))
    assert _ar(4, fully_connected=True).should_custom_ar(torch.empty(64, 2560, dtype=torch.bfloat16))
    assert _ar(2, fully_connected=False).should_custom_ar(torch.empty(64, 2560, dtype=torch.bfloat16))


def test_zero_cap_means_no_extra_limit(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_CUSTOM_AR_ALLOW_PCIE", True)
    monkeypatch.setattr(envs, "VLLM_CUSTOM_AR_PCIE_MAX_BYTES", 0)
    assert _ar(4, fully_connected=False).should_custom_ar(torch.empty(64, 2560, dtype=torch.bfloat16))
