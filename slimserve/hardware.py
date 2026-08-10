# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What accelerators this machine actually has.

Deliberately does not import torch or vllm: the profile picker runs before any
engine exists and must stay instant. amdsmi, pynvml and nvidia-smi all answer
in well under a second and none initializes a CUDA/HIP context; the Apple probe
is two sysctl reads.
"""

from __future__ import annotations

import contextlib
import os
import platform as _platform
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class Machine:
    platform: str | None  # "mi300x", "a100", "metal", or None when unrecognized
    device_name: str
    count: int  # visible devices, after the *_VISIBLE_DEVICES masks
    memory_bytes: int = 0  # unified memory; 0 on the discrete-GPU platforms

    @property
    def known(self) -> bool:
        return self.platform is not None


_VISIBLE_VARS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


def _visible_limit() -> int | None:
    """How many devices the environment masks us down to, if it does."""
    for var in _VISIBLE_VARS:
        value = os.environ.get(var)
        if value:
            entries = [item for item in value.split(",") if item.strip()]
            if entries:
                return len(entries)
    return None


def _probe_amd() -> tuple[str, int] | None:
    try:
        import amdsmi
    except ImportError:
        return None
    try:
        amdsmi.amdsmi_init()
        handles = amdsmi.amdsmi_get_processor_handles()
        if not handles:
            return None
        name = amdsmi.amdsmi_get_gpu_asic_info(handles[0]).get("market_name", "")
        return str(name), len(handles)
    except Exception:
        return None
    finally:
        with contextlib.suppress(Exception):
            amdsmi.amdsmi_shut_down()


def _probe_nvidia_nvml() -> tuple[str, int] | None:
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return None
        name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
        if isinstance(name, bytes):
            name = name.decode()
        return str(name), count
    except Exception:
        return None
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def _probe_nvidia_smi() -> tuple[str, int] | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    names = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    if not names:
        return None
    return names[0], len(names)


def _probe_nvidia() -> tuple[str, int] | None:
    return _probe_nvidia_nvml() or _probe_nvidia_smi()


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = out.stdout.strip()
    return value or None


def _probe_apple() -> Machine | None:
    """Apple Silicon: one GPU, and unified memory is the number that matters.

    Reported memory is physical RAM. Metal will not let the GPU hold all of it
    -- recommendedMaxWorkingSetSize is about 90% -- so the registry's
    min_memory_bytes figures already carry that margin.
    """
    if _platform.system() != "Darwin" or _platform.machine() != "arm64":
        return None
    chip = _sysctl("machdep.cpu.brand_string") or "Apple Silicon"
    raw = _sysctl("hw.memsize")
    try:
        memory = int(raw) if raw else 0
    except ValueError:
        memory = 0
    return Machine(platform="metal", device_name=chip, count=1, memory_bytes=memory)


def _classify(device_name: str) -> str | None:
    lowered = device_name.lower()
    if "mi300x" in lowered or "mi325" in lowered:
        return "mi300x"
    if "a100" in lowered:
        return "a100"
    return None


def detect() -> Machine:
    apple = _probe_apple()
    if apple is not None:
        return apple
    probed = _probe_amd() or _probe_nvidia()
    if probed is None:
        return Machine(platform=None, device_name="none detected", count=0)
    device_name, count = probed
    limit = _visible_limit()
    if limit is not None:
        count = min(count, limit)
    return Machine(_classify(device_name), device_name, count)
