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
    # "mi300x", "a100", "rtx3090", "rtx5090", "rtx6000", "b70", "metal",
    # or None when unrecognized
    platform: str | None
    device_name: str
    count: int  # visible devices, after the *_VISIBLE_DEVICES masks
    memory_bytes: int = 0  # unified memory; 0 on the discrete-GPU platforms
    host_ram_bytes: int = 0  # system RAM; gates host-offload profiles

    @property
    def known(self) -> bool:
        return self.platform is not None


# Each vendor's runtime honours its own device mask, so only that vendor's
# variables may shrink the count: a CUDA_VISIBLE_DEVICES left over in the
# environment says nothing about how many Intel cards Level Zero exposes.
# HIP also honours CUDA_VISIBLE_DEVICES, so it stays in the AMD list.
_VISIBLE_VARS = {
    "amd": ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"),
    "nvidia": ("CUDA_VISIBLE_DEVICES",),
    "intel": ("ZE_AFFINITY_MASK",),
}


def _mask_entries(var: str, value: str) -> int:
    entries = [item.strip() for item in value.split(",") if item.strip()]
    if var == "ZE_AFFINITY_MASK":
        # Level Zero masks address sub-devices as "card.tile"; "0.0,0.1" is
        # both tiles of one card, and the probe counts cards.
        return len({item.split(".", 1)[0] for item in entries})
    return len(entries)


def _visible_limit(vendor: str) -> int | None:
    """How many of this vendor's devices the environment masks us down to."""
    for var in _VISIBLE_VARS.get(vendor, ()):
        value = os.environ.get(var)
        if value:
            count = _mask_entries(var, value)
            if count:
                return count
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


def _probe_intel_xpu() -> tuple[str, int] | None:
    """Intel discrete GPUs via xpu-smi (no torch, no SYCL context)."""
    try:
        out = subprocess.run(
            ["xpu-smi", "discovery"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    names = [
        line.split("Device Name:", 1)[1].split("|")[0].strip()
        for line in out.stdout.splitlines()
        if "Device Name:" in line
    ]
    if not names:
        return None
    return names[0], len(names)


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
    if memory <= 0:
        # Sandboxed macOS processes may be denied access to sysctl even though
        # POSIX sysconf remains available. SlimServe's profile gate must not
        # turn that probe failure into a fictitious zero-memory Mac.
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            memory = pages * page_size
        except (OSError, ValueError, TypeError):
            memory = 0
    return Machine(platform="metal", device_name=chip, count=1, memory_bytes=memory)


def _classify(device_name: str) -> str | None:
    lowered = device_name.lower()
    if "mi300x" in lowered or "mi325" in lowered:
        return "mi300x"
    if "a100" in lowered:
        return "a100"
    if "3090" in lowered:
        return "rtx3090"
    if "5090" in lowered:
        return "rtx5090"
    if "rtx pro 6000" in lowered:
        return "rtx6000"
    if "b70" in lowered and "arc" in lowered:
        return "b70"
    return None


def _host_ram_bytes() -> int:
    """System RAM from /proc/meminfo; 0 when unreadable."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def detect() -> Machine:
    apple = _probe_apple()
    if apple is not None:
        return apple
    probed = None
    vendor = None
    for vendor, probe in (
        ("amd", _probe_amd),
        ("nvidia", _probe_nvidia),
        ("intel", _probe_intel_xpu),
    ):
        probed = probe()
        if probed is not None:
            break
    if probed is None:
        return Machine(platform=None, device_name="none detected", count=0)
    device_name, count = probed
    limit = _visible_limit(vendor)
    if limit is not None:
        count = min(count, limit)
    return Machine(
        _classify(device_name),
        device_name,
        count,
        host_ram_bytes=_host_ram_bytes(),
    )
