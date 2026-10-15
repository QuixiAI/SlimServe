from types import SimpleNamespace

from slimserve import hardware


def test_rtx5090_is_detected_as_its_own_platform():
    assert hardware._classify("NVIDIA GeForce RTX 5090") == "rtx5090"


def test_rtx_pro_6000_is_detected_as_its_own_platform():
    for name in (
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    ):
        assert hardware._classify(name) == "rtx6000"


def test_nvidia_smi_probe_detects_a100s(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="NVIDIA A100-SXM4-80GB\nNVIDIA A100-SXM4-80GB\n",
        )

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware._probe_nvidia_smi() == ("NVIDIA A100-SXM4-80GB", 2)


def test_nvidia_smi_probe_ignores_failures(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware._probe_nvidia_smi() is None


def test_apple_probe_falls_back_to_sysconf_when_sysctl_is_sandboxed(monkeypatch):
    monkeypatch.setattr(hardware._platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware._platform, "machine", lambda: "arm64")
    monkeypatch.setattr(
        hardware,
        "_sysctl",
        lambda name: "Apple M4 Max" if name == "machdep.cpu.brand_string" else None,
    )
    values = {"SC_PHYS_PAGES": 8_388_608, "SC_PAGE_SIZE": 16_384}
    monkeypatch.setattr(hardware.os, "sysconf", values.__getitem__)

    machine = hardware._probe_apple()

    assert machine == hardware.Machine(
        platform="metal",
        device_name="Apple M4 Max",
        count=1,
        memory_bytes=128 * 1024**3,
    )


def test_xpu_smi_probe_detects_b70s(monkeypatch):
    row = "| 0         | Device Name: Intel(R) Arc(TM) Pro B70 Graphics    |\n"

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=row * 4)

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware._probe_intel_xpu() == ("Intel(R) Arc(TM) Pro B70 Graphics", 4)
    assert hardware._classify("Intel(R) Arc(TM) Pro B70 Graphics") == "b70"


_ALL_MASKS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "ZE_AFFINITY_MASK",
)


def _clear_masks(monkeypatch):
    for var in _ALL_MASKS:
        monkeypatch.delenv(var, raising=False)


def test_intel_count_ignores_cuda_and_hip_masks(monkeypatch):
    """A stray CUDA/HIP mask must not shrink the number of Intel cards."""
    _clear_masks(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    assert hardware._visible_limit("intel") is None
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0,1")
    assert hardware._visible_limit("intel") == 2


def test_ze_affinity_mask_counts_cards_not_tiles(monkeypatch):
    _clear_masks(monkeypatch)
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0.0,0.1")
    assert hardware._visible_limit("intel") == 1
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0.0,0.1,1,2.1")
    assert hardware._visible_limit("intel") == 3


def test_nvidia_and_amd_ignore_ze_affinity_mask(monkeypatch):
    _clear_masks(monkeypatch)
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0")
    assert hardware._visible_limit("nvidia") is None
    assert hardware._visible_limit("amd") is None
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    assert hardware._visible_limit("nvidia") == 3
    # HIP honours CUDA_VISIBLE_DEVICES too.
    assert hardware._visible_limit("amd") == 3
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0,1")
    assert hardware._visible_limit("nvidia") == 3
    assert hardware._visible_limit("amd") == 2


def test_detect_applies_only_the_probed_vendors_mask(monkeypatch):
    _clear_masks(monkeypatch)
    monkeypatch.setattr(hardware, "_probe_apple", lambda: None)
    monkeypatch.setattr(hardware, "_probe_amd", lambda: None)
    monkeypatch.setattr(hardware, "_probe_nvidia", lambda: None)
    monkeypatch.setattr(
        hardware,
        "_probe_intel_xpu",
        lambda: ("Intel(R) Arc(TM) Pro B70 Graphics", 4),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    machine = hardware.detect()
    assert (machine.platform, machine.count) == ("b70", 4)
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0.0,0.1,1.0,1.1")
    assert hardware.detect().count == 2


def test_unrecognized_hardware_message_lists_b70(monkeypatch, capsys):
    import pytest

    from slimserve import cli

    monkeypatch.setattr(
        cli.hardware,
        "detect",
        lambda: hardware.Machine(platform=None, device_name="mystery", count=1),
    )
    with pytest.raises(SystemExit):
        cli.main(["dsv4-xxs-1", "--dry-run"])
    captured = capsys.readouterr()
    assert "Intel Arc Pro B70" in captured.out + captured.err
