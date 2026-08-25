from types import SimpleNamespace

from slimserve import hardware


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
