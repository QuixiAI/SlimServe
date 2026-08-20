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


def test_xpu_smi_probe_detects_b70s(monkeypatch):
    row = "| 0         | Device Name: Intel(R) Arc(TM) Pro B70 Graphics    |\n"

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=row * 4)

    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware._probe_intel_xpu() == ("Intel(R) Arc(TM) Pro B70 Graphics", 4)
    assert hardware._classify("Intel(R) Arc(TM) Pro B70 Graphics") == "b70"
