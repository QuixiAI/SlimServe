# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AcceptanceThrottle (vllm/v1/spec_decode/dynamic/utils.py).

The throttle pauses drafting while the measured accepted/drafted EMA says
drafting is a net loss, and re-probes so drafting recovers when content
changes. Ratios in these tests mirror the 2026-08-27 M5 Max sweep: prose
~0.90, hostile verse ~0.07.
"""

from vllm.v1.spec_decode.dynamic.utils import AcceptanceThrottle


def make(**kw):
    kw.setdefault("min_ratio", 0.30)
    kw.setdefault("pause_steps", 8)
    kw.setdefault("warmup_calls", 4)
    return AcceptanceThrottle(**kw)


def drive(throttle, k, ratio, steps):
    """One gate+observe round per step; returns the gated K sequence."""
    out = []
    for _ in range(steps):
        gated = throttle.gate(k)
        out.append(gated)
        if gated > 0:
            throttle.observe(gated, round(gated * ratio))
    return out


def test_healthy_content_never_pauses():
    t = make()
    assert drive(t, 3, 0.9, 50) == [3] * 50


def test_hostile_content_pauses_after_warmup():
    t = make()
    ks = drive(t, 3, 0.0, 20)
    assert ks[:4] == [3, 3, 3, 3]  # warmup drafts
    assert ks[4] == 0  # EMA below threshold -> paused
    assert set(ks[4:12]) == {0}  # pause_steps long


def test_reprobe_recovers_on_content_change():
    t = make()
    drive(t, 3, 0.0, 4)  # hostile warmup
    assert t.gate(3) == 0  # paused
    for _ in range(7):  # burn the rest of the pause
        assert t.gate(3) == 0
    # Content is now healthy: the re-probe warms up and stays open.
    ks = drive(t, 3, 0.9, 20)
    assert ks == [3] * 20


def test_still_hostile_after_reprobe_pauses_again():
    t = make()
    drive(t, 3, 0.0, 4)
    for _ in range(8):
        t.gate(3)
    ks = drive(t, 3, 0.0, 6)
    assert ks[:4] == [3, 3, 3, 3]  # probe drafts
    assert ks[4] == 0  # pauses again


def test_zero_k_passes_through_untouched():
    t = make()
    assert t.gate(0) == 0
    t.observe(0, 0)  # no-draft steps do not poison the EMA
    assert t.gate(3) == 3


def test_disabled_without_env(monkeypatch):
    monkeypatch.delenv("VLLM_SD_ADAPT_THROTTLE", raising=False)
    assert AcceptanceThrottle.from_env() is None
    monkeypatch.setenv("VLLM_SD_ADAPT_THROTTLE", "1")
    monkeypatch.setenv("VLLM_SD_ADAPT_MIN_RATIO", "0.5")
    t = AcceptanceThrottle.from_env()
    assert t is not None and t.min_ratio == 0.5
