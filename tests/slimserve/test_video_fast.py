# SPDX-License-Identifier: Apache-2.0
"""The LTX-2.5 fast tier: settings, pass selection, tiled attention indices
and the first-block step cache. Small arrays, no weights, no GPU; the
measured A/B lives in perf/ltx25_metal_campaign.md section 24."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from slimserve.video import server  # noqa: E402
from slimserve.video.ltx25 import dit, pipeline, sampling  # noqa: E402


class _Cfg:
    num_layers = 48


class _Dit:
    cfg = _Cfg()


def _texts():
    cond = (mx.ones((1, 4, 8)), mx.ones((1, 4, 6)))
    neg = (mx.zeros((1, 4, 8)), mx.zeros((1, 4, 6)))
    return cond, neg


def test_fast_defaults_are_the_exact_pipeline():
    fast = pipeline.Fast.from_config(None)
    assert not fast.active
    assert fast.cache() is None
    assert fast.tiles(2, 4, 4) is None
    assert fast.sigmas1(sampling.DISTILLED_SIGMAS) == sampling.DISTILLED_SIGMAS
    assert server.fast_settings({"pipeline": "distilled"}) is None
    assert server.fast_settings({"pipeline": "distilled", "fast": {}}) is None


def test_fast_from_config_converts_lists_and_rejects_unknown_keys():
    fast = pipeline.Fast.from_config(
        {"attention_tiles": [2, 2], "stage2_sigmas": [0.9, 0.4, 0.0], "step_cache": 0.1}
    )
    assert fast.active
    assert fast.attention_tiles == (2, 2)
    assert fast.sigmas2([1.0]) == [0.9, 0.4, 0.0]
    assert isinstance(fast.cache(), dit.StepCache)
    assert server.fast_settings({"fast": {"step_cache": 0.1}}) == fast.__class__(
        step_cache=0.1
    )
    with pytest.raises(ValueError, match="unknown fast settings"):
        pipeline.Fast.from_config({"teacache": 0.1})


def test_guided_denoiser_runs_only_the_passes_with_a_non_neutral_scale():
    cond, neg = _texts()
    full = pipeline.GuidedDenoiser(
        _Dit(), cond, neg, sampling.Guidance(cfg=3.0), sampling.Guidance(cfg=7.0)
    )
    assert full.kinds == ["cond", "neg", "stg", "mod"]
    assert full.share == (28, 0, 2)
    assert full.text_rows.tolist() == [0, 1, 0, 0]
    assert full.video_text.shape[0] == 2

    no_mod = pipeline.GuidedDenoiser(
        _Dit(),
        cond,
        neg,
        sampling.Guidance(cfg=3.0, modality=1.0),
        sampling.Guidance(cfg=7.0, modality=1.0),
    )
    assert no_mod.kinds == ["cond", "neg", "stg"]
    assert no_mod.passes == 3
    assert ("a2v", 0) not in no_mod.stg
    assert no_mod.stg[("video_self", 28)].tolist() == [1.0, 1.0, 0.0]

    plain = pipeline.GuidedDenoiser(
        _Dit(),
        cond,
        neg,
        sampling.Guidance(cfg=1.0, stg=0.0, modality=1.0),
        sampling.Guidance(cfg=1.0, stg=0.0, modality=1.0),
    )
    assert plain.kinds == ["cond"]
    assert plain.share is None
    assert plain.video_text.shape[0] == 1 and plain.stg == {}


def test_guidance_combine_treats_a_missing_pass_as_its_neutral_term():
    g = sampling.Guidance(cfg=3.0, stg=1.0, modality=3.0, rescale=0.7)
    c, u, p, m = (mx.random.normal((1, 5, 3)) for _ in range(4))
    neutral_mod = sampling.Guidance(cfg=3.0, stg=1.0, modality=1.0, rescale=0.7)
    assert np.allclose(
        np.array(neutral_mod.combine(c, u, p, m)),
        np.array(neutral_mod.combine(c, u, p, None)),
    )
    assert np.allclose(
        np.array(sampling.Guidance(cfg=1.0, stg=0.0, modality=1.0).combine(c, u, p, m)),
        np.array(
            sampling.Guidance(cfg=1.0, stg=0.0, modality=1.0).combine(
                c, None, None, None
            )
        ),
    )
    assert not np.allclose(
        np.array(g.combine(c, u, p, m)), np.array(g.combine(c, u, p, None))
    )


def test_attention_tiles_cover_every_query_once_and_widen_keys_by_the_halo():
    f, h, w, extra = 2, 6, 8, 5
    n = f * h * w
    qi, ki, inverse = dit.attention_tiles(f, h, w, (2, 2), 1, total=n + extra)
    assert len(qi) == 5  # four tiles plus the global tokens
    order = np.concatenate([np.array(q) for q in qi])
    assert sorted(order.tolist()) == list(range(n + extra))
    assert (order[np.array(inverse)] == np.arange(n + extra)).all()
    grid = np.arange(n).reshape(f, h, w)
    # tile (0, 0): rows 0..3, cols 0..4; keys rows 0..4, cols 0..5, plus extras
    assert set(np.array(qi[0]).tolist()) == set(grid[:, :3, :4].reshape(-1).tolist())
    keys = set(np.array(ki[0]).tolist())
    assert keys == set(grid[:, :4, :5].reshape(-1).tolist()) | set(range(n, n + extra))
    assert np.array(ki[-1]).tolist() == list(range(n + extra))
    # no halo and one tile: the full attention
    qi1, ki1, _ = dit.attention_tiles(f, h, w, (1, 1), 0)
    assert len(qi1) == 1 and np.array(ki1[0]).tolist() == list(range(n))


def test_step_cache_skips_within_the_threshold_and_never_on_a_forced_step():
    cache = dit.StepCache(0.1)
    v1, a1 = mx.ones((1, 4, 3)), mx.ones((1, 2, 3))
    signal = mx.full((1, 4, 3), 2.0)
    assert not cache.hit(signal)  # nothing cached yet
    cache.store(v1, a1, v1 + 5.0, a1 + 7.0)
    assert cache.computed == 1
    assert cache.hit(signal * 1.02)  # rel-L1 0.02, accumulated 0.02
    assert cache.hit(signal * 1.05)  # 0.05 -> 0.07
    assert not cache.hit(signal * 1.05)  # 0.05 -> 0.12 crosses 0.1
    cache.store(v1, a1, v1 + 1.0, a1 + 1.0)  # resets the accumulator
    rv, ra = cache.reuse(v1, a1)
    assert np.allclose(np.array(rv), 2.0) and np.allclose(np.array(ra), 2.0)
    cache.force = True
    assert not cache.hit(signal * 1.05)
    assert (
        cache.skipped == 2 and len(cache.history) == 4
    )  # the first call had no reference


def test_samplers_force_the_last_step_and_hand_the_cache_to_the_denoiser():
    seen = []

    def denoise(video, audio, vx, ax, sigma, step_cache=None, step=None):
        seen.append((sigma, step_cache.force if step_cache else None))
        return vx * 0.5, ax * 0.5

    state = sampling.noised_state((1, 3, 2), mx.zeros((1, 3, 3)), 0)
    audio = sampling.noised_state((1, 2, 2), mx.zeros((1, 2, 1)), 1)
    cache = dit.StepCache(0.5)
    sampling.euler_loop(denoise, state, audio, [1.0, 0.5, 0.0], step_cache=cache)
    assert seen == [(1.0, False), (0.5, True)]
    seen.clear()
    sampling.euler_ancestral_loop(
        denoise, state, audio, [1.0, 0.5, 0.0], noise_seed=3, step_cache=cache
    )
    assert seen == [(1.0, False), (0.5, True)]
    seen.clear()
    sampling.euler_loop(denoise, state, audio, [1.0, 0.0])
    assert seen == [(1.0, None)]
