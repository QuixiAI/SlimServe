# SPDX-License-Identifier: Apache-2.0
"""DFR temporal rounds: the window layout, prefixes and audio resampling against
upstream's pure-Python helpers (tests/slimserve/data/ltx25_temporal_layout.json
was produced by ltx_pipelines.dfr_helpers on Lightricks/LTX-2 9ec55f9). The
denoising itself is checked by perf/ltx25_harness/n11_temporal_round_parity.py."""

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from slimserve.video.ltx25 import sampling  # noqa: E402

FIXTURE = json.loads(
    (Path(__file__).parent / "data" / "ltx25_temporal_layout.json").read_text()
)


@pytest.mark.parametrize(
    "case", FIXTURE["plans"], ids=lambda c: f"{c['requested']}f-{c['num_tiles']}t"
)
def test_temporal_tile_plan_matches_upstream(case):
    tiles = sampling.temporal_tile_plan(
        case["seams"], case["frames"], case["num_tiles"]
    )
    assert len(tiles) == len(case["tiles"])
    for ours, ref in zip(tiles, case["tiles"]):
        assert (ours.start, ours.end, ours.pixel_start, ours.pixel_end) == (
            ref["start"],
            ref["end"],
            ref["ps"],
            ref["pe"],
        )
        assert list(ours.anchors) == ref["anchors"]
        assert list(ours.slots) == ref["slots"]


@pytest.mark.parametrize("case", FIXTURE["prefixes"], ids=lambda c: f"seam{c['seam']}")
def test_tile_prefix_matches_upstream(case):
    p = sampling.tile_prefix(case["seam"], case["planes"])
    assert (p.keyframe_position, p.video_start_cell, p.cells, p.resume_pixel) == (
        case["keyframe"],
        case["video_start_cell"],
        case["cells"],
        case["resume"],
    )


def test_tile_plan_rejects_a_canvas_that_does_not_end_on_a_seam():
    with pytest.raises(ValueError):
        sampling.temporal_tile_plan([48, 96], 105, 2)


def test_audio_resampling_is_linear_and_clamped():
    tokens = mx.arange(10).astype(mx.float32)[None, :, None] * mx.ones((1, 10, 3))
    out = sampling.resample_audio_tokens(tokens, 2.0, 6.0, 4)  # step 1: 2, 3, 4, 5
    assert np.allclose(np.array(out)[0, :, 0], [2, 3, 4, 5])
    out = sampling.resample_audio_tokens(tokens, 0.0, 10.0, 20)  # step 0.5
    assert np.allclose(np.array(out)[0, :4, 0], [0, 0.5, 1, 1.5])
    assert np.allclose(np.array(out)[0, -1, 0], 9.0)  # clamped at the last cell
    # a 49-frame window at 48 fps playback out of a 49/24 s source: all 51 source
    # tokens, 20 output tokens at the 60 fps conditioning rate
    src = mx.random.normal((1, 51, 128))
    tile = sampling.audio_tokens_for_tile(src, 0, 49, 48.0, 49 / 24, 60.0)
    assert tile.shape == (1, 20, 128)
    assert np.allclose(np.array(tile[0, 0]), np.array(src[0, 0]))


def test_anchor_keyframes_are_appended_near_clean_and_unmarked():
    h, w = 2, 3
    state = sampling.noised_state(
        (1, 2 * h * w, 4),
        sampling.video_positions(2, h, w, 24.0),
        0,
        tokens_per_frame=h * w,
    )
    planes = mx.ones((1, 4, 2, h, w))
    out = sampling.append_anchor_keyframes(state, planes, [8, 16], h, w, 24.0, 0.975, 5)
    n = 2 * h * w
    assert out.latent.shape[1] == n + 2 * h * w
    assert np.allclose(np.array(out.denoise_mask[0, n:, 0]), 0.05)
    assert np.allclose(np.array(out.clean[0, n:]), 1.0)
    # the noisy start is 0.95 * clean + 0.05 * sigma * noise: close to clean
    assert np.abs(np.array(out.latent[0, n:]) - 0.95).max() < 0.05 * 0.975 * 6
    assert np.allclose(np.array(out.keyframes_mask[0, n:, 0]), 0.0)
    assert np.allclose(np.array(out.positions[0, n, 0]), 8.5 / 24.0)
    assert np.allclose(np.array(out.positions[0, n + h * w, 0]), 16.5 / 24.0)


SPATIAL = json.loads(
    (Path(__file__).parent / "data" / "ltx25_spatial_tiles.json").read_text()
)


@pytest.mark.parametrize(
    "case", SPATIAL, ids=lambda c: f"{c['dim']}c-{c['n']}t-o{c['overlap']}"
)
def test_split_by_count_and_trapezoids_match_upstream(case):
    tiles = sampling.split_by_count(case["dim"], case["n"], case["overlap"])
    assert [list(t) for t in tiles] == case["tiles"]
    if "masks" in case:
        for (start, end, left, right), mask in zip(tiles, case["masks"]):
            assert np.allclose(
                sampling.trapezoid_mask(end - start, left, right), mask, atol=1e-6
            )


def test_spatial_tiles_blend_exactly_as_upstream_does():
    """2x2 tiles blend to 1 everywhere; 4x4 with a 10-cell overlap triple-covers
    a few cells (upstream adds premultiplied tiles and never renormalizes), so
    the 2-D sum must equal the product of upstream's 1-D mask sums."""
    h, w = 32, 48
    total = np.zeros((h, w), dtype=np.float32)
    for t in sampling.spatial_tiles(h, w, 2, 10):
        total[t.h0 : t.h1, t.w0 : t.w1] += t.weights
    assert np.allclose(total, 1.0, atol=1e-5)

    def upstream_sum(dim, n):
        case = next(
            c for c in SPATIAL if (c["dim"], c["n"], c["overlap"]) == (dim, n, 10)
        )
        out = np.zeros(dim, dtype=np.float32)
        for (start, end, _, _), mask in zip(case["tiles"], case["masks"]):
            out[start:end] += np.asarray(mask, dtype=np.float32)
        return out

    total = np.zeros((h, w), dtype=np.float32)
    for t in sampling.spatial_tiles(h, w, 4, 10):
        total[t.h0 : t.h1, t.w0 : t.w1] += t.weights
    assert np.allclose(
        total, np.outer(upstream_sum(h, 4), upstream_sum(w, 4)), atol=1e-5
    )
    # fewer cells than tiles on an axis: that axis is one tile (3 rows); the
    # other keeps its count with the overlap clamped (5 columns, overlap 1)
    small = sampling.spatial_tiles(3, 5, 4, 10)
    assert len(small) == 4 and all((t.h0, t.h1) == (0, 3) for t in small)


def test_res2s_coefficients_match_the_exponential_integrator():
    import math

    a21, b1, b2 = sampling.res2s_coefficients(0.0)  # phi_1(0) = 1, phi_2(0) = 1/2
    assert np.allclose([a21, b1, b2], [0.5, 0.0, 1.0])
    h = 0.7
    phi1 = (math.exp(-h) - 1) / (-h)
    phi2 = (math.exp(-h) - 1 + h) / (h * h)
    a21, b1, b2 = sampling.res2s_coefficients(h)
    assert np.isclose(a21, 0.5 * (math.exp(-h / 2) - 1) / (-h / 2))
    assert np.isclose(b2, phi2 / 0.5) and np.isclose(b1, phi1 - b2)


def test_res2s_loop_ends_on_the_terminal_x0_and_draws_twice_per_modality_per_step():
    calls = []

    def denoise(video, audio, vx, ax, sigma, step=None):
        calls.append(sigma)
        return vx * 0.0, ax * 0.0  # x0 = 0 everywhere

    v = sampling.noised_state((1, 6, 4), mx.zeros((1, 6, 3)), 0)
    a = sampling.noised_state((1, 2, 4), mx.zeros((1, 2, 1)), 1)
    draws = []
    real = mx.random.normal

    def normal(shape, *args, **kw):
        draws.append(tuple(shape))
        return real(shape, *args, **kw)

    mx.random.normal = normal
    try:
        vx, ax = sampling.res2s_loop(denoise, v, a, [1.0, 0.5, 0.0], noise_seed=7)
        mx.eval(vx, ax)
    finally:
        mx.random.normal = real
    # two steps (1.0 -> 0.5 -> 0.0011) each with a midpoint call, then the terminal call
    assert len(calls) == 5 and calls[-1] == sampling.RES2S_TERMINAL_SIGMA
    assert draws.count((1, 6, 4)) == 4 and draws.count((1, 2, 4)) == 4
    assert np.allclose(np.array(vx), 0.0) and np.allclose(np.array(ax), 0.0)
