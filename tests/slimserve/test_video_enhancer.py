# SPDX-License-Identifier: Apache-2.0
"""The prompt enhancer's model-free pieces: the chat template against
transformers' rendering (tests/slimserve/data/ltx25_enhancer_template.json),
upstream's response cleanup, the no-repeat-n-gram ban, and the Gemma-4 E2B
layer wiring (K/V sharing, RoPE flavours, masks) on a synthetic config. The
generation itself is checked against transformers by
perf/ltx25_harness/n11_enhancer_parity.py."""

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from slimserve.video.ltx25 import enhancer  # noqa: E402

FIXTURE = json.loads(
    (Path(__file__).parent / "data" / "ltx25_enhancer_template.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["user"][:12])
def test_chat_text_matches_transformers_rendering(case):
    assert "<bos>" + enhancer.chat_text(case["system"], case["user"]) == case["text"]


def test_system_prompts_are_upstreams():
    t2v = enhancer.system_prompt("t2v")
    assert t2v.startswith("You are given a user's short text-to-video request.")
    assert "audio" in t2v.lower()
    assert enhancer.system_prompt("i2v") != t2v


def test_clean_response_matches_upstream():
    assert enhancer.clean_response("“A man’s dog” — runs") == ("A man's dog\" - runs")
    assert enhancer.clean_response("...\n\n*** Caption: ok") == "Caption: ok"
    assert enhancer.clean_response("123 456") == "123 456"  # no letters: unchanged


def test_banned_tokens_complete_a_seen_ngram_only():
    seq = [1, 2, 3, 4, 5, 9, 2, 3, 4]
    assert enhancer.banned_tokens(seq, 4) == {5}  # (2 3 4) was followed by 5
    assert enhancer.banned_tokens(seq, 5) == set()  # (9 2 3 4) never occurred before
    assert enhancer.banned_tokens([1, 2, 3, 4, 5, 1, 2, 3, 4], 5) == {5}
    assert enhancer.banned_tokens([1, 2, 3, 1, 2], 3) == {3}
    assert enhancer.banned_tokens([1, 2], 5) == set()
    assert enhancer.banned_tokens([1, 2, 3, 4, 5, 6], 0) == set()


def _toy() -> enhancer.PromptEnhancer:
    """A PromptEnhancer with the E2B config and no weights."""
    e = enhancer.PromptEnhancer()
    kinds = (["sliding_attention"] * 4 + ["full_attention"]) * 7
    e.cfg = {
        "layer_types": kinds,
        "num_kv_shared_layers": 20,
        "head_dim": 256,
        "global_head_dim": 512,
        "sliding_window": 512,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1000000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        },
    }
    return e


def test_kv_sharing_maps_the_tail_onto_the_last_unshared_layer_of_its_kind():
    e = _toy()
    assert [e._kv_source(i) for i in range(15)] == list(range(15))
    tail = {e._kv_source(i) for i in range(15, 35)}
    assert tail == {13, 14}  # last sliding (13) and last global (14) before the tail
    assert e._kv_source(19) == 14 and e._kv_source(20) == 13


def test_rotary_flavours():
    e = _toy()
    pos = mx.arange(8).astype(mx.float32)
    hd, cos, sin = e._rotary(True, pos)
    assert hd == 256 and cos.shape == (1, 1, 8, 128)
    hd, cos, sin = e._rotary(False, pos)
    assert hd == 512 and cos.shape == (1, 1, 8, 256)
    # proportional RoPE: only the first quarter of the head (64 angles) rotates
    c = np.array(cos)[0, 0]
    assert np.allclose(c[:, 64:], 1.0) and not np.allclose(c[1, :64], 1.0)


def test_masks_are_causal_and_windowed():
    e = _toy()
    e.cfg["sliding_window"] = 3
    m = np.array(e._mask(mx.arange(5).astype(mx.float32), 5, True))
    visible = np.isfinite(m)
    assert visible.tolist() == [
        [True, False, False, False, False],
        [True, True, False, False, False],
        [True, True, True, False, False],
        [False, True, True, True, False],
        [False, False, True, True, True],
    ]
    # one decode step at position 7 over 8 cached keys
    step = np.isfinite(np.array(e._mask(mx.array([7.0]), 8, True)))
    assert step.tolist() == [[False] * 5 + [True] * 3]
    assert e._mask(mx.array([7.0]), 8, False) is None
    full = np.isfinite(np.array(e._mask(mx.arange(3).astype(mx.float32), 3, False)))
    assert full.tolist() == [[True, False, False], [True, True, False], [True] * 3]


def test_checkpoint_registry_names_the_enhancer_next_to_its_configs():
    from slimserve.video.ltx25 import checkpoints

    p = checkpoints.path_of("enhancer", Path("/root"))
    assert p == Path("/root/prompt_enhancer/gemma-4-E2B-it/model.safetensors")


# ---- the image path (enhance_i2v) ------------------------------------------


def test_fit_long_side_is_upstreams_geometry():
    """resize_aspect_ratio_preserving: long side 896, short side truncated."""
    still = enhancer.fit_long_side(np.zeros((768, 1152, 3), dtype=np.uint8))
    assert still.shape == (597, 896, 3) and still.dtype == np.uint8
    tall = enhancer.fit_long_side(np.zeros((1000, 300, 3), dtype=np.uint8))
    assert tall.shape == (896, 268, 3)


def test_patch_grid_matches_gemma4_image_processor():
    """get_aspect_ratio_preserving_size (values from transformers 5.10.1):
    aspect kept, 48-pixel grid, at most 2520 patches of 16 (280 soft tokens
    after the 3x3 pool); a hairline image gets one row of patches capped at
    the longest side the budget allows."""
    grid = lambda h, w: enhancer.patch_grid(h, w, 16, 2520, 3)  # noqa: E731
    assert grid(597, 896) == (624, 960)  # upscaled onto the grid: 39 x 60 = 2340
    assert grid(896, 896) == (768, 768)  # 48 x 48 = 2304
    assert grid(896, 268) == (1440, 432)
    assert grid(10, 2000) == (48, 11328)
    assert grid(1, 100000) == (48, 13440)  # 280 * 48
    for h, w in [(597, 896), (896, 268), (10, 2000), (480, 480), (1, 100000)]:
        th, tw = grid(h, w)
        assert th % 48 == 0 and tw % 48 == 0 and (th // 16) * (tw // 16) <= 2520


def test_image_patches_are_row_major_with_pixel_then_channel_order():
    """convert_image_to_patches: patch (py, px) holds its pixels as
    (row, column, channel) in [0, 1]. A 768x768 still already sits on the
    grid (48 x 48 patches), so no resize intervenes."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(768, 768, 3), dtype=np.uint8)
    patches, ph, pw = enhancer.image_patches(img, 16, 280, 3)
    assert (ph, pw) == (48, 48) and patches.shape == (2304, 768)
    want = img[16:32, 32:48].reshape(-1) / 255.0  # patch row 1, column 2
    np.testing.assert_allclose(patches[1 * 48 + 2], want, atol=1e-6)


def test_i2v_turn_is_the_processors_expansion():
    """enhance_i2v: the system prompt, then one <|image|> from the template
    expanded to boi + a placeholder per soft token + eoi, and the raw prompt
    with upstream's framing and trailing period."""
    text = enhancer.chat_text(
        enhancer.system_prompt("i2v"),
        enhancer.BOI_TOKEN
        + enhancer.IMAGE_TOKEN * 3
        + enhancer.EOI_TOKEN
        + "User Raw Input Prompt: a cat.",
    )
    assert text.endswith(
        "<|turn>user\n<|image><|image|><|image|><|image|><image|>"
        "User Raw Input Prompt: a cat.<turn|>\n<|turn>model\n"
    )


def test_vision_rotary_and_2d_rope_split_the_head_between_axes():
    e = enhancer.PromptEnhancer()
    e.vcfg = {"head_dim": 64, "rope_parameters": {"rope_theta": 100.0}}
    cos, sin = e._vision_rotary(mx.array([0, 1]))
    assert cos.shape == (2, 1, 16)
    np.testing.assert_allclose(np.array(cos[0, 0]), 1.0)
    inv = 1.0 / 100.0 ** (np.arange(0, 32, 2) / 32)
    np.testing.assert_allclose(np.array(cos[1, 0]), np.cos(inv), rtol=2e-3)  # fp16
    # a position-0 column leaves the first half alone while the row turns the second
    x = mx.ones((1, 1, 64))
    y = e._rope2d(x, e._vision_rotary(mx.array([0])), e._vision_rotary(mx.array([2])))
    y = np.array(y)[0, 0]
    np.testing.assert_allclose(y[:32], 1.0)
    assert not np.allclose(y[32:], 1.0)


def test_uint8_bilinear_is_torchs_fixed_point_arithmetic():
    """image.resize_bilinear_uint8: Pillow-style int16 taps, horizontal then
    vertical, uint8 between the passes. [0, 255] -> 3 samples: taps (1, 0),
    (.5, .5), (1, 0) at precision 14 give 0, 128, 255; the float path rounded
    agrees here but not everywhere (verified bit-exact against torch by the
    N11 harness)."""
    from slimserve.video.ltx25 import image as image_mod

    row = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)  # 1 x 2 x 3
    out = image_mod.resize_bilinear_uint8(row, 1, 3)
    assert out.dtype == np.uint8 and out.shape == (1, 3, 3)
    assert out[0, :, 0].tolist() == [0, 128, 255]
    i0, q, precision = image_mod._uint8_taps(2, 3)
    assert precision == 14 and i0.tolist() == [0, 0, 1]
    assert q.tolist() == [[16384, 0], [8192, 8192], [16384, 0]]
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
    ours = image_mod.resize_bilinear_uint8(img, 20, 70).astype(int)
    flt = np.rint(image_mod.resize_bilinear(img, 20, 70)).astype(int)
    assert np.abs(ours - flt).max() <= 1
