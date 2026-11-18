"""Prompt planning for image inputs on the q38 native engine (no GPU, no weights)."""
import numpy as np
import pytest

from slimserve.native import vision

IMG = 248056


def _rows(n, hidden=8, fill=1):
    return np.full((n, hidden), fill, dtype=np.uint16)


def _reference_positions(types, grids):
    """Qwen3.5 get_rope_index for one sequence; grids are merged (h, w)."""
    out, cur, gi, i = [], 0, 0, 0
    while i < len(types):
        if types[i] == 0:
            out.append((cur, cur, cur))
            cur += 1
            i += 1
            continue
        h, w = grids[gi]
        gi += 1
        for y in range(h):
            for x in range(w):
                out.append((cur, cur + y, cur + x))
        cur += max(h, w)
        i += h * w
    return np.asarray(out, dtype=np.int32)


def test_text_after_an_image_continues_past_its_longer_side():
    ids = [1, 2] + [IMG] * 6 + [3, 4]
    tokens, embeds, pos3 = vision.plan(ids, [(2, 6, None, "a", _rows(6), (2, 3))])
    assert pos3.tolist() == _reference_positions([0, 0] + [1] * 6 + [0, 0], [(2, 3)]).tolist()
    assert pos3[8].tolist() == [5, 5, 5]  # 2 + max(2, 3)
    assert (tokens[2:8] < 0).all() and tokens[[0, 1, 8, 9]].tolist() == [1, 2, 3, 4]
    assert embeds.shape == (6, 8)


def test_two_images_match_the_reference_and_keep_row_order():
    ids = [9] + [IMG] * 4 + [9, 9] + [IMG] * 6 + [9]
    a, b = _rows(4, fill=1), _rows(6, fill=2)
    # given out of order on purpose
    tokens, embeds, pos3 = vision.plan(ids, [(7, 6, None, "b", b, (3, 2)), (1, 4, None, "a", a, (2, 2))])
    types = [0] + [1] * 4 + [0, 0] + [1] * 6 + [0]
    assert pos3.tolist() == _reference_positions(types, [(2, 2), (3, 2)]).tolist()
    assert embeds[:4].tolist() == a.tolist() and embeds[4:].tolist() == b.tolist()


def test_row_ids_identify_the_image():
    ids = [IMG] * 4 + [5]
    t1, _, _ = vision.plan(ids, [(0, 4, None, "one", _rows(4), (2, 2))])
    t2, _, _ = vision.plan(ids, [(0, 4, None, "two", _rows(4), (2, 2))])
    t3, _, _ = vision.plan(ids, [(0, 4, None, "one", _rows(4), (2, 2))])
    assert t1.tolist() == t3.tolist()
    assert not set(t1[:4].tolist()) & set(t2[:4].tolist())
    assert len(set(t1[:4].tolist())) == 4 and (t1[:4] < 0).all()


def test_mask_selects_the_embedding_rows():
    ids = [7, IMG, IMG, IMG, IMG, 7, 5]  # range covers a start and an end marker too
    mask = np.array([False, True, True, True, True, False])
    tokens, embeds, pos3 = vision.plan(ids, [(0, 6, mask, "a", _rows(4), (2, 2))])
    assert tokens[[0, 5, 6]].tolist() == [7, 7, 5] and (tokens[1:5] < 0).all()
    assert pos3.tolist() == _reference_positions([0, 1, 1, 1, 1, 0, 0], [(2, 2)]).tolist()


def test_a_grid_that_does_not_fit_its_range_is_rejected():
    with pytest.raises(ValueError):
        vision.plan([IMG] * 5 + [1], [(0, 5, None, "a", _rows(4), (2, 2))])


def test_video_frames_are_positioned_as_separate_grids():
    # <t0> [start] frame0 [end] <t1> [start] frame1 [end] tail: 2 frame groups of 2 x 3
    h, w, t = 2, 3, 2
    frame = [IMG] * (h * w)
    ids = [10, 11, 20, *frame, 21, 12, 20, *frame, 21, 30, 31]
    offset, length = 2, len(ids) - 4
    mask = np.asarray([i == IMG for i in ids[offset:offset + length]])
    tokens, embeds, pos3 = vision.plan(ids, [(offset, length, mask, "vid", _rows(t * h * w), (t, h, w))])
    types = [1 if i == IMG else 0 for i in ids]
    np.testing.assert_array_equal(pos3, _reference_positions(types, [(h, w), (h, w)]))
    assert embeds.shape == (t * h * w, 8)
    assert (tokens[np.asarray(types) == 1] < 0).all() and len(set(tokens[tokens < 0].tolist())) == t * h * w
    assert (tokens[np.asarray(types) == 0] == np.asarray(ids)[np.asarray(types) == 0]).all()
