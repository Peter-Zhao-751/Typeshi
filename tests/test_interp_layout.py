import numpy as np
import pytest
from typeshi.interp import layout


def test_twenty_seven_keys_all_placed_and_assigned():
    assert len(layout.KEYS) == 27
    assert layout.KEYS[-1] == " "
    pos, fingers = layout.key_positions(), layout.finger_of()
    assert all(k in pos and k in fingers for k in layout.KEYS)


@pytest.mark.parametrize("key,expected", [
    ("q", (2.00, 1.0)),    # Tab is 1.5u wide, so Q's center lands at 2.0
    ("p", (11.00, 1.0)),   # ninth key along the top row
    ("a", (2.25, 0.0)),    # Caps is 1.75u
    ("l", (10.25, 0.0)),
    ("z", (2.75, -1.0)),   # LShift is 2.25u
    ("m", (8.75, -1.0)),
    (" ", (6.875, -2.0)),  # 6.25u space bar starting at x=3.75
])
def test_key_centers_match_hand_measured_ansi_stagger(key, expected):
    assert layout.key_positions()[key] == pytest.approx(expected)


@pytest.mark.parametrize("a,b,expected", [
    ("e", "d", "same_finger"),   # both left middle
    ("t", "h", "alternate"),     # left index -> right index
    ("e", "r", "same_hand"),     # left middle -> left index
    ("s", "s", "repeat"),
    (" ", "a", "alternate"),     # thumb counts as its own hand
])
def test_bigram_classes_hand_checked(a, b, expected):
    assert layout.bigram_class(a, b) == expected


def test_truth_coords_follows_KEYS_order():
    coords = layout.truth_coords()
    assert coords.shape == (27, 2)
    pos = layout.key_positions()
    assert np.allclose(coords[0], pos["a"])   # KEYS is alphabetical, then space
    assert np.allclose(coords[-1], pos[" "])
