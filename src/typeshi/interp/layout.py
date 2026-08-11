"""Physical QWERTY ground truth: where the keys sit, and which finger types them.

Pure data -- no model, no corpus. Every reconstruction metric is scored against
this file, so a wrong entry here silently invalidates the whole result. That is
why the tests hand-check individual keys instead of round-tripping the tables
against themselves.
"""

from __future__ import annotations

import numpy as np

# Canonical key order: 26 lowercase letters then space. Every matrix in this
# package is indexed by this tuple, in this order.
KEYS: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwxyz") + (" ",)

# Standard ANSI stagger in key-width units (1u = one key), y increasing upward.
# The x offsets are the real ones: Tab is 1.5u wide so Q's center lands at 2.0,
# Caps is 1.75u so A lands at 2.25, LShift is 2.25u so Z lands at 2.75.
_ROWS = (
    ("qwertyuiop", 2.00, 1.0),
    ("asdfghjkl", 2.25, 0.0),
    ("zxcvbnm", 2.75, -1.0),
)
# The space bar is 6.25u starting at x=3.75, so its center is 3.75 + 6.25/2.
_SPACE_POS = (6.875, -2.0)

# Standard touch-typing assignment; pinky=4 ... index=1, thumb=0. The thumb is
# its own "hand": calling space left- or right-handed would make every space
# bigram same-hand for one half of the board and alternate for the other, which
# is not how a thumb behaves.
_FINGERS = {
    ("L", 4): "qaz",
    ("L", 3): "wsx",
    ("L", 2): "edc",
    ("L", 1): "rfvtgb",
    ("R", 1): "yhnujm",
    ("R", 2): "ik",
    ("R", 3): "ol",
    ("R", 4): "p",
    ("T", 0): " ",
}


def key_positions() -> dict[str, tuple[float, float]]:
    """Key center coordinates in key-width units."""
    pos: dict[str, tuple[float, float]] = {}
    for chars, x0, y in _ROWS:
        for i, c in enumerate(chars):
            pos[c] = (x0 + float(i), y)
    pos[" "] = _SPACE_POS
    return pos


def truth_coords() -> np.ndarray:
    """(27, 2) ground-truth coordinates in KEYS order."""
    pos = key_positions()
    return np.array([pos[k] for k in KEYS], dtype=float)


def finger_of() -> dict[str, tuple[str, int]]:
    """key -> (hand, finger index)."""
    return {c: hand_finger for hand_finger, chars in _FINGERS.items() for c in chars}


def bigram_class(a: str, b: str) -> str:
    """One of: repeat, same_finger, same_hand, alternate.

    `repeat` is separate from `same_finger` deliberately -- a double tap is one
    of the FASTEST digraphs while a same-finger move is the slowest, so folding
    them together would cancel the very effect this probe is looking for.
    """
    if a == b:
        return "repeat"
    fingers = finger_of()
    (hand_a, fin_a), (hand_b, fin_b) = fingers[a], fingers[b]
    if hand_a != hand_b:
        return "alternate"
    return "same_finger" if fin_a == fin_b else "same_hand"
