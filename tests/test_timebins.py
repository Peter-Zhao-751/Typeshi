import numpy as np
import pytest
from typeshi import config
from typeshi.timebins import bin_edges, to_bin, from_bin


def test_edges_span_configured_range():
    e = bin_edges()
    assert len(e) == config.TIME_BINS + 1
    assert e[0] == pytest.approx(config.MIN_MS)
    assert e[-1] == pytest.approx(config.MAX_MS)


def test_edges_are_monotonic():
    e = bin_edges()
    assert np.all(np.diff(e) > 0)


def test_bins_are_log_spaced_not_linear():
    e = bin_edges()
    first_width = e[1] - e[0]
    last_width = e[-1] - e[-2]
    assert last_width > first_width * 100


def test_to_bin_is_in_range():
    for dt in [0, 1, 50, 200, 5_000, 500_000]:
        assert 0 <= to_bin(dt) < config.TIME_BINS


def test_to_bin_clamps_extremes():
    assert to_bin(-5) == 0
    assert to_bin(10_000_000) == config.TIME_BINS - 1


def test_to_bin_is_monotonic_nondecreasing():
    bins = [to_bin(dt) for dt in range(1, 5000, 7)]
    assert bins == sorted(bins)


def test_round_trip_error_is_bounded():
    """A value recovered from its bin should be within ~15% of the original."""
    for dt in [5, 40, 120, 900, 4_000, 30_000]:
        recovered = from_bin(to_bin(dt))
        assert abs(recovered - dt) / dt < 0.15


def test_from_bin_returns_float():
    # from_bin returns float to guarantee idempotence: to_bin(from_bin(k)) == k.
    # Integer rounding breaks this for low-valued bins (geometric mean of
    # [1.096, 1.201] is 1.148, rounds to 1, which bins to 0).
    assert isinstance(from_bin(10), float)


def test_bin_roundtrip_is_idempotent():
    """Symmetrized eval feeds deserialize(serialize(x)) to the discriminator;
    that is only a fixed point if re-binning a bin center returns its bin."""
    for k in range(config.TIME_BINS):
        assert to_bin(from_bin(k)) == k
