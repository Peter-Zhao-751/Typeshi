# scripts/shuffle_diagnostic.py
"""One-off diagnostic backing docs/results-tiny-poc.md §5.2: why the
serial-dependence gate (real vs timing-shuffled real, >= 0.75) cannot pass.

The doc originally claimed lag-1 log-IKI autocorrelation was "the only one of
33 [featurize()] features that shuffling can change." That is false: the
burst block (5 quantiles + mean + std of run lengths, indices 24-30) is
order-sensitive too, because run boundaries fall wherever a pause lands in
the shuffled order -- only the burst COUNT (index 31) is shuffle-invariant,
since permuting gaps does not change how many of them exceed the pause
threshold. So 8 of 33 features are order-sensitive, not 1. This script makes
that measurable and reproduces the numbers §5.2 cites: real-vs-shuffled
discriminator accuracy for both raw-millisecond and round-tripped timings,
mean lag-1 log-IKI autocorrelation for real and shuffled sessions, and the
fraction of sessions whose burst-block features actually change under
shuffling. Written in the style of harness_control.py (§5.1's script). Spec
2026-08-10, §5.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from typeshi.adapters import aalto
from typeshi.eval.discriminator import featurize, shuffle_timing, train_discriminator
from typeshi.eval.distributional import timing_features
from typeshi.serialize import deserialize, serialize, unsupported_chars

# featurize()'s layout (count_features=True): four 8-wide groups -- 5
# quantiles + mean + std + count -- for iki/hold/pause/burst, then one
# trailing lag-1 autocorrelation feature. This slice is the burst block's 7
# order-sensitive stats (quantiles + mean + std); index 31, the burst COUNT,
# is deliberately excluded because it is shuffle-invariant (see module
# docstring).
_BURST_STATS = slice(24, 31)


def _lag1_autocorr(events) -> float:
    """Mirrors featurize()'s trailing feature, exposed standalone for
    per-session reporting instead of buried in a 33-wide vector."""
    iki = timing_features(events)["iki"]
    if iki.size <= 2:
        return 0.0
    li = np.log1p(iki)
    return float(np.corrcoef(li[:-1], li[1:])[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", type=Path,
                    default=Path("data/processed/heldout_aalto"))
    ap.add_argument("--split", type=Path,
                    default=Path("data/processed/split.json"))
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", type=Path, default=Path("shuffle_diagnostic.json"))
    args = ap.parse_args()

    payload = json.loads(args.split.read_text())
    test_writers = set(payload["test_writers"])

    real = []
    for writer, _target, events in aalto.iter_sessions(args.held_out):
        if len(real) >= args.n:
            break
        if f"aalto:{writer}" not in test_writers:
            continue
        if not events or unsupported_chars(events):
            continue
        real.append(events)

    # Raw milliseconds (as logged) vs round-tripped (bin-center timings, the
    # basis run_eval.py actually scores against -- see harness_control.py).
    roundtripped = [deserialize(serialize(e)) for e in real]
    shuffled_raw = [shuffle_timing(e, seed=i) for i, e in enumerate(real)]
    shuffled_roundtripped = [
        shuffle_timing(e, seed=i) for i, e in enumerate(roundtripped)
    ]

    _, acc_raw = train_discriminator(real, shuffled_raw, paired=True)
    _, acc_roundtripped = train_discriminator(
        roundtripped, shuffled_roundtripped, paired=True
    )

    autocorr_real = float(np.mean([_lag1_autocorr(e) for e in real]))
    autocorr_shuffled = float(np.mean([_lag1_autocorr(e) for e in shuffled_raw]))

    changed = sum(
        not np.allclose(featurize(r)[_BURST_STATS], featurize(s)[_BURST_STATS])
        for r, s in zip(real, shuffled_raw)
    )

    report = {
        "sessions": len(real),
        "discriminator_accuracy_real_vs_shuffled_raw_ms": acc_raw,
        "discriminator_accuracy_real_vs_shuffled_roundtripped": acc_roundtripped,
        "mean_lag1_log_iki_autocorr_real": autocorr_real,
        "mean_lag1_log_iki_autocorr_shuffled": autocorr_shuffled,
        "burst_features_changed_by_shuffling": changed,
        "burst_features_changed_fraction": changed / len(real) if real else float("nan"),
        "interpretation": (
            "8 of featurize()'s 33 features are order-sensitive (the 7 "
            "burst-block quantile/mean/std stats plus lag-1 autocorrelation). "
            "Long pauses are rare in Aalto transcription sessions (short "
            "single sentences), so per-session burst run-lengths are near-"
            "degenerate -- usually one burst covering the whole session -- "
            "and lag-1 autocorrelation is ~0. Between them the order-"
            "sensitive features carry almost no per-session signal, so "
            "pass_serial_dependence_teeth (>= 0.75) cannot pass regardless "
            "of the generator."
        ),
    }
    text = json.dumps(report, indent=2)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
