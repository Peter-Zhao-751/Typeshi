# scripts/harness_control.py
"""One-off harness-ceiling control: real vs serialize->deserialize(real).

The realism gates compare generations carrying BIN-CENTER timings (they exit
deserialize()) against real sessions carrying raw milliseconds. Whether that
asymmetry alone is discriminable has never been measured -- the documented
0.085-on-exact-copies calibration used raw copies, not round-tripped ones.
If this lands outside ~[0.40, 0.55], pass_model measures the harness, not
the generator, and featurization must change before ANY realism number (tiny
or 7B) is interpreted. Spec 2026-08-10, §5a."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typeshi.adapters import aalto
from typeshi.eval.discriminator import train_discriminator
from typeshi.serialize import deserialize, serialize, unsupported_chars


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", type=Path,
                    default=Path("data/processed/heldout_aalto"))
    ap.add_argument("--split", type=Path,
                    default=Path("data/processed/split.json"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("harness_control.json"))
    args = ap.parse_args()

    payload = json.loads(args.split.read_text())
    test_writers = set(payload["test_writers"])

    real, roundtripped = [], []
    for writer, _target, events in aalto.iter_sessions(args.held_out):
        if len(real) >= args.n:
            break
        if f"aalto:{writer}" not in test_writers:
            continue
        if not events or unsupported_chars(events):
            continue
        real.append(events)
        # The same session through the token format: identical chars and
        # event counts, timings snapped to bin centers. Any accuracy above
        # chance here is pure quantization signal.
        roundtripped.append(deserialize(serialize(events)))

    _, acc_full = train_discriminator(real, roundtripped, paired=True)
    _, acc_timing = train_discriminator(
        real, roundtripped, paired=True, count_features=False
    )
    report = {
        "sessions": len(real),
        "ceiling_full_features": acc_full,
        "ceiling_timing_only": acc_timing,
        "band": [0.40, 0.55],
        "interpretation": (
            "outside the band, pass_model measures harness quantization, "
            "not the generator -- fix featurization before reading realism"
        ),
    }
    text = json.dumps(report, indent=2)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
