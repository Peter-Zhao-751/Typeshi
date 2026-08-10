"""Scores a trained checkpoint: distributional metrics + discriminator.

Tier-1 passes when the discriminator has teeth (>= 0.9 against the naive
heuristic baseline) and still cannot separate our output from real typing
(<= 0.55).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import math

from typeshi.adapters import aalto
from typeshi.eval.discriminator import (
    heuristic_baseline,
    real_vs_real_control,
    train_discriminator,
)
from typeshi.eval.distributional import compare_sessions
from typeshi.generate import generate
from typeshi.labels import compute_labels

PASS_MODEL_MAX = 0.55
PASS_TEETH_MIN = 0.90


def load_test_writers(split_path: Path, allow_unsplit: bool) -> set[str] | None:
    """Writer IDs held out at dataset-build time, or None to score everything.

    The plan holds out by writer, never by session. The eval has to honour the
    *same* split, otherwise it scores realism on writers the model trained on
    and Tier-1 passes for the wrong reason.
    """
    if split_path.exists():
        payload = json.loads(split_path.read_text())
        writers = set(payload["test_writers"])
        print(f"scoring {len(writers)} held-out writers from {split_path}")
        return writers
    if allow_unsplit:
        print(f"WARNING: {split_path} missing and --allow-unsplit set; results "
              "may include writers seen during training")
        return None
    raise SystemExit(
        f"{split_path} not found. It is written by scripts/build_dataset.py and "
        "is what keeps the eval off training writers. Rebuild the dataset, copy "
        "the file across, or pass --allow-unsplit if you accept a leaky number."
    )


def jsonable(value):
    """Turns NaN into null.

    A feature can be absent entirely -- the heuristic baseline never produces
    a pause over one second, so its pause metrics are NaN. `json.dumps` writes
    a bare `NaN`, which is not valid JSON and is rejected by strict parsers.
    """
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("checkpoints/motor"))
    ap.add_argument("--held-out", type=Path, default=Path("data/raw/aalto"))
    ap.add_argument(
        "--split",
        type=Path,
        default=Path("data/processed/split.json"),
        help="writer split written by build_dataset.py; scoring is restricted "
             "to its test writers so the model is never evaluated on writers "
             "it trained on",
    )
    ap.add_argument(
        "--allow-unsplit",
        action="store_true",
        help="score every session found, even training writers (debug only)",
    )
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("eval_report.json"))
    ap.add_argument("--temperature", type=float, default=1.0)
    args = ap.parse_args()

    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoPeftModelForCausalLM.from_pretrained(args.checkpoint, device_map="auto")

    test_writers = load_test_writers(args.split, args.allow_unsplit)

    real: list = []
    fake: list = []
    baseline: list = []
    skipped = 0
    skipped_train_writer = 0

    for i, (writer, target, events) in enumerate(aalto.iter_sessions(args.held_out)):
        if len(real) >= args.n:
            break
        if test_writers is not None and f"aalto:{writer}" not in test_writers:
            skipped_train_writer += 1
            continue
        labels = compute_labels(events, target)
        try:
            generated = generate(
                model, tok, target, labels,
                mode="transcription", temperature=args.temperature, seed=i,
            )
        except ValueError:
            # The model emitted something outside the event grammar.
            skipped += 1
            continue
        if not generated:
            skipped += 1
            continue
        real.append(events)
        fake.append(generated)
        baseline.append(heuristic_baseline(target, wpm=labels.wpm or 60, seed=i))

    if not real:
        raise SystemExit("no sessions scored; check --held-out and the checkpoint")

    _, acc_model = train_discriminator(real, fake)
    _, acc_baseline = train_discriminator(real, baseline)
    control = real_vs_real_control(real)

    report = {
        "sessions_scored": len(real),
        "sessions_skipped_not_held_out": skipped_train_writer,
        "held_out_writers_only": test_writers is not None,
        "generations_rejected_as_malformed": skipped,
        "temperature": args.temperature,
        "distributional": compare_sessions(real, fake),
        "discriminator_accuracy_vs_model": acc_model,
        "discriminator_accuracy_vs_heuristic_baseline": acc_baseline,
        "discriminator_accuracy_real_vs_real_control": control,
        "pass_model": acc_model <= PASS_MODEL_MAX,
        "pass_discriminator_has_teeth": acc_baseline >= PASS_TEETH_MIN,
    }
    report["tier1_met"] = bool(
        report["pass_model"] and report["pass_discriminator_has_teeth"]
    )
    # The control bounds how far below the model score can meaningfully sit:
    # if real-vs-real already scores well above chance, the writer population
    # itself is separable and the model number is inflated.
    report["control_is_near_chance"] = bool(control < 0.60)

    payload = json.dumps(jsonable(report), indent=2)
    args.out.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
