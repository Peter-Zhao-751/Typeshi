"""Scores a trained checkpoint: distributional metrics + discriminator.

Tier-1 passes only when ALL of the following hold (each exists because a
review found the gate exploitable without it):

- teeth: the discriminator separates real sessions from the naive heuristic
  baseline (>= 0.90) AND from timing-shuffled real sessions (>= 0.75) -- the
  second catches discriminators that only read marginal distributions.
- model: paired, group-aware CV accuracy vs our generations sits in
  [0.40, 0.55]. Paired grouping is mandatory (unpaired CV on paired data
  scores 0.085 on EXACT COPIES); the lower bound exists because below-chance
  accuracy means leakage, not realism.
- validity: >= 90% of generation attempts parse AND actually type the target
  (transcription events only, replay similarity >= 0.8). Without this a model
  that fails 99% of the time is judged on its cherry-picked survivors, and
  realistic-timing garbage that never types the target can pass.
- control: real-vs-real sits in [0.40, 0.60] -- outside that band the
  featurization or population is broken and no other number is meaningful.
- symmetry: real and baseline sessions are round-tripped through
  serialize->deserialize before featurization, so every discriminator input
  carries bin-center timings. Without this the quantization alone separates
  real from generated at 0.915 (measured, harness_control.json) and pass_model
  is unreachable for any generator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from typeshi.adapters import aalto
from typeshi.buffer import replay
from typeshi.dataset import build_prompt
from typeshi.events import EventType
from typeshi.eval.discriminator import (
    heuristic_baseline,
    real_vs_real_control,
    shuffle_timing,
    train_discriminator,
)
from typeshi.config import REPLAY_SIM_MIN
from typeshi.eval.distributional import compare_sessions
from typeshi.generate import generate
from typeshi.labels import _levenshtein, compute_labels
from typeshi.serialize import codec_roundtrip

PASS_MODEL_MAX = 0.55
PASS_MODEL_MIN = 0.40          # below-chance = leakage, never realism
PASS_TEETH_MIN = 0.90
PASS_SHUFFLE_TEETH_MIN = 0.75  # serial-dependence sensitivity
PASS_VALID_MIN = 0.90
CONTROL_BAND = (0.40, 0.60)
MIN_PAIRS = 5  # StratifiedGroupKFold(n_splits=5) needs 5 members per class


def load_test_writers(split_path: Path, allow_unsplit: bool) -> set[str] | None:
    """Writer IDs held out at dataset-build time, or None to score everything.

    The plan holds out by writer, never by session. The eval has to honour the
    *same* split, otherwise it scores realism on writers the model trained on
    and Tier-1 passes for the wrong reason.
    """
    if split_path.exists():
        payload = json.loads(split_path.read_text())
        train = set(payload["train_writers"])
        writers = set(payload["test_writers"])
        overlap = train & writers
        if overlap:
            raise SystemExit(
                f"{split_path} is corrupt: {len(overlap)} writers appear in both "
                "train and test sets"
            )
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


def resolve_split_path(checkpoint: Path, explicit: Path) -> Path:
    """Prefers the split the checkpoint was trained against.

    train_motor copies split.json into the checkpoint directory precisely so
    a later rebuild of data/processed cannot silently swap the writer split
    under an existing checkpoint.
    """
    bound = checkpoint / "split.json"
    if bound.exists():
        print(f"using the split bound to the checkpoint: {bound}")
        return bound
    return explicit


def transcription_generation_ok(events, target_text: str) -> bool:
    """A transcription generation must actually type the target.

    Parsing alone is not enough: a stream of plausible timings over the wrong
    characters is valid grammar and perfect nonsense. Require transcription
    event types only, and replayed text within edit distance of the target.
    """
    if not events:
        return False
    if any(e.type not in (EventType.KEY, EventType.BACKSPACE) for e in events):
        return False
    produced = replay(events)
    similarity = 1 - _levenshtein(produced, target_text) / max(len(target_text), 1)
    return similarity >= REPLAY_SIM_MIN


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
        help="writer split written by build_dataset.py; a copy bound to the "
             "checkpoint takes precedence when present",
    )
    ap.add_argument(
        "--allow-unsplit",
        action="store_true",
        help="score every session found, even training writers (debug only)",
    )
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument(
        "--max-per-writer", type=int, default=3,
        help="cap sessions taken from any one participant; Aalto gives each "
             "~15 and the sweep is file-ordered, so an uncapped run scores a "
             "handful of writers many times over",
    )
    ap.add_argument("--out", type=Path, default=Path("eval_report.json"))
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument(
        "--unconstrained", action="store_true",
        help="disable grammar-constrained decoding (diagnostic; the product "
             "path constrains, per the design spec)",
    )
    ap.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="generation budget per attempt. A model that never emits EOS "
             "burns the whole budget every time; transcription completions "
             "average ~92 tokens, so 512 is generous while keeping a failed "
             "generation at seconds rather than minutes",
    )
    args = ap.parse_args()

    import torch

    from typeshi.eval.load import load_checkpoint_model, load_checkpoint_tokenizer
    from typeshi.train_motor import _detect_backend

    # Same placement rules as training: device_map="auto" on MPS aborts for
    # hybrid-attention architectures, so load on CPU there and move after.
    backend = _detect_backend()
    tok = load_checkpoint_tokenizer(args.checkpoint)
    model = load_checkpoint_model(args.checkpoint, backend)
    if backend["device_map"] is None and torch.backends.mps.is_available():
        model = model.to("mps")
    model.eval()

    split_path = resolve_split_path(args.checkpoint, args.split)
    test_writers = load_test_writers(split_path, args.allow_unsplit)

    real: list = []
    fake: list = []
    baseline: list = []
    attempts = 0
    rejected_malformed = 0
    rejected_wrong_text = 0
    skipped_train_writer = 0
    skipped_unencodable = 0
    scored_writers: list[str] = []

    max_attempts = 3 * args.n
    per_writer: dict[str, int] = {}
    for i, (writer, target, events) in enumerate(aalto.iter_sessions(args.held_out)):
        if len(real) >= args.n:
            break
        if attempts >= max_attempts:
            # A mostly-invalid model must not turn the eval into an unbounded
            # sweep: n counts VALID sessions, so without this cap a model with
            # zero valid output would grind through every held-out session.
            # 3n attempts still bounds the success-rate estimate well below
            # the 0.90 gate.
            print(f"stopping after {attempts} attempts with {len(real)} valid",
                  flush=True)
            break
        if test_writers is not None and f"aalto:{writer}" not in test_writers:
            skipped_train_writer += 1
            continue
        # Aalto gives each participant ~15 sessions and this loop walks them
        # file by file, so an uncapped sweep drew 200 sessions from 14
        # writers -- too few groups for a writer-grouped judgement and a
        # thin sample of the population besides.
        if per_writer.get(writer, 0) >= args.max_per_writer:
            continue
        labels = compute_labels(events, target)
        try:
            # The closed char vocabulary raises on prompts the build never
            # gated (it checks typed chars, not the target sentence). A
            # marker-containing target lands here too instead of being
            # miscounted as a malformed *generation*.
            tok(build_prompt(target, labels, "transcription"))
        except Exception:  # noqa: BLE001 - pyo3 raises plain Exception
            skipped_unencodable += 1
            continue
        attempts += 1
        if attempts % 10 == 0:
            print(f"  attempt {attempts}: {len(real)} valid, "
                  f"{rejected_malformed} malformed, "
                  f"{rejected_wrong_text} wrong-text", flush=True)
        try:
            # Two tokens per keystroke plus margin for corrections: a
            # tight per-target budget keeps failed attempts at seconds.
            budget = min(args.max_new_tokens, 4 * len(target) + 64)
            generated = generate(
                model, tok, target, labels,
                mode="transcription", temperature=args.temperature,
                max_new_tokens=budget, seed=i,
                constrained=not args.unconstrained,
            )
        except ValueError:
            rejected_malformed += 1
            continue
        if not transcription_generation_ok(generated, target):
            rejected_wrong_text += 1
            continue
        # Timing-basis symmetrization happens once, uniformly, in the
        # codec_roundtrip pass below -- raw appends here keep labels and
        # roundtripping cleanly separated.
        per_writer[writer] = per_writer.get(writer, 0) + 1
        scored_writers.append(writer)
        real.append(events)
        fake.append(generated)
        baseline.append(heuristic_baseline(target, wpm=labels.wpm or 60, seed=i))

    success_rate = len(real) / attempts if attempts else 0.0

    if len(real) < MIN_PAIRS:
        # Fewer than MIN_PAIRS valid generations: report honestly without
        # running the discriminator (StratifiedGroupKFold needs 5 members per class).
        report = {
            "sessions_scored": len(real),
            "generation_attempts": attempts,
            "generation_success_rate": success_rate,
            "generations_rejected_as_malformed": rejected_malformed,
            "generations_rejected_wrong_text": rejected_wrong_text,
            "sessions_skipped_not_held_out": skipped_train_writer,
            "sessions_skipped_unencodable_prompt": skipped_unencodable,
            "held_out_writers_only": test_writers is not None,
            "temperature": args.temperature,
            "constrained_decoding": not args.unconstrained,
            "pass_discriminator_has_teeth": False,
            "pass_serial_dependence_teeth": False,
            "pass_model": False,
            # Also require the pair count: with too few pairs to score at all
            # (this branch), a 4/4 success_rate would otherwise read as a
            # passing gate right next to sessions_scored: 4.
            "pass_generation_validity": (
                success_rate >= PASS_VALID_MIN and len(real) >= MIN_PAIRS
            ),
            "pass_control_near_chance": False,
            "tier1_met": False,
            "note": (
                f"only {len(real)} valid generation(s); discriminator metrics "
                f"need at least {MIN_PAIRS} pairs and were not computed"
            ),
        }
        payload = json.dumps(jsonable(report), indent=2)
        args.out.write_text(payload)
        print(payload)
        return

    # Every session is projected onto the codec's timing grid before scoring.
    # Generated sessions are born there (they decode from tokens); raw corpus
    # sessions are not, and that alone separates them: real holds take
    # hundreds of distinct ms values, decoded ones ~40 bin centers, and a
    # GBM reads the comb -- measured 0.8275 paired CV against raw real
    # falling to 0.6200 against roundtripped real, with 0.438 of the
    # importance on one hold quantile. A model emitting tokens can never
    # beat the raw-real comparison, so it would gate the codec, not the
    # model. The raw number is still reported below for transparency.
    _, acc_model_raw_real = train_discriminator(real, fake, paired=True)
    real = [codec_roundtrip(s) for s in real]
    fake = [codec_roundtrip(s) for s in fake]  # idempotent, kept uniform
    baseline = [codec_roundtrip(s) for s in baseline]

    # All comparisons against generations/baselines are PAIRED (same targets).
    _, acc_model = train_discriminator(
        real, fake, paired=True, writers=scored_writers
    )
    _, acc_model_pair_grouped = train_discriminator(real, fake, paired=True)
    _, acc_model_timing = train_discriminator(
        real, fake, paired=True, count_features=False, writers=scored_writers
    )
    _, acc_baseline = train_discriminator(
        real, baseline, paired=True, writers=scored_writers
    )
    shuffled = [shuffle_timing(s, seed=i) for i, s in enumerate(real)]
    _, acc_shuffled = train_discriminator(
        real, shuffled, paired=True, writers=scored_writers
    )
    control = real_vs_real_control(real)

    gates = {
        "pass_discriminator_has_teeth": acc_baseline >= PASS_TEETH_MIN,
        "pass_serial_dependence_teeth": acc_shuffled >= PASS_SHUFFLE_TEETH_MIN,
        "pass_model": PASS_MODEL_MIN <= acc_model <= PASS_MODEL_MAX,
        "pass_generation_validity": success_rate >= PASS_VALID_MIN,
        "pass_control_near_chance": CONTROL_BAND[0] <= control <= CONTROL_BAND[1],
    }

    report = {
        "sessions_scored": len(real),
        "generation_attempts": attempts,
        "generation_success_rate": success_rate,
        "generations_rejected_as_malformed": rejected_malformed,
        "generations_rejected_wrong_text": rejected_wrong_text,
        "sessions_skipped_not_held_out": skipped_train_writer,
        "sessions_skipped_unencodable_prompt": skipped_unencodable,
        "held_out_writers_only": test_writers is not None,
        "temperature": args.temperature,
        "constrained_decoding": not args.unconstrained,
        "codec_roundtripped_real": True,
        "distributional": compare_sessions(real, fake),
        "writer_grouped_cv": True,
        "distinct_writers_scored": len(set(scored_writers)),
        "discriminator_accuracy_vs_model": acc_model,
        "discriminator_accuracy_vs_model_pair_grouped": acc_model_pair_grouped,
        "discriminator_accuracy_vs_model_raw_real": acc_model_raw_real,
        "discriminator_accuracy_vs_model_timing_only": acc_model_timing,
        "discriminator_accuracy_vs_heuristic_baseline": acc_baseline,
        "discriminator_accuracy_vs_shuffled_real": acc_shuffled,
        "discriminator_accuracy_real_vs_real_control": control,
        **gates,
        "tier1_met": all(gates.values()),
        "note_below_chance": (
            "model accuracy under 0.40 indicates leakage, not realism"
            if acc_model < PASS_MODEL_MIN else None
        ),
    }
    payload = json.dumps(jsonable(report), indent=2)
    args.out.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
