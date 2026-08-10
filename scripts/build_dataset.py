"""Parses both corpora into data/processed/{train,test}.jsonl.

Writer IDs are namespaced by corpus (`aalto:123`, `klicke:456`) so the
holdout split cannot accidentally treat a colliding numeric ID as one person.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from typeshi import config
from typeshi.adapters import aalto, klicke
from typeshi.dataset import build_examples, split_by_writer
from typeshi.labels import compute_labels
from typeshi.serialize import unsupported_chars


def session_examples(target: str, events, mode: str) -> list[dict] | None:
    """Examples for one session, or None if the session cannot be represented.

    Two integrity gates live here rather than in the adapters, because they are
    properties of the token format, not of any corpus: characters without a
    registered token (English-only vocabulary), and text containing a literal
    prompt marker.
    """
    if unsupported_chars(events):
        return None
    labels = compute_labels(events, target)
    try:
        return build_examples(target, events, labels, mode)
    except ValueError:  # prompt marker inside corpus text
        return None


def collect_aalto(root: Path, limit: int | None, seed: int) -> list[tuple[str, dict]]:
    """Aalto transcription sessions from physical-keyboard participants only."""
    files = sorted(root.rglob("*_keystrokes.txt"))
    metadata = next(root.rglob("metadata_participants.txt"), None)
    allowed = aalto.physical_keyboard_participants(metadata) if metadata else None
    if allowed is not None:
        files = [f for f in files if f.stem.split("_")[0] in allowed]
    if limit:
        random.Random(seed).shuffle(files)
        files = files[:limit]

    rows: list[tuple[str, dict]] = []
    dropped = 0
    for path in files:
        for participant, target, events in aalto.iter_sessions(path):
            examples = session_examples(target, events, "transcription")
            if examples is None:
                dropped += 1
                continue
            rows += [(f"aalto:{participant}", ex) for ex in examples]
    if dropped:
        print(f"aalto: dropped {dropped} sessions (unsupported chars/markers)")
    return rows


def collect_klicke(root: Path, limit: int | None, seed: int) -> list[tuple[str, dict]]:
    """KLiCKe composition sessions that replay exactly."""
    files = sorted(root.rglob("*.csv"))
    files = [f for f in files if klicke.gold_text_path(f) is not None]
    if limit:
        random.Random(seed).shuffle(files)
        files = files[:limit]

    rows: list[tuple[str, dict]] = []
    dropped = 0
    for path in files:
        for writer, final_text, events in klicke.iter_sessions(path):
            examples = session_examples(final_text, events, "composition")
            if examples is None:
                dropped += 1
                continue
            rows += [(f"klicke:{writer}", ex) for ex in examples]
    if dropped:
        print(f"klicke: dropped {dropped} sessions (unsupported chars/markers)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aalto", type=Path, default=Path("data/raw/aalto"))
    ap.add_argument("--klicke", type=Path, default=Path("data/raw/klicke"))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--seed", type=int, default=config.DEFAULT_SEED)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument(
        "--limit-aalto",
        type=int,
        default=None,
        help="sample this many participant files (the corpus has 168k)",
    )
    ap.add_argument("--limit-klicke", type=int, default=None)
    args = ap.parse_args()

    rows: list[tuple[str, dict]] = []
    if args.aalto.exists():
        rows += collect_aalto(args.aalto, args.limit_aalto, args.seed)
        print(f"aalto: {len(rows)} examples")
    if args.klicke.exists():
        before = len(rows)
        rows += collect_klicke(args.klicke, args.limit_klicke, args.seed)
        print(f"klicke: {len(rows) - before} examples")

    if not rows:
        print("no examples produced; check the corpus paths")
        return

    train_ids, test_ids = split_by_writer(
        (w for w, _ in rows), test_frac=args.test_frac, seed=args.seed
    )
    args.out.mkdir(parents=True, exist_ok=True)

    for name, ids in (("train", train_ids), ("test", test_ids)):
        path = args.out / f"{name}.jsonl"
        written = 0
        with path.open("w", encoding="utf-8") as fh:
            for writer, example in rows:
                if writer in ids:
                    fh.write(json.dumps(example) + "\n")
                    written += 1
        print(f"wrote {written} examples from {len(ids)} writers to {path}")

    # The eval needs to know which writers were held out. Without this it would
    # re-derive the split from whatever corpus files happen to be on the eval
    # machine, which is not the same population and would silently score the
    # model on writers it trained on.
    split_path = args.out / "split.json"
    split_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "test_frac": args.test_frac,
                "train_writers": sorted(train_ids),
                "test_writers": sorted(test_ids),
            }
        )
    )
    print(f"wrote writer split to {split_path}")


if __name__ == "__main__":
    main()
