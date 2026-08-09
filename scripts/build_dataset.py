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
    for path in files:
        for participant, target, events in aalto.iter_sessions(path):
            labels = compute_labels(events, target)
            for example in build_examples(target, events, labels, "transcription"):
                rows.append((f"aalto:{participant}", example))
    return rows


def collect_klicke(root: Path, limit: int | None, seed: int) -> list[tuple[str, dict]]:
    """KLiCKe composition sessions that replay exactly."""
    files = sorted(root.rglob("*.csv"))
    files = [f for f in files if klicke.gold_text_path(f) is not None]
    if limit:
        random.Random(seed).shuffle(files)
        files = files[:limit]

    rows: list[tuple[str, dict]] = []
    for path in files:
        for writer, final_text, events in klicke.iter_sessions(path):
            labels = compute_labels(events, final_text)
            for example in build_examples(final_text, events, labels, "composition"):
                rows.append((f"klicke:{writer}", example))
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


if __name__ == "__main__":
    main()
