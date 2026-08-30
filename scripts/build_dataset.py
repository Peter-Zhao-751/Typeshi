"""Parses both corpora into data/processed/{train,test}.jsonl.

Writer IDs are namespaced by corpus (`aalto:123`, `klicke:456`) so the
holdout split cannot accidentally treat a colliding numeric ID as one person.

Collection is parallelized at the file level (see typeshi.corpus_build); the
output is byte-identical for any --workers value.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from typeshi import config
from typeshi.corpus_build import (
    collect_aalto,
    collect_iterater,
    collect_klicke,
    default_workers,
)
from typeshi.dataset import revision_repeats, split_by_writer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aalto", type=Path, default=Path("data/raw/aalto"))
    ap.add_argument("--klicke", type=Path, default=Path("data/raw/klicke"))
    ap.add_argument(
        "--iterater",
        type=Path,
        default=None,
        help="IteraTeR root (data/raw/iterater) to synthesize revision "
        "sessions from. Opt-in so the phase-3 label-fix export stays "
        "attributable; export it alone (nonexistent --aalto/--klicke paths, "
        "--iterater-timing-from for the pools) for a concatenable shard",
    )
    ap.add_argument(
        "--iterater-timing-from",
        type=Path,
        default=None,
        help="KLiCKe root for the synthesis timing pools (default: --klicke)",
    )
    ap.add_argument("--limit-iterater", type=int, default=None)
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
    ap.add_argument(
        "--oversample-revisions",
        type=int,
        default=1,
        help="duplicate high-revision TRAIN windows this many times. Deliberate "
        "revision is too rare in the corpus to learn (87%% of composition "
        "windows sat at the bottom of the old scale); 1 (default) leaves the "
        "export byte-identical",
    )
    ap.add_argument(
        "--oversample-min-bin",
        type=int,
        default=17,
        help="the GEOMETRIC <REV:> bin at or above which a window counts as "
        "high-revision. 17 is ~2.3%% revision rate, above real writers' "
        "1.1-1.3%%; the old whole-percent default of 5 would be ~0.22%% on "
        "this scale and duplicate the majority of revising windows",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="worker processes for file parsing; output is identical for any "
        "value (default: cores minus two)",
    )
    args = ap.parse_args()

    rows = []
    if args.aalto.exists():
        rows += collect_aalto(args.aalto, args.limit_aalto, args.seed, args.workers)
        print(f"aalto: {len(rows)} examples")
    if args.klicke.exists():
        before = len(rows)
        rows += collect_klicke(args.klicke, args.limit_klicke, args.seed, args.workers)
        print(f"klicke: {len(rows) - before} examples")
    if args.iterater and args.iterater.exists():
        before = len(rows)
        timing_root = args.iterater_timing_from or args.klicke
        rows += collect_iterater(
            args.iterater, timing_root, args.limit_iterater, args.seed
        )
        print(f"iterater: {len(rows) - before} examples")

    if not rows:
        print("no examples produced; check the corpus paths")
        return

    train_ids, test_ids = split_by_writer(
        (w for w, _ in rows), test_frac=args.test_frac, seed=args.seed
    )
    # The hash split assigns writers independently, so tiny builds can land
    # every writer on one side. An empty holdout must fail HERE: written to
    # split.json it would ride into the checkpoint, and the eval would score
    # zero held-out sessions after parsing the whole corpus looking for them.
    if rows and args.test_frac > 0 and not test_ids:
        raise SystemExit(
            f"writer split held out 0 of {len(train_ids)} writers "
            f"(test_frac={args.test_frac}); build with more files"
        )
    if rows and args.test_frac < 1 and not train_ids:
        raise SystemExit(
            f"writer split kept 0 of {len(test_ids)} writers for training "
            f"(test_frac={args.test_frac}); build with more files"
        )
    args.out.mkdir(parents=True, exist_ok=True)

    for name, ids in (("train", train_ids), ("test", test_ids)):
        path = args.out / f"{name}.jsonl"
        written = 0
        duplicated = 0
        with path.open("w", encoding="utf-8") as fh:
            for writer, example in rows:
                if writer not in ids:
                    continue
                # Oversampling is TRAIN-only on purpose: duplicating held-out
                # windows would tilt the very distribution the eval measures
                # against, and the discriminator gate reads that distribution.
                repeats = (
                    revision_repeats(
                        example["prompt"],
                        args.oversample_revisions,
                        args.oversample_min_bin,
                    )
                    if name == "train"
                    else 1
                )
                for _ in range(repeats):
                    fh.write(json.dumps(example) + "\n")
                written += repeats
                duplicated += repeats - 1
        extra = f" (+{duplicated} oversampled)" if duplicated else ""
        print(f"wrote {written} examples from {len(ids)} writers to {path}{extra}")

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
