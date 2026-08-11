"""Corpus files -> training-example rows, parallelized at the file level.

The per-file work -- polars parsing, replay verification, a pure-Python
Levenshtein -- is embarrassingly parallel across files, and the sequential
sweep measures 13-14 files/s (~3.3 h for the full 165k-file corpus). Workers
map whole files to rows and the parent consumes results in submission order,
so the output is byte-identical to a sequential run regardless of worker
count: same rows, same order, same writer split.

This lives in the package rather than in scripts/build_dataset.py because
spawned workers import their worker function by qualified name, which needs
an importable module -- and because the parallel map must be unit-testable
without the real corpora.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Callable, Iterable

from typeshi.adapters import aalto, klicke
from typeshi.dataset import build_examples
from typeshi.labels import compute_labels
from typeshi.serialize import unsupported_chars

Row = tuple[str, dict]  # (namespaced writer ID, training example)


def default_workers() -> int:
    """Leave two cores for the OS and whatever else shares the box."""
    return max(1, (os.cpu_count() or 3) - 2)


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


def aalto_file_rows(path: Path) -> tuple[list[Row], int]:
    """(rows, dropped-session count) for one Aalto log. Runs in workers."""
    rows: list[Row] = []
    dropped = 0
    for participant, target, events in aalto.iter_sessions(path):
        examples = session_examples(target, events, "transcription")
        if examples is None:
            dropped += 1
            continue
        rows += [(f"aalto:{participant}", ex) for ex in examples]
    return rows, dropped


def klicke_file_rows(path: Path) -> tuple[list[Row], int]:
    """(rows, dropped-session count) for one KLiCKe log. Runs in workers."""
    rows: list[Row] = []
    dropped = 0
    for writer, final_text, events in klicke.iter_sessions(path):
        examples = session_examples(final_text, events, "composition")
        if examples is None:
            dropped += 1
            continue
        rows += [(f"klicke:{writer}", ex) for ex in examples]
    return rows, dropped


def map_file_rows(
    files: list[Path],
    file_rows: Callable[[Path], tuple[list[Row], int]],
    workers: int,
    label: str = "corpus",
    progress_every: int = 2000,
    chunksize: int | None = None,
) -> tuple[list[Row], int]:
    """Order-preserving map of `file_rows` over `files`, across processes.

    spawn, not fork: the parent's polars runtime has live rayon threads, and
    forking a threaded process can deadlock the child. POLARS_MAX_THREADS=1
    reaches the children through the inherited environment (each re-imports
    polars fresh; the parent's already-started pool is unaffected) -- per-file
    frames are tiny, so intra-file threading buys nothing, and N workers each
    spawning a full-width rayon pool would oversubscribe the box.

    `chunksize=None` scales the imap chunk to the work: a fixed 16 starved
    small builds (a 100-file --limit-aalto dispatched at most 7 of 24
    workers) and, worse, let the tests run every "parallel" case as a single
    chunk on a single worker. Ordering is by submission either way.
    """
    total = len(files)
    rows: list[Row] = []
    dropped = 0

    def consume(results: Iterable[tuple[list[Row], int]]) -> None:
        nonlocal dropped
        for done, (new_rows, new_dropped) in enumerate(results, 1):
            rows.extend(new_rows)
            dropped += new_dropped
            if progress_every and done % progress_every == 0:
                print(f"{label}: {done}/{total} files", flush=True)

    if workers <= 1 or total <= 1:
        consume(map(file_rows, files))
        return rows, dropped

    if chunksize is None:
        chunksize = max(1, min(16, total // (workers * 4)))
    os.environ["POLARS_MAX_THREADS"] = "1"
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, total)) as pool:
        consume(pool.imap(file_rows, files, chunksize=chunksize))
    return rows, dropped


def collect_aalto(
    root: Path, limit: int | None, seed: int, workers: int = 1
) -> list[Row]:
    """Aalto transcription sessions from physical-keyboard participants only."""
    files = sorted(root.rglob("*_keystrokes.txt"))
    metadata = next(root.rglob("metadata_participants.txt"), None)
    allowed = aalto.physical_keyboard_participants(metadata) if metadata else None
    if allowed is not None:
        files = [f for f in files if f.stem.split("_")[0] in allowed]
    if limit:
        random.Random(seed).shuffle(files)
        files = files[:limit]

    rows, dropped = map_file_rows(files, aalto_file_rows, workers, label="aalto")
    if dropped:
        print(f"aalto: dropped {dropped} sessions (unsupported chars/markers)")
    return rows


def collect_klicke(
    root: Path, limit: int | None, seed: int, workers: int = 1
) -> list[Row]:
    """KLiCKe composition sessions that replay exactly."""
    files = sorted(root.rglob("*.csv"))
    files = [f for f in files if klicke.gold_text_path(f) is not None]
    if limit:
        random.Random(seed).shuffle(files)
        files = files[:limit]

    rows, dropped = map_file_rows(files, klicke_file_rows, workers, label="klicke")
    if dropped:
        print(f"klicke: dropped {dropped} sessions (unsupported chars/markers)")
    return rows
