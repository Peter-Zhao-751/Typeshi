"""The parallel corpus build must be indistinguishable from the sequential one.

The dataset is rebuilt at several worker counts depending on the box; if
worker count changed the rows or their order, the JSONL (and downstream
training) would silently depend on the machine that built it.

An adversarial review showed the first version of these tests proved nothing:
with the then-hardcoded chunksize=16 and 6-8 test files, every "parallel"
case travelled as ONE chunk to ONE worker, where even imap_unordered
preserves order. The tests below force many chunks across several workers,
and the ordering test gives file 0 ~25x the work so that completion order
provably differs from submission order.
"""

import re
from pathlib import Path

from typeshi.corpus_build import (
    aalto_file_rows,
    collect_aalto,
    map_file_rows,
)

FIXTURE = Path(__file__).parent / "fixtures" / "aalto_sample.txt"


def _make_corpus(tmp_path: Path, n: int = 6, heavy_files: int = 0, copies: int = 25) -> Path:
    """N single-participant Aalto-format logs derived from the fixture.

    The fixture holds sessions from two participants; every leading ID is
    rewritten so each generated file belongs to exactly one. The first
    `heavy_files` files get `copies` copies of the fixture's sessions (with
    distinct section IDs), making them far slower to parse than the rest.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    src = FIXTURE.read_text()
    header, _, body = src.partition("\n")
    for i in range(n):
        pid = f"77{i:04d}"
        ncopies = copies if i < heavy_files else 1
        parts = [header]
        for k in range(ncopies):
            parts.append(
                re.sub(r"^\d+\t(\d+)\t", rf"{pid}\t9{k:02d}\g<1>\t", body, flags=re.M)
            )
        (tmp_path / f"{pid}_keystrokes.txt").write_text("\n".join(parts))
    return tmp_path


def test_fixture_rewrite_produces_rows(tmp_path):
    """Guards the test corpus itself: silence here would vacuously pass below."""
    corpus = _make_corpus(tmp_path, n=1)
    rows, dropped = aalto_file_rows(next(corpus.glob("*_keystrokes.txt")))
    assert rows, f"fixture produced no rows (dropped={dropped})"
    assert all(w == "aalto:770000" for w, _ in rows)


def test_heavy_rewrite_multiplies_sessions(tmp_path):
    """Guards the slow-file trick the ordering test depends on."""
    light = aalto_file_rows(next(_make_corpus(tmp_path / "l", n=1).glob("*.txt")))[0]
    heavy = aalto_file_rows(
        next(_make_corpus(tmp_path / "h", n=1, heavy_files=1).glob("*.txt"))
    )[0]
    assert len(heavy) >= 20 * len(light)


def test_parallel_collect_matches_sequential_exactly(tmp_path):
    corpus = _make_corpus(tmp_path, n=40)  # >2 chunks per worker at any chunksize<=3
    seq = collect_aalto(corpus, limit=None, seed=0, workers=1)
    par = collect_aalto(corpus, limit=None, seed=0, workers=3)
    assert seq  # nonempty, or the comparison proves nothing
    assert seq == par  # same rows, same order


def test_limit_selects_the_same_files_regardless_of_workers(tmp_path):
    corpus = _make_corpus(tmp_path, n=12)
    seq = collect_aalto(corpus, limit=8, seed=0, workers=1)
    par = collect_aalto(corpus, limit=8, seed=0, workers=2)
    assert seq
    assert seq == par


def test_map_file_rows_preserves_submission_order(tmp_path):
    """Order must come from the file list, not from worker completion times.

    File 0 carries ~25x the sessions of the rest and chunksize=1 hands every
    file to a different worker, so files 1..3 finish while file 0 is still
    parsing; an imap_unordered regression would yield them first.
    """
    corpus = _make_corpus(tmp_path, n=12, heavy_files=1)
    files = sorted(corpus.glob("*_keystrokes.txt"))
    rows, _ = map_file_rows(
        files, aalto_file_rows, workers=4, progress_every=0, chunksize=1
    )
    writers = [w for w, _ in rows]
    seen_order = list(dict.fromkeys(writers))
    assert seen_order == [f"aalto:{f.stem.split('_')[0]}" for f in files]
