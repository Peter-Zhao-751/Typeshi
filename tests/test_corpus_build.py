"""The parallel corpus build must be indistinguishable from the sequential one.

The dataset is rebuilt at several worker counts depending on the box; if
worker count changed the rows or their order, the JSONL (and downstream
training) would silently depend on the machine that built it.
"""

import re
from pathlib import Path

from typeshi.corpus_build import (
    aalto_file_rows,
    collect_aalto,
    map_file_rows,
)

FIXTURE = Path(__file__).parent / "fixtures" / "aalto_sample.txt"


def _make_corpus(tmp_path: Path, n: int = 6) -> Path:
    """N single-participant Aalto-format logs, derived from the fixture.

    The fixture holds sessions from two participants; every leading ID is
    rewritten so each generated file belongs to exactly one, which the
    order-preservation test relies on.
    """
    src = FIXTURE.read_text()
    for i in range(n):
        pid = f"77{i:04d}"
        body = re.sub(r"^\d+\t", f"{pid}\t", src, flags=re.M)
        (tmp_path / f"{pid}_keystrokes.txt").write_text(body)
    return tmp_path


def test_fixture_rewrite_produces_rows(tmp_path):
    """Guards the test corpus itself: silence here would vacuously pass below."""
    corpus = _make_corpus(tmp_path, n=1)
    rows, dropped = aalto_file_rows(next(corpus.glob("*_keystrokes.txt")))
    assert rows, f"fixture produced no rows (dropped={dropped})"
    assert all(w == "aalto:770000" for w, _ in rows)


def test_parallel_collect_matches_sequential_exactly(tmp_path):
    corpus = _make_corpus(tmp_path)
    seq = collect_aalto(corpus, limit=None, seed=0, workers=1)
    par = collect_aalto(corpus, limit=None, seed=0, workers=3)
    assert seq  # nonempty, or the comparison proves nothing
    assert seq == par  # same rows, same order


def test_limit_selects_the_same_files_regardless_of_workers(tmp_path):
    corpus = _make_corpus(tmp_path)
    seq = collect_aalto(corpus, limit=3, seed=0, workers=1)
    par = collect_aalto(corpus, limit=3, seed=0, workers=2)
    assert seq
    assert seq == par


def test_map_file_rows_preserves_submission_order(tmp_path):
    """Order must come from the file list, not from worker completion times."""
    corpus = _make_corpus(tmp_path, n=8)
    files = sorted(corpus.glob("*_keystrokes.txt"))
    rows, _ = map_file_rows(files, aalto_file_rows, workers=4, progress_every=0)
    writers = [w for w, _ in rows]
    # Each file holds one participant, so writers must appear in file order.
    seen_order = list(dict.fromkeys(writers))
    assert seen_order == [f"aalto:{f.stem.split('_')[0]}" for f in files]
