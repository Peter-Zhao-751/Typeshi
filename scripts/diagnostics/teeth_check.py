"""Verifies the ordering test catches an imap->imap_unordered regression."""
import multiprocessing.pool as mpool
import re
import sys
import tempfile
from pathlib import Path

mpool.Pool.imap = mpool.Pool.imap_unordered  # simulate the regression

from typeshi.corpus_build import aalto_file_rows, map_file_rows

FIXTURE = Path("tests/fixtures/aalto_sample.txt")


def make_corpus(root: Path, n: int, heavy_files: int, copies: int = 25) -> Path:
    src = FIXTURE.read_text()
    header, _, body = src.partition("\n")
    for i in range(n):
        pid = f"77{i:04d}"
        parts = [header]
        for k in range(copies if i < heavy_files else 1):
            parts.append(re.sub(r"^\d+\t(\d+)\t", rf"{pid}\t9{k:02d}\g<1>\t", body, flags=re.M))
        (root / f"{pid}_keystrokes.txt").write_text("\n".join(parts))
    return root


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        corpus = make_corpus(Path(td), n=12, heavy_files=1)
        files = sorted(corpus.glob("*_keystrokes.txt"))
        rows, _ = map_file_rows(files, aalto_file_rows, workers=4, progress_every=0, chunksize=1)
        seen = list(dict.fromkeys(w for w, _ in rows))
        expected = [f"aalto:{f.stem.split('_')[0]}" for f in files]
        if seen == expected:
            print("NO TEETH: order preserved even under imap_unordered")
            sys.exit(1)
        print(f"TEETH CONFIRMED: unordered run reordered results (first: {seen[:3]})")
