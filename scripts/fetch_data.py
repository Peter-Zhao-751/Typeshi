"""Inspects downloaded corpora and prints their real schemas.

Run this before writing any adapter. Paste the output into docs/data-schemas.md.

Some KLiCKe logs contain invalid UTF-8 byte sequences, so reads use polars'
lossy decoding rather than failing the whole scan on one bad file.
"""

import sys
from collections import Counter
from pathlib import Path

import polars as pl


def read(path: Path, **kw) -> pl.DataFrame:
    return pl.read_csv(
        path,
        separator="\t" if path.suffix == ".tsv" else ",",
        infer_schema_length=5000,
        ignore_errors=True,
        encoding="utf8-lossy",
        **kw,
    )


def describe(path: Path, n: int = 3) -> None:
    print(f"\n{'=' * 70}\n{path}\n{'=' * 70}")
    try:
        df = read(path, n_rows=5000)
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print(f"  could not parse: {exc}")
        return
    print(f"columns ({len(df.columns)}):")
    for name, dtype in zip(df.columns, df.dtypes):
        print(f"  {name:<28} {dtype}")
    print(f"\nfirst {n} rows:")
    print(df.head(n))


def vocab(files: list[Path], column: str, limit: int = 25) -> None:
    """Value counts for a categorical column across many files."""
    counter: Counter = Counter()
    for path in files:
        try:
            counter.update(read(path)[column].to_list())
        except Exception:  # noqa: BLE001 - diagnostic script
            continue
    print(f"\n{column} vocabulary across {len(files)} files:")
    for value, count in counter.most_common(limit):
        print(f"  {str(value)[:50]:<52} {count}")


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
    files = sorted(p for p in root.rglob("*") if p.suffix in {".csv", ".tsv", ".txt"})
    if not files:
        print(f"no CSV/TSV files under {root}")
        return
    for path in files[:5]:
        describe(path)

    csvs = [p for p in files if p.suffix == ".csv"]
    if csvs:
        sample = csvs[: min(200, len(csvs))]
        if "Activity" in read(sample[0], n_rows=1).columns:
            vocab(sample, "Activity")
            vocab(sample, "DownEvent", limit=30)

    print(f"\n{len(files)} data files found under {root}")


if __name__ == "__main__":
    main()
