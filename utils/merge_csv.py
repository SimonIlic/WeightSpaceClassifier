# NOTE: This file was entirely vibe-coded

"""
Utility for merging two CSV files of model metadata and results.

Typical usage::

    from merge_csv import merge_csv

    df = merge_csv(
        "metrics.csv",
        "fashion_mnist_model_results.csv",
        key_column="model_name",
        output_file="metrics_with_results.csv",
    )

This keeps all rows (and their original ordering) from *file_a* and tacks on
any matching columns from *file_b*, leaving unmatched cells as NaN.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_csv(path: str | Path, key_column: str) -> pd.DataFrame:
    """
    Read *path* into a :class:`pandas.DataFrame`, ensuring *key_column* exists
    as an ordinary column (not an index). If the CSV was written with the
    index, that index is brought back as a column, and its name is forced to
    *key_column* if missing.

    Parameters
    ----------
    path
        CSV file to load.
    key_column
        Name of the key column we want to guarantee.

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(path)

    if key_column not in df.columns:
        # Try again, assuming the first column is the index
        df = pd.read_csv(path, index_col=0)
        if df.index.name in (None, ""):
            df.index.name = key_column
        df = df.reset_index()

    return df


def merge_csv(
    file_a: str | Path,
    file_b: str | Path,
    *,
    key_column: str = "model_name",
    output_file: str | Path | None = None,
) -> pd.DataFrame:
    """
    Merge two CSV files on *key_column* using a **left join** that preserves the
    order of *file_a*.

    Parameters
    ----------
    file_a
        The "left" CSV - every row from this file is retained.
    file_b
        The "right" CSV - its columns are appended when a match is found.
    key_column
        Column name (or index) shared by both CSVs.
    output_file
        If supplied, saves the merged result to this path. When *None* (default),
        no file is written.

    Returns
    -------
    pd.DataFrame
        The merged dataframe.
    """
    df_a = _load_csv(file_a, key_column)
    df_b = _load_csv(file_b, key_column)

    merged = df_a.merge(df_b, on=key_column, how="left", suffixes=("", "_b"))

    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)

    return merged


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Merge two CSV files on a given key column.")
    parser.add_argument("file_a", help="Primary (left) CSV file.")
    parser.add_argument("file_b", help="Secondary (right) CSV file to merge in.")
    parser.add_argument("--key", default="model_name", help="Column to join on (default: model_name).")
    parser.add_argument(
        "-o",
        "--output",
        default="merged.csv",
        help="Destination CSV path for the merged result (default: merged.csv).",
    )

    args = parser.parse_args()
    merge_csv(args.file_a, args.file_b, key_column=args.key, output_file=args.output)
    print(f"Merged CSV written to {args.output}")


if __name__ == "__main__":
    _cli()
