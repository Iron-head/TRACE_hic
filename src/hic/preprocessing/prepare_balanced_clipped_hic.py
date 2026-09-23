#!/usr/bin/env python3
"""Export clipped Cooler-balanced contacts to SUCCEED diagonal NPZ labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cooler
import numpy as np


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cool", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--balance-name", default="weight")
    parser.add_argument("--clip-threshold", required=True, type=float)
    parser.add_argument("--max-diagonals", type=int, default=256)
    parser.add_argument(
        "--row-block-bins",
        type=int,
        default=2048,
        help="Number of matrix rows per balanced diagonal-band query.",
    )
    parser.add_argument(
        "--chromosomes",
        default="all",
        help="Comma-separated Cooler chromosome names or 'all'.",
    )
    parser.add_argument("--output-dtype", choices=("float16", "float32"), default="float32")
    return parser.parse_args(argv)


def _chromosomes(clr: cooler.Cooler, value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(clr.chromnames)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    missing = sorted(set(selected).difference(clr.chromnames))
    if missing:
        raise ValueError(f"Chromosomes are absent from Cooler: {missing}")
    return selected


def _clipped_diagonals(
    matrix_selector,
    *,
    chromosome_bin_start: int,
    chromosome_bin_end: int,
    max_diagonals: int,
    row_block_bins: int,
    clip_threshold: float,
    output_dtype: np.dtype,
) -> dict[str, np.ndarray]:
    chromosome_bins = chromosome_bin_end - chromosome_bin_start
    output: dict[str, np.ndarray] = {
        str(signed_offset): np.zeros(
            chromosome_bins - abs(signed_offset), dtype=output_dtype
        )
        for offset in range(max_diagonals)
        for signed_offset in ({0} if offset == 0 else (offset, -offset))
    }
    # Query only a band around each block of rows. Cooler still applies its
    # balance column, while distant contacts that cannot enter the exported
    # diagonals are never materialized as one huge chromosome sparse matrix.
    for row_start in range(
        chromosome_bin_start, chromosome_bin_end, row_block_bins
    ):
        row_end = min(row_start + row_block_bins, chromosome_bin_end)
        column_start = max(
            chromosome_bin_start, row_start - max_diagonals + 1
        )
        column_end = min(
            chromosome_bin_end, row_end + max_diagonals - 1
        )
        block = matrix_selector[
            row_start:row_end,
            column_start:column_end,
        ].tocoo()
        if not block.nnz:
            continue
        rows = row_start + block.row - chromosome_bin_start
        columns = column_start + block.col - chromosome_bin_start
        offsets = columns - rows
        within_band = np.abs(offsets) < max_diagonals
        rows = rows[within_band]
        columns = columns[within_band]
        offsets = offsets[within_band]
        values = np.asarray(block.data[within_band], dtype=np.float64)
        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=clip_threshold,
            neginf=0.0,
        )
        values = np.clip(values, 0.0, clip_threshold).astype(
            output_dtype, copy=False
        )
        for signed_offset in np.unique(offsets):
            selected = offsets == signed_offset
            diagonal_indices = np.minimum(rows[selected], columns[selected])
            output[str(int(signed_offset))][diagonal_indices] = values[selected]
    return output


def _full_matrix_clipped_diagonals(
    matrix,
    *,
    max_diagonals: int,
    clip_threshold: float,
    output_dtype: np.dtype,
) -> dict[str, np.ndarray]:
    """Reference implementation retained for tests and numerical checks."""
    output: dict[str, np.ndarray] = {}
    for offset in range(max_diagonals):
        for signed_offset in ({0} if offset == 0 else (offset, -offset)):
            values = np.asarray(matrix.diagonal(signed_offset), dtype=np.float64)
            values = np.nan_to_num(values, nan=0.0, posinf=clip_threshold, neginf=0.0)
            values = np.clip(values, 0.0, clip_threshold).astype(
                output_dtype, copy=False
            )
            output[str(signed_offset)] = values
    return output


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.clip_threshold <= 0:
        raise ValueError("--clip-threshold must be positive")
    if args.max_diagonals <= 0:
        raise ValueError("--max-diagonals must be positive")
    if args.row_block_bins <= 0:
        raise ValueError("--row-block-bins must be positive")
    cool_path = args.cool.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    clr = cooler.Cooler(str(cool_path))
    if clr.binsize is None:
        raise ValueError("Variable-bin Coolers are not supported")
    if args.balance_name not in clr.bins()[:1].columns:
        raise KeyError(
            f"Cooler does not contain balance column {args.balance_name!r}"
        )
    selected = _chromosomes(clr, args.chromosomes)
    dtype = np.dtype(args.output_dtype)
    matrix_selector = clr.matrix(balance=args.balance_name, sparse=True)
    outputs = []
    for position, chromosome in enumerate(selected, start=1):
        print(
            f"[{position}/{len(selected)}] Reading balanced {chromosome}",
            flush=True,
        )
        bin_start, bin_end = clr.extent(chromosome)
        diagonals = _clipped_diagonals(
            matrix_selector,
            chromosome_bin_start=bin_start,
            chromosome_bin_end=bin_end,
            max_diagonals=args.max_diagonals,
            row_block_bins=args.row_block_bins,
            clip_threshold=args.clip_threshold,
            output_dtype=dtype,
        )
        filename = chromosome if chromosome.startswith("chr") else f"chr{chromosome}"
        destination = output / f"{filename}.npz"
        np.savez(destination, **diagonals)
        outputs.append(str(destination))
        print(f"[{position}/{len(selected)}] Wrote {destination}", flush=True)
    metadata = {
        "schema": "balanced_clipped_hic_diagonal_npz/1",
        "source_cooler": str(cool_path),
        "cooler_bin_size": int(clr.binsize),
        "balance_name": args.balance_name,
        "clip_threshold": float(args.clip_threshold),
        "max_diagonals": int(args.max_diagonals),
        "row_block_bins": int(args.row_block_bins),
        "output_dtype": dtype.name,
        "chromosomes": selected,
        "operation_order_at_export": ["balance", "clip"],
        "training_operations": ["resize", "log1p", "window_minmax"],
        "training_normalization": "per_window_supervised_valid_pixels",
        "files": outputs,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
