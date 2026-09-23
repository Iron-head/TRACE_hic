"""Shared deterministic transforms for project RNA quantification files."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Mapping

import numpy as np


PROTOCOL_TO_ID = {"total_rna": 0, "polyA_plus_rna": 1}
SAMPLE_SHEET_FIELDS = ("celltype", "rna_protocol", "expression_tsv")


def clean(value: object | None) -> str:
    if value is None:
        return ""
    result = str(value).strip()
    return "" if result in {"", "nan", "NaN", "None", "none"} else result


def canonical_ensg(value: object | None) -> str:
    gene_id = clean(value)
    if not gene_id.startswith("ENSG"):
        return ""
    gene_id = gene_id.split(".", 1)[0]
    if gene_id.endswith("_PAR_Y"):
        gene_id = gene_id[: -len("_PAR_Y")]
    return gene_id


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No TSV header found in {path}")
        return [dict(row) for row in reader]


def parse_quantification_file(
    path: Path,
    gene_to_index: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    values = np.zeros(len(gene_to_index), dtype=np.float32)
    present = np.zeros(len(gene_to_index), dtype=np.bool_)
    counters = {
        "rows": 0,
        "matched_rows": 0,
        "ignored_non_ensg": 0,
        "outside_vocabulary": 0,
        "duplicates": 0,
        "bad_tpm": 0,
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"gene_id", "TPM"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            counters["rows"] += 1
            gene_id = canonical_ensg(row.get("gene_id"))
            if not gene_id:
                counters["ignored_non_ensg"] += 1
                continue
            index = gene_to_index.get(gene_id)
            if index is None:
                counters["outside_vocabulary"] += 1
                continue
            try:
                tpm = float(clean(row.get("TPM")))
            except ValueError:
                counters["bad_tpm"] += 1
                continue
            if not math.isfinite(tpm) or tpm < 0:
                counters["bad_tpm"] += 1
                continue
            if present[index]:
                counters["duplicates"] += 1
            values[index] += np.float32(tpm)
            present[index] = True
            counters["matched_rows"] += 1
    return values, present, counters


def tie_aware_percentiles(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.float32)
    valid_values = np.asarray(values[mask], dtype=np.float64)
    count = len(valid_values)
    if count == 0:
        raise ValueError("Cannot rank a context with zero mapped genes")
    if count == 1:
        result[mask] = 0.5
        return result
    ordered = np.sort(valid_values)
    left = np.searchsorted(ordered, valid_values, side="left")
    right = np.searchsorted(ordered, valid_values, side="right")
    average_rank = (left + right - 1.0) / 2.0
    result[mask] = (average_rank / (count - 1.0)).astype(np.float32)
    return result


def single_context_features(
    mean_tpm: np.ndarray,
    gene_mask: np.ndarray,
    *,
    min_tpm: float,
    log2_tpm_cap: float,
) -> np.ndarray:
    log_tpm = np.log2(mean_tpm.astype(np.float64) + 1.0)
    features = np.zeros((len(mean_tpm), 3), dtype=np.float32)
    features[:, 0] = np.minimum(log_tpm / log2_tpm_cap, 1.0).astype(np.float32)
    features[:, 1] = tie_aware_percentiles(log_tpm, gene_mask)
    features[:, 2] = (mean_tpm >= min_tpm).astype(np.float32)
    features[~gene_mask] = 0.0
    return features


def resolve_sample_sheet(
    sample_sheet: Path,
    data_root: Path,
) -> list[dict[str, str | Path]]:
    rows = read_tsv(sample_sheet)
    if not rows:
        raise ValueError(f"RNA sample sheet is empty: {sample_sheet}")
    missing = set(SAMPLE_SHEET_FIELDS) - set(rows[0])
    if missing:
        raise ValueError(f"RNA sample sheet is missing columns: {sorted(missing)}")
    resolved: list[dict[str, str | Path]] = []
    seen_paths: set[Path] = set()
    for row_index, row in enumerate(rows, start=2):
        celltype = clean(row.get("celltype"))
        protocol = clean(row.get("rna_protocol"))
        raw_path = clean(row.get("expression_tsv"))
        if not celltype or protocol not in PROTOCOL_TO_ID or not raw_path:
            raise ValueError(
                f"Invalid RNA sample sheet row {row_index}: celltype={celltype!r}, "
                f"rna_protocol={protocol!r}, expression_tsv={raw_path!r}"
            )
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = data_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path in seen_paths:
            raise ValueError(f"RNA quantification is listed more than once: {path}")
        seen_paths.add(path)
        resolved.append(
            {
                "celltype": celltype,
                "rna_protocol": protocol,
                "expression_tsv": path,
            }
        )
    return resolved


__all__ = [
    "PROTOCOL_TO_ID",
    "canonical_ensg",
    "clean",
    "parse_quantification_file",
    "read_tsv",
    "resolve_sample_sheet",
    "single_context_features",
    "tie_aware_percentiles",
]
