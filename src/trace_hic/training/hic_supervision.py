"""Fixed legacy geometry for 10-kb Hi-C supervision labels."""

from __future__ import annotations

import argparse


RAW_HIC_MATRIX_DIR = "hic_matrix"
LEGACY_BIN_SIZE = 10_000
DEFAULT_TARGET_WINDOW_BP = 2_097_152
DEFAULT_TARGET_MATRIX_SIZE = 256


def hic_geometry_kwargs(args) -> dict[str, int | bool]:
    """Return dataset constructor kwargs from a namespace or plain mapping."""
    getter = args.get if isinstance(args, dict) else lambda key: getattr(args, key)
    return {
        "hic_bin_size": int(getter("hic_bin_size")),
        "hic_target_window_bp": int(getter("hic_target_window_bp")),
        "hic_target_matrix_size": int(getter("hic_target_matrix_size")),
        "hic_requires_resize": bool(getter("hic_requires_resize")),
    }


def attach_hic_supervision_metadata(
    args: argparse.Namespace, *, celltype: str | None = None
) -> None:
    """Attach the fixed 10-kb resize geometry without reading a JSON report.

    The live and cached SUCCEED trainers now support only the legacy 10-kb
    supervision path.  ``hic_matrix_dir`` still selects which set of NPZ
    labels to load (for example raw counts or raw common-scale counts), but
    every selected directory is interpreted as 10-kb diagonals that are
    resized from 209x209 to the 256x256 model output.
    """
    label_celltype = str(celltype if celltype is not None else args.celltype)
    matrix_dir = str(args.hic_matrix_dir)
    args.hic_label_format = (
        "raw_count_10kb_resize"
        if matrix_dir == RAW_HIC_MATRIX_DIR
        else "preprocessed_count_10kb_resize"
    )
    args.hic_label_metadata_path = None
    args.hic_label_target_marginal = None
    args.hic_label_scale_factor = None
    args.hic_label_source_cooler = None
    args.hic_bin_size = LEGACY_BIN_SIZE
    args.hic_target_window_bp = DEFAULT_TARGET_WINDOW_BP
    args.hic_target_matrix_size = DEFAULT_TARGET_MATRIX_SIZE
    args.hic_requires_resize = True
    source_bins = args.hic_target_window_bp // args.hic_bin_size
    print(
        f"Using fixed 10-kb resize labels for {label_celltype}: "
        f"directory={matrix_dir}, "
        f"geometry={args.hic_bin_size}-bp x "
        f"{source_bins} -> {args.hic_target_matrix_size}, "
        f"resize={args.hic_requires_resize}, JSON metadata disabled",
        flush=True,
    )
