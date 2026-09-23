#!/usr/bin/env python3
"""Train one bounded sparse-transcriptome Hi-C model across cell types.

This entrypoint is the multi-cell counterpart of
``train_project_sparse_transcriptome_enformer_hic_balanced_clip_minmax_live.py``.
Each cell type contributes its own balanced/clipped Hi-C labels and one local
``RNA/context.npz``.  A single Hi-C head is shared across cell types, while the
matching frozen RNA state tokens are selected per sample.

The target pipeline is fixed and auditable:

``balanced contacts -> 209x209 to 256x256 resize -> log1p -> valid-pixel
window min-max``

The decoder output is constrained to ``[0, 1]`` with ``(tanh(x) + 1) / 2``.
The clipping threshold is read from each cell type's label metadata because a
93rd-percentile threshold is generally cell-type-specific.  The thresholds
are recorded in the run configuration and checkpoint provenance; they are not
used as a hidden runtime normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning import callbacks
from torch.utils.data import Dataset


_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_DEFAULT_ENFORMER_ROOT = _SRC_DIR

from trace_hic.data.multicell_live_enformer_dataset import (  # noqa: E402
    BalancedMultiCellLiveEnformerDataset,
    CellTypeContextSpec,
)
from trace_hic.model.enformer_hic_standalone import (  # noqa: E402
    HIC_MATRIX_SIZE,
    HIC_WINDOW_BP,
)
from trace_hic.model.sparse_transcriptome_model_contract import (  # noqa: E402
    SparseTranscriptomeModelContract,
)
from trace_hic.model.unit_interval_enformer_hic import (  # noqa: E402
    UNIT_INTERVAL_OUTPUT_ACTIVATION,
    UnitIntervalEnformerHiCModel,
)
from trace_hic.training.hic_supervision import (  # noqa: E402
    DEFAULT_TARGET_MATRIX_SIZE,
    DEFAULT_TARGET_WINDOW_BP,
    LEGACY_BIN_SIZE,
)
from trace_hic.training.train_enformer_hic_live import (  # noqa: E402
    _oe_pearson,
)
from trace_hic.training.train_project_sparse_transcriptome_enformer_hic_multicell import (  # noqa: E402
    LiveProjectSparseTranscriptomeMultiCellHiCModule,
    parse_project_celltypes,
)
from trace_hic.training.train_sparse_transcriptome_enformer_hic_multicell import (  # noqa: E402
    _invalid_regions_path as _base_invalid_regions_path,
    _preflight_celltype_data,
)


BALANCED_CLIP_MINMAX_MULTICELL_CHECKPOINT_SCHEMA = (
    "project_sparse_transcriptome_hic_multicell_balanced_clip_minmax/1"
)
BALANCED_LABEL_SCHEMA = "balanced_clipped_hic_diagonal_npz/1"
BALANCED_LABEL_TRAINING_OPERATIONS = ("resize", "log1p", "window_minmax")
BALANCED_LABEL_OPERATION_ORDER = ("balance", "clip")
BALANCED_LABEL_BIN_SIZE = 10_000


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Convert run arguments and manifests into strict JSON values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _checkpoint_label_metadata(
    entry: dict[str, Any],
) -> dict[str, Any]:
    """Return path-independent label provenance for a checkpoint."""

    return {
        "celltype": str(entry["celltype"]),
        "schema": str(entry["schema"]),
        "balance_name": str(entry["balance_name"]),
        "clip_threshold": float(entry["clip_threshold"]),
        "max_diagonals": int(entry["max_diagonals"]),
        "metadata_sha256": str(entry["metadata_sha256"]),
    }


def collect_balanced_label_metadata(
    args: argparse.Namespace,
    specs: Sequence[CellTypeContextSpec],
) -> list[dict[str, Any]]:
    """Validate and collect one balanced-label manifest entry per cell type."""

    data_root = Path(args.data_root).expanduser().resolve()
    entries: list[dict[str, Any]] = []
    for spec in specs:
        label_root = data_root / spec.name / args.hic_matrix_dir
        metadata_path = label_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                "Balanced multi-cell training requires metadata.json for every "
                f"label directory: {metadata_path}"
            )
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid Hi-C metadata JSON: {metadata_path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"Hi-C metadata must contain an object: {metadata_path}")

        if payload.get("schema") != BALANCED_LABEL_SCHEMA:
            raise ValueError(
                f"{metadata_path} is not a balanced clipped label archive: "
                f"{payload.get('schema')!r}"
            )
        if int(payload.get("cooler_bin_size", -1)) != BALANCED_LABEL_BIN_SIZE:
            raise ValueError(
                f"{metadata_path} has unsupported bin size: "
                f"{payload.get('cooler_bin_size')!r}"
            )
        balance_name = str(payload.get("balance_name", ""))
        if balance_name != str(args.hic_balance_name):
            raise ValueError(
                f"{metadata_path} uses balance={balance_name!r}, expected "
                f"{args.hic_balance_name!r}"
            )
        operation_order = tuple(payload.get("operation_order_at_export", ()))
        if operation_order != BALANCED_LABEL_OPERATION_ORDER:
            raise ValueError(
                f"{metadata_path} has operation order {operation_order!r}; "
                f"expected {BALANCED_LABEL_OPERATION_ORDER!r}"
            )
        training_operations = tuple(payload.get("training_operations", ()))
        if training_operations != BALANCED_LABEL_TRAINING_OPERATIONS:
            raise ValueError(
                f"{metadata_path} has training operations {training_operations!r}; "
                f"expected {BALANCED_LABEL_TRAINING_OPERATIONS!r}"
            )

        try:
            clip_threshold = float(payload["clip_threshold"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{metadata_path} has no valid clip_threshold"
            ) from error
        if not np.isfinite(clip_threshold) or clip_threshold <= 0:
            raise ValueError(
                f"{metadata_path} has an invalid clip_threshold={clip_threshold!r}"
            )
        try:
            max_diagonals = int(payload["max_diagonals"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{metadata_path} has no valid max_diagonals") from error
        if max_diagonals < HIC_WINDOW_BP // BALANCED_LABEL_BIN_SIZE:
            raise ValueError(
                f"{metadata_path} contains only {max_diagonals} diagonals, but "
                f"the 2-Mb target needs {HIC_WINDOW_BP // BALANCED_LABEL_BIN_SIZE}"
            )

        entries.append(
            {
                "celltype": spec.name,
                "metadata_path": str(metadata_path.resolve()),
                "metadata_sha256": _sha256_file(metadata_path),
                "schema": str(payload["schema"]),
                "balance_name": balance_name,
                "clip_threshold": clip_threshold,
                "max_diagonals": max_diagonals,
                "source_cooler": payload.get("source_cooler"),
            }
        )
    return entries


def attach_balanced_clip_minmax_multicell_metadata(
    args: argparse.Namespace,
    specs: Sequence[CellTypeContextSpec],
    label_metadata: Sequence[dict[str, Any]],
) -> None:
    """Attach fixed geometry and explicit multi-cell target provenance."""

    celltype = "+".join(spec.name for spec in specs)
    percentile = float(args.hic_label_clip_percentile)
    if not np.isfinite(percentile) or not 0.0 < percentile <= 100.0:
        raise ValueError("--hic-label-clip-percentile must be in (0, 100]")
    thresholds = [float(entry["clip_threshold"]) for entry in label_metadata]
    args.hic_bin_size = LEGACY_BIN_SIZE
    args.hic_target_window_bp = DEFAULT_TARGET_WINDOW_BP
    args.hic_target_matrix_size = DEFAULT_TARGET_MATRIX_SIZE
    args.hic_requires_resize = True
    args.hic_label_metadata_path = None
    args.hic_label_target_marginal = None
    args.hic_label_scale_factor = None
    args.hic_label_source_cooler = None
    args.hic_label_format = "balanced_clip_window_minmax_10kb_resize"
    args.hic_label_balance_name = str(args.hic_balance_name)
    args.hic_label_clip_percentile = percentile
    # A single scalar is only meaningful when every label archive has the same
    # threshold.  Keep it nullable and always record the per-cell values below.
    args.hic_label_clip_threshold = (
        thresholds[0]
        if thresholds and all(np.isclose(value, thresholds[0]) for value in thresholds)
        else None
    )
    args.hic_label_clip_thresholds = thresholds
    args.hic_label_metadata_paths = [
        str(entry["metadata_path"]) for entry in label_metadata
    ]
    args.hic_label_normalization = "window_minmax_valid_pixels"
    args.hic_model_output_activation = UNIT_INTERVAL_OUTPUT_ACTIVATION
    print(
        "Using balanced clipped multi-cell unit-interval supervision: "
        f"cells={celltype}, balance={args.hic_label_balance_name}, "
        f"clip=P{percentile:g}, thresholds={thresholds}, "
        "operations=resize->log1p->window_minmax(supervised valid pixels), "
        f"output_activation={UNIT_INTERVAL_OUTPUT_ACTIVATION}",
        flush=True,
    )


def _normalize_window(
    target: np.ndarray,
    valid_mask: np.ndarray,
    *,
    min_diagonal_offset: int,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.float32, np.float32]:
    """Normalize a target window and return the inverse-transform statistics."""

    target = np.asarray(target, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if target.ndim != 2 or target.shape[0] != target.shape[1]:
        raise ValueError(f"Hi-C target must be square, got {target.shape}")
    if valid_mask.shape != target.shape:
        raise ValueError(
            f"Hi-C validity mask {valid_mask.shape} does not match target "
            f"{target.shape}"
        )
    if min_diagonal_offset < 0:
        raise ValueError("Minimum diagonal offset cannot be negative")
    valid = valid_mask & np.isfinite(target)
    if min_diagonal_offset:
        coordinates = np.arange(target.shape[-1])
        diagonal_keep = (
            np.abs(coordinates[:, None] - coordinates[None, :])
            >= min_diagonal_offset
        )
        valid &= diagonal_keep
    valid_values = target[valid]
    if not valid_values.size:
        raise RuntimeError("Window contains no valid Hi-C pixels")
    minimum = np.float32(valid_values.min())
    maximum = np.float32(valid_values.max())
    scale = float(maximum - minimum)
    if scale <= epsilon:
        normalized = np.zeros_like(target, dtype=np.float32)
    else:
        normalized = np.clip(
            (target - minimum) / scale,
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
    return normalized, minimum, maximum


class BalancedClipMinMaxMultiCellDataset(Dataset):
    """Apply the single-cell window transform while preserving cell indices."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        min_diagonal_offset: int,
        epsilon: float = 1e-12,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.min_diagonal_offset = int(min_diagonal_offset)
        self.epsilon = float(epsilon)
        if self.min_diagonal_offset < 0:
            raise ValueError("Minimum diagonal offset cannot be negative")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Window min-max epsilon must be positive")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        values = list(self.dataset[index])
        if len(values) < 4:
            raise ValueError("Multi-cell samples must include a cell index")
        # The multi-cell base dataset deliberately puts celltype index last.
        celltype_index = values.pop()
        normalized, minimum, maximum = _normalize_window(
            values[1],
            values[2],
            min_diagonal_offset=self.min_diagonal_offset,
            epsilon=self.epsilon,
        )
        values[1] = normalized
        # Keep celltype_index last so the inherited Lightning batch parser sees
        # it after the two inverse-transform statistics.
        values.extend((minimum, maximum, celltype_index))
        return tuple(values)


def _centrotelo_path(
    args: argparse.Namespace,
    specs: Sequence[CellTypeContextSpec],
) -> str:
    """Find one assembly-level BED, even if the first cell lacks a copy."""

    if args.centrotelo_bed:
        path = Path(args.centrotelo_bed).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Centromere/telomere BED was not found: {path}")
        return str(path)
    data_root = Path(args.data_root).expanduser().resolve()
    candidates = [data_root / "centrotelo.bed"] + [
        data_root / spec.name / "centrotelo.bed" for spec in specs
    ]
    for path in candidates:
        if path.is_file():
            return str(path.resolve())
    raise FileNotFoundError(
        "No centromere/telomere BED was found. Supply --centrotelo-bed; "
        f"checked {candidates}"
    )


class BalancedClipMinMaxProjectSparseTranscriptomeMultiCellHiCModule(
    LiveProjectSparseTranscriptomeMultiCellHiCModule
):
    """Shared project-RNA Hi-C model with bounded predictions."""

    def __init__(
        self,
        args: argparse.Namespace,
        celltype_specs: tuple[CellTypeContextSpec, ...],
        label_metadata: Sequence[dict[str, Any]],
    ) -> None:
        super().__init__(args, celltype_specs)
        current_head = self.model.hic_model
        with torch.random.fork_rng(devices=[]):
            bounded_head = UnitIntervalEnformerHiCModel(
                encoder_hidden=current_head.encoder_hidden,
                encoder_token_bp=current_head.encoder_token_bp,
                hidden=args.hidden,
                projection_dropout=args.dropout,
                pool_heads=args.pool_heads,
                native_global_layers=args.native_global_layers,
                native_global_latents=args.native_global_latents,
                native_global_heads=args.native_global_heads,
                native_global_dropout=args.dropout,
            )
        bounded_head.load_state_dict(current_head.state_dict(), strict=True)
        self.model.hic_model = bounded_head
        self.output_activation = UNIT_INTERVAL_OUTPUT_ACTIVATION
        self.balanced_label_metadata = tuple(
            _checkpoint_label_metadata(entry) for entry in label_metadata
        )
        self._validation_label_mins: list[torch.Tensor] = []
        self._validation_label_maxs: list[torch.Tensor] = []

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self._validation_label_mins.clear()
        self._validation_label_maxs.clear()

    def validation_step(self, batch, batch_idx):
        del batch_idx
        sequence, target, valid_mask, celltype_indices = self._prepare_batch(batch)
        prediction = self(sequence, celltype_indices)
        loss = self._masked_mse(prediction, target, valid_mask)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=sequence.shape[0],
        )
        self._validation_predictions.append(prediction.detach().float().cpu())
        self._validation_targets.append(target.detach().float().cpu())
        self._validation_masks.append(valid_mask.detach().cpu())
        self._validation_cell_indices.append(celltype_indices.detach().cpu())
        self._validation_label_mins.append(batch[-3].detach().float().cpu())
        self._validation_label_maxs.append(batch[-2].detach().float().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._validation_predictions:
            return
        predictions = torch.cat(self._validation_predictions)
        targets = torch.cat(self._validation_targets)
        masks = torch.cat(self._validation_masks).bool()
        cell_indices = torch.cat(self._validation_cell_indices).long()
        minima = torch.cat(self._validation_label_mins).numpy()[:, None, None]
        maxima = torch.cat(self._validation_label_maxs).numpy()[:, None, None]
        scales = maxima - minima

        # O/E is computed in the original log1p contact domain, as in the
        # single-cell balanced trainer.  The training loss remains in [0, 1]
        # window-minmax space.
        predictions_np = predictions.numpy()
        targets_np = targets.numpy()
        masks_np = masks.numpy().astype(bool)
        predictions_log = predictions_np * scales + minima
        targets_log = targets_np * scales + minima

        local_statistics = torch.zeros(
            (len(self.celltype_names), 4),
            dtype=torch.float64,
        )
        for cell_index, celltype in enumerate(self.celltype_names):
            selected = cell_indices == cell_index
            if not selected.any():
                continue
            selected_np = selected.numpy()
            cell_predictions = predictions[selected]
            cell_targets = targets[selected]
            cell_masks = masks[selected]
            valid = (
                cell_masks
                & torch.isfinite(cell_predictions)
                & torch.isfinite(cell_targets)
            )
            local_statistics[cell_index, 0] = (
                (cell_predictions - cell_targets).square()[valid].sum().double()
            )
            local_statistics[cell_index, 1] = valid.sum().double()
            oe_scores = []
            for prediction, target, mask in zip(
                predictions_log[selected_np],
                targets_log[selected_np],
                masks_np[selected_np],
            ):
                score = _oe_pearson(
                    prediction[None, ...],
                    target[None, ...],
                    mask[None, ...],
                    self.min_diagonal_offset,
                )
                if np.isfinite(score):
                    oe_scores.append(score)
            local_statistics[cell_index, 2] = float(np.sum(oe_scores))
            local_statistics[cell_index, 3] = float(len(oe_scores))

        statistics = local_statistics.to(self.device)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
        if torch.any(statistics[:, 1] <= 0) or torch.any(statistics[:, 3] <= 0):
            raise RuntimeError(
                "At least one cell type has no globally valid validation pixels "
                "or O/E windows"
            )
        per_cell_losses = statistics[:, 0] / statistics[:, 1]
        per_cell_oe_scores = statistics[:, 2] / statistics[:, 3]
        for cell_index, celltype in enumerate(self.celltype_names):
            metric_suffix = "".join(
                character if character.isalnum() else "_" for character in celltype
            ).strip("_")
            self.log(
                f"val_loss_{metric_suffix}",
                per_cell_losses[cell_index].float(),
                sync_dist=False,
            )
            self.log(
                f"val_oe_pearson_{metric_suffix}",
                per_cell_oe_scores[cell_index].float(),
                sync_dist=False,
            )

        self.log(
            "val_macro_loss",
            per_cell_losses.mean().float(),
            prog_bar=True,
            sync_dist=False,
        )
        self.log(
            "val_macro_oe_pearson",
            per_cell_oe_scores.mean().float(),
            prog_bar=True,
            sync_dist=False,
        )
        self.log(
            "val_oe_pearson",
            (statistics[:, 2].sum() / statistics[:, 3].sum()).float(),
            sync_dist=False,
        )

    def on_save_checkpoint(self, checkpoint) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint[
            "balanced_clip_minmax_multicell_checkpoint_schema"
        ] = BALANCED_CLIP_MINMAX_MULTICELL_CHECKPOINT_SCHEMA
        checkpoint["hic_label_format"] = self.hparams["hic_label_format"]
        checkpoint["hic_label_balance_name"] = self.hparams[
            "hic_label_balance_name"
        ]
        checkpoint["hic_label_clip_percentile"] = self.hparams[
            "hic_label_clip_percentile"
        ]
        checkpoint["hic_label_clip_thresholds"] = list(
            self.hparams["hic_label_clip_thresholds"]
        )
        checkpoint["hic_label_normalization"] = "window_minmax_valid_pixels"
        checkpoint["hic_model_output_activation"] = self.output_activation
        checkpoint["balanced_label_metadata"] = list(self.balanced_label_metadata)

    def on_load_checkpoint(self, checkpoint) -> None:
        schema = checkpoint.get("balanced_clip_minmax_multicell_checkpoint_schema")
        if schema != BALANCED_CLIP_MINMAX_MULTICELL_CHECKPOINT_SCHEMA:
            raise ValueError(
                "Checkpoint is not a balanced clipped multi-cell min-max model: "
                f"{schema!r}"
            )
        super().on_load_checkpoint(checkpoint)
        expected = {
            "hic_label_format": self.hparams["hic_label_format"],
            "hic_label_balance_name": self.hparams["hic_label_balance_name"],
            "hic_label_clip_percentile": self.hparams[
                "hic_label_clip_percentile"
            ],
            "hic_label_clip_thresholds": list(
                self.hparams["hic_label_clip_thresholds"]
            ),
            "hic_label_normalization": "window_minmax_valid_pixels",
            "hic_model_output_activation": self.output_activation,
            "balanced_label_metadata": list(self.balanced_label_metadata),
        }
        for key, expected_value in expected.items():
            if checkpoint.get(key) != expected_value:
                raise ValueError(
                    f"Checkpoint {key} does not match current labels: "
                    f"{checkpoint.get(key)!r} vs {expected_value!r}"
                )


def _dataset(
    args: argparse.Namespace,
    specs: tuple[CellTypeContextSpec, ...],
    mode: str,
) -> BalancedClipMinMaxMultiCellDataset:
    if args.hic_bin_size != BALANCED_LABEL_BIN_SIZE:
        raise ValueError(
            f"Balanced clipped labels require {BALANCED_LABEL_BIN_SIZE}-bp bins, "
            f"got {args.hic_bin_size}"
        )
    if args.hic_target_window_bp != HIC_WINDOW_BP:
        raise ValueError(
            f"Sparse Enformer predicts {HIC_WINDOW_BP} bp, but labels describe "
            f"{args.hic_target_window_bp} bp"
        )
    if args.hic_target_matrix_size != HIC_MATRIX_SIZE:
        raise ValueError(
            f"Sparse Enformer predicts {HIC_MATRIX_SIZE}x{HIC_MATRIX_SIZE}, "
            f"but labels describe {args.hic_target_matrix_size}x"
            f"{args.hic_target_matrix_size}"
        )
    if not args.hic_requires_resize:
        raise ValueError("Balanced clipped 10-kb labels require the 209->256 resize")
    base = BalancedMultiCellLiveEnformerDataset(
        args.data_root,
        args.assembly,
        specs,
        mode=mode,
        centrotelo_bed=_centrotelo_path(args, specs),
        invalid_regions_bed=_base_invalid_regions_path(args),
        max_invalid_bin_fraction=args.max_invalid_bin_fraction,
        hic_matrix_dir=args.hic_matrix_dir,
        hic_bin_size=args.hic_bin_size,
        hic_target_window_bp=args.hic_target_window_bp,
        hic_target_matrix_size=args.hic_target_matrix_size,
        hic_requires_resize=args.hic_requires_resize,
    )
    return BalancedClipMinMaxMultiCellDataset(
        base,
        min_diagonal_offset=args.min_diagonal_offset,
    )


def _loader(
    args: argparse.Namespace,
    specs: tuple[CellTypeContextSpec, ...],
    mode: str,
) -> torch.utils.data.DataLoader:
    workers = int(args.num_workers)
    kwargs = {
        "batch_size": args.batch_size_per_device,
        "shuffle": mode == "train",
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 1
    return torch.utils.data.DataLoader(_dataset(args, specs, mode), **kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TRACE_hic with the frozen SUCCEED backbone and "
            "balanced clipped multi-cell Hi-C"
        )
    )
    parser.add_argument(
        "--data-root", default=str(_WORKSPACE_ROOT / "data" / "hg38")
    )
    parser.add_argument("--assembly", default="hg38")
    parser.add_argument(
        "--celltypes",
        default="GM12878,IMR90",
        help="Comma-separated project cell-type directory names.",
    )
    parser.add_argument(
        "--model-contract",
        default=str(
            _WORKSPACE_ROOT
            / "assets"
            / "model_contract"
            / "model_contract.json"
        ),
    )
    parser.add_argument(
        "--succeed-root", dest="enformer_root", metavar="PATH",
        default=str(_DEFAULT_ENFORMER_ROOT),
    )
    parser.add_argument(
        "--succeed-checkpoint", dest="enformer_checkpoint", metavar="PATH",
        default=str(
            _WORKSPACE_ROOT / "weights" / "stage1.pt"
        ),
    )
    parser.add_argument(
        "--save-path",
        default=str(
            _WORKSPACE_ROOT
            / "results"
            / "TRACE_hic"
        ),
    )
    parser.add_argument(
        "--centrotelo-bed",
        default=str(_WORKSPACE_ROOT / "assets" / "regions" / "centrotelo.bed"),
    )
    parser.add_argument(
        "--invalid-regions-bed",
        default=str(_WORKSPACE_ROOT / "assets" / "regions" / "hg38_gap.bed"),
    )
    parser.add_argument(
        "--hic-matrix-dir", default="hic_matrix_balanced_clip93"
    )
    parser.add_argument("--hic-balance-name", default="weight")
    parser.add_argument("--hic-label-clip-percentile", type=float, default=93.0)
    parser.add_argument("--max-invalid-bin-fraction", type=float, default=0.05)
    parser.add_argument("--min-diagonal-offset", type=int, default=2)
    parser.add_argument(
        "--succeed-tile-batch-size", dest="enformer_tile_batch_size",
        type=int, default=2, metavar="N",
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--pool-heads", type=int, default=8)
    parser.add_argument("--native-global-layers", type=int, default=2)
    parser.add_argument("--native-global-latents", type=int, default=128)
    parser.add_argument("--native-global-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2179)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--batch-size-per-device", type=int, default=1)
    parser.add_argument("--accumulate-grad-batches", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    contract = SparseTranscriptomeModelContract(args.model_contract)
    specs = parse_project_celltypes(args.celltypes, args.data_root, contract)
    _preflight_celltype_data(args, specs)
    label_metadata = collect_balanced_label_metadata(args, specs)
    args.celltype = "+".join(spec.name for spec in specs)
    # Compatibility aliases used by the inherited checkpoint hook.  The
    # project-local subclass removes the manifest fields before saving.
    args.transcriptome_manifest = args.model_contract
    attach_balanced_clip_minmax_multicell_metadata(args, specs, label_metadata)
    args.centrotelo_bed = _centrotelo_path(args, specs)
    args.invalid_regions_bed = _base_invalid_regions_path(args)
    pl.seed_everything(args.seed, workers=True)

    output = Path(args.save_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    label_manifest_path = output / "balanced_label_metadata.json"
    label_manifest_path.write_text(
        json.dumps(_jsonable(label_metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.hic_label_manifest_path = str(label_manifest_path)
    configuration = _jsonable(vars(args))
    configuration["entrypoint"] = str(Path(__file__).resolve())
    configuration["celltypes"] = [spec.name for spec in specs]
    configuration["balanced_label_metadata"] = _jsonable(label_metadata)
    (output / "balanced_clip_minmax_multicell_config.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    module = BalancedClipMinMaxProjectSparseTranscriptomeMultiCellHiCModule(
        args,
        specs,
        label_metadata,
    )
    checkpoint = callbacks.ModelCheckpoint(
        dirpath=output / "models",
        monitor="val_macro_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy=(
            "ddp_find_unused_parameters_false" if args.devices > 1 else "auto"
        ),
        sync_batchnorm=args.devices > 1,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        logger=pl.loggers.CSVLogger(save_dir=output / "csv"),
        callbacks=[
            checkpoint,
            callbacks.EarlyStopping(
                monitor="val_macro_loss", mode="min", patience=args.patience
            ),
            callbacks.LearningRateMonitor(logging_interval="epoch"),
        ],
        default_root_dir=output,
        log_every_n_steps=10,
        num_sanity_val_steps=1,
    )
    trainer.fit(
        module,
        train_dataloaders=_loader(args, specs, "train"),
        val_dataloaders=_loader(args, specs, "val"),
        ckpt_path=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "BALANCED_CLIP_MINMAX_MULTICELL_CHECKPOINT_SCHEMA",
    "BalancedClipMinMaxMultiCellDataset",
    "BalancedClipMinMaxProjectSparseTranscriptomeMultiCellHiCModule",
    "_normalize_window",
    "attach_balanced_clip_minmax_multicell_metadata",
    "collect_balanced_label_metadata",
    "parse_args",
]
