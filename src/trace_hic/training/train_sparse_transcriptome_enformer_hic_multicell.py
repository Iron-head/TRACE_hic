"""Train one sparse-transcriptome Enformer Hi-C model across cell types.

Each cell type supplies its own Hi-C targets and one full-transcriptome context
ID.  RNA contexts are encoded once into a small ``[C,32,256]`` state bank;
every batch selects the matching state per sample before evaluating the shared
frozen Enformer and trainable Hi-C head.  Training examples are balanced by
cell type and validation reports both per-cell and macro metrics.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning import callbacks


_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from trace_hic.data.multicell_live_enformer_dataset import (
    BalancedMultiCellLiveEnformerDataset,
    CellTypeContextSpec,
    parse_celltype_context_specs,
)
from trace_hic.model.enformer_hic_standalone import (
    HIC_MATRIX_SIZE,
    HIC_WINDOW_BP,
    EnformerHiCModel,
)
from trace_hic.model.sparse_transcriptome_enformer_hic_standalone import (
    SPARSE_TRANSCRIPTOME_FRAMEWORK,
    FrozenSparseTranscriptomeEnformerExtractor,
    SparseTranscriptomeEndToEndHiCModel,
    SparseTranscriptomeTwoMegabaseEncoder,
)
from trace_hic.training.hic_supervision import attach_hic_supervision_metadata
from trace_hic.training.train_enformer_hic_live import (
    _apply_diagonal_mask,
    _oe_pearson,
)


MULTICELL_CHECKPOINT_SCHEMA = "sparse_transcriptome_hic_multicell/1"


def _metric_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


class LiveSparseTranscriptomeMultiCellHiCModule(pl.LightningModule):
    """Shared Hi-C head with per-sample selection from a frozen RNA bank."""

    strict_loading = False

    def __init__(
        self,
        args: argparse.Namespace,
        celltype_specs: tuple[CellTypeContextSpec, ...] | None = None,
    ) -> None:
        super().__init__()
        specs = (
            parse_celltype_context_specs(args.celltypes)
            if celltype_specs is None
            else tuple(celltype_specs)
        )
        self.save_hyperparameters(vars(args))

        extractor = FrozenSparseTranscriptomeEnformerExtractor(
            args.enformer_root,
            args.enformer_checkpoint,
            args.transcriptome_manifest,
            freeze=True,
        )
        context_ids = tuple(spec.context_id for spec in specs)
        context_rows = tuple(extractor.context_row(value) for value in context_ids)
        context_states = extractor.encode_context_ids(context_ids)
        encoder = SparseTranscriptomeTwoMegabaseEncoder(
            extractor,
            tile_batch_size=args.enformer_tile_batch_size,
        )
        hic_model = EnformerHiCModel(
            encoder_hidden=extractor.output_hidden,
            encoder_token_bp=extractor.token_bp,
            hidden=args.hidden,
            projection_dropout=args.dropout,
            pool_heads=args.pool_heads,
            native_global_layers=args.native_global_layers,
            native_global_latents=args.native_global_latents,
            native_global_heads=args.native_global_heads,
            native_global_dropout=args.dropout,
        )
        self.model = SparseTranscriptomeEndToEndHiCModel(encoder, hic_model)
        self.register_buffer(
            "context_state_bank",
            context_states,
            persistent=True,
        )
        self.celltype_names = tuple(spec.name for spec in specs)
        self.celltype_context_ids = context_ids
        self.celltype_context_rows = context_rows
        self.transcriptome_manifest_sha256 = extractor.transcriptome_manifest_sha256
        self.external_encoder_framework = SPARSE_TRANSCRIPTOME_FRAMEWORK
        self.external_encoder_global_step = extractor.checkpoint_global_step
        self.learning_rate = float(args.learning_rate)
        self.weight_decay = float(args.weight_decay)
        self.warmup_epochs = int(args.warmup_epochs)
        self.max_epochs = int(args.max_epochs)
        self.min_diagonal_offset = int(args.min_diagonal_offset)
        self._validation_predictions: list[torch.Tensor] = []
        self._validation_targets: list[torch.Tensor] = []
        self._validation_masks: list[torch.Tensor] = []
        self._validation_cell_indices: list[torch.Tensor] = []
        print(
            "Using frozen sparse multi-cell transcriptomes: "
            + ", ".join(
                f"{name}=context_id:{context_id}/row:{row}"
                for name, context_id, row in zip(
                    self.celltype_names,
                    self.celltype_context_ids,
                    self.celltype_context_rows,
                )
            )
            + f", state_bank={tuple(context_states.shape)}, "
            f"external_step={extractor.checkpoint_global_step}, "
            f"manifest_sha256={self.transcriptome_manifest_sha256}",
            flush=True,
        )

    def states_for_celltypes(self, celltype_indices: torch.Tensor) -> torch.Tensor:
        indices = torch.as_tensor(
            celltype_indices,
            device=self.context_state_bank.device,
            dtype=torch.long,
        )
        if indices.ndim != 1:
            raise ValueError(
                "celltype_indices must have shape [B], got "
                f"{tuple(indices.shape)}"
            )
        if torch.any(indices < 0) or torch.any(
            indices >= len(self.celltype_names)
        ):
            raise ValueError("celltype_indices contains an out-of-range value")
        return self.context_state_bank.index_select(0, indices)

    def forward(
        self,
        context_sequence: torch.Tensor,
        celltype_indices: torch.Tensor | None = None,
        context_state_tokens: torch.Tensor | None = None,
        *,
        disable_context: bool = False,
    ) -> torch.Tensor:
        if context_state_tokens is None:
            if celltype_indices is None:
                raise ValueError(
                    "celltype_indices or explicit context_state_tokens are required"
                )
            states = self.states_for_celltypes(celltype_indices)
        else:
            states = context_state_tokens
        return self.model(
            context_sequence,
            states,
            disable_context=disable_context,
        )

    @staticmethod
    def _masked_mse(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid_mask & torch.isfinite(prediction) & torch.isfinite(target)
        if not valid.any():
            raise RuntimeError("Batch contains no valid Hi-C target pixels")
        return ((prediction.float() - target.float()) ** 2)[valid].mean()

    def _prepare_batch(self, batch):
        sequence, target, valid_mask, *_, celltype_indices = batch
        target = target.float()
        valid_mask = _apply_diagonal_mask(
            valid_mask.bool(),
            self.min_diagonal_offset,
        )
        return sequence, target, valid_mask, celltype_indices.long()

    def training_step(self, batch, batch_idx):
        del batch_idx
        sequence, target, valid_mask, celltype_indices = self._prepare_batch(batch)
        prediction = self(sequence, celltype_indices)
        loss = self._masked_mse(prediction, target, valid_mask)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=sequence.shape[0],
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self._validation_predictions.clear()
        self._validation_targets.clear()
        self._validation_masks.clear()
        self._validation_cell_indices.clear()

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

    def on_validation_epoch_end(self) -> None:
        if not self._validation_predictions:
            return
        predictions = torch.cat(self._validation_predictions)
        targets = torch.cat(self._validation_targets)
        masks = torch.cat(self._validation_masks).bool()
        cell_indices = torch.cat(self._validation_cell_indices).long()
        # Columns are squared-error sum, valid-pixel count, O/E score sum and
        # valid O/E-window count. Explicit reduction is robust when a DDP rank
        # happens to receive samples from only one cell type.
        local_statistics = torch.zeros(
            (len(self.celltype_names), 4),
            dtype=torch.float64,
        )
        for cell_index, celltype in enumerate(self.celltype_names):
            selected = cell_indices == cell_index
            if not selected.any():
                continue
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
            prediction_values = cell_predictions.numpy()
            target_values = cell_targets.numpy()
            mask_values = cell_masks.numpy().astype(bool)
            for prediction, target, mask in zip(
                prediction_values,
                target_values,
                mask_values,
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
            metric_suffix = _metric_name(celltype)
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

        macro_loss = per_cell_losses.mean().float()
        macro_oe = per_cell_oe_scores.mean().float()
        overall_oe = statistics[:, 2].sum() / statistics[:, 3].sum()
        self.log(
            "val_macro_loss",
            macro_loss,
            prog_bar=True,
            sync_dist=False,
        )
        self.log(
            "val_macro_oe_pearson",
            macro_oe,
            prog_bar=True,
            sync_dist=False,
        )
        self.log(
            "val_oe_pearson",
            overall_oe.float(),
            sync_dist=False,
        )

    def configure_optimizers(self):
        parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("No trainable Hi-C parameters remain")
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        def schedule(epoch: int) -> float:
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / max(1, self.warmup_epochs)
            if self.max_epochs <= self.warmup_epochs:
                return 1.0
            progress = (epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            return 0.5 * (
                1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "name": "lr",
            },
        }

    def on_save_checkpoint(self, checkpoint) -> None:
        prefix = "model.encoder.extractor."
        state_dict = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(prefix)
        }
        checkpoint["multicell_checkpoint_schema"] = MULTICELL_CHECKPOINT_SCHEMA
        checkpoint["external_encoder_framework"] = self.external_encoder_framework
        checkpoint["external_encoder_global_step"] = (
            self.external_encoder_global_step
        )
        checkpoint["external_enformer_root"] = self.hparams["enformer_root"]
        checkpoint["external_enformer_checkpoint"] = self.hparams[
            "enformer_checkpoint"
        ]
        checkpoint["transcriptome_manifest"] = self.hparams[
            "transcriptome_manifest"
        ]
        checkpoint["transcriptome_manifest_sha256"] = (
            self.transcriptome_manifest_sha256
        )
        checkpoint["celltype_names"] = list(self.celltype_names)
        checkpoint["celltype_context_ids"] = list(self.celltype_context_ids)
        checkpoint["celltype_context_rows"] = list(self.celltype_context_rows)

    def on_load_checkpoint(self, checkpoint) -> None:
        schema = checkpoint.get("multicell_checkpoint_schema")
        if schema is not None and schema != MULTICELL_CHECKPOINT_SCHEMA:
            raise ValueError(
                f"Unsupported multi-cell checkpoint schema: {schema!r}"
            )
        saved_framework = checkpoint.get("external_encoder_framework")
        if (
            saved_framework is not None
            and saved_framework != self.external_encoder_framework
        ):
            raise ValueError("External sparse Enformer framework does not match")
        saved_hash = checkpoint.get("transcriptome_manifest_sha256")
        if saved_hash is not None and saved_hash != self.transcriptome_manifest_sha256:
            raise ValueError("Transcriptome manifest does not match the checkpoint")
        expected = (
            list(self.celltype_names),
            list(self.celltype_context_ids),
            list(self.celltype_context_rows),
        )
        saved = (
            checkpoint.get("celltype_names"),
            checkpoint.get("celltype_context_ids"),
            checkpoint.get("celltype_context_rows"),
        )
        for saved_value, expected_value, label in zip(
            saved,
            expected,
            ("celltype names", "context IDs", "context rows"),
        ):
            if saved_value is not None and list(saved_value) != expected_value:
                raise ValueError(
                    f"Checkpoint {label} do not match: "
                    f"{saved_value!r} vs {expected_value!r}"
                )


def _invalid_regions_path(args: argparse.Namespace) -> str | None:
    if args.invalid_regions_bed:
        return str(Path(args.invalid_regions_bed).expanduser().resolve())
    default = _REPO_ROOT / "data" / f"{args.assembly}_gap.bed"
    return str(default) if default.is_file() else None


def _centrotelo_path(
    args: argparse.Namespace,
    specs: tuple[CellTypeContextSpec, ...],
) -> str:
    if args.centrotelo_bed:
        return str(Path(args.centrotelo_bed).expanduser().resolve())
    default = Path(args.data_root) / specs[0].name / "centrotelo.bed"
    if not default.is_file():
        raise FileNotFoundError(
            "No shared centromere/telomere BED was supplied and the first "
            f"cell type has none: {default}"
        )
    return str(default.resolve())


def _dataset(
    args: argparse.Namespace,
    specs: tuple[CellTypeContextSpec, ...],
    mode: str,
) -> BalancedMultiCellLiveEnformerDataset:
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
    return BalancedMultiCellLiveEnformerDataset(
        args.data_root,
        args.assembly,
        specs,
        mode=mode,
        centrotelo_bed=_centrotelo_path(args, specs),
        invalid_regions_bed=_invalid_regions_path(args),
        max_invalid_bin_fraction=args.max_invalid_bin_fraction,
        hic_matrix_dir=args.hic_matrix_dir,
        hic_bin_size=args.hic_bin_size,
        hic_target_window_bp=args.hic_target_window_bp,
        hic_target_matrix_size=args.hic_target_matrix_size,
        hic_requires_resize=args.hic_requires_resize,
    )


def _preflight_celltype_data(
    args: argparse.Namespace,
    specs: tuple[CellTypeContextSpec, ...],
) -> None:
    """Reject missing or differently named supervision before model loading."""

    data_root = Path(args.data_root).expanduser().resolve()
    missing = [
        str(data_root / spec.name / args.hic_matrix_dir)
        for spec in specs
        if not (data_root / spec.name / args.hic_matrix_dir).is_dir()
    ]
    if missing:
        raise FileNotFoundError(
            "Every cell type must provide the same Hi-C supervision directory "
            f"name ({args.hic_matrix_dir!r}). Missing: {missing}. Mixing raw, "
            "ICE or common-scale labels in one run is not allowed."
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanced multi-cell sparse-transcriptome Enformer -> Hi-C"
    )
    parser.add_argument(
        "--data-root",
        default=str(_WORKSPACE_ROOT / "data" / "hg38"),
    )
    parser.add_argument("--assembly", default="hg38")
    parser.add_argument(
        "--celltypes",
        default="GM12878:44,IMR90:57",
        help="Comma-separated CELLTYPE:TRANSCRIPTOME_CONTEXT_ID entries.",
    )
    parser.add_argument(
        "--enformer-root",
        default=str(_REPO_ROOT / "src"),
    )
    parser.add_argument(
        "--enformer-checkpoint",
        default=(
            str(_REPO_ROOT / "weights" / "stage1.pt")
        ),
    )
    parser.add_argument(
        "--transcriptome-manifest",
        default=(
            str(_REPO_ROOT / "assets" / "model_contract" / "model_contract.json")
        ),
    )
    parser.add_argument(
        "--save-path",
        default=str(
            _WORKSPACE_ROOT
            / "results"
            / "GM12878_IMR90_sparse_transcriptome_rna_hic_live"
        ),
    )
    parser.add_argument("--centrotelo-bed", default=None)
    parser.add_argument("--invalid-regions-bed", default=None)
    parser.add_argument(
        "--hic-matrix-dir",
        default="hic_matrix",
        help=(
            "One common label directory name required under every cell type. "
            "Different supervision preprocessing must not be mixed."
        ),
    )
    parser.add_argument("--max-invalid-bin-fraction", type=float, default=0.05)
    parser.add_argument("--min-diagonal-offset", type=int, default=2)
    parser.add_argument("--enformer-tile-batch-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--pool-heads", type=int, default=8)
    parser.add_argument("--native-global-layers", type=int, default=2)
    parser.add_argument("--native-global-latents", type=int, default=128)
    parser.add_argument("--native-global-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2237)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = parse_celltype_context_specs(args.celltypes)
    _preflight_celltype_data(args, specs)
    # Keep the existing fixed-label metadata helper and checkpoint schema.
    args.celltype = "+".join(spec.name for spec in specs)
    attach_hic_supervision_metadata(args, celltype=args.celltype)
    pl.seed_everything(args.seed, workers=True)
    output = Path(args.save_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    module = LiveSparseTranscriptomeMultiCellHiCModule(args, specs)
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
                monitor="val_macro_loss",
                mode="min",
                patience=args.patience,
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
