"""Shared live DNA and Hi-C utilities used by TRACE_hic training."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks
from scipy.stats import pearsonr


_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from trace_hic.data.chromosome_dataset import ChromosomeDataset
from trace_hic.data.genome_dataset import GenomeDataset
from trace_hic.model.enformer_hic_standalone import (
    ENFORMER_CONTEXT_BP,
    ENFORMER_FLANK_BP,
    HIC_MATRIX_SIZE,
    HIC_WINDOW_BP,
    EnformerEndToEndHiCModel,
    EnformerHiCModel,
    EnformerTwoMegabaseEncoder,
    FrozenMultiLayerEnformerExtractor,
    load_expression_npz,
)
from trace_hic.training.hic_supervision import attach_hic_supervision_metadata


class LiveEnformerChromosomeDataset(ChromosomeDataset):
    """Return padded Enformer context plus the existing Hi-C target."""

    _DNA_LUT = np.zeros((256, 4), dtype=np.uint8)
    _DNA_LUT[ord("A"), 0] = _DNA_LUT[ord("a"), 0] = 1
    _DNA_LUT[ord("C"), 1] = _DNA_LUT[ord("c"), 1] = 1
    _DNA_LUT[ord("G"), 2] = _DNA_LUT[ord("g"), 2] = 1
    _DNA_LUT[ord("T"), 3] = _DNA_LUT[ord("t"), 3] = 1

    def _get_padded_context(self, target_start: int) -> np.ndarray:
        # The first Enformer center starts exactly at target_start.  Its input
        # begins ENFORMER_FLANK_BP bases earlier, so all returned hidden tokens
        # have the same genomic origin as the Hi-C target.
        context_start = int(target_start) - ENFORMER_FLANK_BP
        context_end = context_start + ENFORMER_CONTEXT_BP
        chromosome_length = len(self.seq)
        clipped_start = max(0, context_start)
        clipped_end = min(chromosome_length, context_end)

        encoded = np.zeros((ENFORMER_CONTEXT_BP, 4), dtype=np.uint8)
        if clipped_end <= clipped_start:
            return encoded
        sequence = self.seq.seq[clipped_start:clipped_end]
        sequence_bytes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        destination_start = clipped_start - context_start
        encoded[
            destination_start:destination_start + len(sequence_bytes)
        ] = self._DNA_LUT[sequence_bytes]
        return encoded

    def get_data_at_interval(self, start, end, interval_idx=None):
        del end, interval_idx
        start = int(start)
        self.validate_target_start(start)
        sequence = self._get_padded_context(start)
        matrix = self.mat.get(
            start,
            window=self.target_window_bp,
            res=self.res,
        )
        matrix = self.transform_hic_matrix(matrix)
        valid_mask = self.get_valid_pixel_mask(start)
        return sequence, [], matrix, valid_mask


class LiveEnformerGenomeDataset(GenomeDataset):
    """Genome split matching the existing live-SUCCEED training protocol."""

    def load_chrs(self, chr_names, genomic_features):
        print("Loading live multi-layer Enformer chromosome datasets...")
        chromosome_datasets = {}
        lengths = []
        for chromosome in chr_names:
            chromosome_datasets[chromosome] = LiveEnformerChromosomeDataset(
                self.data_root,
                chromosome,
                self.centrotelo_dict[chromosome],
                genomic_features,
                use_aug=False,
                max_invalid_bin_fraction=self.max_invalid_bin_fraction,
                hic_matrix_dir=self.hic_matrix_dir,
                hic_bin_size=self.hic_bin_size,
                hic_target_window_bp=self.hic_target_window_bp,
                hic_target_matrix_size=self.hic_target_matrix_size,
                hic_requires_resize=self.hic_requires_resize,
            )
            lengths.append(len(chromosome_datasets[chromosome]))
        print("Live multi-layer Enformer chromosome datasets loaded")
        return chromosome_datasets, lengths


def _apply_diagonal_mask(mask: torch.Tensor, min_diagonal_offset: int) -> torch.Tensor:
    if min_diagonal_offset <= 0:
        return mask
    coordinates = torch.arange(mask.shape[-1], device=mask.device)
    keep = (coordinates[:, None] - coordinates[None, :]).abs() >= min_diagonal_offset
    return mask & keep


def _log_oe(matrix: np.ndarray, valid_mask: np.ndarray, min_diag: int) -> np.ndarray:
    contact = np.expm1(np.maximum(np.asarray(matrix, dtype=np.float64), 0.0))
    size = contact.shape[0]
    output = np.full((size, size), np.nan, dtype=np.float64)
    for diagonal in range(min_diag, size):
        row = np.arange(size - diagonal)
        column = row + diagonal
        valid = valid_mask[row, column] & np.isfinite(contact[row, column])
        if not valid.any():
            continue
        expected = contact[row[valid], column[valid]].mean()
        if not np.isfinite(expected) or expected <= 0:
            continue
        values = np.log2(
            (contact[row[valid], column[valid]] + 1e-6)
            / (expected + 1e-6)
        )
        output[row[valid], column[valid]] = values
    return output


def _oe_pearson(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    min_diag: int,
) -> float:
    scores = []
    for prediction, target, mask in zip(predictions, targets, masks):
        prediction_oe = _log_oe(prediction, mask, min_diag)
        target_oe = _log_oe(target, mask, min_diag)
        valid = np.isfinite(prediction_oe) & np.isfinite(target_oe)
        if valid.sum() < 2:
            continue
        x, y = prediction_oe[valid], target_oe[valid]
        if x.std() == 0 or y.std() == 0:
            continue
        scores.append(float(pearsonr(x, y).statistic))
    return float(np.mean(scores)) if scores else float("nan")


class LiveEnformerHiCModule(pl.LightningModule):
    """Lightning wrapper with external frozen Enformer weights."""

    strict_loading = False

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(vars(args))

        extractor = FrozenMultiLayerEnformerExtractor(
            args.enformer_root,
            args.enformer_checkpoint,
            freeze=True,
        )
        encoder = EnformerTwoMegabaseEncoder(
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
        self.model = EnformerEndToEndHiCModel(encoder, hic_model)
        expression = load_expression_npz(
            args.expression_npz,
            args.expression_key,
            expected_size=extractor.context_gene_count,
        )
        self.register_buffer(
            "expression",
            torch.from_numpy(expression),
            persistent=True,
        )
        self.learning_rate = float(args.learning_rate)
        self.weight_decay = float(args.weight_decay)
        self.warmup_epochs = int(args.warmup_epochs)
        self.max_epochs = int(args.max_epochs)
        self.min_diagonal_offset = int(args.min_diagonal_offset)
        self._validation_predictions = []
        self._validation_targets = []
        self._validation_masks = []

    def forward(self, context_sequence: torch.Tensor) -> torch.Tensor:
        return self.model(context_sequence, self.expression)

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
        sequence, target, valid_mask, *_ = batch
        target = target.float()
        valid_mask = _apply_diagonal_mask(
            valid_mask.bool(),
            self.min_diagonal_offset,
        )
        return sequence, target, valid_mask

    def training_step(self, batch, batch_idx):
        del batch_idx
        sequence, target, valid_mask = self._prepare_batch(batch)
        prediction = self(sequence)
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

    def validation_step(self, batch, batch_idx):
        del batch_idx
        sequence, target, valid_mask = self._prepare_batch(batch)
        prediction = self(sequence)
        loss = self._masked_mse(prediction, target, valid_mask)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=sequence.shape[0],
        )
        self._validation_predictions.append(prediction.detach().float().cpu())
        self._validation_targets.append(target.detach().float().cpu())
        self._validation_masks.append(valid_mask.detach().cpu())

    def on_validation_epoch_end(self) -> None:
        if not self._validation_predictions:
            return
        predictions = torch.cat(self._validation_predictions).numpy()
        targets = torch.cat(self._validation_targets).numpy()
        masks = torch.cat(self._validation_masks).numpy().astype(bool)
        score = _oe_pearson(
            predictions,
            targets,
            masks,
            self.min_diagonal_offset,
        )
        self.log(
            "val_oe_pearson",
            torch.tensor(score, device=self.device, dtype=torch.float32),
            prog_bar=True,
            sync_dist=True,
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
            return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

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
        # The external Enformer is reconstructed from its trusted path on
        # resume.  Do not duplicate roughly 1 GB of frozen weights in every
        # downstream Hi-C checkpoint.
        prefix = "model.encoder.extractor."
        state_dict = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith(prefix)
        }
        checkpoint["external_enformer_root"] = self.hparams["enformer_root"]
        checkpoint["external_enformer_checkpoint"] = self.hparams[
            "enformer_checkpoint"
        ]


def _invalid_regions_path(args: argparse.Namespace) -> str | None:
    if args.invalid_regions_bed:
        return str(Path(args.invalid_regions_bed).expanduser().resolve())
    default = _REPO_ROOT / "data" / f"{args.assembly}_gap.bed"
    return str(default) if default.is_file() else None


def _dataset(args: argparse.Namespace, mode: str) -> LiveEnformerGenomeDataset:
    if args.hic_target_window_bp != HIC_WINDOW_BP:
        raise ValueError(
            f"Enformer Hi-C model predicts {HIC_WINDOW_BP} bp, but labels "
            f"describe {args.hic_target_window_bp} bp"
        )
    if args.hic_target_matrix_size != HIC_MATRIX_SIZE:
        raise ValueError(
            f"Enformer Hi-C model predicts {HIC_MATRIX_SIZE}x{HIC_MATRIX_SIZE}, "
            f"but labels describe {args.hic_target_matrix_size}x"
            f"{args.hic_target_matrix_size}"
        )
    return LiveEnformerGenomeDataset(
        os.path.join(args.data_root, args.celltype),
        args.assembly,
        {},
        mode=mode,
        include_sequence=True,
        include_genomic_features=False,
        use_aug=False,
        invalid_regions_bed=_invalid_regions_path(args),
        max_invalid_bin_fraction=args.max_invalid_bin_fraction,
        hic_matrix_dir=args.hic_matrix_dir,
        hic_bin_size=args.hic_bin_size,
        hic_target_window_bp=args.hic_target_window_bp,
        hic_target_matrix_size=args.hic_target_matrix_size,
        hic_requires_resize=args.hic_requires_resize,
    )


def _loader(args: argparse.Namespace, mode: str) -> torch.utils.data.DataLoader:
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
    return torch.utils.data.DataLoader(_dataset(args, mode), **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live multi-layer RNA-conditioned Enformer -> Hi-C training"
    )
    parser.add_argument("--data-root", default=str(_WORKSPACE_ROOT / "data" / "hg38"))
    parser.add_argument("--assembly", default="hg38")
    parser.add_argument("--celltype", default="GM12878")
    parser.add_argument(
        "--enformer-root",
        default=str(_REPO_ROOT / "src"),
        help="Checkout containing succeed_backbone/",
    )
    parser.add_argument(
        "--enformer-checkpoint",
        default=(
            str(_REPO_ROOT / "weights" / "stage1.pt")
        ),
    )
    parser.add_argument("--expression-npz", default=None)
    parser.add_argument("--expression-key", default="expression")
    parser.add_argument(
        "--save-path",
        default=str(_WORKSPACE_ROOT / "results" / "GM12878_enformer_multilayer_hic_live"),
    )
    parser.add_argument("--invalid-regions-bed", default=None)
    parser.add_argument("--hic-matrix-dir", default="hic_matrix")
    parser.add_argument("--max-invalid-bin-fraction", type=float, default=0.05)
    parser.add_argument("--min-diagonal-offset", type=int, default=2)
    parser.add_argument(
        "--enformer-tile-batch-size",
        type=int,
        default=1,
        help=(
            "Number of the 19 center-valid Enformer windows evaluated per "
            "frozen forward. Use 1 first to limit peak memory."
        ),
    )
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--pool-heads", type=int, default=8)
    parser.add_argument("--native-global-layers", type=int, default=2)
    parser.add_argument("--native-global-latents", type=int, default=128)
    parser.add_argument("--native-global-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2077)
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
    args = parser.parse_args()
    if args.expression_npz is None:
        args.expression_npz = os.path.join(
            args.data_root,
            args.celltype,
            "RNA",
            "trans_regulator_expression.npz",
        )
    return args


def main() -> None:
    args = parse_args()
    attach_hic_supervision_metadata(args)
    pl.seed_everything(args.seed, workers=True)
    output = Path(args.save_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    module = LiveEnformerHiCModule(args)
    checkpoint = callbacks.ModelCheckpoint(
        dirpath=output / "models",
        monitor="val_loss",
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
                monitor="val_loss",
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
        train_dataloaders=_loader(args, "train"),
        val_dataloaders=_loader(args, "val"),
        ckpt_path=args.resume_from_checkpoint,
    )


if __name__ == "__main__":
    main()
