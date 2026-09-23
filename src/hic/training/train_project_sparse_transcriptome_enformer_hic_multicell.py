#!/usr/bin/env python3
"""Train multi-cell Hi-C from RNA contexts prepared under project ``data``.

Unlike the reference-context trainer, ``--celltypes`` contains only directory
names. RNA features and protocol IDs are looked up by those names in the
corresponding ``RNA/context.npz``, so no external context ID is required.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import callbacks


_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hic.data.multicell_live_enformer_dataset import CellTypeContextSpec
from hic.model.enformer_hic_standalone import EnformerHiCModel
from hic.model.project_sparse_transcriptome_enformer_hic_standalone import (
    FrozenProjectSparseTranscriptomeEnformerExtractor,
)
from hic.model.sparse_transcriptome_enformer_hic_standalone import (
    SPARSE_TRANSCRIPTOME_FRAMEWORK,
    SparseTranscriptomeEndToEndHiCModel,
    SparseTranscriptomeTwoMegabaseEncoder,
)
from hic.model.sparse_transcriptome_model_contract import (
    CellTranscriptomeContext,
    SparseTranscriptomeModelContract,
)
from hic.training.hic_supervision import attach_hic_supervision_metadata
from hic.training.train_sparse_transcriptome_enformer_hic_multicell import (
    LiveSparseTranscriptomeMultiCellHiCModule,
    _loader,
    _preflight_celltype_data,
)


PROJECT_MULTICELL_CHECKPOINT_SCHEMA = "project_sparse_transcriptome_hic_multicell/1"


def parse_project_celltypes(
    value: str,
    data_root: str | Path,
    contract: SparseTranscriptomeModelContract,
) -> tuple[CellTypeContextSpec, ...]:
    names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if len(names) < 2:
        raise ValueError("Multi-cell training requires at least two cell types")
    if len(set(names)) != len(names):
        raise ValueError(f"Cell type names must be unique: {names}")
    root = Path(data_root).expanduser().resolve()
    contexts = tuple(
        CellTranscriptomeContext(root / name / "RNA" / "context.npz", contract)
        for name in names
    )
    actual_names = tuple(context.celltype for context in contexts)
    if actual_names != names:
        raise ValueError(
            f"RNA context names do not match requested cell types: "
            f"{actual_names} vs {names}"
        )
    return tuple(
        CellTypeContextSpec(name=name, context_id=row)
        for row, name in enumerate(names)
    )


class LiveProjectSparseTranscriptomeMultiCellHiCModule(
    LiveSparseTranscriptomeMultiCellHiCModule
):
    """Shared Hi-C head conditioned on project-local RNA measurements."""

    def __init__(
        self,
        args: argparse.Namespace,
        celltype_specs: tuple[CellTypeContextSpec, ...],
    ) -> None:
        # Do not call the parent initializer: it interprets context IDs as rows
        # in the legacy pretraining sidecar. All training/validation methods are
        # inherited after constructing the equivalent model and state buffer.
        pl.LightningModule.__init__(self)
        self.save_hyperparameters(vars(args))
        extractor = FrozenProjectSparseTranscriptomeEnformerExtractor(
            args.enformer_root,
            args.enformer_checkpoint,
            args.model_contract,
            freeze=True,
        )
        celltype_names = tuple(spec.name for spec in celltype_specs)
        context_paths = tuple(
            Path(args.data_root).expanduser().resolve()
            / name
            / "RNA"
            / "context.npz"
            for name in celltype_names
        )
        context_states, contexts = extractor.encode_cell_contexts(
            context_paths,
            expected_celltypes=celltype_names,
        )
        cell_indices = tuple(range(len(celltype_names)))
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
        self.register_buffer("context_state_bank", context_states, persistent=True)
        self.celltype_names = celltype_names
        # Retain inherited checkpoint field names, but values are local state
        # indices rather than external biological context IDs.
        self.celltype_context_ids = cell_indices
        self.celltype_context_rows = cell_indices
        self.model_contract_sha256 = extractor.model_contract_sha256
        self.cell_context_sha256 = tuple(context.sha256 for context in contexts)
        # Compatibility field used by the inherited checkpoint hook.
        self.transcriptome_manifest_sha256 = extractor.model_contract_sha256
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
            "Using project-local sparse transcriptomes: "
            + ", ".join(
                f"{name}=cell_index:{index}"
                for name, index in zip(celltype_names, cell_indices)
            )
            + f", state_bank={tuple(context_states.shape)}, "
            f"external_step={extractor.checkpoint_global_step}, "
            f"model_contract_sha256={self.model_contract_sha256}",
            flush=True,
        )

    def on_save_checkpoint(self, checkpoint) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint["multicell_checkpoint_schema"] = (
            PROJECT_MULTICELL_CHECKPOINT_SCHEMA
        )
        checkpoint["model_contract"] = self.hparams["model_contract"]
        checkpoint["model_contract_sha256"] = self.model_contract_sha256
        checkpoint["cell_context_sha256"] = list(self.cell_context_sha256)
        checkpoint.pop("transcriptome_manifest", None)
        checkpoint.pop("transcriptome_manifest_sha256", None)

    def on_load_checkpoint(self, checkpoint) -> None:
        schema = checkpoint.get("multicell_checkpoint_schema")
        if schema is not None and schema != PROJECT_MULTICELL_CHECKPOINT_SCHEMA:
            raise ValueError(f"Unsupported project multi-cell schema: {schema!r}")
        saved_framework = checkpoint.get("external_encoder_framework")
        if saved_framework is not None and saved_framework != self.external_encoder_framework:
            raise ValueError("External sparse Enformer framework does not match")
        expected_hashes = {"model_contract_sha256": self.model_contract_sha256}
        for key, expected in expected_hashes.items():
            saved = checkpoint.get(key)
            if saved is not None and saved != expected:
                raise ValueError(f"Checkpoint {key} does not match current input")
        expected_names = list(self.celltype_names)
        saved_names = checkpoint.get("celltype_names")
        if saved_names is not None and list(saved_names) != expected_names:
            raise ValueError(
                f"Checkpoint cell types do not match: {saved_names} vs {expected_names}"
            )
        expected_rows = list(self.celltype_context_rows)
        saved_rows = checkpoint.get("celltype_context_rows")
        if saved_rows is not None and list(saved_rows) != expected_rows:
            raise ValueError(
                f"Checkpoint cell indices do not match: {saved_rows} vs {expected_rows}"
            )
        saved_context_hashes = checkpoint.get("cell_context_sha256")
        expected_context_hashes = list(self.cell_context_sha256)
        if saved_context_hashes is not None and (
            list(saved_context_hashes) != expected_context_hashes
        ):
            raise ValueError("Checkpoint per-cell RNA contexts do not match")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project-RNA sparse-transcriptome Enformer -> multi-cell Hi-C"
    )
    parser.add_argument(
        "--data-root", default=str(_WORKSPACE_ROOT / "data" / "hg38")
    )
    parser.add_argument("--assembly", default="hg38")
    parser.add_argument(
        "--celltypes",
        default="GM12878,IMR90",
        help="Comma-separated project cell-type directory names (no context IDs).",
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
        "--save-path",
        default=str(
            _WORKSPACE_ROOT
            / "results"
            / "GM12878_IMR90_project_sparse_transcriptome_rna_hic_live"
        ),
    )
    parser.add_argument("--centrotelo-bed", default=None)
    parser.add_argument("--invalid-regions-bed", default=None)
    parser.add_argument("--hic-matrix-dir", default="hic_matrix")
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
    contract = SparseTranscriptomeModelContract(args.model_contract)
    specs = parse_project_celltypes(args.celltypes, args.data_root, contract)
    _preflight_celltype_data(args, specs)
    args.celltype = "+".join(spec.name for spec in specs)
    # Compatibility alias used only while the inherited hook removes the
    # external frozen encoder state; it is removed from the saved checkpoint.
    args.transcriptome_manifest = args.model_contract
    attach_hic_supervision_metadata(args, celltype=args.celltype)
    pl.seed_everything(args.seed, workers=True)
    output = Path(args.save_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    module = LiveProjectSparseTranscriptomeMultiCellHiCModule(args, specs)
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
