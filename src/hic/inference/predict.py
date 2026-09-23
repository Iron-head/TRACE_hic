"""Predict one label-free Hi-C window from hg38 DNA and an RNA context."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from hic.model.enformer_hic_standalone import (
    ENFORMER_CONTEXT_BP,
    ENFORMER_FLANK_BP,
    HIC_MATRIX_SIZE,
    HIC_WINDOW_BP,
)
from hic.model.project_sparse_transcriptome_enformer_hic_standalone import (
    FrozenProjectSparseTranscriptomeEnformerExtractor,
)
from hic.model.sparse_transcriptome_enformer_hic_standalone import (
    SparseTranscriptomeTwoMegabaseEncoder,
)
from hic.model.sparse_transcriptome_model_contract import (
    CellTranscriptomeContext,
    SparseTranscriptomeModelContract,
)
from hic.model.unit_interval_enformer_hic import UnitIntervalEnformerHiCModel
from hic.training.train_project_sparse_transcriptome_enformer_hic_balanced_clip_minmax_multicell import (
    BALANCED_CLIP_MINMAX_MULTICELL_CHECKPOINT_SCHEMA,
)
from hic.training.train_project_sparse_transcriptome_enformer_hic_multicell import (
    PROJECT_MULTICELL_CHECKPOINT_SCHEMA,
)

ROOT = Path(__file__).resolve().parents[3]
DNA_LUT = np.zeros((256, 4), dtype=np.uint8)
for base, column in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    DNA_LUT[ord(base), column] = 1
    DNA_LUT[ord(base.lower()), column] = 1


def sequence_context(fasta: Path, start: int) -> np.ndarray:
    """Use the exact padding and alignment of the training dataset."""
    if start < 0:
        raise ValueError("Window start must be nonnegative")
    opener = gzip.open if fasta.suffix == ".gz" else open
    with opener(fasta, "rt", encoding="ascii") as handle:
        header = handle.readline()
        if not header.startswith(">"):
            raise ValueError(f"Not a FASTA file: {fasta}")
        sequence = "".join(line.strip() for line in handle)
    if start + HIC_WINDOW_BP > len(sequence):
        raise ValueError(
            f"Window [{start}, {start + HIC_WINDOW_BP}) exceeds FASTA length {len(sequence)}"
        )
    context_start = start - ENFORMER_FLANK_BP
    context_end = context_start + ENFORMER_CONTEXT_BP
    clipped_start = max(0, context_start)
    clipped_end = min(len(sequence), context_end)
    encoded = np.zeros((ENFORMER_CONTEXT_BP, 4), dtype=np.uint8)
    bases = np.frombuffer(
        sequence[clipped_start:clipped_end].encode("ascii"), dtype=np.uint8
    )
    encoded[clipped_start - context_start : clipped_end - context_start] = DNA_LUT[bases]
    return encoded


def load_checkpoint(path: Path) -> dict:
    # Only load checkpoints from a source you trust: torch checkpoints may use pickle.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("balanced_clip_minmax_multicell_checkpoint_schema") != BALANCED_CLIP_MINMAX_MULTICELL_CHECKPOINT_SCHEMA:
        raise ValueError("Expected the balanced, clipped multi-cell Hi-C checkpoint")
    if payload.get("multicell_checkpoint_schema") != PROJECT_MULTICELL_CHECKPOINT_SCHEMA:
        raise ValueError("Unexpected multi-cell checkpoint schema")
    if not payload.get("hyper_parameters"):
        raise ValueError("Checkpoint has no model hyperparameters")
    return payload


def build_model(
    payload: dict,
    stage1: Path,
    contract: Path,
    enformer_root: Path,
    tile_batch_size: int,
    device: torch.device,
):
    extractor = FrozenProjectSparseTranscriptomeEnformerExtractor(
        enformer_root, stage1, contract, freeze=True
    )
    if payload.get("model_contract_sha256") != extractor.model_contract_sha256:
        raise ValueError("Stage-I model contract hash differs from Stage-II checkpoint")
    if payload.get("external_encoder_global_step") != extractor.checkpoint_global_step:
        raise ValueError("Stage-I checkpoint step differs from Stage-II provenance")
    h = payload["hyper_parameters"]
    encoder = SparseTranscriptomeTwoMegabaseEncoder(
        extractor, tile_batch_size=tile_batch_size
    )
    head = UnitIntervalEnformerHiCModel(
        encoder_hidden=extractor.output_hidden,
        encoder_token_bp=extractor.token_bp,
        hidden=h["hidden"],
        projection_dropout=h["dropout"],
        pool_heads=h["pool_heads"],
        native_global_layers=h["native_global_layers"],
        native_global_latents=h["native_global_latents"],
        native_global_heads=h["native_global_heads"],
        native_global_dropout=h["dropout"],
    )
    prefix = "model.hic_model."
    state = {
        key[len(prefix) :]: value
        for key, value in payload["state_dict"].items()
        if key.startswith(prefix)
    }
    if not state:
        raise ValueError("Stage-II checkpoint has no Hi-C head weights")
    head.load_state_dict(state, strict=True)
    return encoder.to(device).eval(), head.to(device).eval(), extractor


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "weights/stage2.ckpt")
    parser.add_argument("--stage1", type=Path, default=ROOT / "weights/stage1.pt")
    parser.add_argument("--model-contract", type=Path, default=ROOT / "assets/model_contract/model_contract.json")
    parser.add_argument("--context-npz", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path, help="One chromosome FASTA, optionally gzip compressed")
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--start", required=True, type=int, help="Zero-based start of a 2,097,152-bp window")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--tile-batch-size", type=int, default=2)
    parser.add_argument("--enformer-root", type=Path, default=ROOT / "src")
    return parser.parse_args(argv)


def predict(args):
    if args.tile_batch_size < 1:
        raise ValueError("--tile-batch-size must be positive")
    device = torch.device(args.device)
    if args.precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 prediction requires a CUDA device; use --precision float32")
    contract = SparseTranscriptomeModelContract(args.model_contract)
    context = CellTranscriptomeContext(args.context_npz, contract)
    payload = load_checkpoint(args.checkpoint)
    sequence = sequence_context(args.fasta, args.start)
    encoder, head, extractor = build_model(
        payload, args.stage1, args.model_contract, args.enformer_root,
        args.tile_batch_size, device,
    )
    states, _ = extractor.encode_cell_contexts(
        (args.context_npz,), expected_celltypes=(context.celltype,)
    )
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=args.precision == "bf16",
    ):
        tokens = encoder(torch.from_numpy(sequence[None]).to(device), states.to(device).float())
        prediction = head(tokens).float().cpu().numpy()[0]
    if prediction.shape != (HIC_MATRIX_SIZE, HIC_MATRIX_SIZE):
        raise RuntimeError(f"Unexpected prediction shape: {prediction.shape}")
    if not np.isfinite(prediction).all():
        raise RuntimeError("Prediction contains nonfinite values")
    metadata = {
        "schema": "multicell_rna_hic_prediction/1",
        "celltype": context.celltype,
        "chromosome": args.chromosome,
        "start": args.start,
        "end": args.start + HIC_WINDOW_BP,
        "assembly": "hg38",
        "shape": list(prediction.shape),
        "value_domain": "window_minmax_normalized_0_1",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "rna_context_sha256": context.sha256,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, prediction=prediction, metadata_json=json.dumps(metadata))
    return metadata


def main(argv=None):
    metadata = predict(parse_args(argv))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
