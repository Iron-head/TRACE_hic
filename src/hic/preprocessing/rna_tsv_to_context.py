#!/usr/bin/env python3
"""Convert one cell type's RNA-seq TPM TSV file(s) into one context NPZ."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKSPACE_ROOT = _REPO_ROOT
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from hic.model.sparse_transcriptome_model_contract import (
    CELL_CONTEXT_SCHEMA,
    CellTranscriptomeContext,
    SparseTranscriptomeModelContract,
    sha256_file,
)
from hic.preprocessing.transcriptome_context_utils import (
    parse_quantification_file,
    single_context_features,
)


def build_context_npz(
    *,
    rna_tsv_paths: Sequence[str | Path],
    celltype: str,
    rna_protocol: str,
    model_contract_path: str | Path,
    output_path: str | Path,
    feature_dtype: str = "float16",
    overwrite: bool = False,
    source_display_paths: Sequence[str] | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> CellTranscriptomeContext:
    """Build and validate one self-contained context NPZ."""

    name = str(celltype).strip()
    if not name:
        raise ValueError("celltype must be non-empty")
    if feature_dtype not in {"float16", "float32"}:
        raise ValueError(f"Unsupported feature dtype: {feature_dtype}")
    contract = SparseTranscriptomeModelContract(model_contract_path)
    if rna_protocol not in contract.protocol_to_id:
        raise ValueError(
            f"Unsupported RNA protocol {rna_protocol!r}; expected one of "
            f"{list(contract.protocol_to_id)}"
        )
    sources = tuple(Path(path).expanduser().resolve() for path in rna_tsv_paths)
    if not sources:
        raise ValueError("At least one --rna-tsv file is required")
    if len(set(sources)) != len(sources):
        raise ValueError("RNA TSV paths must be unique")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    if source_display_paths is None:
        display_paths = tuple(str(path) for path in sources)
    else:
        display_paths = tuple(str(value) for value in source_display_paths)
        if len(display_paths) != len(sources):
            raise ValueError("source_display_paths must match rna_tsv_paths")

    genes = tuple(str(value) for value in contract.gene_ids.tolist())
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    sum_tpm = np.zeros(contract.gene_count, dtype=np.float64)
    observed_count = np.zeros(contract.gene_count, dtype=np.uint16)
    source_records: list[dict[str, object]] = []
    for source, display_path in zip(sources, display_paths):
        values, present, counters = parse_quantification_file(
            source,
            gene_to_index,
        )
        sum_tpm[present] += values[present]
        observed_count[present] += 1
        source_records.append(
            {
                "path": display_path,
                "sha256": sha256_file(source),
                **counters,
                "mapped_gene_count": int(present.sum()),
            }
        )
    gene_mask = observed_count > 0
    if not gene_mask.any():
        raise ValueError(f"No genes from {name} map to the model contract")
    mean_tpm = np.zeros(contract.gene_count, dtype=np.float32)
    mean_tpm[gene_mask] = (
        sum_tpm[gene_mask] / observed_count[gene_mask]
    ).astype(np.float32)
    min_tpm = float(contract.context_transform["min_tpm"])
    log2_tpm_cap = float(contract.context_transform["log2_tpm_cap"])
    storage_dtype = np.float16 if feature_dtype == "float16" else np.float32
    gene_features = single_context_features(
        mean_tpm,
        gene_mask,
        min_tpm=min_tpm,
        log2_tpm_cap=log2_tpm_cap,
    ).astype(storage_dtype)
    protocol_id = int(contract.protocol_to_id[rna_protocol])
    metadata: dict[str, object] = {
        "schema_version": CELL_CONTEXT_SCHEMA,
        "celltype": name,
        "model_contract_sha256": contract.sha256,
        "gene_vocabulary_sha256": contract.gene_vocabulary_sha256,
        "gene_count": contract.gene_count,
        "feature_dim": contract.feature_dim,
        "feature_names": list(contract.feature_names),
        "feature_dtype": feature_dtype,
        "selected_protocol": rna_protocol,
        "protocol_id": protocol_id,
        "selected_file_count": len(sources),
        "mapped_gene_count": int(gene_mask.sum()),
        "detected_gene_count": int((mean_tpm >= min_tpm).sum()),
        "context_transform": dict(contract.context_transform),
        "source_files": source_records,
    }
    if extra_metadata:
        protected = set(metadata)
        overlap = protected & set(extra_metadata)
        if overlap:
            raise ValueError(f"extra_metadata cannot replace fields: {sorted(overlap)}")
        metadata.update(extra_metadata)

    output = Path(output_path).expanduser().resolve()
    if output.suffix != ".npz":
        raise ValueError(f"Output must end in .npz: {output}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.npz")
    np.savez_compressed(
        temporary,
        schema_version=np.asarray(CELL_CONTEXT_SCHEMA),
        celltype=np.asarray(name),
        model_contract_sha256=np.asarray(contract.sha256),
        gene_vocabulary_sha256=np.asarray(contract.gene_vocabulary_sha256),
        feature_names=np.asarray(contract.feature_names),
        gene_features=gene_features,
        gene_mask=gene_mask,
        protocol_id=np.asarray(protocol_id, dtype=np.int64),
        protocol_name=np.asarray(rna_protocol),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    temporary.replace(output)
    context = CellTranscriptomeContext(output, contract)
    if context.celltype != name:
        raise RuntimeError("Written context failed its celltype validation")
    print(
        f"Saved {name} RNA context to {output}: protocol={rna_protocol}, "
        f"replicates={len(sources)}, mapped={int(gene_mask.sum())}, "
        f"detected={int((mean_tpm >= min_tpm).sum())}",
        flush=True,
    )
    return context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rna-tsv", nargs="+", required=True, type=Path)
    parser.add_argument("--celltype", required=True)
    parser.add_argument(
        "--rna-protocol",
        required=True,
        choices=("total_rna", "polyA_plus_rna"),
    )
    parser.add_argument(
        "--model-contract",
        type=Path,
        default=(
            _WORKSPACE_ROOT
            / "assets"
            / "model_contract"
            / "model_contract.json"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--feature-dtype", choices=("float16", "float32"), default="float16"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_context_npz(
        rna_tsv_paths=args.rna_tsv,
        celltype=args.celltype,
        rna_protocol=args.rna_protocol,
        model_contract_path=args.model_contract,
        output_path=args.output,
        feature_dtype=args.feature_dtype,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
