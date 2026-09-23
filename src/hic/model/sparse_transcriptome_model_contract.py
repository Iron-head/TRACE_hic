"""Local input contract for the pretrained sparse-transcriptome encoder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


MODEL_CONTRACT_SCHEMA = "succeed_sparse_transcriptome_model_contract/1"
CELL_CONTEXT_SCHEMA = "succeed_cell_transcriptome_context/1"
EXPECTED_FEATURE_NAMES = (
    "scaled_log2_tpm",
    "within_sample_percentile",
    "detected",
)
EXPECTED_PROTOCOLS = {"total_rna": 0, "polyA_plus_rna": 1}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SparseTranscriptomeModelContract:
    """Validated gene vocabulary and feature schema carried by the model."""

    def __init__(self, path: str | Path) -> None:
        contract_path = Path(path).expanduser().resolve()
        if contract_path.is_dir():
            contract_path = contract_path / "model_contract.json"
        if not contract_path.is_file():
            raise FileNotFoundError(f"Model contract not found: {contract_path}")
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != MODEL_CONTRACT_SCHEMA:
            raise ValueError(
                f"Expected contract schema {MODEL_CONTRACT_SCHEMA!r}, got "
                f"{payload.get('schema_version')!r}"
            )
        gene_file = str(payload.get("gene_vocabulary", {}).get("file", ""))
        expected_hash = str(
            payload.get("gene_vocabulary", {}).get("sha256", "")
        )
        if not gene_file or not expected_hash:
            raise ValueError("Model contract has no gene vocabulary file/hash")
        gene_path = (contract_path.parent / gene_file).resolve()
        actual_hash = sha256_file(gene_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Gene vocabulary SHA256 mismatch: {actual_hash} vs {expected_hash}"
            )
        gene_ids = np.load(gene_path, mmap_mode="r", allow_pickle=False)
        gene_count = int(payload.get("gene_count", -1))
        if gene_ids.shape != (gene_count,):
            raise ValueError(
                f"Gene vocabulary shape {gene_ids.shape} != ({gene_count},)"
            )
        genes = [str(value) for value in gene_ids.tolist()]
        if any(not value.startswith("ENSG") for value in genes):
            raise ValueError("Model contract contains a non-ENSG gene ID")
        if len(set(genes)) != len(genes):
            raise ValueError("Model contract gene IDs are not unique")
        feature_names = tuple(str(x) for x in payload.get("feature_names", ()))
        protocols = {
            str(key): int(value)
            for key, value in payload.get("protocol_to_id", {}).items()
        }
        if feature_names != EXPECTED_FEATURE_NAMES:
            raise ValueError(f"Unsupported feature names: {feature_names}")
        if protocols != EXPECTED_PROTOCOLS:
            raise ValueError(f"Unsupported protocol mapping: {protocols}")
        if int(payload.get("feature_dim", -1)) != len(feature_names):
            raise ValueError("Contract feature_dim does not match feature_names")
        transform = payload.get("context_transform")
        if not isinstance(transform, Mapping):
            raise ValueError("Model contract has no context_transform")
        if float(transform.get("min_tpm", -1)) < 0:
            raise ValueError("Model contract has invalid min_tpm")
        if float(transform.get("log2_tpm_cap", 0)) <= 0:
            raise ValueError("Model contract has invalid log2_tpm_cap")
        self.path = contract_path
        self.payload: Mapping[str, Any] = payload
        self.gene_ids = gene_ids
        self.gene_count = gene_count
        self.feature_names = feature_names
        self.feature_dim = len(feature_names)
        self.protocol_to_id = protocols
        self.context_transform = transform
        self.sha256 = sha256_file(contract_path)
        self.gene_vocabulary_sha256 = expected_hash


class CellTranscriptomeContext:
    """One self-contained ``context.npz`` stored beside raw RNA files."""

    def __init__(
        self,
        path: str | Path,
        contract: SparseTranscriptomeModelContract,
    ) -> None:
        array_path = Path(path).expanduser().resolve()
        if array_path.is_dir():
            array_path = array_path / "context.npz"
        if not array_path.is_file():
            raise FileNotFoundError(f"Cell RNA context not found: {array_path}")
        with np.load(array_path, allow_pickle=False) as arrays:
            required = {
                "schema_version",
                "celltype",
                "model_contract_sha256",
                "gene_vocabulary_sha256",
                "feature_names",
                "gene_features",
                "gene_mask",
                "protocol_id",
                "protocol_name",
                "metadata_json",
            }
            missing = required - set(arrays.files)
            if missing:
                raise ValueError(f"Cell context is missing arrays: {sorted(missing)}")
            schema_version = str(np.asarray(arrays["schema_version"]).item())
            celltype = str(np.asarray(arrays["celltype"]).item())
            model_contract_sha256 = str(
                np.asarray(arrays["model_contract_sha256"]).item()
            )
            gene_vocabulary_sha256 = str(
                np.asarray(arrays["gene_vocabulary_sha256"]).item()
            )
            feature_names = tuple(
                str(value) for value in np.asarray(arrays["feature_names"]).tolist()
            )
            gene_features = np.asarray(arrays["gene_features"])
            gene_mask = np.asarray(arrays["gene_mask"], dtype=np.bool_)
            protocol_id = int(np.asarray(arrays["protocol_id"]).item())
            protocol_name = str(np.asarray(arrays["protocol_name"]).item())
            metadata_json = str(np.asarray(arrays["metadata_json"]).item())
        if schema_version != CELL_CONTEXT_SCHEMA:
            raise ValueError(
                f"Expected cell context schema {CELL_CONTEXT_SCHEMA!r}, got "
                f"{schema_version!r}"
            )
        if model_contract_sha256 != contract.sha256:
            raise ValueError(
                "Cell RNA context was built for a different model contract: "
                f"{model_contract_sha256} vs {contract.sha256}"
            )
        if gene_vocabulary_sha256 != contract.gene_vocabulary_sha256:
            raise ValueError("Cell RNA context uses a different gene vocabulary")
        if feature_names != contract.feature_names:
            raise ValueError(
                f"Cell RNA feature names differ from model contract: "
                f"{feature_names} vs {contract.feature_names}"
            )
        expected_features = (contract.gene_count, contract.feature_dim)
        if gene_features.shape != expected_features:
            raise ValueError(
                f"Cell feature shape {gene_features.shape} != {expected_features}"
            )
        if gene_mask.shape != (contract.gene_count,):
            raise ValueError(f"Cell mask has invalid shape: {gene_mask.shape}")
        if not gene_mask.any():
            raise ValueError("Cell RNA context has no mapped genes")
        if not np.isfinite(gene_features).all():
            raise ValueError("Cell RNA context contains non-finite features")
        if gene_features.min(initial=0.0) < 0.0 or (
            gene_features.max(initial=0.0) > 1.0
        ):
            raise ValueError("Cell RNA features must be in [0, 1]")
        if protocol_id not in contract.protocol_to_id.values():
            raise ValueError(f"Cell context has invalid protocol_id={protocol_id}")
        if contract.protocol_to_id.get(protocol_name) != protocol_id:
            raise ValueError(
                f"Cell RNA protocol name/ID disagree: {protocol_name!r}/{protocol_id}"
            )
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as error:
            raise ValueError("Cell context metadata_json is invalid") from error
        if not isinstance(metadata, Mapping):
            raise ValueError("Cell context metadata_json must contain an object")
        expected_metadata = {
            "schema_version": schema_version,
            "celltype": celltype,
            "model_contract_sha256": model_contract_sha256,
            "gene_vocabulary_sha256": gene_vocabulary_sha256,
            "protocol_id": protocol_id,
            "selected_protocol": protocol_name,
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Cell context metadata {key} disagrees with NPZ field: "
                    f"{metadata.get(key)!r} vs {expected!r}"
                )
        if not celltype:
            raise ValueError("Cell RNA context has an empty celltype")
        self.path = array_path
        self.metadata: Mapping[str, Any] = metadata
        self.celltype = celltype
        self.gene_features = gene_features
        self.gene_mask = gene_mask
        self.protocol_id = protocol_id
        self.protocol_name = protocol_name
        self.array_path = array_path
        self.sha256 = sha256_file(array_path)


__all__ = [
    "CELL_CONTEXT_SCHEMA",
    "EXPECTED_FEATURE_NAMES",
    "EXPECTED_PROTOCOLS",
    "MODEL_CONTRACT_SCHEMA",
    "CellTranscriptomeContext",
    "SparseTranscriptomeModelContract",
    "sha256_file",
]
