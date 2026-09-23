#!/usr/bin/env python3
"""Dataset and GPU context bank for full-transcriptome sparse routing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .assay_collapsed_dataset import CorgiEnformerAssayDataset


FULL_TRANSCRIPTOME_CONTEXT_SCHEMA = "full_transcriptome_context_bank/1"


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FullTranscriptomeContextBatch:
    gene_features: torch.Tensor
    gene_mask: torch.Tensor
    protocol_ids: torch.Tensor


class TorchFullTranscriptomeContextBank:
    """Immutable full-transcriptome bank copied once to each training device."""

    def __init__(
        self,
        *,
        gene_features: torch.Tensor,
        gene_mask: torch.Tensor,
        protocol_ids: torch.Tensor,
    ) -> None:
        self.gene_features = gene_features
        self.gene_mask = gene_mask
        self.protocol_ids = protocol_ids

    @property
    def device(self) -> torch.device:
        return self.gene_features.device

    @property
    def context_count(self) -> int:
        return int(self.gene_features.shape[0])

    @property
    def gene_count(self) -> int:
        return int(self.gene_features.shape[1])

    def gather(self, context_rows: torch.Tensor) -> FullTranscriptomeContextBatch:
        rows = context_rows.to(device=self.device, dtype=torch.long)
        return FullTranscriptomeContextBatch(
            gene_features=self.gene_features.index_select(0, rows),
            gene_mask=self.gene_mask.index_select(0, rows),
            protocol_ids=self.protocol_ids.index_select(0, rows),
        )

    def all(self) -> FullTranscriptomeContextBatch:
        return FullTranscriptomeContextBatch(
            gene_features=self.gene_features,
            gene_mask=self.gene_mask,
            protocol_ids=self.protocol_ids,
        )


class FullTranscriptomeContextBank:
    """Validated mmap sidecar produced by prepare_full_transcriptome_context."""

    def __init__(self, manifest_path: str | Path) -> None:
        path = Path(manifest_path).expanduser().resolve()
        if path.is_dir():
            path = path / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Full-transcriptome manifest not found: {path}")
        self.manifest_path = path
        self.manifest_dir = path.parent
        self.manifest: Mapping[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        schema = str(self.manifest.get("schema_version", ""))
        if schema != FULL_TRANSCRIPTOME_CONTEXT_SCHEMA:
            raise ValueError(
                f"Expected schema {FULL_TRANSCRIPTOME_CONTEXT_SCHEMA!r}, got {schema!r}"
            )
        arrays = self.manifest.get("arrays")
        if not isinstance(arrays, Mapping):
            raise ValueError("Full-transcriptome manifest has no arrays mapping")

        def load(name: str) -> np.ndarray:
            if name not in arrays:
                raise ValueError(f"Manifest is missing arrays.{name}")
            return np.load(
                _resolve(self.manifest_dir, str(arrays[name])),
                mmap_mode="r",
                allow_pickle=False,
            )

        self.context_ids = load("context_ids")
        self.gene_features = load("gene_features")
        self.gene_mask = load("gene_mask")
        self.protocol_ids = load("protocol_ids")
        self.feature_names = tuple(str(value) for value in self.manifest["feature_names"])
        self.gene_count = int(self.manifest["gene_count"])
        self.protocol_count = len(self.manifest["protocol_to_id"])
        self.manifest_sha256 = _sha256(self.manifest_path)
        self._validate()

    @property
    def context_count(self) -> int:
        return int(self.context_ids.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.gene_features.shape[-1])

    def _validate(self) -> None:
        context_count = int(self.manifest["context_count"])
        feature_dim = int(self.manifest["feature_dim"])
        if self.context_ids.shape != (context_count,):
            raise ValueError(f"Invalid context_ids shape: {self.context_ids.shape}")
        if len(np.unique(np.asarray(self.context_ids))) != context_count:
            raise ValueError("Context IDs contain duplicates")
        if self.gene_features.shape != (context_count, self.gene_count, feature_dim):
            raise ValueError(f"Invalid gene_features shape: {self.gene_features.shape}")
        if self.gene_mask.shape != (context_count, self.gene_count):
            raise ValueError(f"Invalid gene_mask shape: {self.gene_mask.shape}")
        if self.protocol_ids.shape != (context_count,):
            raise ValueError(f"Invalid protocol_ids shape: {self.protocol_ids.shape}")
        if feature_dim != len(self.feature_names):
            raise ValueError("Feature names do not match feature dimension")
        mask = np.asarray(self.gene_mask, dtype=np.bool_)
        if not np.all(mask.any(axis=1)):
            raise ValueError("Every context must contain at least one mapped gene")
        for start in range(0, context_count, 8):
            features = np.asarray(self.gene_features[start : start + 8])
            if not np.isfinite(features).all():
                raise ValueError("Gene features contain non-finite values")
            if features.min(initial=0.0) < 0.0 or features.max(initial=0.0) > 1.0:
                raise ValueError("Single-sample gene features must be in [0,1]")
        protocols = np.asarray(self.protocol_ids)
        if np.any(protocols < 0) or np.any(protocols >= self.protocol_count):
            raise ValueError("Protocol IDs are outside the manifest vocabulary")

    def validate_context_ids(self, expected: Sequence[int] | np.ndarray) -> None:
        expected_array = np.asarray(expected, dtype=np.int64)
        if not np.array_equal(np.asarray(self.context_ids, dtype=np.int64), expected_array):
            raise ValueError(
                "Full-transcriptome context IDs/order do not match the Enformer Dataset"
            )

    def to(
        self,
        device: torch.device | str,
        *,
        feature_dtype: torch.dtype = torch.float32,
    ) -> TorchFullTranscriptomeContextBank:
        target = torch.device(device)
        return TorchFullTranscriptomeContextBank(
            gene_features=torch.as_tensor(
                np.array(self.gene_features, dtype=np.float32, copy=True),
                device=target,
                dtype=feature_dtype,
            ),
            gene_mask=torch.as_tensor(
                np.array(self.gene_mask, dtype=np.bool_, copy=True),
                device=target,
                dtype=torch.bool,
            ),
            protocol_ids=torch.as_tensor(
                np.array(self.protocol_ids, dtype=np.int64, copy=True),
                device=target,
                dtype=torch.long,
            ),
        )


class FullTranscriptomeCorgiEnformerAssayDataset(CorgiEnformerAssayDataset):
    """Reuse packed DNA/targets and retain only the full-context row index."""

    def __init__(
        self,
        *args: Any,
        transcriptome_manifest: str | Path,
        return_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        self._return_full_transcriptome_metadata = bool(return_metadata)
        super().__init__(*args, return_metadata=True, **kwargs)
        self.transcriptome_bank = FullTranscriptomeContextBank(
            transcriptome_manifest
        )
        self.transcriptome_bank.validate_context_ids(self.context_ids_all)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = super().__getitem__(index)
        sample.pop("context_vector", None)
        if not self._return_full_transcriptome_metadata:
            for key in (
                "context_id",
                "region_index",
                "target_count",
                "chrom",
                "start",
                "end",
            ):
                sample.pop(key, None)
        return sample


__all__ = [
    "FULL_TRANSCRIPTOME_CONTEXT_SCHEMA",
    "FullTranscriptomeContextBatch",
    "FullTranscriptomeContextBank",
    "TorchFullTranscriptomeContextBank",
    "FullTranscriptomeCorgiEnformerAssayDataset",
]
