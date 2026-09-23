#!/usr/bin/env python3
"""Lazy, context-conditioned access to the local Enformer H5 dataset.

The Dataset treats one training example as a pair of:

* one genomic interval from an Enformer split, and
* one Corgi biological context.

The H5 target tensor still contains all 5,313 Enformer human channels.  The
``target_mask`` identifies the channels that were paired to the selected
context through ``target_context_map.tsv``.  A future context-conditioned
Enformer can therefore keep its original human output head and apply the
mask in its loss.

No existing Enformer project files are imported or modified here.  H5 files
are opened lazily per worker process, which avoids sharing an HDF5 handle
across DataLoader workers.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_MANIFEST_NAME = "dataset_manifest.json"
SUPPORTED_OUTPUT_FORMATS = {"named", "enformer"}


def _as_path(value: str | os.PathLike[str]) -> Path:
    return Path(value).expanduser()


def _resolve_manifest_path(manifest_path: str | os.PathLike[str]) -> Path:
    path = _as_path(manifest_path)
    if path.is_dir():
        path = path / DEFAULT_MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {path}")
    return path.resolve()


def _resolve_relative(base_dir: Path, value: str | os.PathLike[str]) -> Path:
    path = _as_path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _read_bed_rows(path: Path) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    """Read the BED-like coordinate table and verify H5 row ordering."""

    chroms: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"BED file has no header: {path}")
        required = {"chrom", "start", "end"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"BED file {path} is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader):
            try:
                start = int(row["start"])
                end = int(row["end"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid interval at {path}:{row_number + 2}") from exc
            if start < 0 or end <= start:
                raise ValueError(f"Invalid interval {row['chrom']}:{start}-{end} at {path}:{row_number + 2}")
            if "h5_idx" in row and row["h5_idx"] not in {"", None}:
                try:
                    h5_idx = int(row["h5_idx"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid h5_idx at {path}:{row_number + 2}") from exc
                if h5_idx != row_number:
                    raise ValueError(
                        f"BED/H5 row order mismatch at {path}:{row_number + 2}: "
                        f"h5_idx={h5_idx}, expected={row_number}"
                    )
            chroms.append(str(row["chrom"]))
            starts.append(start)
            ends.append(end)

    return tuple(chroms), np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def _mix_uint64(value: int) -> int:
    """A deterministic 64-bit mixer used for bounded random pair sampling."""

    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (value ^ (value >> 31)) & 0xFFFFFFFFFFFFFFFF


class CorgiEnformerH5Dataset(Dataset):
    """A lazy Dataset over genomic intervals crossed with Corgi contexts.

    Parameters
    ----------
    manifest_path:
        Path to the generated ``dataset_manifest.json`` or its containing
        directory.
    split:
        One of the splits present in the manifest, normally ``train``,
        ``valid`` or ``test``.
    context_ids:
        Optional subset of Corgi ``context_id`` values.  The order supplied
        here becomes the context order in the Cartesian product.
    expression:
        Expression matrix key from the manifest.  ``preferred_corgi`` is
        the default matrix and follows the public Corgi normalization path;
        ``preferred_raw`` is available for ablations.
    epoch_size:
        Optional number of examples exposed by this Dataset.  If smaller
        than the full Cartesian product, deterministic seeded sampling is
        used.  This is useful because the full train product is millions of
        region-context pairs.
    seed:
        Seed for bounded pair sampling.
    output_format:
        ``named`` returns ``sequence``, ``context_vector``, ``targets`` and
        ``target_mask``.  ``enformer`` returns ``x``, ``context_vector``,
        ``labels`` and ``target_mask`` for a future Enformer-style forward.
    return_metadata:
        Add context IDs and genomic coordinates.  Keep this false for a
        minimal training batch and enable it for inspection/debugging.
    cache_last_region:
        Cache the last H5 region per worker.  Cartesian sampling visits all
        contexts for a region consecutively, so this substantially reduces
        repeated compressed-H5 reads.
    contexts_per_region:
        For bounded epochs, emit this many consecutive contexts for one
        randomly sampled genomic region.  Keep DataLoader shuffling disabled
        so a batch worker can reuse the cached sequence and target tensors.
    """

    def __init__(
        self,
        manifest_path: str | os.PathLike[str],
        split: str = "train",
        context_ids: Sequence[int] | None = None,
        expression: str | None = None,
        epoch_size: int | None = None,
        seed: int = 0,
        output_format: str = "named",
        return_metadata: bool = False,
        cache_last_region: bool = True,
        contexts_per_region: int = 1,
    ) -> None:
        if output_format not in SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Unsupported output_format={output_format!r}; "
                f"choose one of {sorted(SUPPORTED_OUTPUT_FORMATS)}"
            )

        self.manifest_path = _resolve_manifest_path(manifest_path)
        self.manifest_dir = self.manifest_path.parent
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            self.manifest: Mapping[str, Any] = json.load(handle)

        schema_version = str(self.manifest.get("schema_version", ""))
        if not schema_version.startswith("corgi_enformer_dataset/"):
            raise ValueError(f"Unsupported dataset manifest schema: {schema_version!r}")

        split_info = self.manifest.get("splits", {}).get(split)
        if not isinstance(split_info, Mapping):
            available = sorted(self.manifest.get("splits", {}).keys())
            raise ValueError(f"Unknown split={split!r}; available splits: {available}")
        self.split = split
        self.split_info = split_info
        self.h5_path = _resolve_relative(self.manifest_dir, str(split_info["h5_path"]))
        self.bed_path = _resolve_relative(self.manifest_dir, str(split_info["bed_path"]))
        storage = split_info.get("storage")
        external_storage_formats = {"sharded_npy", "context_packed_npy"}
        uses_external_storage = (
            isinstance(storage, Mapping)
            and storage.get("format") in external_storage_formats
        )
        if not uses_external_storage and not self.h5_path.is_file():
            raise FileNotFoundError(f"H5 file not found for split {split}: {self.h5_path}")
        if not self.bed_path.is_file():
            raise FileNotFoundError(f"BED file not found for split {split}: {self.bed_path}")

        contexts_info = self.manifest["contexts"]
        self.context_ids_all = np.asarray(
            np.load(_resolve_relative(self.manifest_dir, str(contexts_info["ids_path"]))),
            dtype=np.int64,
        )
        self.context_target_mask_all = np.load(
            _resolve_relative(self.manifest_dir, str(contexts_info["target_mask_path"])),
            mmap_mode="r",
        )
        self.context_target_indices_all = np.load(
            _resolve_relative(self.manifest_dir, str(contexts_info["target_indices_path"])),
            mmap_mode="r",
        )

        expression_info = self.manifest["expression"]
        expression_key = expression or str(expression_info.get("default", "preferred_corgi"))
        expression_paths = expression_info.get("matrices", {})
        if expression_key not in expression_paths:
            raise ValueError(
                f"Unknown expression={expression_key!r}; "
                f"available matrices: {sorted(expression_paths)}"
            )
        self.expression_name = expression_key
        self.expression_matrix = np.load(
            _resolve_relative(self.manifest_dir, str(expression_paths[expression_key])),
            mmap_mode="r",
        )

        self.sequence_length = int(self.manifest["shape"]["sequence_length"])
        self.target_length = int(self.manifest["shape"]["target_length"])
        self.num_target_channels = int(self.manifest["shape"]["num_target_channels"])
        self._validate_context_arrays()
        self.context_rows = self._select_context_rows(context_ids)
        self.num_contexts = int(len(self.context_rows))
        if self.num_contexts == 0:
            raise ValueError("At least one context must be selected")
        if int(contexts_per_region) <= 0:
            raise ValueError("contexts_per_region must be positive")
        if int(contexts_per_region) > self.num_contexts:
            raise ValueError(
                "contexts_per_region cannot exceed the selected context count: "
                f"{contexts_per_region} vs {self.num_contexts}"
            )
        self.contexts_per_region = int(contexts_per_region)

        self.chroms, self.starts, self.ends = _read_bed_rows(self.bed_path)
        expected_regions = int(split_info["num_regions"])
        if len(self.chroms) != expected_regions:
            raise ValueError(
                f"BED row count mismatch for {split}: {len(self.chroms)} vs manifest {expected_regions}"
            )
        self.num_regions = expected_regions

        total_pairs = self.num_regions * self.num_contexts
        if epoch_size is None:
            self.epoch_size = total_pairs
        else:
            if int(epoch_size) <= 0:
                raise ValueError("epoch_size must be positive when provided")
            self.epoch_size = int(epoch_size)
        self.total_pairs = total_pairs
        self.seed = int(seed)
        self.epoch = 0
        self.output_format = output_format
        self.return_metadata = bool(return_metadata)
        self.cache_last_region = bool(cache_last_region)

        self._h5_file: h5py.File | None = None
        self._h5_pid: int | None = None
        self._cached_region_index: int | None = None
        self._cached_sequence: np.ndarray | None = None
        self._cached_targets: np.ndarray | None = None

    def _validate_context_arrays(self) -> None:
        context_count = len(self.context_ids_all)
        if self.context_ids_all.ndim != 1:
            raise ValueError(f"Context IDs must be one-dimensional, got {self.context_ids_all.shape}")
        if len(np.unique(self.context_ids_all)) != context_count:
            raise ValueError("Context IDs contain duplicates")
        expected_targets = self.num_target_channels
        expected_genes = int(self.manifest["expression"]["gene_count"])
        if self.context_target_mask_all.shape != (context_count, expected_targets):
            raise ValueError(
                "Context target mask shape mismatch: "
                f"{self.context_target_mask_all.shape} vs {(context_count, expected_targets)}"
            )
        if self.context_target_indices_all.ndim != 2 or self.context_target_indices_all.shape[0] != context_count:
            raise ValueError(
                f"Context target index table shape mismatch: {self.context_target_indices_all.shape}"
            )
        if self.expression_matrix.ndim != 2 or self.expression_matrix.shape != (context_count, expected_genes):
            raise ValueError(
                f"Expression matrix shape mismatch: {self.expression_matrix.shape} "
                f"vs {(context_count, expected_genes)}"
            )
        if not np.isfinite(np.asarray(self.expression_matrix[: min(context_count, 2)])).all():
            raise ValueError("Expression matrix contains non-finite values")

    def _select_context_rows(self, context_ids: Sequence[int] | None) -> np.ndarray:
        if context_ids is None:
            return np.arange(len(self.context_ids_all), dtype=np.int64)
        requested = [int(value) for value in context_ids]
        if len(set(requested)) != len(requested):
            raise ValueError("context_ids contains duplicates")
        lookup = {int(context_id): row for row, context_id in enumerate(self.context_ids_all)}
        missing = [context_id for context_id in requested if context_id not in lookup]
        if missing:
            raise ValueError(f"Requested context IDs are absent from the manifest: {missing[:10]}")
        return np.asarray([lookup[context_id] for context_id in requested], dtype=np.int64)

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used by bounded deterministic pair sampling."""

        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size

    def _resolve_pair(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"Dataset index out of range: {index}")

        if self.epoch_size < self.total_pairs and self.contexts_per_region > 1:
            group_size = self.contexts_per_region
            group_index = index // group_size
            context_offset = index % group_size
            groups_per_epoch = math.ceil(self.epoch_size / group_size)
            region_mixed = _mix_uint64(
                self.seed
                + group_index
                + self.epoch * max(groups_per_epoch, 1) * 0x9E3779B1
            )
            region_index = int(region_mixed % self.num_regions)
            context_mixed = _mix_uint64(region_mixed ^ 0xD1B54A32D192ED03)
            context_start = int(context_mixed % self.num_contexts)
            context_local_index = (context_start + context_offset) % self.num_contexts
            context_row = int(self.context_rows[context_local_index])
            return region_index, context_row

        if self.epoch_size < self.total_pairs:
            mixed = _mix_uint64(
                self.seed
                + index
                + self.epoch * max(self.epoch_size, 1) * 0x9E3779B1
            )
            pair_index = int(mixed % self.total_pairs)
        else:
            # The normal case is epoch_size == total_pairs.  Cycling also
            # keeps the class well-defined if a caller intentionally asks
            # for a longer repeated epoch.
            pair_index = index % self.total_pairs

        region_index = pair_index // self.num_contexts
        context_local_index = pair_index % self.num_contexts
        context_row = int(self.context_rows[context_local_index])
        return region_index, context_row

    def _get_h5_file(self) -> h5py.File:
        pid = os.getpid()
        if self._h5_file is None or self._h5_pid != pid:
            self.close()
            self._h5_file = h5py.File(self.h5_path, "r")
            self._h5_pid = pid
            sequences_shape = tuple(self._h5_file["sequences"].shape)
            targets_shape = tuple(self._h5_file["targets"].shape)
            expected_sequence_shape = (self.num_regions, self.sequence_length, 4)
            expected_target_shape = (self.num_regions, self.target_length, self.num_target_channels)
            if sequences_shape != expected_sequence_shape:
                raise ValueError(f"Unexpected sequence shape in {self.h5_path}: {sequences_shape}")
            if targets_shape != expected_target_shape:
                raise ValueError(f"Unexpected target shape in {self.h5_path}: {targets_shape}")
        return self._h5_file

    def _load_region(self, region_index: int) -> tuple[np.ndarray, np.ndarray]:
        if (
            self.cache_last_region
            and self._cached_region_index == region_index
            and self._cached_sequence is not None
            and self._cached_targets is not None
        ):
            return self._cached_sequence, self._cached_targets

        h5_file = self._get_h5_file()
        sequence = np.ascontiguousarray(h5_file["sequences"][region_index].astype(np.float32, copy=False))
        targets = np.ascontiguousarray(h5_file["targets"][region_index].astype(np.float32, copy=False))
        if not np.isfinite(targets).all():
            raise ValueError(f"Non-finite target values at {self.split} region {region_index}")
        if self.cache_last_region:
            self._cached_region_index = region_index
            self._cached_sequence = sequence
            self._cached_targets = targets
        return sequence, targets

    def _format_sample(
        self,
        sequence: np.ndarray,
        context_vector: np.ndarray,
        targets: np.ndarray,
        target_mask: np.ndarray,
        region_index: int,
        context_row: int,
    ) -> dict[str, Any]:
        sequence_tensor = torch.from_numpy(sequence)
        # The expression matrix is memory-mapped and therefore read-only;
        # make a writable sample-owned array before exposing it to PyTorch.
        context_tensor = torch.from_numpy(np.array(context_vector, dtype=np.float32, copy=True))
        targets_tensor = torch.from_numpy(targets)
        mask_tensor = torch.from_numpy(np.array(target_mask, dtype=np.bool_, copy=True))

        if self.output_format == "named":
            sample: dict[str, Any] = {
                "sequence": sequence_tensor,
                "context_vector": context_tensor,
                "targets": targets_tensor,
                "target_mask": mask_tensor,
            }
        else:
            sample = {
                "x": sequence_tensor,
                "context_vector": context_tensor,
                "labels": targets_tensor,
                "target_mask": mask_tensor,
            }

        if self.return_metadata:
            sample.update(
                {
                    "context_id": torch.tensor(int(self.context_ids_all[context_row]), dtype=torch.long),
                    "context_row": torch.tensor(context_row, dtype=torch.long),
                    "region_index": torch.tensor(region_index, dtype=torch.long),
                    "target_count": torch.tensor(int(mask_tensor.sum().item()), dtype=torch.long),
                    "chrom": self.chroms[region_index],
                    "start": torch.tensor(int(self.starts[region_index]), dtype=torch.long),
                    "end": torch.tensor(int(self.ends[region_index]), dtype=torch.long),
                }
            )
        return sample

    def __getitem__(self, index: int) -> dict[str, Any]:
        region_index, context_row = self._resolve_pair(int(index))
        sequence, targets = self._load_region(region_index)
        context_vector = np.asarray(self.expression_matrix[context_row], dtype=np.float32)
        target_mask = np.asarray(self.context_target_mask_all[context_row], dtype=np.bool_)
        return self._format_sample(
            sequence=sequence,
            context_vector=context_vector,
            targets=targets,
            target_mask=target_mask,
            region_index=region_index,
            context_row=context_row,
        )

    def close(self) -> None:
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            finally:
                self._h5_file = None
                self._h5_pid = None
        self._cached_region_index = None
        self._cached_sequence = None
        self._cached_targets = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5_file"] = None
        state["_h5_pid"] = None
        state["_cached_region_index"] = None
        state["_cached_sequence"] = None
        state["_cached_targets"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
