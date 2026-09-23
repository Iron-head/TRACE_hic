#!/usr/bin/env python3
"""Context-contiguous packed-target backend for Corgi-Enformer training."""

from __future__ import annotations

import bisect
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .dataset import CorgiEnformerH5Dataset, _resolve_relative


CONTEXT_PACKED_NPY_SCHEMA = "corgi_enformer_dataset/context_packed_npy/1"


class CorgiEnformerContextPackedNpyDataset(CorgiEnformerH5Dataset):
    """Read one context's compact targets instead of all 5,313 tracks.

    Target shards use ``[region, packed_track, bin]`` layout.  Tracks for a
    context are contiguous, so one sample performs a small sequential mmap
    read.  Returned labels are padded to ``max_targets_per_context`` for the
    default PyTorch collator; ``target_indices`` identifies which full-head
    model channels correspond to those labels.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        schema = str(self.manifest.get("schema_version", ""))
        if schema != CONTEXT_PACKED_NPY_SCHEMA:
            raise ValueError(
                f"Expected schema {CONTEXT_PACKED_NPY_SCHEMA!r}, got {schema!r}"
            )

        packing = self.manifest.get("packing")
        if not isinstance(packing, Mapping):
            raise ValueError("Packed Dataset manifest has no packing metadata")
        if packing.get("layout") != "region_packed_track_bin":
            raise ValueError(f"Unsupported packed target layout: {packing.get('layout')!r}")
        self.packed_target_indices = np.load(
            _resolve_relative(
                self.manifest_dir, str(packing["packed_target_indices_path"])
            ),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.context_packed_offsets = np.load(
            _resolve_relative(self.manifest_dir, str(packing["context_offsets_path"])),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.context_packed_counts = np.load(
            _resolve_relative(self.manifest_dir, str(packing["context_counts_path"])),
            mmap_mode="r",
            allow_pickle=False,
        )
        self.packed_target_count = int(packing["packed_target_count"])
        self.max_targets_per_context = int(packing["max_targets_per_context"])
        self._validate_packing_tables()

        storage = self.split_info.get("storage")
        if (
            not isinstance(storage, Mapping)
            or storage.get("format") != "context_packed_npy"
            or storage.get("target_layout") != "region_packed_track_bin"
        ):
            raise ValueError(
                f"Split {self.split!r} has no supported context-packed storage metadata"
            )
        raw_records = storage.get("shards")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(f"Split {self.split!r} contains no packed shard records")

        self._packed_shards: list[dict[str, Any]] = []
        expected_start = 0
        for expected_index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, Mapping):
                raise ValueError(f"Invalid packed shard record at index {expected_index}")
            record = dict(raw_record)
            shard_index = int(record["index"])
            start = int(record["start"])
            end = int(record["end"])
            if shard_index != expected_index or start != expected_start or end <= start:
                raise ValueError(
                    f"Non-contiguous packed shard metadata at index {expected_index}: {record}"
                )
            sequence_path = _resolve_relative(
                self.manifest_dir, str(record["sequence_path"])
            )
            target_path = _resolve_relative(
                self.manifest_dir, str(record["target_path"])
            )
            if not sequence_path.is_file() or not target_path.is_file():
                raise FileNotFoundError(
                    f"Missing packed shard for {self.split} rows [{start},{end}): "
                    f"{sequence_path}, {target_path}"
                )
            record.update(
                {
                    "start": start,
                    "end": end,
                    "sequence_path": sequence_path,
                    "target_path": target_path,
                }
            )
            self._packed_shards.append(record)
            expected_start = end
        if expected_start != self.num_regions:
            raise ValueError(
                f"Packed shards cover {expected_start} {self.split} regions, "
                f"expected {self.num_regions}"
            )

        self._packed_shard_ends = [int(record["end"]) for record in self._packed_shards]
        self._active_packed_shard_index: int | None = None
        self._active_sequence_shard: np.ndarray | None = None
        self._active_target_shard: np.ndarray | None = None

    @staticmethod
    def _close_memmap(array: np.ndarray | None) -> None:
        if array is None:
            return
        mmap = getattr(array, "_mmap", None)
        if mmap is not None:
            mmap.close()

    def _validate_packing_tables(self) -> None:
        context_count = len(self.context_ids_all)
        if self.packed_target_indices.shape != (self.packed_target_count,):
            raise ValueError(
                "Packed target index shape mismatch: "
                f"{self.packed_target_indices.shape} vs {(self.packed_target_count,)}"
            )
        if self.context_packed_offsets.shape != (context_count + 1,):
            raise ValueError(
                "Context offset shape mismatch: "
                f"{self.context_packed_offsets.shape} vs {(context_count + 1,)}"
            )
        if self.context_packed_counts.shape != (context_count,):
            raise ValueError(
                "Context packed count shape mismatch: "
                f"{self.context_packed_counts.shape} vs {(context_count,)}"
            )
        offsets = np.asarray(self.context_packed_offsets, dtype=np.int64)
        counts = np.asarray(self.context_packed_counts, dtype=np.int64)
        packed_indices = np.asarray(self.packed_target_indices, dtype=np.int64)
        if offsets[0] != 0 or offsets[-1] != self.packed_target_count:
            raise ValueError("Packed context offsets do not cover all packed tracks")
        if not np.array_equal(np.diff(offsets), counts):
            raise ValueError("Packed context offsets and counts disagree")
        if np.any(counts <= 0) or int(np.max(counts)) != self.max_targets_per_context:
            raise ValueError("Invalid packed target counts")
        if np.any(packed_indices < 0) or np.any(
            packed_indices >= self.num_target_channels
        ):
            raise ValueError("Packed target indices contain an out-of-range channel")
        if len(np.unique(packed_indices)) != self.packed_target_count:
            raise ValueError("Packed target indices contain duplicates")
        for context_row in range(context_count):
            start = int(offsets[context_row])
            end = int(offsets[context_row + 1])
            expected = np.asarray(
                self.context_target_indices_all[context_row, : end - start],
                dtype=np.int64,
            )
            if not np.array_equal(packed_indices[start:end], expected):
                raise ValueError(
                    f"Packed target mapping disagrees at context row {context_row}"
                )

    def _open_packed_shard(self, shard_index: int) -> tuple[np.ndarray, np.ndarray]:
        if (
            self._active_packed_shard_index == shard_index
            and self._active_sequence_shard is not None
            and self._active_target_shard is not None
        ):
            return self._active_sequence_shard, self._active_target_shard

        self._close_memmap(self._active_sequence_shard)
        self._close_memmap(self._active_target_shard)
        record = self._packed_shards[shard_index]
        sequences = np.load(
            record["sequence_path"], mmap_mode="r", allow_pickle=False
        )
        targets = np.load(record["target_path"], mmap_mode="r", allow_pickle=False)
        rows = int(record["end"]) - int(record["start"])
        expected_sequence_shape = (rows, self.sequence_length, 4)
        expected_target_shape = (
            rows,
            self.packed_target_count,
            self.target_length,
        )
        if tuple(sequences.shape) != expected_sequence_shape or sequences.dtype != np.bool_:
            self._close_memmap(sequences)
            self._close_memmap(targets)
            raise ValueError(
                f"Unexpected sequence shard {record['sequence_path']}: "
                f"shape={sequences.shape}, dtype={sequences.dtype}"
            )
        if tuple(targets.shape) != expected_target_shape or targets.dtype != np.float32:
            self._close_memmap(sequences)
            self._close_memmap(targets)
            raise ValueError(
                f"Unexpected packed target shard {record['target_path']}: "
                f"shape={targets.shape}, dtype={targets.dtype}"
            )
        self._active_packed_shard_index = shard_index
        self._active_sequence_shard = sequences
        self._active_target_shard = targets
        return sequences, targets

    def _load_packed_pair(
        self, region_index: int, context_row: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shard_index = bisect.bisect_right(self._packed_shard_ends, region_index)
        if shard_index >= len(self._packed_shards):
            raise IndexError(f"Region index out of packed shard range: {region_index}")
        record = self._packed_shards[shard_index]
        local_index = region_index - int(record["start"])
        sequence_shard, target_shard = self._open_packed_shard(shard_index)

        if (
            self.cache_last_region
            and self._cached_region_index == region_index
            and self._cached_sequence is not None
        ):
            sequence = self._cached_sequence
        else:
            sequence = np.array(
                sequence_shard[local_index], dtype=np.float32, order="C", copy=True
            )
            if self.cache_last_region:
                self._cached_region_index = region_index
                self._cached_sequence = sequence

        packed_start = int(self.context_packed_offsets[context_row])
        count = int(self.context_packed_counts[context_row])
        packed_end = packed_start + count
        targets = np.zeros(
            (self.target_length, self.max_targets_per_context), dtype=np.float32
        )
        # This is the only target I/O performed for a sample.  In the on-disk
        # layout this slice is contiguous and averages only ~20 KiB.
        targets[:, :count] = target_shard[
            local_index, packed_start:packed_end, :
        ].T
        if not np.isfinite(targets[:, :count]).all():
            raise ValueError(
                f"Non-finite packed targets at {self.split} region {region_index}, "
                f"context row {context_row}"
            )
        target_indices = np.full(self.max_targets_per_context, -1, dtype=np.int64)
        target_indices[:count] = self.packed_target_indices[packed_start:packed_end]
        target_mask = np.zeros(self.max_targets_per_context, dtype=np.bool_)
        target_mask[:count] = True
        return sequence, targets, target_indices, target_mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        region_index, context_row = self._resolve_pair(int(index))
        sequence, targets, target_indices, target_mask = self._load_packed_pair(
            region_index, context_row
        )
        context_vector = np.asarray(self.expression_matrix[context_row], dtype=np.float32)

        sequence_tensor = torch.from_numpy(sequence)
        context_tensor = torch.from_numpy(
            np.array(context_vector, dtype=np.float32, copy=True)
        )
        targets_tensor = torch.from_numpy(targets)
        indices_tensor = torch.from_numpy(target_indices)
        mask_tensor = torch.from_numpy(target_mask)
        if self.output_format == "named":
            sample: dict[str, Any] = {
                "sequence": sequence_tensor,
                "context_vector": context_tensor,
                "targets": targets_tensor,
                "target_indices": indices_tensor,
                "target_mask": mask_tensor,
            }
        else:
            sample = {
                "x": sequence_tensor,
                "context_vector": context_tensor,
                "labels": targets_tensor,
                "target_indices": indices_tensor,
                "target_mask": mask_tensor,
            }

        if self.return_metadata:
            sample.update(
                {
                    "context_id": torch.tensor(
                        int(self.context_ids_all[context_row]), dtype=torch.long
                    ),
                    "context_row": torch.tensor(context_row, dtype=torch.long),
                    "region_index": torch.tensor(region_index, dtype=torch.long),
                    "target_count": torch.tensor(
                        int(target_mask.sum()), dtype=torch.long
                    ),
                    "chrom": self.chroms[region_index],
                    "start": torch.tensor(
                        int(self.starts[region_index]), dtype=torch.long
                    ),
                    "end": torch.tensor(int(self.ends[region_index]), dtype=torch.long),
                }
            )
        return sample

    def close(self) -> None:
        super().close()
        self._close_memmap(getattr(self, "_active_sequence_shard", None))
        self._close_memmap(getattr(self, "_active_target_shard", None))
        self._active_packed_shard_index = None
        self._active_sequence_shard = None
        self._active_target_shard = None

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state["_active_packed_shard_index"] = None
        state["_active_sequence_shard"] = None
        state["_active_target_shard"] = None
        return state


__all__ = ["CONTEXT_PACKED_NPY_SCHEMA", "CorgiEnformerContextPackedNpyDataset"]
