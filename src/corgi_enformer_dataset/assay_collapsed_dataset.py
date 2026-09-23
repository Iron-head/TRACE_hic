#!/usr/bin/env python3
"""Collapse paired Enformer tracks into fixed Corgi assay channels.

The context-packed files retain every paired Enformer experiment.  Corgi's
prediction channels, however, represent assay semantics rather than individual
experiments.  This Dataset keeps the existing mmap storage and averages
replicate tracks that belong to the same ``(context, assay)`` pair at read time.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .context_packed_npy_dataset import CorgiEnformerContextPackedNpyDataset
from .dataset import _resolve_relative


ASSAY_NAMES: tuple[str, ...] = (
    "dnase",
    "h3k4me1",
    "h3k4me2",
    "h3k4me3",
    "h3k9ac",
    "h3k9me3",
    "h3k27ac",
    "h3k27me3",
    "h3k36me3",
    "h3k79me2",
    "ctcf",
)


class CorgiEnformerAssayDataset(CorgiEnformerContextPackedNpyDataset):
    """Return one fixed channel per assay, averaging same-context replicates."""

    def __init__(
        self,
        *args: Any,
        assay_names: Sequence[str] = ASSAY_NAMES,
        target_context_map: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        names = tuple(str(name).strip().lower() for name in assay_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("assay_names must be non-empty and unique")
        self.assay_names = names
        self.assay_to_index = {name: index for index, name in enumerate(names)}
        self.raw_num_target_channels = int(self.num_target_channels)

        if target_context_map is None:
            sources = self.manifest.get("sources")
            if not isinstance(sources, Mapping) or "target_context_map" not in sources:
                raise ValueError("Dataset manifest has no sources.target_context_map")
            map_path = _resolve_relative(
                self.manifest_dir, str(sources["target_context_map"])
            )
        else:
            map_path = Path(target_context_map).expanduser()
            if not map_path.is_absolute():
                map_path = (self.manifest_dir / map_path).resolve()
        if not map_path.is_file():
            raise FileNotFoundError(f"Target-context map not found: {map_path}")
        self.target_context_map_path = map_path

        target_to_assay, target_to_context = self._read_target_context_map(map_path)
        packed_indices = np.asarray(self.packed_target_indices, dtype=np.int64)
        missing = [int(value) for value in packed_indices if int(value) not in target_to_assay]
        if missing:
            raise ValueError(
                "Packed targets are absent from target_context_map.tsv: "
                f"{missing[:10]}"
            )

        self.assay_target_indices: dict[str, tuple[int, ...]] = {
            name: tuple(
                sorted(target for target, assay in target_to_assay.items() if assay == name)
            )
            for name in self.assay_names
        }
        empty_assays = [name for name, values in self.assay_target_indices.items() if not values]
        if empty_assays:
            raise ValueError(f"No paired targets were found for assays: {empty_assays}")

        # Each entry is a tuple of raw packed-label column positions for one
        # assay in one context. Empty tuples become masked output channels.
        context_groups: list[tuple[tuple[int, ...], ...]] = []
        context_assay_mask = np.zeros(
            (len(self.context_ids_all), len(self.assay_names)), dtype=np.bool_
        )
        for context_row, context_id_value in enumerate(self.context_ids_all):
            start = int(self.context_packed_offsets[context_row])
            count = int(self.context_packed_counts[context_row])
            raw_indices = packed_indices[start : start + count]
            grouped: dict[int, list[int]] = defaultdict(list)
            for raw_position, target_index_value in enumerate(raw_indices):
                target_index = int(target_index_value)
                expected_context = target_to_context[target_index]
                context_id = int(context_id_value)
                if expected_context != context_id:
                    raise ValueError(
                        "Packed context mapping disagrees with target_context_map.tsv: "
                        f"target={target_index}, packed_context={context_id}, "
                        f"mapped_context={expected_context}"
                    )
                assay_index = self.assay_to_index[target_to_assay[target_index]]
                grouped[assay_index].append(raw_position)
            row_groups = tuple(
                tuple(grouped.get(assay_index, ()))
                for assay_index in range(len(self.assay_names))
            )
            context_groups.append(row_groups)
            context_assay_mask[context_row] = np.asarray(
                [bool(group) for group in row_groups], dtype=np.bool_
            )
        self.context_assay_positions = tuple(context_groups)
        self.context_assay_mask_all = context_assay_mask
        self.num_target_channels = len(self.assay_names)

    def _read_target_context_map(
        self, path: Path
    ) -> tuple[dict[int, str], dict[int, int]]:
        target_to_assay: dict[int, str] = {}
        target_to_context: dict[int, int] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"target_index", "context_id", "manifest_track_name"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"Target-context map {path} is missing columns: {sorted(missing)}"
                )
            for line_number, row in enumerate(reader, start=2):
                target_index = int(row["target_index"])
                context_id = int(row["context_id"])
                assay = str(row["manifest_track_name"]).strip().lower()
                if assay not in self.assay_to_index:
                    raise ValueError(
                        f"Unsupported assay {assay!r} at {path}:{line_number}"
                    )
                if target_index in target_to_assay:
                    raise ValueError(
                        f"Duplicate target_index={target_index} at {path}:{line_number}"
                    )
                target_to_assay[target_index] = assay
                target_to_context[target_index] = context_id
        return target_to_assay, target_to_context

    def __getitem__(self, index: int) -> dict[str, Any]:
        region_index, context_row = self._resolve_pair(int(index))
        sequence, raw_targets, _, _ = self._load_packed_pair(region_index, context_row)
        context_vector = np.asarray(self.expression_matrix[context_row], dtype=np.float32)

        targets = np.zeros(
            (self.target_length, len(self.assay_names)), dtype=np.float32
        )
        target_mask = self.context_assay_mask_all[context_row].copy()
        for assay_index, raw_positions in enumerate(
            self.context_assay_positions[context_row]
        ):
            if raw_positions:
                # Mean merging is the Corgi convention for multiple experiments
                # measuring the same assay in the same context.
                targets[:, assay_index] = raw_targets[:, raw_positions].mean(
                    axis=1, dtype=np.float32
                )

        sequence_tensor = torch.from_numpy(sequence)
        context_tensor = torch.from_numpy(
            np.array(context_vector, dtype=np.float32, copy=True)
        )
        targets_tensor = torch.from_numpy(targets)
        mask_tensor = torch.from_numpy(target_mask)
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

    def expression_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-gene mean/std over the contexts selected for this Dataset."""

        values = np.asarray(self.expression_matrix[self.context_rows], dtype=np.float32)
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = values.std(axis=0, dtype=np.float64).astype(np.float32)
        std = np.maximum(std, np.float32(1e-6))
        return mean, std

    def assay_summary(self) -> dict[str, Any]:
        selected_mask = self.context_assay_mask_all[self.context_rows]
        replicate_pairs = 0
        for context_row in self.context_rows:
            replicate_pairs += sum(
                max(len(group) - 1, 0)
                for group in self.context_assay_positions[int(context_row)]
            )
        return {
            "assays": list(self.assay_names),
            "assay_count": len(self.assay_names),
            "selected_context_count": int(len(self.context_rows)),
            "observed_context_assay_pairs": int(selected_mask.sum()),
            "replicate_tracks_merged": int(replicate_pairs),
            "raw_paired_track_count": int(self.packed_target_count),
            "raw_targets_per_assay": {
                name: len(self.assay_target_indices[name]) for name in self.assay_names
            },
        }


__all__ = ["ASSAY_NAMES", "CorgiEnformerAssayDataset"]
