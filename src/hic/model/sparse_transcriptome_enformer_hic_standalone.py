"""Sparse full-transcriptome Enformer encoder for live Hi-C prediction.

This additive adapter reconstructs a
``pure_pytorch_sparse_full_transcriptome_enformer`` checkpoint, validates its
full-transcriptome sidecar, and exposes the same ``[B,896,3072]`` hidden
representation consumed by the existing Enformer Hi-C head.

The 64,217-gene sparse router is independent of genomic position.  A cellular
context is encoded once into ``[B,32,256]`` state tokens and reused for all 19
Enformer windows covering a 2-Mb Hi-C target.  Only the selected context rows
are copied out of the mmap bank; the complete 232-context bank is never moved
to the accelerator by this adapter.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from hic.model.enformer_hic_standalone import (
    ENFORMER_INPUT_BP,
    ENFORMER_OUTPUT_TOKENS,
    ENFORMER_TOKEN_BP,
)
from hic.model.transcriptome_enformer_hic_standalone import (
    TranscriptomeEndToEndHiCModel,
    TranscriptomeTwoMegabaseEncoder,
)


SPARSE_TRANSCRIPTOME_FRAMEWORK = (
    "pure_pytorch_sparse_full_transcriptome_enformer"
)


def _load_external_sparse_symbols(
    enformer_root: str | Path,
) -> tuple[type[nn.Module], type[Any], type[Any], type[Any]]:
    """Import the exact sparse model, configs and mmap context bank."""

    root = Path(enformer_root).expanduser().resolve()
    package_dir = root / "corgi_enformer_model"
    if not package_dir.is_dir():
        raise FileNotFoundError(
            "Enformer source package was not found under "
            f"{root}: expected {package_dir}"
        )
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    try:
        from corgi_enformer_dataset.full_transcriptome_context_dataset import (
            FullTranscriptomeContextBank,
        )
        from corgi_enformer_model.pure_pytorch_enformer import PureEnformerConfig
        from corgi_enformer_model.sparse_transcriptome_enformer import (
            SparseTranscriptomeContextEnformer,
        )
        from corgi_enformer_model.sparse_transcriptome_router import (
            SparseTranscriptomeRouterConfig,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Could not import the local sparse-transcriptome Enformer from "
            f"{root}. Check --enformer-root and the torch environment."
        ) from error
    return (
        SparseTranscriptomeContextEnformer,
        PureEnformerConfig,
        SparseTranscriptomeRouterConfig,
        FullTranscriptomeContextBank,
    )


def _load_sparse_checkpoint(
    path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Sparse-transcriptome checkpoint not found: {checkpoint_path}"
        )
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Expected a mapping checkpoint at {checkpoint_path}, "
            f"got {type(payload)!r}"
        )
    framework = payload.get("framework")
    if framework != SPARSE_TRANSCRIPTOME_FRAMEWORK:
        raise ValueError(
            "The supplied checkpoint is not a sparse full-transcriptome "
            f"Enformer: framework={framework!r}, expected "
            f"{SPARSE_TRANSCRIPTOME_FRAMEWORK!r}"
        )
    required = {
        "format_version",
        "model_state_dict",
        "model_config",
        "router_config",
        "assay_names",
        "film_transformer_layers",
        "film_bound",
        "routing_diversity_weight",
        "transcriptome_manifest_sha256",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            "Sparse-transcriptome checkpoint is missing fields: "
            f"{sorted(missing)}"
        )
    raw_state = payload["model_state_dict"]
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError("model_state_dict must be a non-empty mapping")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError(
                "Sparse state_dict must map string names to tensors; "
                f"found key={key!r}, value={type(value)!r}"
            )
        state[key] = value
    metadata_keys = (
        "format_version",
        "framework",
        "model_config",
        "router_config",
        "assay_names",
        "film_transformer_layers",
        "film_bound",
        "routing_diversity_weight",
        "run_args",
        "transcriptome_manifest",
        "transcriptome_manifest_sha256",
        "global_step",
        "best_valid_loss",
    )
    metadata = {key: payload[key] for key in metadata_keys if key in payload}
    metadata["checkpoint_path"] = str(checkpoint_path)
    return state, metadata


class FrozenSparseTranscriptomeEnformerExtractor(nn.Module):
    """Frozen sparse-transcriptome Enformer exposing central hidden tokens."""

    def __init__(
        self,
        enformer_root: str | Path,
        checkpoint_path: str | Path,
        transcriptome_manifest: str | Path,
        *,
        freeze: bool = True,
        expected_input_bp: int = ENFORMER_INPUT_BP,
        expected_output_tokens: int = ENFORMER_OUTPUT_TOKENS,
    ) -> None:
        super().__init__()
        state_dict, metadata = _load_sparse_checkpoint(checkpoint_path)
        (
            SparseTranscriptomeContextEnformer,
            PureEnformerConfig,
            SparseTranscriptomeRouterConfig,
            FullTranscriptomeContextBank,
        ) = _load_external_sparse_symbols(enformer_root)

        config_values = metadata["model_config"]
        router_values = metadata["router_config"]
        if not isinstance(config_values, Mapping) or not isinstance(
            router_values, Mapping
        ):
            raise ValueError("model_config and router_config must be mappings")
        config = PureEnformerConfig.from_dict(config_values)
        # The complete upstream model is frozen in the first Hi-C stage.
        config.use_checkpointing = False
        router_config = SparseTranscriptomeRouterConfig.from_dict(router_values)
        router_config.validate()
        if int(config.target_length) != int(expected_output_tokens):
            raise ValueError(
                "Unexpected Enformer target_length: checkpoint has "
                f"{config.target_length}, expected {expected_output_tokens}"
            )

        assay_names = metadata["assay_names"]
        film_layers = metadata["film_transformer_layers"]
        if isinstance(assay_names, str) or not isinstance(assay_names, Sequence):
            raise ValueError(f"Invalid assay_names metadata: {assay_names!r}")
        if isinstance(film_layers, str):
            film_layers = [
                item.strip() for item in film_layers.split(",") if item.strip()
            ]
        try:
            film_layers_tuple = tuple(int(value) for value in film_layers)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid film_transformer_layers metadata: {film_layers!r}"
            ) from error
        film_bound = float(metadata["film_bound"])
        diversity_weight = float(metadata["routing_diversity_weight"])

        manifest_path = Path(transcriptome_manifest).expanduser().resolve()
        bank = FullTranscriptomeContextBank(manifest_path)
        expected_manifest_hash = str(metadata["transcriptome_manifest_sha256"])
        if bank.manifest_sha256 != expected_manifest_hash:
            raise ValueError(
                "Full-transcriptome manifest does not match the checkpoint: "
                f"actual sha256={bank.manifest_sha256}, "
                f"expected={expected_manifest_hash}"
            )
        if router_config.gene_count != bank.gene_count:
            raise ValueError(
                "Router gene_count does not match the transcriptome bank: "
                f"{router_config.gene_count} vs {bank.gene_count}"
            )
        if router_config.feature_dim != bank.feature_dim:
            raise ValueError(
                "Router feature_dim does not match the transcriptome bank: "
                f"{router_config.feature_dim} vs {bank.feature_dim}"
            )
        if router_config.protocol_count != bank.protocol_count:
            raise ValueError(
                "Router protocol_count does not match the transcriptome bank: "
                f"{router_config.protocol_count} vs {bank.protocol_count}"
            )

        model = SparseTranscriptomeContextEnformer(
            config,
            assay_names=tuple(str(name) for name in assay_names),
            router_config=router_config,
            film_transformer_layers=film_layers_tuple,
            film_bound=film_bound,
            routing_diversity_weight=diversity_weight,
        )
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "The sparse-transcriptome checkpoint does not exactly match "
                "the local SparseTranscriptomeContextEnformer implementation."
            ) from error

        self.enformer = model
        self.transcriptome_bank = bank
        self.freeze = bool(freeze)
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.enformer_root = str(Path(enformer_root).expanduser().resolve())
        self.transcriptome_manifest = str(manifest_path)
        self.transcriptome_manifest_sha256 = bank.manifest_sha256
        self.checkpoint_format_version = int(metadata["format_version"])
        self.checkpoint_global_step = int(metadata.get("global_step", -1))
        self.checkpoint_best_valid_loss = float(
            metadata.get("best_valid_loss", float("nan"))
        )
        self.input_bp = int(expected_input_bp)
        self.output_tokens = int(expected_output_tokens)
        self.token_bp = ENFORMER_TOKEN_BP
        self.output_hidden = int(config.dim) * 2
        self.context_state_token_count = int(router_config.state_token_count)
        self.context_state_dim = int(router_config.state_dim)
        self.gene_count = int(bank.gene_count)
        self.candidate_gene_count = int(bank.gene_count)
        self.feature_names = tuple(bank.feature_names)
        self.film_transformer_layers = film_layers_tuple
        self.film_bound = film_bound
        self.routing_activation = str(router_config.routing_activation)
        self._context_id_to_row = {
            int(context_id): row
            for row, context_id in enumerate(np.asarray(bank.context_ids))
        }
        if self.input_bp != ENFORMER_INPUT_BP:
            raise ValueError(
                f"Only 131,072-bp Enformer inputs are supported, got {self.input_bp}"
            )
        if self.output_hidden != 3072:
            raise ValueError(
                "The Hi-C head expects sparse Enformer hidden=3072, got "
                f"{self.output_hidden}"
            )
        if self.freeze:
            for parameter in self.enformer.parameters():
                parameter.requires_grad_(False)
        self.enformer.eval()
        del state_dict

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.enformer.eval()
        return self

    def context_row(self, context_id: int) -> int:
        try:
            return self._context_id_to_row[int(context_id)]
        except KeyError as error:
            available = sorted(self._context_id_to_row)
            raise KeyError(
                f"Sparse transcriptome context_id={context_id} is unavailable; "
                f"bank contains {len(available)} IDs"
            ) from error

    def encode_context_ids(self, context_ids: int | Sequence[int]) -> torch.Tensor:
        """Encode selected mmap rows once and return normal ``[C,S,D]`` tensors."""

        if isinstance(context_ids, (int, np.integer)):
            values = (int(context_ids),)
        else:
            values = tuple(int(value) for value in context_ids)
        if not values:
            raise ValueError("At least one transcriptome context ID is required")
        rows = np.asarray(
            [self.context_row(value) for value in values],
            dtype=np.int64,
        )
        bank = self.transcriptome_bank
        parameter = next(self.enformer.context_router.parameters())
        device = parameter.device
        # Copy only requested contexts out of mmap.  Calling bank.to(device)
        # here would unnecessarily materialize all 232 transcriptomes.
        gene_features = torch.as_tensor(
            np.array(bank.gene_features[rows], dtype=np.float32, copy=True),
            device=device,
            dtype=parameter.dtype,
        )
        gene_mask = torch.as_tensor(
            np.array(bank.gene_mask[rows], dtype=np.bool_, copy=True),
            device=device,
            dtype=torch.bool,
        )
        protocol_ids = torch.as_tensor(
            np.array(bank.protocol_ids[rows], dtype=np.int64, copy=True),
            device=device,
            dtype=torch.long,
        )
        with torch.inference_mode():
            encoded = self.enformer.encode_context(
                gene_features=gene_features,
                gene_mask=gene_mask,
                protocol_ids=protocol_ids,
            )
        if not isinstance(encoded, torch.Tensor):
            raise TypeError("Sparse encode_context returned diagnostics unexpectedly")
        # Exit inference_mode before cloning so the persistent Lightning buffer
        # is a regular tensor that can be used by the trainable downstream head.
        states = encoded.clone().float()
        expected = (
            len(values),
            self.context_state_token_count,
            self.context_state_dim,
        )
        if tuple(states.shape) != expected:
            raise RuntimeError(
                f"Expected sparse context states {expected}, got {tuple(states.shape)}"
            )
        if not torch.isfinite(states).all():
            raise RuntimeError("Sparse transcriptome router produced non-finite states")
        return states

    @staticmethod
    def _prepare_sequence(sequence: torch.Tensor) -> torch.Tensor:
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.as_tensor(sequence)
        if sequence.ndim != 3:
            raise ValueError(
                "Enformer DNA must have shape [B,L,4] or [B,4,L], got "
                f"{tuple(sequence.shape)}"
            )
        if sequence.shape[-1] == 4:
            return sequence
        if sequence.shape[1] == 4:
            return sequence.transpose(1, 2).contiguous()
        raise ValueError("Enformer DNA input must contain exactly four channels")

    def _prepare_context_states(self, states: torch.Tensor) -> torch.Tensor:
        if not isinstance(states, torch.Tensor):
            states = torch.as_tensor(states)
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if states.ndim != 3:
            raise ValueError(
                "Context states must have shape [S,D] or [B,S,D], got "
                f"{tuple(states.shape)}"
            )
        expected = (self.context_state_token_count, self.context_state_dim)
        if tuple(states.shape[1:]) != expected:
            raise ValueError(
                f"Expected sparse state shape [B,{expected[0]},{expected[1]}], "
                f"got {tuple(states.shape)}"
            )
        return states

    def forward(
        self,
        sequence: torch.Tensor,
        context_state_tokens: torch.Tensor | None,
        *,
        disable_context: bool = False,
    ) -> torch.Tensor:
        """Return ``[B,896,3072]`` sparse-RNA-conditioned hidden tokens."""

        sequence = self._prepare_sequence(sequence)
        if sequence.shape[1] != self.input_bp:
            raise ValueError(
                f"Expected {self.input_bp} DNA bases, got {sequence.shape[1]}"
            )
        states = None
        if not disable_context:
            if context_state_tokens is None:
                raise ValueError("Context state tokens are required")
            states = self._prepare_context_states(context_state_tokens)
        gradient_context = (
            torch.inference_mode() if self.freeze else contextlib.nullcontext()
        )
        with gradient_context:
            if disable_context:
                hidden = self.enformer(
                    sequence,
                    disable_context=True,
                    return_only_embeddings=True,
                )
            else:
                hidden = self.enformer(
                    sequence,
                    context_state_tokens=states,
                    return_only_embeddings=True,
                )
        if self.freeze:
            hidden = hidden.clone()
        expected = (self.output_tokens, self.output_hidden)
        if tuple(hidden.shape[1:]) != expected:
            raise RuntimeError(
                f"Expected hidden [B,{expected[0]},{expected[1]}], "
                f"got {tuple(hidden.shape)}"
            )
        return hidden


class SparseTranscriptomeTwoMegabaseEncoder(TranscriptomeTwoMegabaseEncoder):
    """Typed sparse-RNA alias for the shared exact 2-Mb tile stitcher."""

    def __init__(
        self,
        extractor: FrozenSparseTranscriptomeEnformerExtractor,
        *,
        tile_batch_size: int = 1,
    ) -> None:
        super().__init__(extractor, tile_batch_size=tile_batch_size)


class SparseTranscriptomeEndToEndHiCModel(TranscriptomeEndToEndHiCModel):
    """Sparse-RNA Enformer connected to the unchanged trainable Hi-C head."""


__all__ = [
    "SPARSE_TRANSCRIPTOME_FRAMEWORK",
    "FrozenSparseTranscriptomeEnformerExtractor",
    "SparseTranscriptomeTwoMegabaseEncoder",
    "SparseTranscriptomeEndToEndHiCModel",
]
