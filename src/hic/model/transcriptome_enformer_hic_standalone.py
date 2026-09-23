"""Full-transcriptome Enformer encoder for live Hi-C prediction.

This module is additive and leaves the existing 2,891-gene multi-layer FiLM
adapter unchanged.  It reconstructs the local
``pure_pytorch_transcriptome_context_enformer`` checkpoint, validates the
transcriptome sidecar used by that checkpoint, and exposes the same
``[B,896,3072]`` hidden representation consumed by the existing Enformer Hi-C
head.

The expensive 4,096-gene tokenizer is independent of genomic position.  A
cellular context is therefore encoded once into ``[B,16,256]`` state tokens,
then reused for every one of the 19 Enformer windows covering a 2-Mb Hi-C
target.  This is a small in-memory cache (16 KiB per float32 context), not a
cache of genomic embeddings.
"""

from __future__ import annotations

import contextlib
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from hic.model.enformer_hic_standalone import (
    ENFORMER_CONTEXT_BP,
    ENFORMER_INPUT_BP,
    ENFORMER_OUTPUT_TOKENS,
    ENFORMER_TOKEN_BP,
    HIC_WINDOW_BP,
    EnformerHiCModel,
)


TRANSCRIPTOME_FRAMEWORK = "pure_pytorch_transcriptome_context_enformer"


def _load_external_transcriptome_symbols(
    enformer_root: str | Path,
) -> tuple[type[nn.Module], type[Any], type[Any], type[Any]]:
    """Import the exact model and transcriptome-bank implementations."""

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
        from corgi_enformer_dataset.transcriptome_context_dataset import (
            TranscriptomeContextBank,
        )
        from corgi_enformer_model.pure_pytorch_enformer import PureEnformerConfig
        from corgi_enformer_model.transcriptome_context_enformer import (
            TranscriptomeContextEnformer,
        )
        from corgi_enformer_model.transcriptome_tokenizer import (
            TranscriptomeTokenizerConfig,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Could not import the local transcriptome-context Enformer from "
            f"{root}. Check --enformer-root and the torch environment."
        ) from error
    return (
        TranscriptomeContextEnformer,
        PureEnformerConfig,
        TranscriptomeTokenizerConfig,
        TranscriptomeContextBank,
    )


def _load_transcriptome_checkpoint(
    path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Transcriptome Enformer checkpoint not found: {checkpoint_path}"
        )
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Expected a mapping checkpoint at {checkpoint_path}, got {type(payload)!r}"
        )
    framework = payload.get("framework")
    if framework != TRANSCRIPTOME_FRAMEWORK:
        raise ValueError(
            "The supplied checkpoint is not a transcriptome-context Enformer: "
            f"framework={framework!r}, expected {TRANSCRIPTOME_FRAMEWORK!r}"
        )
    required = {
        "model_state_dict",
        "model_config",
        "tokenizer_config",
        "assay_names",
        "film_transformer_layers",
        "transcriptome_manifest_sha256",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(
            "Transcriptome Enformer checkpoint is missing fields: "
            f"{sorted(missing)}"
        )
    raw_state = payload["model_state_dict"]
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError("model_state_dict must be a non-empty mapping")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError(
                "Transcriptome state_dict must map string names to tensors; "
                f"found key={key!r}, value={type(value)!r}"
            )
        state[key] = value
    metadata = {
        key: payload[key]
        for key in (
            "framework",
            "model_config",
            "tokenizer_config",
            "assay_names",
            "film_transformer_layers",
            "film_bound",
            "run_args",
            "transcriptome_manifest",
            "transcriptome_manifest_sha256",
            "global_step",
            "best_valid_loss",
        )
        if key in payload
    }
    metadata["checkpoint_path"] = str(checkpoint_path)
    return state, metadata


class FrozenTranscriptomeContextEnformerExtractor(nn.Module):
    """Frozen transcriptome-token Enformer exposing central hidden tokens."""

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
        state_dict, metadata = _load_transcriptome_checkpoint(checkpoint_path)
        (
            TranscriptomeContextEnformer,
            PureEnformerConfig,
            TranscriptomeTokenizerConfig,
            TranscriptomeContextBank,
        ) = _load_external_transcriptome_symbols(enformer_root)

        config_values = metadata["model_config"]
        tokenizer_values = metadata["tokenizer_config"]
        if not isinstance(config_values, Mapping) or not isinstance(
            tokenizer_values, Mapping
        ):
            raise ValueError("model_config and tokenizer_config must be mappings")
        config = PureEnformerConfig.from_dict(config_values)
        # The downstream encoder is always frozen, so activation checkpointing
        # adds overhead without reducing retained autograd state.
        config.use_checkpointing = False
        tokenizer_config = TranscriptomeTokenizerConfig.from_dict(tokenizer_values)
        tokenizer_config.validate()
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
        film_bound = float(metadata.get("film_bound", 3.0))
        run_args = metadata.get("run_args")
        if isinstance(run_args, Mapping) and run_args.get("film_bound") is not None:
            film_bound = float(run_args["film_bound"])

        manifest_path = Path(transcriptome_manifest).expanduser().resolve()
        bank = TranscriptomeContextBank(manifest_path)
        expected_manifest_hash = str(metadata["transcriptome_manifest_sha256"])
        if bank.manifest_sha256 != expected_manifest_hash:
            raise ValueError(
                "Transcriptome manifest does not match the checkpoint: "
                f"actual sha256={bank.manifest_sha256}, "
                f"expected={expected_manifest_hash}"
            )
        if tokenizer_config.gene_count != bank.candidate_gene_count:
            raise ValueError(
                "Tokenizer gene_count does not match transcriptome bank: "
                f"{tokenizer_config.gene_count} vs {bank.candidate_gene_count}"
            )
        if tokenizer_config.feature_dim != bank.feature_dim:
            raise ValueError(
                "Tokenizer feature_dim does not match transcriptome bank: "
                f"{tokenizer_config.feature_dim} vs {bank.feature_dim}"
            )
        if tokenizer_config.protocol_count != bank.protocol_count:
            raise ValueError(
                "Tokenizer protocol_count does not match transcriptome bank: "
                f"{tokenizer_config.protocol_count} vs {bank.protocol_count}"
            )

        model = TranscriptomeContextEnformer(
            config,
            assay_names=tuple(str(name) for name in assay_names),
            tokenizer_config=tokenizer_config,
            film_transformer_layers=film_layers_tuple,
            film_bound=film_bound,
        )
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "The transcriptome checkpoint does not exactly match the local "
                "TranscriptomeContextEnformer implementation."
            ) from error

        self.enformer = model
        self.transcriptome_bank = bank
        self.freeze = bool(freeze)
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.enformer_root = str(Path(enformer_root).expanduser().resolve())
        self.transcriptome_manifest = str(manifest_path)
        self.transcriptome_manifest_sha256 = bank.manifest_sha256
        self.input_bp = int(expected_input_bp)
        self.output_tokens = int(expected_output_tokens)
        self.token_bp = ENFORMER_TOKEN_BP
        self.output_hidden = int(config.dim) * 2
        self.context_state_token_count = int(tokenizer_config.state_token_count)
        self.context_state_dim = int(tokenizer_config.token_dim)
        self.gene_token_count = int(bank.gene_token_count)
        self.candidate_gene_count = int(bank.candidate_gene_count)
        self.feature_names = tuple(bank.feature_names)
        self.film_transformer_layers = film_layers_tuple
        self.film_bound = film_bound
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
                "The Hi-C head expects transcriptome Enformer hidden=3072, got "
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
                f"Transcriptome context_id={context_id} is unavailable; "
                f"bank contains {len(available)} IDs"
            ) from error

    def encode_context_ids(self, context_ids: int | Sequence[int]) -> torch.Tensor:
        """Encode bank contexts once and return normal ``[C,S,D]`` tensors."""

        if isinstance(context_ids, (int, np.integer)):
            values = (int(context_ids),)
        else:
            values = tuple(int(value) for value in context_ids)
        if not values:
            raise ValueError("At least one transcriptome context ID is required")
        rows = np.asarray([self.context_row(value) for value in values], dtype=np.int64)
        bank = self.transcriptome_bank
        parameter = next(self.enformer.context_tokenizer.parameters())
        device = parameter.device
        gene_indices = torch.as_tensor(
            np.array(bank.gene_indices[rows], dtype=np.int64, copy=True),
            device=device,
            dtype=torch.long,
        )
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
                gene_indices=gene_indices,
                gene_features=gene_features,
                gene_mask=gene_mask,
                protocol_ids=protocol_ids,
            )
        # Leave inference_mode before cloning so the persistent downstream
        # buffer is a regular tensor and can safely participate in autograd
        # operations performed by the trainable Hi-C head.
        states = encoded.clone().float()
        expected = (
            len(values),
            self.context_state_token_count,
            self.context_state_dim,
        )
        if tuple(states.shape) != expected:
            raise RuntimeError(
                f"Expected context states {expected}, got {tuple(states.shape)}"
            )
        if not torch.isfinite(states).all():
            raise RuntimeError("Transcriptome tokenizer produced non-finite states")
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
                f"Expected transcriptome state shape [B,{expected[0]},{expected[1]}], "
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
        """Return ``[B,896,3072]`` transcriptome-conditioned hidden tokens."""

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


class TranscriptomeTwoMegabaseEncoder(nn.Module):
    """Stitch transcriptome-conditioned Enformer windows into exact 2 Mb."""

    def __init__(
        self,
        extractor: FrozenTranscriptomeContextEnformerExtractor,
        *,
        tile_batch_size: int = 1,
    ) -> None:
        super().__init__()
        if tile_batch_size <= 0:
            raise ValueError("tile_batch_size must be positive")
        self.extractor = extractor
        self.tile_batch_size = int(tile_batch_size)
        self.input_bp = int(extractor.input_bp)
        self.output_tokens = int(extractor.output_tokens)
        self.token_bp = int(extractor.token_bp)
        self.central_bp = self.output_tokens * self.token_bp
        self.flank_bp = (self.input_bp - self.central_bp) // 2
        if self.input_bp - self.central_bp != 2 * self.flank_bp:
            raise ValueError("Enformer central output is not symmetrically aligned")
        self.tile_count = math.ceil(HIC_WINDOW_BP / self.central_bp)
        self.context_bp = (self.tile_count - 1) * self.central_bp + self.input_bp
        self.target_tokens = HIC_WINDOW_BP // self.token_bp
        if self.target_tokens * self.token_bp != HIC_WINDOW_BP:
            raise ValueError("Hi-C window is not aligned to Enformer token size")
        if self.context_bp != ENFORMER_CONTEXT_BP:
            raise ValueError(
                f"Computed context {self.context_bp}, expected {ENFORMER_CONTEXT_BP}"
            )

    @staticmethod
    def _prepare_sequence(sequence: torch.Tensor) -> torch.Tensor:
        if not isinstance(sequence, torch.Tensor):
            sequence = torch.as_tensor(sequence)
        if sequence.ndim != 3:
            raise ValueError(
                "2-Mb DNA context must have shape [B,L,4] or [B,4,L], got "
                f"{tuple(sequence.shape)}"
            )
        if sequence.shape[-1] == 4:
            return sequence
        if sequence.shape[1] == 4:
            return sequence.transpose(1, 2).contiguous()
        raise ValueError("2-Mb DNA context must contain exactly four channels")

    @staticmethod
    def _repeat_states_for_tiles(
        states: torch.Tensor,
        batch_size: int,
        tile_count: int,
    ) -> torch.Tensor:
        if states.ndim == 2:
            states = states.unsqueeze(0)
        if states.ndim != 3:
            raise ValueError(
                "Context states must be [S,D], [1,S,D], or [B,S,D], got "
                f"{tuple(states.shape)}"
            )
        if states.shape[0] == 1:
            return states
        if states.shape[0] != batch_size:
            raise ValueError(
                "DNA and transcriptome-state batch sizes do not match: "
                f"{batch_size} vs {states.shape[0]}"
            )
        # DNA windows are tile-major: [tile0(batch), tile1(batch), ...].
        return states.repeat(tile_count, 1, 1)

    def forward(
        self,
        context_sequence: torch.Tensor,
        context_state_tokens: torch.Tensor | None,
        *,
        disable_context: bool = False,
    ) -> torch.Tensor:
        sequence = self._prepare_sequence(context_sequence)
        if sequence.shape[1] != self.context_bp:
            raise ValueError(
                f"Expected {self.context_bp} context bases, got {sequence.shape[1]}"
            )
        batch_size = sequence.shape[0]
        states = None
        if not disable_context:
            if context_state_tokens is None:
                raise ValueError("Context state tokens are required")
            states = torch.as_tensor(context_state_tokens, device=sequence.device)

        stitched: torch.Tensor | None = None
        for first_tile in range(0, self.tile_count, self.tile_batch_size):
            last_tile = min(first_tile + self.tile_batch_size, self.tile_count)
            tile_count = last_tile - first_tile
            windows = torch.cat(
                [
                    sequence[
                        :,
                        tile_index * self.central_bp:
                        tile_index * self.central_bp + self.input_bp,
                    ]
                    for tile_index in range(first_tile, last_tile)
                ],
                dim=0,
            )
            tiled_states = (
                None
                if states is None
                else self._repeat_states_for_tiles(states, batch_size, tile_count)
            )
            hidden = self.extractor(
                windows,
                tiled_states,
                disable_context=disable_context,
            )
            hidden = hidden.reshape(
                tile_count,
                batch_size,
                hidden.shape[1],
                hidden.shape[2],
            ).permute(1, 0, 2, 3)
            hidden = hidden.reshape(batch_size, tile_count * self.output_tokens, -1)
            if stitched is None:
                stitched = hidden.new_empty(
                    batch_size,
                    self.tile_count * self.output_tokens,
                    hidden.shape[-1],
                )
            stitched[
                :,
                first_tile * self.output_tokens:
                last_tile * self.output_tokens,
            ] = hidden

        if stitched is None:
            raise RuntimeError("No Enformer tiles were evaluated")
        if stitched.shape[1] < self.target_tokens:
            raise RuntimeError(
                f"Produced {stitched.shape[1]} tokens, need {self.target_tokens}"
            )
        return stitched[:, : self.target_tokens].contiguous()


class TranscriptomeEndToEndHiCModel(nn.Module):
    """Connect the frozen transcriptome encoder to the trainable Hi-C head."""

    def __init__(
        self,
        encoder: TranscriptomeTwoMegabaseEncoder,
        hic_model: EnformerHiCModel,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hic_model = hic_model

    def forward(
        self,
        context_sequence: torch.Tensor,
        context_state_tokens: torch.Tensor | None,
        *,
        disable_context: bool = False,
    ) -> torch.Tensor:
        tokens = self.encoder(
            context_sequence,
            context_state_tokens,
            disable_context=disable_context,
        )
        return self.hic_model(tokens)


__all__ = [
    "TRANSCRIPTOME_FRAMEWORK",
    "FrozenTranscriptomeContextEnformerExtractor",
    "TranscriptomeTwoMegabaseEncoder",
    "TranscriptomeEndToEndHiCModel",
]
