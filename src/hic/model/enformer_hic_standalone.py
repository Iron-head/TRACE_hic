"""RNA-conditioned multi-layer Enformer encoder for live Hi-C prediction.

This module is intentionally additive.  It does not alter the existing Corgi
implementation in :mod:`hic.model.corgi_hic_standalone` and is loaded by the
separate ``train_enformer_hic_live.py`` entrypoint.

The external checkpoint used by the first implementation is the local
``pure_pytorch_multilayer_context_enformer`` checkpoint produced by the
``enfpcot`` project.  Its sequence trunk is a 131,072-bp Enformer, with a
2,891-dimensional expression context injected by FiLM after the convolution
tower and inside transformer blocks 2, 5, and 8.  The assay head is not used
for Hi-C; the hidden representation immediately before that head is returned.
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

from hic.model import blocks
from hic.model.native_global import NativeGlobalTransformer
from hic.model.succeed_hic import AttentionPool1D


# ---------------------------------------------------------------------------
# Hi-C and Enformer geometry
# ---------------------------------------------------------------------------

HIC_WINDOW_BP = 2_097_152
HIC_MATRIX_SIZE = 256
HIC_TOKEN_BP = HIC_WINDOW_BP // HIC_MATRIX_SIZE

ENFORMER_INPUT_BP = 131_072
ENFORMER_TOKEN_BP = 128
ENFORMER_OUTPUT_TOKENS = 896
ENFORMER_CENTRAL_BP = ENFORMER_OUTPUT_TOKENS * ENFORMER_TOKEN_BP
ENFORMER_FLANK_BP = (ENFORMER_INPUT_BP - ENFORMER_CENTRAL_BP) // 2
ENFORMER_TILES_PER_HIC_WINDOW = math.ceil(HIC_WINDOW_BP / ENFORMER_CENTRAL_BP)
ENFORMER_CONTEXT_BP = (
    (ENFORMER_TILES_PER_HIC_WINDOW - 1) * ENFORMER_CENTRAL_BP
    + ENFORMER_INPUT_BP
)
ENFORMER_TOKENS_PER_HIC_WINDOW = HIC_WINDOW_BP // ENFORMER_TOKEN_BP

if ENFORMER_INPUT_BP % ENFORMER_TOKEN_BP != 0:
    raise RuntimeError("Enformer input length must be divisible by its token size")
if HIC_WINDOW_BP % ENFORMER_TOKEN_BP != 0:
    raise RuntimeError("Hi-C window must be divisible by Enformer token size")
if HIC_TOKEN_BP % ENFORMER_TOKEN_BP != 0:
    raise RuntimeError("Hi-C token size must be divisible by Enformer token size")


def _load_external_enformer_symbols(
    enformer_root: str | Path,
) -> tuple[type[nn.Module], type[Any]]:
    """Import the exact Enformer implementation used by the checkpoint.

    The two projects are intentionally kept separate.  The root is inserted
    into ``sys.path`` only for this process; no external source file is copied
    or edited.  Returning the classes instead of importing them at module load
    time keeps cached/Corgi-only workflows independent of the Enformer
    dependency.
    """

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
        from corgi_enformer_model.multilayer_context_enformer import (
            MultiLayerContextEnformer,
        )
        from corgi_enformer_model.pure_pytorch_enformer import PureEnformerConfig
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Could not import the local multi-layer Enformer implementation "
            f"from {root}. Check --enformer-root and the torch environment."
        ) from error
    return MultiLayerContextEnformer, PureEnformerConfig


def _checkpoint_state_and_metadata(
    path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read a full training checkpoint or a model-only exported checkpoint."""

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Enformer checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:  # torch versions before the weights_only argument
        payload = torch.load(checkpoint_path, map_location="cpu")

    metadata: dict[str, Any] = {}
    if isinstance(payload, Mapping):
        for key in (
            "framework",
            "model_config",
            "run_args",
            "assay_names",
            "film_transformer_layers",
            "source_checkpoint",
        ):
            if key in payload:
                metadata[key] = payload[key]
        state = payload.get("model_state_dict")
        if state is None:
            state = payload.get("state_dict")
        if state is None:
            # A model-only file may itself be a raw state dict.
            state = payload
    else:
        state = payload

    if not isinstance(state, Mapping) or not state:
        raise ValueError(
            f"No model state_dict found in local Enformer checkpoint {checkpoint_path}"
        )
    state_dict: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError(
                "Enformer state_dict must map string names to tensors; "
                f"found key={key!r}, value={type(value)!r}"
            )
        state_dict[key] = value
    metadata["checkpoint_path"] = str(checkpoint_path)
    return state_dict, metadata


def _parse_film_layers(
    metadata: Mapping[str, Any],
) -> tuple[int, ...]:
    value = metadata.get("film_transformer_layers")
    if value is None:
        run_args = metadata.get("run_args")
        if isinstance(run_args, Mapping):
            value = run_args.get("film_transformer_layers")
    if value is None:
        return (2, 5, 8)
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    try:
        layers = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid film_transformer_layers metadata: {value!r}") from error
    if not layers:
        raise ValueError("film_transformer_layers must not be empty")
    return layers


class FrozenMultiLayerEnformerExtractor(nn.Module):
    """Load a frozen RNA-conditioned multi-layer Enformer and expose hidden tokens.

    Parameters are loaded strictly, including the learned FiLM projectors and
    the expression normalization buffers.  The assay heads remain in the
    reconstructed model so that the checkpoint is structurally exact, but the
    forward pass stops at the 3,072-dimensional hidden representation.
    """

    def __init__(
        self,
        enformer_root: str | Path,
        checkpoint_path: str | Path,
        *,
        freeze: bool = True,
        expected_input_bp: int = ENFORMER_INPUT_BP,
        expected_output_tokens: int = ENFORMER_OUTPUT_TOKENS,
    ) -> None:
        super().__init__()
        state_dict, metadata = _checkpoint_state_and_metadata(checkpoint_path)
        config_values = metadata.get("model_config")
        if not isinstance(config_values, Mapping):
            raise ValueError(
                "The multi-layer Enformer checkpoint must contain a mapping "
                "under 'model_config'."
            )
        MultiLayerContextEnformer, PureEnformerConfig = _load_external_enformer_symbols(
            enformer_root
        )
        config = PureEnformerConfig.from_dict(config_values)
        if int(config.target_length) != int(expected_output_tokens):
            raise ValueError(
                "Unexpected Enformer target_length: checkpoint has "
                f"{config.target_length}, expected {expected_output_tokens}."
            )

        assay_names = metadata.get("assay_names")
        if assay_names is None:
            raise ValueError(
                "The multi-layer Enformer checkpoint is missing assay_names; "
                "use its full training checkpoint or export it with the new "
                "encoder export utility."
            )
        if isinstance(assay_names, str) or not isinstance(assay_names, Sequence):
            raise ValueError(f"Invalid assay_names metadata: {assay_names!r}")
        film_layers = _parse_film_layers(metadata)
        run_args = metadata.get("run_args")
        film_bound = 3.0
        if isinstance(run_args, Mapping) and run_args.get("film_bound") is not None:
            film_bound = float(run_args["film_bound"])

        context_gene_count = int(getattr(config, "context_gene_count", 2891))
        context_hidden_dim = int(getattr(config, "context_hidden_dim", 512))
        context_dropout = float(getattr(config, "context_dropout", 0.1))
        model = MultiLayerContextEnformer(
            config,
            assay_names=tuple(str(name) for name in assay_names),
            context_gene_count=context_gene_count,
            context_hidden_dim=context_hidden_dim,
            context_dropout=context_dropout,
            film_transformer_layers=film_layers,
            film_bound=film_bound,
        )
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "The supplied checkpoint does not exactly match the local "
                "multi-layer Enformer implementation. Verify --enformer-root "
                "points to the enfpcot checkout used to create the checkpoint."
            ) from error

        self.enformer = model
        self.freeze = bool(freeze)
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.enformer_root = str(Path(enformer_root).expanduser().resolve())
        self.input_bp = int(expected_input_bp)
        self.output_tokens = int(expected_output_tokens)
        self.token_bp = ENFORMER_TOKEN_BP
        self.output_hidden = int(config.dim) * 2
        self.context_gene_count = context_gene_count
        self.context_hidden_dim = context_hidden_dim
        self.film_transformer_layers = tuple(film_layers)
        self.film_bound = float(film_bound)
        if self.input_bp != ENFORMER_INPUT_BP:
            raise ValueError(
                f"Only the 131,072-bp Enformer geometry is supported, got {self.input_bp}."
            )
        if self.output_hidden != 3072:
            raise ValueError(
                "The Hi-C adapter currently expects Enformer dim=1536 and "
                f"therefore hidden=3072, got {self.output_hidden}."
            )
        if self.freeze:
            for parameter in self.enformer.parameters():
                parameter.requires_grad_(False)
        self.enformer.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            # The context encoder contains dropout and the Enformer trunk has
            # BatchNorm.  Frozen inference must remain deterministic even when
            # the surrounding Lightning module enters train() mode.
            self.enformer.eval()
        return self

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
            channel_last = sequence
        elif sequence.shape[1] == 4:
            channel_last = sequence.transpose(1, 2).contiguous()
        else:
            raise ValueError("Enformer DNA input must contain exactly four channels")
        return channel_last

    @staticmethod
    def _prepare_expression(expression: torch.Tensor) -> torch.Tensor:
        if not isinstance(expression, torch.Tensor):
            expression = torch.as_tensor(expression)
        if expression.ndim == 1:
            expression = expression.unsqueeze(0)
        if expression.ndim != 2:
            raise ValueError(
                "Expression must have shape [R] or [B,R], got "
                f"{tuple(expression.shape)}"
            )
        return expression

    def forward(
        self,
        sequence: torch.Tensor,
        expression: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[B,896,3072]`` RNA-conditioned hidden tokens."""

        sequence = self._prepare_sequence(sequence)
        expression = self._prepare_expression(expression)
        if sequence.shape[1] != self.input_bp:
            raise ValueError(
                f"Expected {self.input_bp} DNA bases per Enformer window, "
                f"got {sequence.shape[1]}"
            )
        if expression.shape[-1] != self.context_gene_count:
            raise ValueError(
                f"Expected {self.context_gene_count} expression features, "
                f"got {expression.shape[-1]}"
            )
        gradient_context = (
            torch.inference_mode() if self.freeze else contextlib.nullcontext()
        )
        with gradient_context:
            hidden = self.enformer(
                sequence,
                expression,
                return_only_embeddings=True,
            )
        if self.freeze:
            # Inference tensors cannot be saved for backward by the trainable
            # Hi-C projection.  Clone after leaving inference_mode to create a
            # normal, graph-free tensor.
            hidden = hidden.clone()
        expected = (self.output_tokens, self.output_hidden)
        if tuple(hidden.shape[1:]) != expected:
            raise RuntimeError(
                "Unexpected multi-layer Enformer hidden shape: expected "
                f"[B,{expected[0]},{expected[1]}], got {tuple(hidden.shape)}"
            )
        return hidden


class EnformerTwoMegabaseEncoder(nn.Module):
    """Stitch center-valid Enformer outputs into one exact 2-Mb sequence.

    A 131,072-bp input produces 1,024 128-bp positions before the model's
    target crop.  The checkpoint was trained with the central 896 positions,
    so this adapter keeps only those 114,688 bp.  Nineteen overlapping input
    windows cover the 2,097,152-bp Hi-C interval; the concatenated output is
    cropped to exactly 16,384 128-bp tokens.
    """

    def __init__(
        self,
        extractor: FrozenMultiLayerEnformerExtractor,
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
            raise ValueError("Enformer input and central output are not symmetrically aligned")
        self.tile_count = math.ceil(HIC_WINDOW_BP / self.central_bp)
        self.context_bp = (self.tile_count - 1) * self.central_bp + self.input_bp
        self.target_tokens = HIC_WINDOW_BP // self.token_bp
        if self.target_tokens * self.token_bp != HIC_WINDOW_BP:
            raise ValueError("Hi-C window is not aligned to Enformer token size")
        if self.context_bp != ENFORMER_CONTEXT_BP:
            raise ValueError(
                "Unexpected Enformer context geometry: computed "
                f"{self.context_bp}, module constant is {ENFORMER_CONTEXT_BP}"
            )

    @staticmethod
    def _repeat_expression_for_tiles(
        expression: torch.Tensor,
        batch_size: int,
        tile_count: int,
    ) -> torch.Tensor:
        if expression.ndim <= 1 or expression.shape[0] == 1:
            return expression
        if expression.ndim != 2 or expression.shape[0] != batch_size:
            raise ValueError(
                "Expression must be [R], [1,R], or [B,R] before tile batching; "
                f"got {tuple(expression.shape)} for DNA batch {batch_size}"
            )
        # Windows are concatenated tile-major: [tile0(batch), tile1(batch), ...].
        return expression.repeat(tile_count, 1)

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

    def forward(
        self,
        context_sequence: torch.Tensor,
        expression: torch.Tensor,
    ) -> torch.Tensor:
        sequence = self._prepare_sequence(context_sequence)
        if sequence.shape[1] != self.context_bp:
            raise ValueError(
                f"Expected {self.context_bp} DNA bases for the Enformer context, "
                f"got {sequence.shape[1]}"
            )
        batch_size = sequence.shape[0]
        expression = torch.as_tensor(expression, device=sequence.device)
        if expression.ndim == 1:
            expression = expression.unsqueeze(0)

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
            tiled_expression = self._repeat_expression_for_tiles(
                expression,
                batch_size,
                tile_count,
            )
            hidden = self.extractor(windows, tiled_expression)
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
                f"Enformer tiles produced only {stitched.shape[1]} tokens; "
                f"need {self.target_tokens}"
            )
        return stitched[:, : self.target_tokens].contiguous()


class EnformerHiCModel(nn.Module):
    """Hi-C decoder operating on 128-bp Enformer hidden tokens."""

    def __init__(
        self,
        *,
        encoder_hidden: int = 3072,
        encoder_token_bp: int = ENFORMER_TOKEN_BP,
        hidden: int = 256,
        projection_dropout: float = 0.1,
        pool_heads: int = 8,
        native_global_layers: int = 2,
        native_global_latents: int = 128,
        native_global_heads: int = 8,
        native_global_dropout: float = 0.1,
        native_global_gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        if encoder_hidden <= 0 or hidden <= 0:
            raise ValueError("Embedding dimensions must be positive")
        if encoder_token_bp <= 0:
            raise ValueError("encoder_token_bp must be positive")
        if hidden % pool_heads != 0:
            raise ValueError("hidden must be divisible by pool_heads")
        if HIC_WINDOW_BP % encoder_token_bp != 0:
            raise ValueError("Hi-C window must be divisible by encoder_token_bp")
        if HIC_TOKEN_BP % encoder_token_bp != 0:
            raise ValueError("Hi-C token size must be divisible by encoder_token_bp")

        self.encoder_hidden = int(encoder_hidden)
        self.encoder_token_bp = int(encoder_token_bp)
        self.hidden = int(hidden)
        self.native_token_count = HIC_WINDOW_BP // self.encoder_token_bp
        self.pool_size = HIC_TOKEN_BP // self.encoder_token_bp
        self.encoder_projection = nn.Sequential(
            nn.LayerNorm(self.encoder_hidden),
            nn.Linear(self.encoder_hidden, self.hidden),
            nn.GELU(),
            nn.Dropout(projection_dropout),
        )
        self.native_global_transformer = NativeGlobalTransformer(
            hidden=self.hidden,
            num_latents=native_global_latents,
            num_layers=native_global_layers,
            num_heads=native_global_heads,
            dropout=native_global_dropout,
            gate_init=native_global_gate_init,
            max_tokens=self.native_token_count,
        )
        self.pool_to_hic = AttentionPool1D(
            dim=self.hidden,
            pool_size=self.pool_size,
            num_heads=pool_heads,
            max_output_tokens=HIC_MATRIX_SIZE,
        )
        self.output_projection = nn.Conv1d(self.hidden, self.hidden, kernel_size=1)
        self.decoder = blocks.Decoder(self.hidden * 2)

    def encode(self, enformer_tokens: torch.Tensor) -> torch.Tensor:
        if enformer_tokens.ndim != 3:
            raise ValueError(
                "Enformer tokens must be [B,L,C], got "
                f"{tuple(enformer_tokens.shape)}"
            )
        expected = (self.native_token_count, self.encoder_hidden)
        if tuple(enformer_tokens.shape[1:]) != expected:
            raise ValueError(
                f"Expected Enformer token shape [B,{expected[0]},{expected[1]}], "
                f"got {tuple(enformer_tokens.shape)}"
            )
        hidden = self.encoder_projection(enformer_tokens.float())
        hidden = self.native_global_transformer(hidden)
        hidden = self.pool_to_hic(hidden.transpose(1, 2).contiguous())
        hidden = self.output_projection(hidden)
        if hidden.shape[-1] != HIC_MATRIX_SIZE:
            raise RuntimeError(
                f"Expected {HIC_MATRIX_SIZE} Hi-C tokens, got {hidden.shape[-1]}"
            )
        return hidden

    @staticmethod
    def diagonalize(hidden: torch.Tensor) -> torch.Tensor:
        hidden_i = hidden.unsqueeze(2).expand(-1, -1, HIC_MATRIX_SIZE, -1)
        hidden_j = hidden.unsqueeze(3).expand(-1, -1, -1, HIC_MATRIX_SIZE)
        return torch.cat([hidden_i, hidden_j], dim=1)

    def forward(self, enformer_tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(enformer_tokens)
        pairwise = self.diagonalize(hidden)
        return self.decoder(pairwise).squeeze(1)


class EnformerEndToEndHiCModel(nn.Module):
    """Convenience wrapper connecting the Enformer encoder and Hi-C head."""

    def __init__(
        self,
        encoder: EnformerTwoMegabaseEncoder,
        hic_model: EnformerHiCModel | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.hic_model = hic_model if hic_model is not None else EnformerHiCModel()

    def forward(
        self,
        context_sequence: torch.Tensor,
        expression: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.encoder(context_sequence, expression)
        return self.hic_model(tokens)


def load_expression_npz(
    path: str | Path,
    key: str = "expression",
    *,
    expected_size: int = 2891,
) -> np.ndarray:
    """Load a prepared Corgi-order expression vector for Enformer context."""

    expression_path = Path(path).expanduser().resolve()
    if not expression_path.is_file():
        raise FileNotFoundError(f"Expression NPZ not found: {expression_path}")
    with np.load(expression_path, allow_pickle=False) as archive:
        if key not in archive:
            raise KeyError(
                f"Expression key {key!r} not found in {expression_path}; "
                f"available keys: {list(archive.keys())}"
            )
        expression = np.asarray(archive[key], dtype=np.float32)
    if expression.ndim != 1 or expression.size != int(expected_size):
        raise ValueError(
            f"Expression must have exactly {expected_size} values, "
            f"found shape {expression.shape} in {expression_path}"
        )
    if not np.isfinite(expression).all():
        raise ValueError(f"Expression contains non-finite values: {expression_path}")
    return expression


__all__ = [
    "HIC_WINDOW_BP",
    "HIC_MATRIX_SIZE",
    "HIC_TOKEN_BP",
    "ENFORMER_INPUT_BP",
    "ENFORMER_TOKEN_BP",
    "ENFORMER_OUTPUT_TOKENS",
    "ENFORMER_CENTRAL_BP",
    "ENFORMER_FLANK_BP",
    "ENFORMER_TILES_PER_HIC_WINDOW",
    "ENFORMER_CONTEXT_BP",
    "ENFORMER_TOKENS_PER_HIC_WINDOW",
    "FrozenMultiLayerEnformerExtractor",
    "EnformerTwoMegabaseEncoder",
    "EnformerHiCModel",
    "EnformerEndToEndHiCModel",
    "load_expression_npz",
]
