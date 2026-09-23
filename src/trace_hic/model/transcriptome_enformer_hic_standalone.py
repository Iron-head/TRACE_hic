"""Exact 2-Mb tiling adapter used by the SUCCEED backbone in TRACE_hic."""

from __future__ import annotations

import math
import torch
from torch import nn

from trace_hic.model.enformer_hic_standalone import (
    ENFORMER_CONTEXT_BP,
    HIC_WINDOW_BP,
    EnformerHiCModel,
)


class TranscriptomeTwoMegabaseEncoder(nn.Module):
    """Stitch transcriptome-conditioned Enformer windows into exact 2 Mb."""

    def __init__(
        self,
        extractor: nn.Module,
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


__all__ = ["TranscriptomeTwoMegabaseEncoder", "TranscriptomeEndToEndHiCModel"]
