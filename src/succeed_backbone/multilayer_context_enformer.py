#!/usr/bin/env python3
"""SUCCEED-style multi-layer context conditioning for the local Enformer.

The base Enformer implementation and its checkpoint-compatible parameter
names remain unchanged.  This subclass inserts residual FiLM adapters after
the convolution tower and inside selected transformer feed-forward branches,
then predicts fixed assay semantics with a compact shared head.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .losses import masked_poisson_loss
from .pure_pytorch_enformer import (
    PureEnformerConfig,
    PureEnformerOutput,
    PurePyTorchEnformer,
)


DEFAULT_FILM_TRANSFORMER_LAYERS: tuple[int, ...] = (2, 5, 8)


class _ContextEncoder(nn.Module):
    def __init__(
        self,
        gene_count: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(gene_count, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class MultiLayerContextEnformer(PurePyTorchEnformer):
    """Enformer with low-resolution multi-layer FiLM and an assay head.

    FiLM projectors are zero initialized, so disabling context or using a
    newly initialized adapter produces exactly the pretrained sequence trunk.
    The selected transformer FiLM is applied after the feed-forward LayerNorm
    and before the first feed-forward Linear, matching the residual structure
    used by SUCCEED.
    """

    def __init__(
        self,
        config: PureEnformerConfig | Mapping[str, Any] | None = None,
        *,
        assay_names: Sequence[str],
        context_gene_count: int | None = None,
        context_hidden_dim: int | None = None,
        context_dropout: float | None = None,
        context_mean: torch.Tensor | Sequence[float] | None = None,
        context_std: torch.Tensor | Sequence[float] | None = None,
        film_transformer_layers: Sequence[int] = DEFAULT_FILM_TRANSFORMER_LAYERS,
        film_bound: float = 3.0,
        tf_gammas_path: str | Path | None = None,
    ) -> None:
        super().__init__(config, tf_gammas_path=tf_gammas_path)
        gene_count = int(
            self.config.context_gene_count
            if context_gene_count is None
            else context_gene_count
        )
        hidden_dim = int(
            self.config.context_hidden_dim
            if context_hidden_dim is None
            else context_hidden_dim
        )
        dropout = float(
            self.config.context_dropout
            if context_dropout is None
            else context_dropout
        )
        names = tuple(str(name).strip().lower() for name in assay_names)
        layers = tuple(int(index) for index in film_transformer_layers)
        if gene_count <= 0 or hidden_dim <= 0:
            raise ValueError("context_gene_count and context_hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"context_dropout must be in [0,1), got {dropout}")
        if not names or len(set(names)) != len(names):
            raise ValueError("assay_names must be non-empty and unique")
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("film_transformer_layers must be non-empty and unique")
        invalid_layers = [index for index in layers if not 0 <= index < len(self.transformer)]
        if invalid_layers:
            raise ValueError(
                f"FiLM transformer layers are outside [0,{len(self.transformer) - 1}]: "
                f"{invalid_layers}"
            )
        if float(film_bound) <= 0:
            raise ValueError("film_bound must be positive")

        self.context_gene_count = gene_count
        self.context_hidden_dim = hidden_dim
        self.context_dropout = dropout
        self.assay_names = names
        self.film_transformer_layers = layers
        self.film_bound = float(film_bound)
        self.config.context_gene_count = gene_count
        self.config.context_hidden_dim = hidden_dim
        self.config.context_dropout = dropout

        mean = torch.zeros(gene_count, dtype=torch.float32)
        std = torch.ones(gene_count, dtype=torch.float32)
        if context_mean is not None:
            mean = torch.as_tensor(context_mean, dtype=torch.float32).clone()
        if context_std is not None:
            std = torch.as_tensor(context_std, dtype=torch.float32).clone()
        if mean.shape != (gene_count,) or std.shape != (gene_count,):
            raise ValueError(
                "context_mean/context_std must have shape "
                f"({gene_count},), got {tuple(mean.shape)} and {tuple(std.shape)}"
            )
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Context normalization statistics contain non-finite values")
        if torch.any(std <= 0):
            raise ValueError("context_std must be strictly positive")
        self.register_buffer("context_mean", mean, persistent=True)
        self.register_buffer("context_std", std.clamp_min(1e-6), persistent=True)

        self.context_encoder = _ContextEncoder(gene_count, hidden_dim, dropout)
        projector_names = ["post_conv", *(f"transformer_{index}" for index in layers)]
        self.film_projectors = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, self.config.dim * 2)
                for name in projector_names
            }
        )
        for projector in self.film_projectors.values():
            nn.init.zeros_(projector.weight)
            nn.init.zeros_(projector.bias)

        self.assay_head = nn.Sequential(
            nn.Linear(self.config.dim * 2, len(names)),
            nn.Softplus(),
        )
        self._freeze_sequence_prefix = False
        self._frozen_transformer_layer_count = 0

    def _prepare_context(
        self,
        context_vector: torch.Tensor | Sequence[float],
        batch_size: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(context_vector, torch.Tensor):
            context_vector = torch.as_tensor(context_vector)
        if context_vector.ndim == 1:
            context_vector = context_vector.unsqueeze(0)
        expected = (batch_size, self.context_gene_count)
        if context_vector.shape[0] == 1 and batch_size > 1:
            context_vector = context_vector.expand(batch_size, -1)
        if tuple(context_vector.shape) != expected:
            raise ValueError(
                f"context_vector must have shape [B,{self.context_gene_count}], "
                f"got {tuple(context_vector.shape)}"
            )
        parameter = next(self.context_encoder.parameters())
        value = context_vector.to(device=parameter.device, dtype=parameter.dtype)
        standardized = (value - self.context_mean) / self.context_std
        return self.context_encoder(standardized).to(dtype=output_dtype)

    def _apply_film(
        self,
        hidden: torch.Tensor,
        context_embedding: torch.Tensor,
        projector_name: str,
    ) -> torch.Tensor:
        film = self.film_projectors[projector_name](context_embedding)
        raw_gamma, raw_beta = film.chunk(2, dim=-1)
        bound = self.film_bound
        gamma = bound * torch.tanh(raw_gamma / bound)
        beta = bound * torch.tanh(raw_beta / bound)
        return hidden * (1.0 + gamma[:, None, :]) + beta[:, None, :]

    def _forward_transformer_block(
        self,
        block: nn.Sequential,
        hidden: torch.Tensor,
        context_embedding: torch.Tensor | None,
        layer_index: int,
    ) -> torch.Tensor:
        # block[0] is the attention residual. block[1].fn is
        # LayerNorm -> Linear -> Dropout -> ReLU -> Linear -> Dropout.
        hidden = block[0](hidden)
        feed_forward = block[1].fn
        feed_forward_hidden = feed_forward[0](hidden)
        if context_embedding is not None and layer_index in self.film_transformer_layers:
            feed_forward_hidden = self._apply_film(
                feed_forward_hidden,
                context_embedding,
                f"transformer_{layer_index}",
            )
        for layer in list(feed_forward.children())[1:]:
            feed_forward_hidden = layer(feed_forward_hidden)
        return hidden + feed_forward_hidden

    def _run_transformer(
        self,
        hidden: torch.Tensor,
        context_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer_index, block in enumerate(self.transformer):
            if self.use_checkpointing and self.training and torch.is_grad_enabled():
                def block_forward(
                    value: torch.Tensor,
                    embedding: torch.Tensor,
                    *,
                    current_block: nn.Sequential = block,
                    current_index: int = layer_index,
                ) -> torch.Tensor:
                    return self._forward_transformer_block(
                        current_block, value, embedding, current_index
                    )

                if context_embedding is None:
                    # A dummy tensor keeps a single checkpoint signature while
                    # disable_context bypasses every FiLM projector.
                    dummy = hidden.new_zeros((hidden.shape[0], self.context_hidden_dim))

                    def no_context_forward(
                        value: torch.Tensor,
                        _: torch.Tensor,
                        *,
                        current_block: nn.Sequential = block,
                        current_index: int = layer_index,
                    ) -> torch.Tensor:
                        return self._forward_transformer_block(
                            current_block, value, None, current_index
                        )

                    hidden = activation_checkpoint(
                        no_context_forward, hidden, dummy, use_reentrant=False
                    )
                else:
                    hidden = activation_checkpoint(
                        block_forward,
                        hidden,
                        context_embedding,
                        use_reentrant=False,
                    )
            else:
                hidden = self._forward_transformer_block(
                    block, hidden, context_embedding, layer_index
                )
        return hidden

    def initialize_assay_head_from_human_head(
        self,
        assay_target_indices: Mapping[str, Sequence[int]],
    ) -> dict[str, Any]:
        """Initialize each assay row from matching pretrained human-head rows."""

        if "human" not in self.heads:
            raise ValueError("Pretrained assay initialization requires the human head")
        source = self.heads["human"][0]
        target = self.assay_head[0]
        report: dict[str, Any] = {"source_head": "human", "assays": {}}
        with torch.no_grad():
            for assay_index, assay_name in enumerate(self.assay_names):
                indices = tuple(int(value) for value in assay_target_indices.get(assay_name, ()))
                if not indices:
                    raise ValueError(f"No pretrained target indices supplied for {assay_name}")
                if min(indices) < 0 or max(indices) >= source.out_features:
                    raise ValueError(
                        f"Out-of-range human-head target index for {assay_name}: "
                        f"range=({min(indices)},{max(indices)}), size={source.out_features}"
                    )
                index_tensor = torch.as_tensor(indices, device=source.weight.device)
                target.weight[assay_index].copy_(source.weight.index_select(0, index_tensor).mean(0))
                target.bias[assay_index].copy_(source.bias.index_select(0, index_tensor).mean(0))
                report["assays"][assay_name] = {"source_track_count": len(indices)}
        return report

    def configure_stage1_trainability(
        self,
        *,
        unfreeze_last_transformer_blocks: int = 0,
    ) -> dict[str, Any]:
        """Train adapters/head while freezing the sequence backbone by default."""

        unfreeze_count = int(unfreeze_last_transformer_blocks)
        if not 0 <= unfreeze_count <= len(self.transformer):
            raise ValueError(
                "unfreeze_last_transformer_blocks must be between 0 and "
                f"{len(self.transformer)}"
            )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (self.context_encoder, self.film_projectors, self.assay_head):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if unfreeze_count:
            for block in self.transformer[-unfreeze_count:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        self._freeze_sequence_prefix = True
        self._frozen_transformer_layer_count = len(self.transformer) - unfreeze_count

        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "stage": 1 if unfreeze_count == 0 else 2,
            "unfreeze_last_transformer_blocks": unfreeze_count,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "trainable_fraction": trainable / max(total, 1),
        }

    def enforce_frozen_backbone_eval(self) -> None:
        """Keep frozen backbone dropout/BatchNorm deterministic after train()."""

        if not self._freeze_sequence_prefix:
            return
        self.stem.eval()
        self.conv_tower.eval()
        self.final_pointwise.eval()
        for layer_index, block in enumerate(self.transformer):
            if layer_index < self._frozen_transformer_layer_count:
                block.eval()
        for head in self.heads.values():
            head.eval()

    def forward(
        self,
        x: Any,
        context_vector: torch.Tensor | Sequence[float],
        labels: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        *,
        target_length: int | None = None,
        disable_context: bool = False,
        return_embeddings: bool = False,
        return_only_embeddings: bool = False,
    ) -> PureEnformerOutput | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if context_vector is None and not disable_context:
            raise ValueError("context_vector is required unless disable_context=True")
        sequence, no_batch = self._prepare_sequence(x)
        if target_length is not None:
            self.set_target_length(target_length)

        sequence_channels_first = sequence.permute(0, 2, 1)
        if self._freeze_sequence_prefix:
            with torch.no_grad():
                hidden = self.stem(sequence_channels_first)
                hidden = self.conv_tower(hidden)
            hidden = hidden.detach()
        else:
            hidden = self.stem(sequence_channels_first)
            hidden = self.conv_tower(hidden)
        hidden = hidden.permute(0, 2, 1)

        context_embedding = None
        if not disable_context:
            context_embedding = self._prepare_context(
                context_vector,
                batch_size=hidden.shape[0],
                output_dtype=hidden.dtype,
            )
            hidden = self._apply_film(hidden, context_embedding, "post_conv")
        hidden = self._run_transformer(hidden, context_embedding)
        hidden = self.crop_final(hidden)
        hidden = self.final_pointwise(hidden)
        if return_only_embeddings:
            return hidden[0] if no_batch else hidden

        logits = self.assay_head(hidden)
        loss = None
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels)
            if no_batch and labels.ndim == 2:
                labels = labels.unsqueeze(0)
            labels = labels.to(device=logits.device)
            if not labels.is_floating_point():
                labels = labels.float()
            if tuple(labels.shape) != tuple(logits.shape):
                raise ValueError(
                    f"labels shape {tuple(labels.shape)} does not match assay logits "
                    f"{tuple(logits.shape)}"
                )
            if target_mask is None:
                raise ValueError("target_mask is required when labels are supplied")
            loss = masked_poisson_loss(logits, labels, target_mask)

        if no_batch:
            logits = logits[0]
            hidden = hidden[0]
        if return_embeddings:
            return logits, hidden
        return PureEnformerOutput(loss=loss, logits=logits, hidden_states=hidden)


__all__ = [
    "DEFAULT_FILM_TRANSFORMER_LAYERS",
    "MultiLayerContextEnformer",
]
