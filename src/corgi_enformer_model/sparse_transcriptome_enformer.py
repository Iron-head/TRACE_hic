#!/usr/bin/env python3
"""Enformer conditioned by reference-free sparse full-transcriptome routing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .losses import masked_poisson_loss
from .multilayer_context_enformer import DEFAULT_FILM_TRANSFORMER_LAYERS
from .pure_pytorch_enformer import (
    PureEnformerConfig,
    PureEnformerOutput,
    PurePyTorchEnformer,
)
from .sparse_transcriptome_router import (
    SparseTranscriptomeRouter,
    SparseTranscriptomeRouterConfig,
    SparseTranscriptomeRouterOutput,
)


@dataclass
class SparseTranscriptomeEnformerOutput(PureEnformerOutput):
    """Prediction output with optional sparse-routing diagnostics."""

    prediction_loss: torch.Tensor | None = None
    routing_diversity_loss: torch.Tensor | None = None
    active_gene_counts: torch.Tensor | None = None
    routing_entropy: torch.Tensor | None = None
    routing_weights: torch.Tensor | None = None


class SparseTranscriptomeContextEnformer(PurePyTorchEnformer):
    """Pure-PyTorch Enformer with end-to-end transcriptome gene selection.

    Every valid gene is scored by the context router.  Sparse latent state
    tokens then condition the convolutional tower output and selected
    Transformer feed-forward branches through zero-initialized FiLM adapters.
    The base Enformer therefore remains an exact identity at initialization.
    """

    def __init__(
        self,
        config: PureEnformerConfig | Mapping[str, Any] | None = None,
        *,
        assay_names: Sequence[str],
        router_config: SparseTranscriptomeRouterConfig,
        film_transformer_layers: Sequence[int] = DEFAULT_FILM_TRANSFORMER_LAYERS,
        film_bound: float = 3.0,
        routing_diversity_weight: float = 0.0,
        tf_gammas_path: str | Path | None = None,
    ) -> None:
        super().__init__(config, tf_gammas_path=tf_gammas_path)
        router_config.validate()
        names = tuple(str(name).strip().lower() for name in assay_names)
        layers = tuple(int(index) for index in film_transformer_layers)
        if not names or len(set(names)) != len(names):
            raise ValueError("assay_names must be non-empty and unique")
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("film_transformer_layers must be non-empty and unique")
        invalid = [index for index in layers if not 0 <= index < len(self.transformer)]
        if invalid:
            raise ValueError(f"FiLM transformer layer indices are invalid: {invalid}")
        if film_bound <= 0:
            raise ValueError("film_bound must be positive")
        if routing_diversity_weight < 0:
            raise ValueError("routing_diversity_weight cannot be negative")

        self.assay_names = names
        self.router_config = router_config
        self.film_transformer_layers = layers
        self.film_bound = float(film_bound)
        self.routing_diversity_weight = float(routing_diversity_weight)
        self.context_router = SparseTranscriptomeRouter(router_config)

        projector_names = ["post_conv", *(f"transformer_{index}" for index in layers)]
        state_dim = router_config.state_dim
        self.context_pool_queries = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(state_dim))
                for name in projector_names
            }
        )
        self.context_pool_norms = nn.ModuleDict(
            {name: nn.LayerNorm(state_dim) for name in projector_names}
        )
        self.film_projectors = nn.ModuleDict(
            {
                name: nn.Linear(state_dim, self.config.dim * 2)
                for name in projector_names
            }
        )
        for query in self.context_pool_queries.values():
            nn.init.normal_(query, std=0.02)
        for projector in self.film_projectors.values():
            nn.init.zeros_(projector.weight)
            nn.init.zeros_(projector.bias)

        self.assay_head = nn.Sequential(
            nn.Linear(self.config.dim * 2, len(names)),
            nn.Softplus(),
        )
        self._freeze_sequence_prefix = False
        self._frozen_transformer_layer_count = 0

    def encode_context(
        self,
        *,
        gene_features: torch.Tensor,
        protocol_ids: torch.Tensor,
        gene_indices: torch.Tensor | None = None,
        gene_mask: torch.Tensor | None = None,
        return_routing: bool = False,
    ) -> torch.Tensor | SparseTranscriptomeRouterOutput:
        output = self.context_router(
            gene_features,
            protocol_ids,
            gene_indices=gene_indices,
            gene_mask=gene_mask,
            return_routing=return_routing,
        )
        return output if return_routing else output.state_tokens

    def _pool_context(
        self, state_tokens: torch.Tensor, projector_name: str
    ) -> torch.Tensor:
        if state_tokens.ndim != 3:
            raise ValueError(
                "context state tokens must have shape [B,S,D], got "
                f"{tuple(state_tokens.shape)}"
            )
        query = self.context_pool_queries[projector_name]
        scores = torch.einsum("bsd,d->bs", state_tokens, query)
        scores = scores / math.sqrt(float(state_tokens.shape[-1]))
        weights = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bs,bsd->bd", weights, state_tokens)
        return self.context_pool_norms[projector_name](pooled)

    def _apply_film(
        self,
        hidden: torch.Tensor,
        state_tokens: torch.Tensor,
        projector_name: str,
    ) -> torch.Tensor:
        pooled = self._pool_context(state_tokens, projector_name)
        film = self.film_projectors[projector_name](pooled).to(dtype=hidden.dtype)
        raw_gamma, raw_beta = film.chunk(2, dim=-1)
        bound = self.film_bound
        gamma = bound * torch.tanh(raw_gamma / bound)
        beta = bound * torch.tanh(raw_beta / bound)
        return hidden * (1.0 + gamma[:, None, :]) + beta[:, None, :]

    def _forward_transformer_block(
        self,
        block: nn.Sequential,
        hidden: torch.Tensor,
        state_tokens: torch.Tensor | None,
        layer_index: int,
    ) -> torch.Tensor:
        hidden = block[0](hidden)
        feed_forward = block[1].fn
        feed_forward_hidden = feed_forward[0](hidden)
        if state_tokens is not None and layer_index in self.film_transformer_layers:
            feed_forward_hidden = self._apply_film(
                feed_forward_hidden,
                state_tokens,
                f"transformer_{layer_index}",
            )
        for layer in list(feed_forward.children())[1:]:
            feed_forward_hidden = layer(feed_forward_hidden)
        return hidden + feed_forward_hidden

    def _run_transformer(
        self, hidden: torch.Tensor, state_tokens: torch.Tensor | None
    ) -> torch.Tensor:
        for layer_index, block in enumerate(self.transformer):
            if self.use_checkpointing and self.training and torch.is_grad_enabled():

                def block_forward(
                    value: torch.Tensor,
                    states: torch.Tensor,
                    *,
                    current_block: nn.Sequential = block,
                    current_index: int = layer_index,
                ) -> torch.Tensor:
                    return self._forward_transformer_block(
                        current_block, value, states, current_index
                    )

                if state_tokens is None:
                    dummy = hidden.new_zeros((hidden.shape[0], 1, 1))

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
                        block_forward, hidden, state_tokens, use_reentrant=False
                    )
            else:
                hidden = self._forward_transformer_block(
                    block, hidden, state_tokens, layer_index
                )
        return hidden

    def initialize_assay_head_from_human_head(
        self, assay_target_indices: Mapping[str, Sequence[int]]
    ) -> dict[str, Any]:
        """Initialize each compact assay row from pretrained human tracks."""

        if "human" not in self.heads:
            raise ValueError("Pretrained assay initialization requires the human head")
        source = self.heads["human"][0]
        target = self.assay_head[0]
        report: dict[str, Any] = {"source_head": "human", "assays": {}}
        with torch.no_grad():
            for assay_index, assay_name in enumerate(self.assay_names):
                indices = tuple(
                    int(value) for value in assay_target_indices.get(assay_name, ())
                )
                if not indices:
                    raise ValueError(
                        f"No pretrained target indices supplied for {assay_name}"
                    )
                if min(indices) < 0 or max(indices) >= source.out_features:
                    raise ValueError(
                        f"Out-of-range human-head indices for {assay_name}: {indices}"
                    )
                index = torch.as_tensor(indices, device=source.weight.device)
                target.weight[assay_index].copy_(
                    source.weight.index_select(0, index).mean(0)
                )
                target.bias[assay_index].copy_(
                    source.bias.index_select(0, index).mean(0)
                )
                report["assays"][assay_name] = {
                    "source_track_count": len(indices)
                }
        return report

    def configure_stage1_trainability(
        self, *, unfreeze_last_transformer_blocks: int = 0
    ) -> dict[str, Any]:
        """Train router, FiLM and assay head while freezing Enformer by default."""

        count = int(unfreeze_last_transformer_blocks)
        if not 0 <= count <= len(self.transformer):
            raise ValueError(
                "unfreeze_last_transformer_blocks must be between zero and "
                f"{len(self.transformer)}"
            )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module in (
            self.context_router,
            self.context_pool_queries,
            self.context_pool_norms,
            self.film_projectors,
            self.assay_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        if count:
            for block in self.transformer[-count:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
        self._freeze_sequence_prefix = True
        self._frozen_transformer_layer_count = len(self.transformer) - count
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        return {
            "stage": 1 if count == 0 else 2,
            "unfreeze_last_transformer_blocks": count,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "trainable_fraction": trainable / max(total, 1),
        }

    def enforce_frozen_backbone_eval(self) -> None:
        """Keep frozen sequence modules deterministic after ``model.train()``."""

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
        *,
        gene_features: torch.Tensor | None = None,
        protocol_ids: torch.Tensor | None = None,
        gene_indices: torch.Tensor | None = None,
        gene_mask: torch.Tensor | None = None,
        context_state_tokens: torch.Tensor | None = None,
        context_batch_index: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        target_length: int | None = None,
        disable_context: bool = False,
        return_routing: bool = False,
        return_embeddings: bool = False,
        return_only_embeddings: bool = False,
    ) -> (
        SparseTranscriptomeEnformerOutput
        | torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
    ):
        router_output: SparseTranscriptomeRouterOutput | None = None
        if not disable_context and context_state_tokens is None:
            if gene_features is None or protocol_ids is None:
                raise ValueError(
                    "gene_features and protocol_ids are required unless precomputed "
                    "context_state_tokens are supplied"
                )
            router_output = self.context_router(
                gene_features,
                protocol_ids,
                gene_indices=gene_indices,
                gene_mask=gene_mask,
                return_routing=return_routing,
            )
            context_state_tokens = router_output.state_tokens

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

        states = None if disable_context else context_state_tokens
        if states is not None:
            if context_batch_index is not None:
                index = context_batch_index.to(device=states.device, dtype=torch.long)
                if tuple(index.shape) != (hidden.shape[0],):
                    raise ValueError(
                        "context_batch_index must contain one row per DNA sequence"
                    )
                if torch.any(index < 0) or torch.any(index >= states.shape[0]):
                    raise ValueError("context_batch_index is outside the context batch")
                states = states.index_select(0, index)
            elif states.shape[0] == 1 and hidden.shape[0] > 1:
                states = states.expand(hidden.shape[0], -1, -1)
            if states.shape[0] != hidden.shape[0]:
                raise ValueError("Context-state batch size does not match sequence batch")
            states = states.to(device=hidden.device, dtype=hidden.dtype)
            hidden = self._apply_film(hidden, states, "post_conv")
        hidden = self._run_transformer(hidden, states)
        hidden = self.crop_final(hidden)
        hidden = self.final_pointwise(hidden)
        if return_only_embeddings:
            return hidden[0] if no_batch else hidden

        logits = self.assay_head(hidden)
        prediction_loss = None
        total_loss = None
        diversity_loss = (
            router_output.diversity_loss if router_output is not None else None
        )
        if labels is not None:
            labels = torch.as_tensor(labels, device=logits.device)
            if no_batch and labels.ndim == 2:
                labels = labels.unsqueeze(0)
            labels = labels.float()
            if tuple(labels.shape) != tuple(logits.shape):
                raise ValueError(
                    f"labels shape {tuple(labels.shape)} does not match logits "
                    f"{tuple(logits.shape)}"
                )
            if target_mask is None:
                raise ValueError("target_mask is required when labels are supplied")
            prediction_loss = masked_poisson_loss(logits, labels, target_mask)
            total_loss = prediction_loss
            if diversity_loss is not None and self.routing_diversity_weight > 0.0:
                total_loss = total_loss + self.routing_diversity_weight * diversity_loss
        if no_batch:
            logits = logits[0]
            hidden = hidden[0]
        if return_embeddings:
            return logits, hidden
        return SparseTranscriptomeEnformerOutput(
            loss=total_loss,
            logits=logits,
            hidden_states=hidden,
            prediction_loss=prediction_loss,
            routing_diversity_loss=diversity_loss,
            active_gene_counts=(
                router_output.active_gene_counts if router_output is not None else None
            ),
            routing_entropy=(
                router_output.routing_entropy if router_output is not None else None
            ),
            routing_weights=(
                router_output.routing_weights if router_output is not None else None
            ),
        )


__all__ = [
    "SparseTranscriptomeContextEnformer",
    "SparseTranscriptomeEnformerOutput",
]
