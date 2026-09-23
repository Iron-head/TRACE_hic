#!/usr/bin/env python3
"""Reference-free full-transcriptome routing with sparse latent attention.

The router treats a transcriptome as an unordered set of gene identities and
single-sample expression features.  Every valid gene is scored by every
learned state query.  ``entmax15`` then assigns exact zero weight to genes that
are not useful for a query, avoiding a preprocessing-time hard Top-K panel.

No cross-sample mean, variance, marker list, pathway list, or trans-regulator
list is used by this module.  The gene vocabulary is fixed, but the effective
gene subset is sample-dependent and learned end-to-end from the downstream
prediction objective.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import torch
from torch import nn
from torch.nn import functional as F


RoutingActivation = Literal["entmax15", "softmax"]


class _Entmax15Function(torch.autograd.Function):
    """Exact 1.5-entmax with an analytical, support-aware backward pass."""

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        valid: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        input_dtype = logits.dtype
        values = logits.float().masked_fill(~valid, -1.0e4)
        values = values - values.amax(dim=dim, keepdim=True)
        values = values / 2.0

        sorted_values, _ = torch.sort(values, dim=dim, descending=True)
        dimension_size = values.shape[dim]
        rho_shape = [1] * values.ndim
        rho_shape[dim] = dimension_size
        rho = torch.arange(
            1,
            dimension_size + 1,
            device=values.device,
            dtype=values.dtype,
        ).view(rho_shape)
        cumulative_mean = sorted_values.cumsum(dim=dim) / rho
        cumulative_mean_sq = sorted_values.square().cumsum(dim=dim) / rho
        support_variance = rho * (
            cumulative_mean_sq - cumulative_mean.square()
        )
        delta = (1.0 - support_variance) / rho
        taus = cumulative_mean - torch.sqrt(delta.clamp_min(0.0))
        support_size = (taus <= sorted_values).sum(dim=dim, keepdim=True)
        support_size = support_size.clamp_min(1)
        tau_star = torch.gather(
            taus,
            dim,
            support_size.to(dtype=torch.long) - 1,
        )
        probabilities = (values - tau_star).clamp_min(0.0).square()
        probabilities = probabilities.masked_fill(~valid, 0.0)
        ctx.dim = dim
        ctx.input_dtype = input_dtype
        ctx.save_for_backward(probabilities, valid)
        return probabilities.to(dtype=input_dtype)

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        probabilities, valid = ctx.saved_tensors
        # For alpha=1.5, 1 / g''(p) = sqrt(p).  This analytical JVP avoids
        # differentiating through sort, support selection, clamp and sqrt.
        inverse_curvature = probabilities.sqrt()
        weighted_gradient = grad_output.float() * inverse_curvature
        normalizer = inverse_curvature.sum(dim=ctx.dim, keepdim=True)
        projection = weighted_gradient.sum(
            dim=ctx.dim, keepdim=True
        ) / normalizer.clamp_min(torch.finfo(probabilities.dtype).tiny)
        grad_logits = weighted_gradient - projection * inverse_curvature
        grad_logits = grad_logits.masked_fill(~valid, 0.0)
        return grad_logits.to(dtype=ctx.input_dtype), None, None


def entmax15(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Differentiable 1.5-entmax with optional boolean masking.

    The implementation follows the sorting-based threshold solution for
    alpha=1.5.  Threshold computation is performed in fp32 for numerical
    stability under mixed precision.  Unlike softmax, valid low-scoring
    entries can receive an exact probability of zero.
    """

    if not logits.is_floating_point():
        raise TypeError("entmax15 logits must be floating point")
    if logits.numel() == 0:
        raise ValueError("entmax15 cannot operate on an empty tensor")
    ndim = logits.ndim
    normalized_dim = dim if dim >= 0 else ndim + dim
    if not 0 <= normalized_dim < ndim:
        raise IndexError(f"dim={dim} is invalid for a {ndim}-D tensor")

    valid: torch.Tensor
    if mask is not None:
        valid = mask.to(device=logits.device, dtype=torch.bool)
        try:
            valid = torch.broadcast_to(valid, logits.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} cannot broadcast to "
                f"logits shape {tuple(logits.shape)}"
            ) from exc
        if not valid.any(dim=normalized_dim).all():
            raise ValueError("Every entmax row must contain at least one valid entry")
    else:
        valid = torch.ones_like(logits, dtype=torch.bool)
    return _Entmax15Function.apply(logits, valid, normalized_dim)


@dataclass(frozen=True)
class SparseTranscriptomeRouterConfig:
    """Configuration for full-transcriptome sparse latent routing."""

    gene_count: int
    feature_dim: int
    protocol_count: int = 2
    routing_dim: int = 64
    state_dim: int = 256
    state_token_count: int = 32
    state_attention_heads: int = 8
    state_layers: int = 2
    dropout: float = 0.1
    gene_token_dropout: float = 0.05
    routing_activation: RoutingActivation = "entmax15"

    @classmethod
    def from_dict(
        cls, values: Mapping[str, Any]
    ) -> "SparseTranscriptomeRouterConfig":
        fields = cls.__dataclass_fields__
        return cls(**{key: values[key] for key in fields if key in values})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        for name in (
            "gene_count",
            "feature_dim",
            "protocol_count",
            "routing_dim",
            "state_dim",
            "state_token_count",
            "state_attention_heads",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.state_layers < 0:
            raise ValueError("state_layers cannot be negative")
        if self.state_dim % self.state_attention_heads:
            raise ValueError("state_dim must be divisible by state_attention_heads")
        for name in ("dropout", "gene_token_dropout"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0,1)")
        if self.routing_activation not in {"entmax15", "softmax"}:
            raise ValueError(
                "routing_activation must be either 'entmax15' or 'softmax'"
            )


@dataclass
class SparseTranscriptomeRouterOutput:
    """Latent state plus optional gene-level routing diagnostics."""

    state_tokens: torch.Tensor
    routing_weights: torch.Tensor | None = None
    active_gene_counts: torch.Tensor | None = None
    routing_entropy: torch.Tensor | None = None
    diversity_loss: torch.Tensor | None = None


class SparseTranscriptomeRouter(nn.Module):
    """Compress all valid genes into sample-specific sparse state tokens.

    ``gene_mask`` marks genes that are absent from the quantification
    vocabulary.  A measured gene with TPM=0 remains valid and should be
    represented through the input features rather than masked out.
    """

    def __init__(self, config: SparseTranscriptomeRouterConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        route_dim = config.routing_dim
        state_dim = config.state_dim

        self.gene_embedding = nn.Embedding(config.gene_count, route_dim)
        self.feature_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, route_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(route_dim * 2, route_dim * 2),
        )
        self.gene_norm = nn.LayerNorm(route_dim)
        self.key_projection = nn.Linear(route_dim, route_dim, bias=False)
        self.key_norm = nn.LayerNorm(route_dim)
        self.value_projection = nn.Linear(route_dim, state_dim, bias=False)

        self.routing_queries = nn.Parameter(
            torch.empty(config.state_token_count, route_dim)
        )
        self.protocol_query_embedding = nn.Embedding(
            config.protocol_count, route_dim
        )
        self.query_norm = nn.LayerNorm(route_dim)
        self.state_residuals = nn.Parameter(
            torch.empty(config.state_token_count, state_dim)
        )
        self.protocol_state_embedding = nn.Embedding(
            config.protocol_count, state_dim
        )
        self.state_norm = nn.LayerNorm(state_dim)
        self.state_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=state_dim,
                    nhead=config.state_attention_heads,
                    dim_feedforward=state_dim * 4,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.state_layers)
            ]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.gene_embedding.weight, std=0.02)
        nn.init.normal_(self.routing_queries, std=0.02)
        nn.init.normal_(self.state_residuals, std=0.02)
        nn.init.normal_(self.protocol_query_embedding.weight, std=0.02)
        nn.init.normal_(self.protocol_state_embedding.weight, std=0.02)

    def _prepare_inputs(
        self,
        gene_features: torch.Tensor,
        gene_indices: torch.Tensor | None,
        gene_mask: torch.Tensor | None,
        protocol_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if gene_features.ndim != 3:
            raise ValueError(
                "gene_features must have shape [B,G,F], got "
                f"{tuple(gene_features.shape)}"
            )
        batch_size, token_count, feature_dim = gene_features.shape
        if feature_dim != self.config.feature_dim:
            raise ValueError(
                f"Expected {self.config.feature_dim} gene features, got {feature_dim}"
            )
        if gene_indices is None:
            if token_count != self.config.gene_count:
                raise ValueError(
                    "gene_indices can be omitted only when gene_features contains "
                    "the complete fixed gene vocabulary"
                )
            gene_indices = torch.arange(
                self.config.gene_count, device=gene_features.device
            ).unsqueeze(0).expand(batch_size, -1)
        if tuple(gene_indices.shape) != (batch_size, token_count):
            raise ValueError("gene_indices shape does not match gene_features")
        indices = gene_indices.to(device=gene_features.device, dtype=torch.long)
        if gene_mask is None:
            mask = torch.ones(
                (batch_size, token_count),
                device=gene_features.device,
                dtype=torch.bool,
            )
        else:
            if tuple(gene_mask.shape) != (batch_size, token_count):
                raise ValueError("gene_mask shape does not match gene_features")
            mask = gene_mask.to(device=gene_features.device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("Every sample must contain at least one valid gene")
        active_indices = indices[mask]
        if torch.any(active_indices < 0) or torch.any(
            active_indices >= self.config.gene_count
        ):
            raise ValueError("Active gene indices are outside the fixed vocabulary")
        # Structural padding may use an out-of-range sentinel such as -1.
        # Replace it before the embedding lookup; the corresponding token is
        # still removed by the boolean mask.
        indices = indices.masked_fill(~mask, 0)
        if tuple(protocol_ids.shape) != (batch_size,):
            raise ValueError("protocol_ids must have shape [B]")
        protocols = protocol_ids.to(device=gene_features.device, dtype=torch.long)
        if torch.any(protocols < 0) or torch.any(
            protocols >= self.config.protocol_count
        ):
            raise ValueError("protocol_ids are outside the configured vocabulary")
        features = gene_features.to(dtype=self.gene_embedding.weight.dtype)
        if not torch.isfinite(features[mask]).all():
            raise ValueError("Valid gene features contain non-finite values")
        features = features.masked_fill(~mask.unsqueeze(-1), 0.0)
        return features, indices, mask, protocols

    def _drop_gene_tokens(self, mask: torch.Tensor) -> torch.Tensor:
        probability = self.config.gene_token_dropout
        if not self.training or probability <= 0.0:
            return mask
        keep = torch.rand(mask.shape, device=mask.device) >= probability
        dropped = mask & keep
        empty_rows = ~dropped.any(dim=1)
        if empty_rows.any():
            first_active = mask.to(torch.int64).argmax(dim=1)
            dropped[empty_rows, first_active[empty_rows]] = True
        return dropped

    @staticmethod
    def routing_diversity_loss(weights: torch.Tensor) -> torch.Tensor:
        """Penalize different latent queries for using identical gene mixtures."""

        if weights.ndim != 3:
            raise ValueError("routing weights must have shape [B,S,G]")
        state_count = weights.shape[1]
        if state_count <= 1:
            return weights.sum() * 0.0
        normalized = F.normalize(weights.float(), p=2, dim=-1, eps=1e-8)
        similarity = normalized @ normalized.transpose(1, 2)
        identity = torch.eye(
            state_count, device=weights.device, dtype=similarity.dtype
        ).unsqueeze(0)
        off_diagonal = (similarity - identity).square()
        return off_diagonal.sum() / (
            weights.shape[0] * state_count * (state_count - 1)
        )

    def forward(
        self,
        gene_features: torch.Tensor,
        protocol_ids: torch.Tensor,
        *,
        gene_indices: torch.Tensor | None = None,
        gene_mask: torch.Tensor | None = None,
        return_routing: bool = False,
    ) -> SparseTranscriptomeRouterOutput:
        features, indices, mask, protocols = self._prepare_inputs(
            gene_features, gene_indices, gene_mask, protocol_ids
        )
        route_mask = self._drop_gene_tokens(mask)

        gene_identity = self.gene_embedding(indices)
        raw_gamma, beta = self.feature_encoder(features).chunk(2, dim=-1)
        gamma = 0.5 * torch.tanh(raw_gamma)
        gene_tokens = self.gene_norm(gene_identity * (1.0 + gamma) + beta)
        gene_tokens = gene_tokens.masked_fill(~route_mask.unsqueeze(-1), 0.0)

        keys = self.key_norm(self.key_projection(gene_tokens))
        values = self.value_projection(gene_tokens)
        queries = self.routing_queries.unsqueeze(0).expand(
            features.shape[0], -1, -1
        )
        queries = queries + self.protocol_query_embedding(protocols).unsqueeze(1)
        queries = self.query_norm(queries)
        scores = torch.einsum("bsd,bgd->bsg", queries, keys)
        scores = scores / math.sqrt(float(self.config.routing_dim))
        expanded_mask = route_mask.unsqueeze(1)
        if self.config.routing_activation == "entmax15":
            weights = entmax15(scores, expanded_mask, dim=-1)
        else:
            weights = torch.softmax(
                scores.masked_fill(~expanded_mask, -torch.inf), dim=-1
            )

        routed = torch.einsum(
            "bsg,bgd->bsd", weights.to(dtype=values.dtype), values
        )
        residual = self.state_residuals.unsqueeze(0)
        residual = residual + self.protocol_state_embedding(protocols).unsqueeze(1)
        states = self.state_norm(routed + residual)
        for block in self.state_blocks:
            states = block(states)

        active_counts = (weights > 0).sum(dim=-1)
        entropy = -(
            weights.float()
            * weights.float().clamp_min(torch.finfo(torch.float32).tiny).log()
        ).sum(dim=-1)
        diversity = self.routing_diversity_loss(weights)
        return SparseTranscriptomeRouterOutput(
            state_tokens=states,
            routing_weights=weights if return_routing else None,
            active_gene_counts=active_counts,
            routing_entropy=entropy,
            diversity_loss=diversity,
        )


__all__ = [
    "SparseTranscriptomeRouter",
    "SparseTranscriptomeRouterConfig",
    "SparseTranscriptomeRouterOutput",
    "entmax15",
]
