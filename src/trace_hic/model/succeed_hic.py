"""Hi-C adapters around the native SUCCEED sequence representation.

The pretrained SUCCEED checkpoint used here was trained with 128 bp output
tokens.  A 2 Mb Hi-C window contains 16,384 of those tokens, so running the
full dense SUCCEED transformer over the whole window is impractical.  This
module keeps the native 131,072 bp SUCCEED context by evaluating sixteen
chunks, concatenates their 128 bp embeddings, and then performs local
attention pooling (64 x 128 bp = 8,192 bp) for the Hi-C model.
"""

from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class AttentionPool1D(nn.Module):
    """Pool fixed-size groups of sequence tokens with learned attention.

    Input and output use channel-first tensors:

    ``[batch, dim, length] -> [batch, dim, length // pool_size]``.

    A single learned query is applied independently to every local group.
    Relative positions inside a group and absolute positions of pooled tokens
    are learned separately.  The local formulation avoids constructing a
    quadratic attention matrix over all 16,384 128-bp tokens.
    """

    def __init__(
        self,
        dim: int,
        pool_size: int = 64,
        num_heads: int = 8,
        max_output_tokens: Optional[int] = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if pool_size <= 0:
            raise ValueError(f"pool_size must be positive, got {pool_size}")

        self.dim = int(dim)
        self.pool_size = int(pool_size)
        self.max_output_tokens = max_output_tokens
        self.query = nn.Parameter(torch.empty(1, 1, self.dim))
        self.relative_position = nn.Parameter(
            torch.empty(1, self.pool_size, self.dim)
        )
        self.input_norm = nn.LayerNorm(self.dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(self.dim)
        if max_output_tokens is None:
            self.output_position = None
        else:
            self.output_position = nn.Parameter(
                torch.zeros(1, int(max_output_tokens), self.dim)
            )

        nn.init.normal_(self.query, std=0.02)
        nn.init.normal_(self.relative_position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"AttentionPool1D expects [B,C,L], got {tuple(x.shape)}")
        batch, dim, length = x.shape
        if dim != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {dim}")
        if length % self.pool_size != 0:
            raise ValueError(
                f"Token length {length} must be divisible by pool_size={self.pool_size}"
            )

        output_tokens = length // self.pool_size
        if (
            self.output_position is not None
            and output_tokens > self.output_position.shape[1]
        ):
            raise ValueError(
                f"Need {output_tokens} output positions, but only "
                f"{self.output_position.shape[1]} were allocated"
            )

        # [B,C,L] -> [B,N,G,C] -> [B*N,G,C]
        groups = x.transpose(1, 2).contiguous()
        groups = groups.reshape(batch, output_tokens, self.pool_size, dim)
        groups = groups + self.relative_position.unsqueeze(1)
        groups = groups.reshape(batch * output_tokens, self.pool_size, dim)
        groups = self.input_norm(groups)

        query = self.input_norm(self.query)
        query = query.expand(batch * output_tokens, -1, -1)
        pooled, _ = self.attention(query, groups, groups, need_weights=False)
        pooled = pooled.reshape(batch, output_tokens, dim)

        if self.output_position is not None:
            pooled = pooled + self.output_position[:, :output_tokens]
        pooled = self.output_norm(pooled)
        return pooled.transpose(1, 2).contiguous()


class ChunkedSucceedEncoder(nn.Module):
    """Run native-length SUCCEED chunks and concatenate 128 bp embeddings."""

    def __init__(
        self,
        succeed_model: nn.Module,
        chunk_size: int = 131072,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.base_model = succeed_model
        self.chunk_size = int(chunk_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B,C,L] DNA input, got {tuple(x.shape)}")
        length = x.shape[-1]
        if length % self.chunk_size != 0:
            raise ValueError(
                f"Input length {length} must be divisible by chunk_size={self.chunk_size}"
            )

        embeddings = []
        for start in range(0, length, self.chunk_size):
            chunk = x[..., start : start + self.chunk_size]
            # EncoderSplit intentionally keeps the frozen SUCCEED module in
            # eval() mode so its BatchNorm statistics do not drift.  Use the
            # autograd state, rather than self.training, to decide whether
            # checkpointing is needed.
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                chunk_embedding = checkpoint(
                    self.base_model,
                    chunk,
                    use_reentrant=False,
                )
            else:
                chunk_embedding = self.base_model(chunk)
            if chunk_embedding.ndim != 3:
                raise ValueError(
                    "SUCCEED must be constructed with return_emb=True; "
                    f"got output shape {tuple(chunk_embedding.shape)}"
                )
            embeddings.append(chunk_embedding)

        return torch.cat(embeddings, dim=-1)
