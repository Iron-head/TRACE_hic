"""Efficient full-window context for native 128-bp genomic tokens."""

from __future__ import annotations

import math

import torch
from torch import nn


def _sinusoidal_position_table(max_tokens: int, hidden: int) -> torch.Tensor:
    position = torch.arange(max_tokens, dtype=torch.float32).unsqueeze(1)
    divisor = torch.exp(
        torch.arange(0, hidden, 2, dtype=torch.float32)
        * (-math.log(10000.0) / hidden)
    )
    table = torch.zeros(1, max_tokens, hidden, dtype=torch.float32)
    table[0, :, 0::2] = torch.sin(position * divisor)
    table[0, :, 1::2] = torch.cos(position * divisor)
    return table


class GlobalLatentTransformerBlock(nn.Module):
    """Pre-norm Transformer block operating only on global latent tokens."""

    def __init__(
        self,
        hidden: int = 256,
        num_heads: int = 8,
        feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward, hidden),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        context, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + self.attention_dropout(context)
        return tokens + self.ffn(self.ffn_norm(tokens))


class NativeGlobalTransformer(nn.Module):
    """Inject 2-Mb global context before native-token attention pooling.

    Learned global latents first read all 16,384 native 128-bp tokens.  A
    small dense Transformer models relations among those latents, after which
    every native token queries the globally contextualized latents.  This has
    linear cross-attention cost in the native sequence length and avoids a
    prohibitively large 16,384 x 16,384 attention matrix.
    """

    def __init__(
        self,
        hidden: int = 256,
        num_latents: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        feedforward: int = 512,
        dropout: float = 0.1,
        gate_init: float = 0.05,
        max_tokens: int = 16384,
    ) -> None:
        super().__init__()
        if num_latents <= 0 or num_layers <= 0:
            raise ValueError("num_latents and num_layers must be positive.")
        if hidden % num_heads != 0:
            raise ValueError("hidden must be divisible by num_heads.")
        self.hidden = int(hidden)
        self.num_latents = int(num_latents)
        self.max_tokens = int(max_tokens)
        self.global_latents = nn.Parameter(
            torch.empty(self.num_latents, self.hidden)
        )
        nn.init.normal_(self.global_latents, mean=0.0, std=0.02)

        self.register_buffer(
            "position_table",
            _sinusoidal_position_table(self.max_tokens, self.hidden),
            persistent=False,
        )
        self.read_query_norm = nn.LayerNorm(hidden)
        self.read_key_norm = nn.LayerNorm(hidden)
        self.read_value_norm = nn.LayerNorm(hidden)
        self.read_attention = nn.MultiheadAttention(
            hidden,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.read_dropout = nn.Dropout(dropout)

        self.transformer = nn.ModuleList(
            [
                GlobalLatentTransformerBlock(
                    hidden=hidden,
                    num_heads=num_heads,
                    feedforward=feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.latent_output_norm = nn.LayerNorm(hidden)

        self.write_query_norm = nn.LayerNorm(hidden)
        self.write_key_norm = nn.LayerNorm(hidden)
        self.write_attention = nn.MultiheadAttention(
            hidden,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.write_dropout = nn.Dropout(dropout)
        self.write_gate = nn.Parameter(
            torch.full((hidden,), float(gate_init))
        )

    def forward(self, native_tokens: torch.Tensor) -> torch.Tensor:
        if native_tokens.ndim != 3 or native_tokens.shape[-1] != self.hidden:
            raise ValueError(
                "Native tokens must have shape [batch, tokens, hidden], got "
                f"{tuple(native_tokens.shape)}."
            )
        token_count = native_tokens.shape[1]
        if token_count > self.max_tokens:
            raise ValueError(
                f"Received {token_count} native tokens, maximum is "
                f"{self.max_tokens}."
            )
        positions = self.position_table[:, :token_count].to(
            dtype=native_tokens.dtype
        )
        positioned_tokens = native_tokens + positions
        latents = self.global_latents.unsqueeze(0).expand(
            native_tokens.shape[0], -1, -1
        )
        read_context, _ = self.read_attention(
            query=self.read_query_norm(latents),
            key=self.read_key_norm(positioned_tokens),
            value=self.read_value_norm(native_tokens),
            need_weights=False,
        )
        latents = latents + self.read_dropout(read_context)
        for layer in self.transformer:
            latents = layer(latents)
        latents = self.latent_output_norm(latents)

        write_context, _ = self.write_attention(
            query=self.write_query_norm(positioned_tokens),
            key=self.write_key_norm(latents),
            value=latents,
            need_weights=False,
        )
        gate = torch.tanh(self.write_gate).view(1, 1, -1)
        return native_tokens + gate * self.write_dropout(write_context)
