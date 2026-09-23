#!/usr/bin/env python3
"""Standalone PyTorch implementation of the Enformer architecture.

This module is deliberately independent from Hugging Face.  It contains the
neural network layers used by the local Enformer implementation, but uses
plain ``torch.nn.Module`` objects, a small JSON-serializable configuration,
and a lightweight output dataclass.

The parameter names of the trunk and output heads follow the existing
``enformer_pytorch`` implementation.  That makes a local PyTorch
``state_dict`` usable without downloading a repository or calling
``from_pretrained``.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential

from .losses import masked_poisson_loss


SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896


def _default_output_heads() -> dict[str, int]:
    return {"human": 5313, "mouse": 1643}


@dataclass
class PureEnformerConfig:
    """JSON-friendly Enformer configuration.

    The defaults match the public Enformer architecture.  ``use_tf_gamma``
    is false by default because the analytic gamma positional features are
    self-contained; set it to true when reproducing an official checkpoint
    that was trained with the precomputed TensorFlow gamma table.
    """

    dim: int = 1536
    depth: int = 11
    heads: int = 8
    output_heads: dict[str, int] = field(default_factory=_default_output_heads)
    target_length: int = 896
    attn_dim_key: int = 64
    dropout_rate: float = 0.4
    attn_dropout: float = 0.05
    pos_dropout: float = 0.01
    use_checkpointing: bool = False
    use_convnext: bool = False
    num_downsamples: int = 7
    dim_divisible_by: int = 128
    use_tf_gamma: bool = False

    # These fields are ignored by the base Enformer and used by the SUCCEED
    # context branch. Keeping them in the same config makes full checkpoints
    # self-describing and still allows a plain Enformer JSON to be loaded.
    context_gene_count: int = 2891
    context_hidden_dim: int = 512
    context_dropout: float = 0.1

    def __post_init__(self) -> None:
        self.dim = int(self.dim)
        self.depth = int(self.depth)
        self.heads = int(self.heads)
        self.target_length = int(self.target_length)
        self.attn_dim_key = int(self.attn_dim_key)
        self.num_downsamples = int(self.num_downsamples)
        self.dim_divisible_by = int(self.dim_divisible_by)
        self.context_gene_count = int(self.context_gene_count)
        self.context_hidden_dim = int(self.context_hidden_dim)
        self.output_heads = {
            str(name): int(features) for name, features in dict(self.output_heads).items()
        }

        if self.dim <= 0 or self.dim % 2 != 0:
            raise ValueError(f"dim must be a positive even integer, got {self.dim}")
        if self.depth <= 0:
            raise ValueError(f"depth must be positive, got {self.depth}")
        if self.heads <= 0 or self.dim % self.heads != 0:
            raise ValueError(f"heads must divide dim; got dim={self.dim}, heads={self.heads}")
        if (self.dim // self.heads) % 6 != 0:
            raise ValueError(
                "dim // heads must be divisible by 6 for Enformer's relative positional features; "
                f"got dim={self.dim}, heads={self.heads}"
            )
        if self.attn_dim_key <= 0:
            raise ValueError(f"attn_dim_key must be positive, got {self.attn_dim_key}")
        if self.num_downsamples < 2:
            raise ValueError(
                "num_downsamples must be at least 2; the Enformer convolution tower "
                "needs at least one downsampling block"
            )
        if self.dim_divisible_by <= 0:
            raise ValueError(f"dim_divisible_by must be positive, got {self.dim_divisible_by}")
        if self.target_length == 0 or self.target_length < -1:
            raise ValueError("target_length must be -1 or a positive integer")
        if not self.output_heads or any(features <= 0 for features in self.output_heads.values()):
            raise ValueError(f"output_heads must contain positive sizes, got {self.output_heads}")
        if self.context_gene_count <= 0 or self.context_hidden_dim <= 0:
            raise ValueError("context_gene_count and context_hidden_dim must be positive")
        for name, value in (
            ("dropout_rate", self.dropout_rate),
            ("attn_dropout", self.attn_dropout),
            ("pos_dropout", self.pos_dropout),
            ("context_dropout", self.context_dropout),
        ):
            value = float(value)
            setattr(self, name, value)
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1), got {value}")

        self.use_checkpointing = bool(self.use_checkpointing)
        self.use_convnext = bool(self.use_convnext)
        self.use_tf_gamma = bool(self.use_tf_gamma)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PureEnformerConfig":
        """Create a config while ignoring unrelated HF metadata fields."""

        if not isinstance(values, Mapping):
            raise TypeError(f"Config must be a mapping, got {type(values)!r}")
        nested = values.get("model_config")
        if isinstance(nested, Mapping):
            values = nested
        known = {item.name for item in fields(cls)}
        filtered = {key: value for key, value in values.items() if key in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, path: str | Path) -> "PureEnformerConfig":
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Enformer config JSON not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
        return cls.from_dict(values)

    def save_json(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


@dataclass
class PureEnformerOutput:
    """Small output object replacing ``transformers.ModelOutput``."""

    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


def _exists(value: Any) -> bool:
    return value is not None


def _default(value: Any, fallback: Any) -> Any:
    return value if _exists(value) else fallback


def _exponential_linspace_int(
    start: int,
    end: int,
    num: int,
    divisible_by: int = 1,
) -> list[int]:
    if num <= 0:
        raise ValueError(f"num must be positive, got {num}")
    if num == 1:
        return [int(round(start / divisible_by) * divisible_by)]

    def _round(value: float) -> int:
        return int(round(value / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**index) for index in range(num)]


def _log(value: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return torch.log(value.clamp(min=eps))


def _maybe_sync_batchnorm(is_distributed: bool | None = None):
    if is_distributed is None:
        is_distributed = bool(
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        )
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d


def _get_positional_features_exponential(
    positions: torch.Tensor,
    features: int,
    seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(
        3.0,
        max_range,
        features,
        device=positions.device,
    )
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.0) / half_life * positions).to(dtype)


def _get_positional_features_central_mask(
    positions: torch.Tensor,
    features: int,
    seq_len: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    del seq_len
    center_widths = 2 ** torch.arange(1, features + 1, device=positions.device).to(dtype)
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).to(dtype)


def _gamma_pdf(
    value: torch.Tensor,
    concentration: torch.Tensor,
    rate: torch.Tensor,
) -> torch.Tensor:
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, value) - rate * value
    log_normalization = torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def _get_positional_features_gamma(
    positions: torch.Tensor,
    features: int,
    seq_len: int,
    stddev: float | None = None,
    start_mean: float | None = None,
    eps: float = 1e-8,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if stddev is None:
        stddev = seq_len / (2 * features)
    if start_mean is None:
        start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device=positions.device)[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev**2
    probabilities = _gamma_pdf(
        positions.to(dtype).abs()[..., None], concentration, rate
    )
    probabilities = probabilities + eps
    return (probabilities / torch.amax(probabilities, dim=-1, keepdim=True)).to(dtype)


def _get_positional_embed(
    seq_len: int,
    feature_size: int,
    device: torch.device,
    use_tf_gamma: bool,
    tf_gammas: torch.Tensor | None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    distances = torch.arange(-seq_len + 1, seq_len, device=device)
    if use_tf_gamma:
        if seq_len != 1536:
            raise ValueError("The precomputed TensorFlow gamma table requires sequence length 1536")
        if tf_gammas is None:
            raise RuntimeError("use_tf_gamma=True but no local TensorFlow gamma table was loaded")
    num_components = 6
    if feature_size % num_components != 0:
        raise ValueError(
            f"feature size {feature_size} is not divisible by positional component count {num_components}"
        )

    num_basis_per_class = feature_size // num_components
    embeddings = [
        _get_positional_features_exponential(
            distances, num_basis_per_class, seq_len, dtype=dtype
        ),
        _get_positional_features_central_mask(
            distances, num_basis_per_class, seq_len, dtype=dtype
        ),
    ]
    if use_tf_gamma:
        gamma = tf_gammas.to(device=device, dtype=dtype)
        expected_shape = (2 * seq_len - 1, num_basis_per_class)
        if tuple(gamma.shape) != expected_shape:
            raise ValueError(
                "TensorFlow gamma table shape does not match the current attention layer: "
                f"got {tuple(gamma.shape)}, expected {expected_shape}"
            )
        embeddings.append(gamma)
    else:
        embeddings.append(
            _get_positional_features_gamma(
                distances, num_basis_per_class, seq_len, dtype=dtype
            )
        )

    embeddings = torch.cat(embeddings, dim=-1)
    embeddings = torch.cat(
        (embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1
    )
    return embeddings.to(dtype)


def _relative_shift(value: torch.Tensor) -> torch.Tensor:
    to_pad = torch.zeros_like(value[..., :1])
    value = torch.cat((to_pad, value), dim=-1)
    _, heads, t1, t2 = value.shape
    value = value.reshape(-1, heads, t2, t1)
    value = value[:, :, 1:, :]
    value = value.reshape(-1, heads, t1, t2 - 1)
    return value[..., : ((t2 + 1) // 2)]


class _Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.fn = module

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return self.fn(x, **kwargs) + x


class _GELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(1.702 * x) * x


class _AttentionPool(nn.Module):
    def __init__(self, dim: int, pool_size: int = 2) -> None:
        super().__init__()
        self.pool_size = int(pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)
        nn.init.dirac_(self.to_attn_logits.weight)
        with torch.no_grad():
            self.to_attn_logits.weight.mul_(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, sequence_length = x.shape
        remainder = sequence_length % self.pool_size
        needs_padding = remainder > 0
        if needs_padding:
            x = F.pad(x, (0, remainder), value=0)
            mask = torch.zeros(
                (batch, 1, sequence_length), dtype=torch.bool, device=x.device
            )
            mask = F.pad(mask, (0, remainder), value=True)

        padded_length = x.shape[-1]
        x = x.reshape(batch, x.shape[1], padded_length // self.pool_size, self.pool_size)
        logits = self.to_attn_logits(x)
        if needs_padding:
            mask = mask.reshape(batch, 1, padded_length // self.pool_size, self.pool_size)
            mask_value = -torch.finfo(logits.dtype).max
            logits = logits.masked_fill(mask, mask_value)
        attention = logits.softmax(dim=-1)
        return (x * attention).sum(dim=-1)


class _TargetLengthCrop(nn.Module):
    def __init__(self, target_length: int) -> None:
        super().__init__()
        self.target_length = int(target_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length, target_length = x.shape[-2], self.target_length
        if target_length == -1:
            return x
        if sequence_length < target_length:
            raise ValueError(
                f"sequence length {sequence_length} is less than target length {target_length}"
            )
        trim = (target_length - sequence_length) // 2
        if trim == 0:
            return x
        return x[:, -trim:trim]


def _conv_block(
    dim: int,
    dim_out: int | None = None,
    kernel_size: int = 1,
    is_distributed: bool | None = None,
) -> nn.Sequential:
    batchnorm_class = _maybe_sync_batchnorm(is_distributed)
    return nn.Sequential(
        batchnorm_class(dim),
        _GELU(),
        nn.Conv1d(dim, _default(dim_out, dim), kernel_size, padding=kernel_size // 2),
    )


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_rel_pos_features: int,
        heads: int = 8,
        dim_key: int = 64,
        dim_value: int = 64,
        dropout: float = 0.0,
        pos_dropout: float = 0.0,
        use_tf_gamma: bool = False,
        tf_gammas: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.scale = dim_key**-0.5
        self.heads = int(heads)
        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)
        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)
        self.num_rel_pos_features = int(num_rel_pos_features)
        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.use_tf_gamma = bool(use_tf_gamma)
        # This tensor is intentionally not a parameter or buffer. It is a
        # fixed local lookup table and therefore should not be written into a
        # model weight file.
        self.tf_gammas = tf_gammas

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence_length = x.shape[0], x.shape[-2]
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)
        query = query.reshape(batch, sequence_length, self.heads, -1).permute(0, 2, 1, 3)
        key = key.reshape(batch, sequence_length, self.heads, -1).permute(0, 2, 1, 3)
        value = value.reshape(batch, sequence_length, self.heads, -1).permute(0, 2, 1, 3)
        query = query * self.scale

        content_logits = torch.einsum(
            "bhid,bhjd->bhij", query + self.rel_content_bias, key
        )
        positions = _get_positional_embed(
            sequence_length,
            self.num_rel_pos_features,
            x.device,
            use_tf_gamma=self.use_tf_gamma,
            tf_gammas=self.tf_gammas,
            dtype=self.to_rel_k.weight.dtype,
        )
        positions = self.pos_dropout(positions)
        relative_key = self.to_rel_k(positions)
        relative_key = relative_key.reshape(sequence_length * 2 - 1, self.heads, -1).permute(
            1, 0, 2
        )
        relative_logits = torch.einsum(
            "bhid,hjd->bhij", query + self.rel_pos_bias, relative_key
        )
        relative_logits = _relative_shift(relative_logits)
        attention = (content_logits + relative_logits).softmax(dim=-1)
        attention = self.attn_dropout(attention)
        output = torch.einsum("bhij,bhjd->bhid", attention, value)
        output = output.permute(0, 2, 1, 3).reshape(batch, sequence_length, -1)
        return self.to_out(output)


def _str_to_one_hot(sequences: str | list[str] | tuple[str, ...]) -> torch.Tensor:
    if isinstance(sequences, str):
        sequences = [sequences]
        no_batch = True
    else:
        no_batch = False
    if not sequences:
        raise ValueError("At least one sequence is required")
    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) != 1:
        raise ValueError("All sequences in a batch must have the same length")
    mapping = {"a": 0, "c": 1, "g": 2, "t": 3}
    encoded = torch.zeros((len(sequences), len(sequences[0]), 4), dtype=torch.float32)
    for batch_index, sequence in enumerate(sequences):
        for position, base in enumerate(sequence.lower()):
            if base == ".":
                encoded[batch_index, position] = 0.25
            elif base in mapping:
                encoded[batch_index, position, mapping[base]] = 1.0
    return encoded[0] if no_batch else encoded


def _indices_to_one_hot(indices: torch.Tensor) -> torch.Tensor:
    is_padding = indices == -1
    indices = indices.clamp(min=0)
    output = F.one_hot(indices, num_classes=5)[..., :4].float()
    return output.masked_fill(is_padding[..., None], 0.25)


class PurePyTorchEnformer(nn.Module):
    """The local Enformer trunk and output heads without Transformers."""

    def __init__(
        self,
        config: PureEnformerConfig | Mapping[str, Any] | None = None,
        *,
        tf_gammas_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = PureEnformerConfig()
        elif isinstance(config, Mapping):
            config = PureEnformerConfig.from_dict(config)
        elif not isinstance(config, PureEnformerConfig):
            raise TypeError(f"Unsupported config type: {type(config)!r}")
        self.config = config
        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2
        self.tf_gammas_path = str(tf_gammas_path) if tf_gammas_path is not None else None
        tf_gammas = self._load_tf_gammas(tf_gammas_path) if config.use_tf_gamma else None

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding=7),
            _Residual(_conv_block(half_dim)),
            _AttentionPool(half_dim, pool_size=2),
        )

        filter_list = _exponential_linspace_int(
            half_dim,
            config.dim,
            num=config.num_downsamples - 1,
            divisible_by=config.dim_divisible_by,
        )
        filter_list = [half_dim, *filter_list]
        conv_layers: list[nn.Module] = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                nn.Sequential(
                    _conv_block(dim_in, dim_out, kernel_size=5),
                    _Residual(_conv_block(dim_out, dim_out, 1)),
                    _AttentionPool(dim_out, pool_size=2),
                )
            )
        self.conv_tower = nn.Sequential(*conv_layers)

        transformer: list[nn.Module] = []
        for _ in range(config.depth):
            transformer.append(
                nn.Sequential(
                    _Residual(
                        nn.Sequential(
                            nn.LayerNorm(config.dim),
                            _Attention(
                                config.dim,
                                heads=config.heads,
                                dim_key=config.attn_dim_key,
                                dim_value=config.dim // config.heads,
                                dropout=config.attn_dropout,
                                pos_dropout=config.pos_dropout,
                                num_rel_pos_features=config.dim // config.heads,
                                use_tf_gamma=config.use_tf_gamma,
                                tf_gammas=tf_gammas,
                            ),
                            nn.Dropout(config.dropout_rate),
                        )
                    ),
                    _Residual(
                        nn.Sequential(
                            nn.LayerNorm(config.dim),
                            nn.Linear(config.dim, config.dim * 2),
                            nn.Dropout(config.dropout_rate),
                            nn.ReLU(),
                            nn.Linear(config.dim * 2, config.dim),
                            nn.Dropout(config.dropout_rate),
                        )
                    ),
                )
            )
        self.transformer = nn.Sequential(*transformer)

        self.target_length = config.target_length
        self.crop_final = _TargetLengthCrop(config.target_length)
        # Build this block explicitly so its state-dict names match the
        # established ``final_pointwise.1.*`` names.
        self.final_pointwise = nn.Sequential(
            _TransposeLastToChannel(),
            _conv_block(filter_list[-1], twice_dim, 1),
            _TransposeChannelToLast(),
            nn.Dropout(config.dropout_rate / 8),
            _GELU(),
        )

        # Register the same module aliases as the established implementation.
        # PyTorch consequently emits both component and _trunk keys, which is
        # useful when importing checkpoints produced by either implementation.
        self._trunk = nn.Sequential(
            _TransposeLastToChannel(),
            self.stem,
            self.conv_tower,
            _TransposeChannelToLast(),
            self.transformer,
            self.crop_final,
            self.final_pointwise,
        )
        self.add_heads(**config.output_heads)
        self.use_checkpointing = config.use_checkpointing

    @staticmethod
    def _load_tf_gammas(path: str | Path | None) -> torch.Tensor:
        if path is None:
            path = (
                Path(__file__).resolve().parents[1]
                / "enformer-pytorch"
                / "enformer_pytorch"
                / "precomputed"
                / "tf_gammas.pt"
            )
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                "use_tf_gamma=True requires a local tf_gammas.pt file; "
                f"not found: {path}"
            )
        try:
            table = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            table = torch.load(path, map_location="cpu")
        if not isinstance(table, torch.Tensor):
            raise TypeError(f"Expected a Tensor in {path}, got {type(table)!r}")
        return table.float().contiguous()

    @property
    def trunk(self) -> nn.Sequential:
        return self._trunk

    @property
    def heads(self) -> nn.ModuleDict:
        return self._heads

    def add_heads(self, **kwargs: int) -> None:
        self.output_heads = {str(name): int(features) for name, features in kwargs.items()}
        self._heads = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(self.dim * 2, features), nn.Softplus())
                for name, features in self.output_heads.items()
            }
        )

    def set_target_length(self, target_length: int) -> None:
        target_length = int(target_length)
        if target_length == 0 or target_length < -1:
            raise ValueError("target_length must be -1 or a positive integer")
        self.crop_final.target_length = target_length
        self.target_length = target_length

    def trunk_checkpointed(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.stem(x)
        x = self.conv_tower(x)
        x = x.permute(0, 2, 1)
        x = checkpoint_sequential(self.transformer, len(self.transformer), x)
        x = self.crop_final(x)
        return self.final_pointwise(x)

    def _prepare_sequence(self, x: Any) -> tuple[torch.Tensor, bool]:
        if isinstance(x, (str, list, tuple)):
            x = _str_to_one_hot(x)
        elif isinstance(x, torch.Tensor) and x.dtype == torch.long:
            x = _indices_to_one_hot(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a Tensor or DNA sequence(s), got {type(x)!r}")
        if x.ndim not in {2, 3} or x.shape[-1] != 4:
            raise ValueError(f"x must have shape [L,4] or [B,L,4], got {tuple(x.shape)}")
        no_batch = x.ndim == 2
        if no_batch:
            x = x.unsqueeze(0)
        device = next(self.parameters()).device
        x = x.to(device=device)
        if not x.is_floating_point():
            x = x.float()
        return x, no_batch

    def forward(
        self,
        x: Any,
        labels: torch.Tensor | None = None,
        *,
        head: str = "human",
        target_length: int | None = None,
        return_embeddings: bool = False,
        return_only_embeddings: bool = False,
    ) -> PureEnformerOutput | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if head not in self._heads:
            raise ValueError(f"Unknown head={head!r}; available heads: {sorted(self._heads)}")
        x, no_batch = self._prepare_sequence(x)
        if target_length is not None:
            self.set_target_length(target_length)
        trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
        hidden = trunk_fn(x)
        if return_only_embeddings:
            return hidden[0] if no_batch else hidden
        logits = self._heads[head](hidden)
        loss = None
        if labels is not None:
            if no_batch and labels.ndim == 2:
                labels = labels.unsqueeze(0)
            labels = labels.to(device=logits.device)
            if labels.shape != logits.shape:
                raise ValueError(
                    f"labels shape {tuple(labels.shape)} does not match logits shape {tuple(logits.shape)}"
                )
            loss = _poisson_loss(logits, labels)
        if no_batch:
            logits = logits[0]
            hidden = hidden[0]
        if return_embeddings:
            return logits, hidden
        return PureEnformerOutput(loss=loss, logits=logits, hidden_states=hidden)


class _TransposeLastToChannel(nn.Module):
    """``b n d -> b d n`` without an einops dependency."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)


class _TransposeChannelToLast(nn.Module):
    """``b d n -> b n d`` without an einops dependency."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 2, 1)


def _poisson_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target * _log(prediction)).mean()


class PurePyTorchContextEnformer(PurePyTorchEnformer):
    """Enformer conditioned on a SUCCEED expression vector through FiLM."""

    def __init__(
        self,
        config: PureEnformerConfig | Mapping[str, Any] | None = None,
        *,
        context_gene_count: int | None = None,
        context_hidden_dim: int | None = None,
        context_dropout: float | None = None,
        tf_gammas_path: str | Path | None = None,
    ) -> None:
        super().__init__(config, tf_gammas_path=tf_gammas_path)
        context_gene_count = int(
            self.config.context_gene_count if context_gene_count is None else context_gene_count
        )
        context_hidden_dim = int(
            self.config.context_hidden_dim if context_hidden_dim is None else context_hidden_dim
        )
        context_dropout = float(
            self.config.context_dropout if context_dropout is None else context_dropout
        )
        if context_gene_count <= 0 or context_hidden_dim <= 0:
            raise ValueError("context_gene_count and context_hidden_dim must be positive")
        if not 0 <= context_dropout < 1:
            raise ValueError(f"context_dropout must be in [0,1), got {context_dropout}")

        self.context_gene_count = context_gene_count
        self.context_hidden_dim = context_hidden_dim
        self.context_dropout = context_dropout
        self.context_feature_dim = self.config.dim * 2
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_gene_count),
            nn.Linear(context_gene_count, context_hidden_dim),
            nn.GELU(),
            nn.Dropout(context_dropout),
            nn.Linear(context_hidden_dim, self.context_feature_dim * 2),
        )
        nn.init.zeros_(self.context_encoder[-1].weight)
        nn.init.zeros_(self.context_encoder[-1].bias)
        self.config.context_gene_count = context_gene_count
        self.config.context_hidden_dim = context_hidden_dim
        self.config.context_dropout = context_dropout

    def _prepare_context(
        self,
        context_vector: torch.Tensor,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        if not isinstance(context_vector, torch.Tensor):
            context_vector = torch.as_tensor(context_vector)
        if context_vector.ndim == 1:
            context_vector = context_vector.unsqueeze(0)
        if context_vector.ndim != 2 or context_vector.shape[-1] != self.context_gene_count:
            raise ValueError(
                "context_vector must have shape [G] or [B,G] with "
                f"G={self.context_gene_count}; got {tuple(context_vector.shape)}"
            )
        if context_vector.shape[0] == 1 and batch_size > 1:
            context_vector = context_vector.expand(batch_size, -1)
        elif context_vector.shape[0] != batch_size:
            raise ValueError(
                f"context batch size {context_vector.shape[0]} does not match sequence batch {batch_size}"
            )
        parameter = next(self.context_encoder.parameters())
        context_vector = context_vector.to(device=parameter.device, dtype=parameter.dtype)
        return self.context_encoder(context_vector).to(dtype=hidden_dtype)

    def forward(
        self,
        x: Any,
        context_vector: torch.Tensor,
        labels: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        target_indices: torch.Tensor | None = None,
        *,
        head: str = "human",
        target_length: int | None = None,
        return_embeddings: bool = False,
        return_only_embeddings: bool = False,
    ) -> PureEnformerOutput | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if context_vector is None:
            raise ValueError("context_vector is required for PurePyTorchContextEnformer")
        if head not in self._heads:
            raise ValueError(f"Unknown head={head!r}; available heads: {sorted(self._heads)}")
        x, no_batch = self._prepare_sequence(x)
        if labels is not None:
            if not isinstance(labels, torch.Tensor):
                labels = torch.as_tensor(labels)
            if no_batch and labels.ndim == 2:
                labels = labels.unsqueeze(0)
            labels = labels.to(device=x.device)
            if not labels.is_floating_point():
                labels = labels.float()
            if target_mask is None:
                raise ValueError("target_mask is required when labels are supplied")
        if target_length is not None:
            self.set_target_length(target_length)

        trunk_fn = self.trunk_checkpointed if self.use_checkpointing else self._trunk
        hidden = trunk_fn(x)
        film = self._prepare_context(
            context_vector,
            batch_size=hidden.shape[0],
            hidden_dtype=hidden.dtype,
        )
        gamma, beta = film.chunk(2, dim=-1)
        conditioned_hidden = hidden * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        if return_only_embeddings:
            return conditioned_hidden[0] if no_batch else conditioned_hidden

        logits = self._heads[head](conditioned_hidden)
        loss = None
        if labels is not None:
            if not isinstance(target_mask, torch.Tensor):
                target_mask = torch.as_tensor(target_mask, device=logits.device)
            else:
                target_mask = target_mask.to(device=logits.device)

            loss_logits = logits
            if labels.shape != logits.shape:
                if target_indices is None:
                    raise ValueError(
                        f"labels shape {tuple(labels.shape)} does not match logits shape "
                        f"{tuple(logits.shape)}; target_indices is required for packed labels"
                    )
                if not isinstance(target_indices, torch.Tensor):
                    target_indices = torch.as_tensor(target_indices)
                if no_batch and target_indices.ndim == 1:
                    target_indices = target_indices.unsqueeze(0)
                target_indices = target_indices.to(device=logits.device, dtype=torch.long)
                if labels.ndim != 3:
                    raise ValueError(
                        f"Packed labels must have shape [B,P,K], got {tuple(labels.shape)}"
                    )
                expected_prefix = (logits.shape[0], logits.shape[1])
                if tuple(labels.shape[:2]) != expected_prefix:
                    raise ValueError(
                        f"Packed labels batch/bin dimensions {tuple(labels.shape[:2])} "
                        f"do not match logits {expected_prefix}"
                    )
                expected_index_shape = (logits.shape[0], labels.shape[-1])
                if tuple(target_indices.shape) != expected_index_shape:
                    raise ValueError(
                        f"target_indices shape {tuple(target_indices.shape)} does not "
                        f"match packed labels {expected_index_shape}"
                    )
                if target_mask.ndim == 1:
                    validation_mask = target_mask.unsqueeze(0).expand(
                        logits.shape[0], -1
                    )
                elif target_mask.ndim == 2:
                    validation_mask = target_mask
                else:
                    raise ValueError(
                        "Packed target_mask must have shape [K] or [B,K], got "
                        f"{tuple(target_mask.shape)}"
                    )
                if tuple(validation_mask.shape) != expected_index_shape:
                    raise ValueError(
                        f"Packed target_mask shape {tuple(target_mask.shape)} cannot "
                        f"match target_indices {expected_index_shape}"
                    )
                valid_indices = (target_indices >= 0) & (
                    target_indices < logits.shape[-1]
                )
                if torch.any(validation_mask.bool() & ~valid_indices):
                    raise ValueError("A valid packed target has an out-of-range channel index")
                safe_indices = target_indices.clamp(min=0, max=logits.shape[-1] - 1)
                gather_indices = safe_indices[:, None, :].expand(
                    -1, logits.shape[1], -1
                )
                loss_logits = torch.gather(logits, dim=-1, index=gather_indices)
            loss = masked_poisson_loss(loss_logits, labels, target_mask)

        if no_batch:
            logits = logits[0]
            conditioned_hidden = conditioned_hidden[0]
        if return_embeddings:
            return logits, conditioned_hidden
        return PureEnformerOutput(
            loss=loss,
            logits=logits,
            hidden_states=conditioned_hidden,
        )


__all__ = [
    "PureEnformerConfig",
    "PureEnformerOutput",
    "PurePyTorchEnformer",
    "PurePyTorchContextEnformer",
    "SEQUENCE_LENGTH",
    "TARGET_LENGTH",
]
