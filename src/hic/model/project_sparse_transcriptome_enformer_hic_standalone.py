"""Pretrained sparse Enformer consuming per-cell project RNA contexts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn

from hic.model.enformer_hic_standalone import (
    ENFORMER_INPUT_BP,
    ENFORMER_OUTPUT_TOKENS,
    ENFORMER_TOKEN_BP,
)
from hic.model.sparse_transcriptome_enformer_hic_standalone import (
    FrozenSparseTranscriptomeEnformerExtractor,
    _load_external_sparse_symbols,
    _load_sparse_checkpoint,
)
from hic.model.sparse_transcriptome_model_contract import (
    CellTranscriptomeContext,
    SparseTranscriptomeModelContract,
)


class FrozenProjectSparseTranscriptomeEnformerExtractor(
    FrozenSparseTranscriptomeEnformerExtractor
):
    """Load the checkpoint without loading any ``enfpcot`` dataset sidecar."""

    def __init__(
        self,
        enformer_root: str | Path,
        checkpoint_path: str | Path,
        model_contract: str | Path,
        *,
        freeze: bool = True,
        expected_input_bp: int = ENFORMER_INPUT_BP,
        expected_output_tokens: int = ENFORMER_OUTPUT_TOKENS,
    ) -> None:
        nn.Module.__init__(self)
        state_dict, metadata = _load_sparse_checkpoint(checkpoint_path)
        (
            SparseTranscriptomeContextEnformer,
            PureEnformerConfig,
            SparseTranscriptomeRouterConfig,
            _,
        ) = _load_external_sparse_symbols(enformer_root)
        contract = SparseTranscriptomeModelContract(model_contract)
        config_values = metadata["model_config"]
        router_values = metadata["router_config"]
        if not isinstance(config_values, Mapping) or not isinstance(
            router_values, Mapping
        ):
            raise ValueError("model_config and router_config must be mappings")
        config = PureEnformerConfig.from_dict(config_values)
        config.use_checkpointing = False
        router_config = SparseTranscriptomeRouterConfig.from_dict(router_values)
        router_config.validate()
        if int(config.target_length) != int(expected_output_tokens):
            raise ValueError(
                f"Checkpoint target_length={config.target_length}, expected "
                f"{expected_output_tokens}"
            )
        if str(contract.payload.get("framework", "")) != str(
            metadata["framework"]
        ):
            raise ValueError("Model contract framework does not match checkpoint")
        if int(contract.payload.get("checkpoint_format_version", -1)) != int(
            metadata["format_version"]
        ):
            raise ValueError("Model contract format version does not match checkpoint")
        recorded_sidecar_hash = str(
            contract.payload.get("checkpoint_transcriptome_manifest_sha256", "")
        )
        if recorded_sidecar_hash != str(metadata["transcriptome_manifest_sha256"]):
            raise ValueError("Model contract provenance does not match checkpoint")
        expected_interface = (
            contract.gene_count,
            contract.feature_dim,
            len(contract.protocol_to_id),
        )
        checkpoint_interface = (
            int(router_config.gene_count),
            int(router_config.feature_dim),
            int(router_config.protocol_count),
        )
        if checkpoint_interface != expected_interface:
            raise ValueError(
                "Model contract and checkpoint RNA interfaces differ: "
                f"{expected_interface} vs {checkpoint_interface}"
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
                "Sparse checkpoint does not exactly match the imported model code"
            ) from error

        self.enformer = model
        self.freeze = bool(freeze)
        self.checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        self.enformer_root = str(Path(enformer_root).expanduser().resolve())
        self.model_contract = contract
        self.model_contract_path = str(contract.path)
        self.model_contract_sha256 = contract.sha256
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
        self.gene_count = contract.gene_count
        self.candidate_gene_count = contract.gene_count
        self.feature_names = contract.feature_names
        self.film_transformer_layers = film_layers_tuple
        self.film_bound = film_bound
        self.routing_activation = str(router_config.routing_activation)
        if self.input_bp != ENFORMER_INPUT_BP:
            raise ValueError(
                f"Only {ENFORMER_INPUT_BP}-bp inputs are supported, got "
                f"{self.input_bp}"
            )
        if self.output_hidden != 3072:
            raise ValueError(
                f"The Hi-C head expects hidden=3072, got {self.output_hidden}"
            )
        if self.freeze:
            for parameter in self.enformer.parameters():
                parameter.requires_grad_(False)
        self.enformer.eval()
        del state_dict

    def encode_cell_contexts(
        self,
        context_paths: Sequence[str | Path],
        *,
        expected_celltypes: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, tuple[CellTranscriptomeContext, ...]]:
        contexts = tuple(
            CellTranscriptomeContext(path, self.model_contract)
            for path in context_paths
        )
        if not contexts:
            raise ValueError("At least one cell RNA context is required")
        if expected_celltypes is not None:
            expected = tuple(str(value) for value in expected_celltypes)
            actual = tuple(context.celltype for context in contexts)
            if actual != expected:
                raise ValueError(
                    f"RNA context cell types do not match: {actual} vs {expected}"
                )
        parameter = next(self.enformer.context_router.parameters())
        device = parameter.device
        gene_features = torch.as_tensor(
            np.stack(
                [
                    np.asarray(context.gene_features, dtype=np.float32)
                    for context in contexts
                ]
            ),
            device=device,
            dtype=parameter.dtype,
        )
        gene_mask = torch.as_tensor(
            np.stack([context.gene_mask for context in contexts]),
            device=device,
            dtype=torch.bool,
        )
        protocol_ids = torch.as_tensor(
            [context.protocol_id for context in contexts],
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
            raise TypeError("Sparse transcriptome router returned diagnostics")
        states = encoded.clone().float()
        expected_shape = (
            len(contexts),
            self.context_state_token_count,
            self.context_state_dim,
        )
        if tuple(states.shape) != expected_shape:
            raise RuntimeError(
                f"Expected RNA states {expected_shape}, got {tuple(states.shape)}"
            )
        if not torch.isfinite(states).all():
            raise RuntimeError("Cell transcriptome states are non-finite")
        return states, contexts


__all__ = ["FrozenProjectSparseTranscriptomeEnformerExtractor"]
