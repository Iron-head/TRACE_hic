"""Unit-interval output variant of the Enformer Hi-C decoder."""

from __future__ import annotations

import torch

from trace_hic.model.enformer_hic_standalone import EnformerHiCModel


UNIT_INTERVAL_OUTPUT_ACTIVATION = "(tanh(x)+1)/2"


class UnitIntervalEnformerHiCModel(EnformerHiCModel):
    """Preserve the base head state layout and bound predictions to [0, 1]."""

    def forward(self, enformer_tokens: torch.Tensor) -> torch.Tensor:
        logits = super().forward(enformer_tokens)
        return (torch.tanh(logits) + 1.0) * 0.5


__all__ = ["UNIT_INTERVAL_OUTPUT_ACTIVATION", "UnitIntervalEnformerHiCModel"]
