#!/usr/bin/env python3
"""Loss functions for context-conditioned Enformer training."""

from __future__ import annotations

from typing import Literal

import torch


Reduction = Literal["mean", "sum", "none"]


def poisson_loss_per_element(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return Enformer-style Poisson negative log-likelihood per element.

    Enformer predicts positive values with a Softplus output head.  The
    clamp keeps this function safe when it is called with an independently
    implemented head or with numerically tiny values.  The additive
    ``log(target!)`` term is omitted because it is constant with respect to
    the prediction, matching the existing Enformer implementation.
    """

    if prediction.shape != targets.shape:
        raise ValueError(
            f"prediction and targets must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(targets.shape)}"
        )
    if not prediction.is_floating_point() or not targets.is_floating_point():
        raise TypeError("prediction and targets must be floating-point tensors")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    positive_prediction = prediction.clamp_min(float(eps))
    return positive_prediction - targets * torch.log(positive_prediction)


def _broadcast_channel_mask(
    target_mask: torch.Tensor,
    target_shape: torch.Size,
) -> torch.Tensor:
    """Convert [C] or [B,C] masks to a shape broadcastable to [B,P,C]."""

    mask = target_mask
    if mask.ndim == 0:
        raise ValueError("target_mask must contain a channel dimension")
    while mask.ndim < len(target_shape):
        # For [B,C] -> [B,1,C], and for [C] -> [1,1,C].
        mask = mask.unsqueeze(-2)
    try:
        return torch.broadcast_to(mask, target_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"target_mask shape {tuple(target_mask.shape)} cannot broadcast to "
            f"prediction shape {tuple(target_shape)}"
        ) from exc


def masked_poisson_loss(
    prediction: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    eps: float = 1e-8,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Compute Poisson loss only on context-matched target channels.

    Parameters
    ----------
    prediction, targets:
        Tensors with identical shape, normally ``[batch, 896, 5313]``.
    target_mask:
        Boolean or numeric mask shaped ``[5313]`` or ``[batch, 5313]``.
        It is broadcast over the 896 genomic output bins.  A mask value of
        one includes that target channel; zero ignores it.
    eps:
        Lower bound used inside ``log``.
    reduction:
        ``mean`` divides by the number of valid batch/bin/channel elements;
        ``sum`` returns their sum; ``none`` returns the masked per-element
        loss tensor.

    An all-zero mask returns a differentiable zero scalar for ``mean`` and
    ``sum``.  This avoids NaNs while still producing no gradient for that
    batch; callers may choose to reject such batches at the sampler level.
    """

    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unsupported reduction={reduction!r}")

    # Keep the reduction and logarithm in fp32 when the model is under fp16
    # or bf16 autocast. Gradients still flow back through the cast to the
    # model output, while the loss is less prone to overflow/underflow.
    loss_prediction = (
        prediction.float()
        if prediction.dtype in {torch.float16, torch.bfloat16}
        else prediction
    )
    loss_targets = (
        targets.float()
        if targets.dtype in {torch.float16, torch.bfloat16}
        else targets
    )
    per_element = poisson_loss_per_element(loss_prediction, loss_targets, eps=eps)
    if not isinstance(target_mask, torch.Tensor):
        target_mask = torch.as_tensor(target_mask, device=prediction.device)
    else:
        target_mask = target_mask.to(device=prediction.device)
    mask = _broadcast_channel_mask(target_mask, prediction.shape)
    mask = mask.to(dtype=per_element.dtype)
    masked = per_element * mask

    if reduction == "none":
        return masked
    if reduction == "sum":
        return masked.sum()

    valid_count = mask.sum()
    return masked.sum() / valid_count.clamp_min(1.0)


__all__ = ["poisson_loss_per_element", "masked_poisson_loss"]
