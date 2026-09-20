"""
Gradient Reversal Layer for adversarial training.

Forward pass: identity.
Backward pass: multiplies gradient by -α.

Supports configurable α scheduling (linear warmup) for stable training.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Autograd function implementing gradient reversal."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Module wrapper around GradientReversalFunction.

    Supports alpha scheduling:
        - Fixed alpha mode: set alpha at init
        - Warmup mode: linearly ramp alpha from 0 to target over warmup_steps
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float) -> None:
        """Update the reversal strength."""
        self.alpha = alpha

    @staticmethod
    def compute_warmup_alpha(
        epoch: int,
        warmup_epochs: int,
        target_alpha: float,
    ) -> float:
        """
        Compute α with linear warmup scheduling.

        Args:
            epoch: Current epoch (0-indexed).
            warmup_epochs: Number of warmup epochs.
            target_alpha: Final α value after warmup.

        Returns:
            Current α value.
        """
        if warmup_epochs <= 0:
            return target_alpha
        progress = min(epoch / warmup_epochs, 1.0)
        return progress * target_alpha
