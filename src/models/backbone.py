"""
DenseNet-121 backbone with layer-wise feature extraction hooks.

DenseNet-121 is the standard architecture for CXR classification (used by
CheXpert, Yang et al. Nature Medicine 2024, Jones et al. ICLR 2025).

Provides:
  - Final pooled features (1024-d) for task and adversarial heads
  - Intermediate features from each DenseBlock via forward hooks (for layer-wise
    leakage measurement)
"""

from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class DenseNet121Backbone(nn.Module):
    """
    DenseNet-121 feature extractor with intermediate layer hooks.

    The original classification head is removed. The backbone outputs a
    1024-dimensional feature vector from global average pooling.

    Intermediate features can be captured from each DenseBlock for
    layer-wise leakage analysis.
    """

    LAYER_NAMES = ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]

    def __init__(self, pretrained: bool = True):
        super().__init__()

        if pretrained:
            weights = models.DenseNet121_Weights.DEFAULT
            base = models.densenet121(weights=weights)
        else:
            base = models.densenet121(weights=None)

        # The DenseNet features module contains all convolutional layers
        self.features = base.features  # nn.Sequential of conv/bn/relu/pool + denseblocks

        # Global average pooling (replaces the classifier)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature dimension after pooling
        self.feature_dim = 1024  # DenseNet-121 output channels

        # Storage for intermediate features (populated by hooks)
        self._intermediate_features: Dict[str, torch.Tensor] = {}
        self._hooks = []

    def register_intermediate_hooks(self) -> None:
        """
        Register forward hooks on each DenseBlock to capture intermediate features.
        Call this before running forward passes for leakage analysis.
        """
        self.remove_hooks()  # Clean up any existing hooks

        for name, module in self.features.named_children():
            if "denseblock" in name:
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        """Create a forward hook that captures the output of a named layer."""
        def hook_fn(module, input, output):
            # Apply global average pooling to spatial features for a 1-d vector
            pooled = nn.functional.adaptive_avg_pool2d(output, (1, 1))
            self._intermediate_features[name] = pooled.squeeze(-1).squeeze(-1)
        return hook_fn

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._intermediate_features.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning the final 1024-d feature vector.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            features: (B, 1024) pooled feature tensor.
        """
        features = self.features(x)
        features = nn.functional.relu(features, inplace=True)
        features = self.pool(features)
        features = features.view(features.size(0), -1)  # (B, 1024)
        return features

    def get_intermediate_features(self) -> Dict[str, torch.Tensor]:
        """
        Return intermediate features captured by hooks during the last forward pass.

        Must call register_intermediate_hooks() first.

        Returns:
            Dict mapping layer name → (B, C) pooled feature tensor.
        """
        return dict(self._intermediate_features)

    def forward_with_intermediates(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convenience method: forward pass + return intermediate features.

        Hooks must be registered first via register_intermediate_hooks().

        Returns:
            (final_features, {layer_name: intermediate_features})
        """
        self._intermediate_features.clear()
        final = self.forward(x)
        return final, dict(self._intermediate_features)
