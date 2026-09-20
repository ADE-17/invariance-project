"""
Prediction heads for task classification, adversarial debiasing, and
group-conditioned inference.

  - TaskHead: simple linear binary classifier per pathology
  - AdversaryHead: MLP for demographic prediction (used by DANN/CDANN)
  - GroupConditionedHead: separate prediction heads per demographic group
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class TaskHead(nn.Module):
    """
    Linear binary classification head for a single pathology.

    Outputs raw logits (apply sigmoid externally for probabilities).
    """

    def __init__(self, in_features: int = 1024):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) backbone features.
        Returns:
            logits: (B,) raw logits.
        """
        return self.fc(features).squeeze(-1)


class AdversaryHead(nn.Module):
    """
    MLP head for demographic attribute prediction.

    Used by DANN (one head for all samples) and CDANN (separate heads
    per class label). Architecture: in_features → 128 → ReLU → 64 → ReLU → 1.
    """

    def __init__(self, in_features: int = 1024, hidden_dims: tuple = (128, 64)):
        super().__init__()
        layers = []
        prev_dim = in_features
        for hdim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hdim),
                nn.ReLU(inplace=True),
            ])
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) features (after gradient reversal).
        Returns:
            logits: (B,) demographic prediction logits.
        """
        return self.net(features).squeeze(-1)


class GroupConditionedHead(nn.Module):
    """
    Group-conditioned prediction head with shared backbone.

    Maintains separate linear classifiers for each demographic group.
    At training time, each sample is routed through its group's head.
    At test time, uses the appropriate head based on known group membership.

    This directly tests whether group-aware prediction (without invariance)
    can achieve fairness.
    """

    def __init__(self, in_features: int = 1024, n_groups: int = 2):
        """
        Args:
            in_features: Feature dimension from backbone.
            n_groups: Number of demographic groups (default 2: Male/Female).
        """
        super().__init__()
        self.n_groups = n_groups
        self.heads = nn.ModuleList([
            nn.Linear(in_features, 1) for _ in range(n_groups)
        ])

    def forward(
        self,
        features: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass routing each sample through its group-specific head.

        Args:
            features: (B, D) backbone features.
            group_ids: (B,) integer group labels (0, 1, ..., n_groups-1).

        Returns:
            logits: (B,) task prediction logits.
        """
        logits = torch.zeros(features.size(0), device=features.device, dtype=features.dtype)

        for g in range(self.n_groups):
            mask = group_ids == g
            if mask.any():
                group_features = features[mask]
                group_logits = self.heads[g](group_features).squeeze(-1)
                logits[mask] = group_logits

        return logits
