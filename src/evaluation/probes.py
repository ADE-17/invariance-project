"""
Post-hoc demographic probes: Linear and MLP probes on frozen representations.

For every trained model, these probes measure how much demographic information
remains decodable from the learned representation. This is the core measurement
of "invariance" (or lack thereof).

Probes:
  - LinearProbe: Logistic regression (linear decodability)
  - MLPProbe: 2-layer MLP (nonlinear decodability — stronger test)
  - Class-conditional probes: separate probes for Y=0 and Y=1 subsets

If the training adversary is at chance but a stronger post-hoc probe recovers
sex, the adversary failed rather than producing invariance.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe:
    """
    Logistic regression probe for demographic decodability.

    Uses sklearn LogisticRegression on frozen representations.
    """

    def __init__(self, max_iter: int = 2000, random_state: int = 42):
        self.clf = LogisticRegression(max_iter=max_iter, random_state=random_state, solver="lbfgs")
        self.fitted = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LinearProbe":
        """
        Train the linear probe.

        Args:
            features: (N, D) feature matrix from frozen encoder.
            labels: (N,) binary demographic labels.
        """
        self.clf.fit(features, labels)
        self.fitted = True
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for the positive class."""
        return self.clf.predict_proba(features)[:, 1]

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute AUROC and accuracy."""
        probs = self.predict_proba(features)
        preds = (probs >= 0.5).astype(int)
        try:
            auroc = float(roc_auc_score(labels, probs))
        except ValueError:
            auroc = float("nan")
        acc = float(accuracy_score(labels, preds))
        return {"auroc": auroc, "accuracy": acc}


class MLPProbe(nn.Module):
    """
    2-layer MLP probe for nonlinear demographic decodability.

    Architecture: in_features → hidden1 → ReLU → hidden2 → ReLU → 1.
    Trained with SGD on frozen representations.
    """

    def __init__(
        self,
        in_features: int = 1024,
        hidden_dims: Tuple[int, int] = (256, 128),
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_probe(
    probe: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 256,
    device: str = "cpu",
) -> nn.Module:
    """
    Train an MLP probe on frozen representations.

    Args:
        probe: MLPProbe module.
        features: (N, D) numpy array of frozen features.
        labels: (N,) binary demographic labels.
        lr: Learning rate.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        device: Device string.

    Returns:
        Trained probe module.
    """
    device_obj = torch.device(device)
    probe = probe.to(device_obj)
    probe.train()

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device_obj)
            batch_y = batch_y.to(device_obj)

            optimizer.zero_grad()
            logits = probe(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    return probe


def evaluate_probe(
    probe: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate a trained MLP probe.

    Returns:
        Dict with 'auroc' and 'accuracy'.
    """
    device_obj = torch.device(device)
    probe = probe.to(device_obj)
    probe.eval()

    X = torch.tensor(features, dtype=torch.float32).to(device_obj)

    with torch.no_grad():
        logits = probe(X)
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= 0.5).astype(int)
    try:
        auroc = float(roc_auc_score(labels, probs))
    except ValueError:
        auroc = float("nan")
    acc = float(accuracy_score(labels, preds))

    return {"auroc": auroc, "accuracy": acc}


def run_all_probes(
    features_train: np.ndarray,
    labels_train: np.ndarray,
    features_test: np.ndarray,
    labels_test: np.ndarray,
    task_labels_train: Optional[np.ndarray] = None,
    task_labels_test: Optional[np.ndarray] = None,
    feature_dim: int = 1024,
    hidden_dims: Tuple[int, int] = (256, 128),
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Run all probe types: linear, MLP, and class-conditional.

    Args:
        features_train/test: Frozen representation features.
        labels_train/test: Binary demographic labels.
        task_labels_train/test: Binary task labels (for class-conditional probes).
        feature_dim: Feature dimensionality.
        hidden_dims: MLP probe hidden dimensions.
        device: Compute device.

    Returns:
        Dict with probe type → {auroc, accuracy} on test data.
    """
    results = {}

    # 1. Linear probe
    linear = LinearProbe()
    linear.fit(features_train, labels_train)
    results["linear_probe"] = linear.evaluate(features_test, labels_test)

    # 2. MLP probe (nonlinear — stronger test)
    mlp = MLPProbe(in_features=feature_dim, hidden_dims=hidden_dims)
    mlp = train_probe(mlp, features_train, labels_train, device=device)
    results["mlp_probe"] = evaluate_probe(mlp, features_test, labels_test, device=device)

    # 3. Class-conditional probes (separate for Y=0 and Y=1)
    if task_labels_train is not None and task_labels_test is not None:
        for y_val, y_name in [(0, "Y=0"), (1, "Y=1")]:
            train_mask = task_labels_train == y_val
            test_mask = task_labels_test == y_val

            if train_mask.sum() < 10 or test_mask.sum() < 10:
                results[f"linear_probe_{y_name}"] = {"auroc": float("nan"), "accuracy": float("nan")}
                results[f"mlp_probe_{y_name}"] = {"auroc": float("nan"), "accuracy": float("nan")}
                continue

            # Linear
            cond_linear = LinearProbe()
            cond_linear.fit(features_train[train_mask], labels_train[train_mask])
            results[f"linear_probe_{y_name}"] = cond_linear.evaluate(
                features_test[test_mask], labels_test[test_mask]
            )

            # MLP
            cond_mlp = MLPProbe(in_features=feature_dim, hidden_dims=hidden_dims)
            cond_mlp = train_probe(
                cond_mlp, features_train[train_mask], labels_train[train_mask], device=device
            )
            results[f"mlp_probe_{y_name}"] = evaluate_probe(
                cond_mlp, features_test[test_mask], labels_test[test_mask], device=device
            )

    return results
