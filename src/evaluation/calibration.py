"""
Calibration analysis: ECE, ICI, calibration slope/intercept, and calibration curves.

Group-wise calibration is essential for RQ4: understanding whether invariance
imposes distributional and calibration costs beyond what fairness constraints require.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error with equal-width bins.

    ECE = Σ (|bin| / N) · |acc(bin) - conf(bin)|

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        n_bins: Number of equal-width bins.

    Returns:
        ECE value (lower is better).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == 0:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob > lower) & (y_prob <= upper)

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        avg_confidence = float(y_prob[mask].mean())
        avg_accuracy = float(y_true[mask].mean())
        ece += (n_in_bin / len(y_true)) * abs(avg_accuracy - avg_confidence)

    return ece


def compute_ici(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Integrated Calibration Index: mean absolute calibration error across bins.

    ICI = (1/K) Σ |acc(bin) - conf(bin)| for non-empty bins.

    This differs from ECE by not weighting by bin size.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(y_true) == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    errors = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        mask = (y_prob > lower) & (y_prob <= upper) if i > 0 else (y_prob >= lower) & (y_prob <= upper)

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        avg_confidence = float(y_prob[mask].mean())
        avg_accuracy = float(y_true[mask].mean())
        errors.append(abs(avg_accuracy - avg_confidence))

    return float(np.mean(errors)) if errors else float("nan")


def compute_calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """
    Compute calibration slope and intercept via logistic recalibration
    (Platt scaling diagnostic).

    A perfectly calibrated model has slope=1 and intercept=0.

    Returns:
        Dict with 'slope', 'intercept'.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float).clip(1e-7, 1 - 1e-7)

    if len(y_true) < 10:
        return {"slope": float("nan"), "intercept": float("nan")}

    # Logistic recalibration: fit logistic regression on logit(y_prob) → y_true
    from sklearn.linear_model import LogisticRegression

    logits = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
    clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1e10)  # No regularization
    clf.fit(logits, y_true)

    return {
        "slope": float(clf.coef_[0, 0]),
        "intercept": float(clf.intercept_[0]),
    }


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, List[float]]:
    """
    Compute calibration curve data points for plotting.

    Returns:
        Dict with 'bin_centers', 'bin_accuracies', 'bin_confidences', 'bin_counts'.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = []
    accuracies = []
    confidences = []
    counts = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        mask = (y_prob > lower) & (y_prob <= upper) if i > 0 else (y_prob >= lower) & (y_prob <= upper)

        n_in_bin = mask.sum()
        counts.append(int(n_in_bin))

        if n_in_bin == 0:
            centers.append(float((lower + upper) / 2))
            accuracies.append(float("nan"))
            confidences.append(float("nan"))
        else:
            centers.append(float(y_prob[mask].mean()))
            accuracies.append(float(y_true[mask].mean()))
            confidences.append(float(y_prob[mask].mean()))

    return {
        "bin_centers": centers,
        "bin_accuracies": accuracies,
        "bin_confidences": confidences,
        "bin_counts": counts,
    }


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """
    Compute all calibration metrics in one call.

    Returns:
        Dict with ECE, ICI, slope, intercept, and curve data.
    """
    # Brier and NLL
    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except ValueError:
        brier = float("nan")
        
    try:
        # NLL / Log Loss
        nll = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError:
        nll = float("nan")

    return {
        "ece": compute_ece(y_true, y_prob, n_bins),
        "ici": compute_ici(y_true, y_prob, n_bins),
        "brier": brier,
        "nll": nll,
        **compute_calibration_slope_intercept(y_true, y_prob),
        "curve": compute_calibration_curve(y_true, y_prob, n_bins),
    }


def compute_conditional_score_mmd(
    scores_group_a: np.ndarray,
    scores_group_b: np.ndarray,
    bandwidth: Optional[float] = None,
) -> float:
    """
    MMD between score distributions of two groups.

    Used to measure distributional alignment of predicted scores
    (distinct from representation-level MMD in distribution_metrics.py).

    Args:
        scores_group_a: Predicted scores for group A.
        scores_group_b: Predicted scores for group B.
        bandwidth: RBF kernel bandwidth. If None, uses median heuristic.

    Returns:
        Estimated MMD^2 value.
    """
    a = np.asarray(scores_group_a, dtype=float).reshape(-1, 1)
    b = np.asarray(scores_group_b, dtype=float).reshape(-1, 1)

    if len(a) == 0 or len(b) == 0:
        return float("nan")

    # Median heuristic for bandwidth
    if bandwidth is None:
        combined = np.concatenate([a, b])
        dists = np.abs(combined[:, None] - combined[None, :])
        bandwidth = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1.0

    gamma = 1.0 / (2 * bandwidth ** 2)

    def rbf(x, y):
        return np.exp(-gamma * (x - y.T) ** 2)

    kaa = rbf(a, a).mean()
    kbb = rbf(b, b).mean()
    kab = rbf(a, b).mean()

    return float(kaa + kbb - 2 * kab)
