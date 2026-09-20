"""
Task performance metrics: per-group and overall.

Computes AUROC, AUPRC, sensitivity, specificity, Brier score, NLL,
absolute false negative counts, and change relative to ERM baseline.

All metrics use a single global threshold fixed on validation data.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


def compute_task_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive task metrics for a binary classification problem.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_prob: Predicted probabilities.
        threshold: Decision threshold (fixed on validation data).

    Returns:
        Dict with AUROC, AUPRC, sensitivity, specificity, Brier, NLL,
        accuracy, F1, false negatives, false positives.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    n = len(y_true)
    if n == 0:
        return _empty_metrics()

    # Core discrimination metrics
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = float("nan")

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = float("nan")

    # Confusion matrix derived
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")  # TPR
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")  # TNR
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    # Calibration metrics
    try:
        brier = float(brier_score_loss(y_true, y_prob))
    except ValueError:
        brier = float("nan")

    try:
        nll = float(log_loss(y_true, y_prob))
    except ValueError:
        nll = float("nan")

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,  # TPR
        "specificity": specificity,  # TNR
        "fpr": fpr,
        "ppv": ppv,
        "npv": npv,
        "brier": brier,
        "nll": nll,
        "accuracy": acc,
        "f1": f1,
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "n_samples": n,
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
        "prevalence": float(np.mean(y_true)),
        "threshold": threshold,
    }


def compute_group_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_ids: np.ndarray,
    threshold: float = 0.5,
    group_names: Optional[Dict[int, str]] = None,
    erm_metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Compute task metrics separately for each demographic group.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        group_ids: Integer group identifiers (0=Female, 1=Male).
        threshold: Decision threshold (global, from validation).
        group_names: Optional mapping from group ID to name.
        erm_metrics: Optional ERM baseline metrics per group for delta computation.

    Returns:
        Dict with 'overall', per-group metrics, and 'worst_group' metrics.
    """
    if group_names is None:
        group_names = {0: "Female", 1: "Male"}

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    group_ids = np.asarray(group_ids, dtype=int)

    result = {
        "overall": compute_task_metrics(y_true, y_prob, threshold),
        "groups": {},
    }

    worst_auroc = float("inf")
    worst_group = None

    for gid, gname in group_names.items():
        mask = group_ids == gid
        if mask.sum() == 0:
            continue

        group_m = compute_task_metrics(y_true[mask], y_prob[mask], threshold)

        # Compute delta relative to ERM if provided
        if erm_metrics is not None and gname in erm_metrics:
            erm_g = erm_metrics[gname]
            group_m["delta_auroc_vs_erm"] = group_m["auroc"] - erm_g.get("auroc", 0)
            group_m["delta_sensitivity_vs_erm"] = group_m["sensitivity"] - erm_g.get("sensitivity", 0)
            group_m["delta_specificity_vs_erm"] = group_m["specificity"] - erm_g.get("specificity", 0)

        result["groups"][gname] = group_m

        if group_m["auroc"] < worst_auroc:
            worst_auroc = group_m["auroc"]
            worst_group = gname

    result["worst_group_name"] = worst_group
    result["worst_group_auroc"] = worst_auroc

    return result


def select_threshold_on_validation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    criterion: str = "youden",
) -> float:
    """
    Select a global threshold on validation data.

    Args:
        y_true: Validation ground truth.
        y_prob: Validation predicted probabilities.
        criterion: Selection criterion:
            "youden" — maximizes sensitivity + specificity - 1
            "f1" — maximizes F1 score

    Returns:
        Optimal threshold.
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if criterion == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])
    elif criterion == "f1":
        best_f1 = 0.0
        best_t = 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        return float(best_t)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def _empty_metrics() -> Dict[str, float]:
    """Return a metrics dict with NaN values for empty inputs."""
    return {
        "auroc": float("nan"), "auprc": float("nan"),
        "sensitivity": float("nan"), "specificity": float("nan"),
        "fpr": float("nan"), "ppv": float("nan"), "npv": float("nan"),
        "brier": float("nan"), "nll": float("nan"),
        "accuracy": float("nan"), "f1": float("nan"),
        "false_negatives": 0, "false_positives": 0,
        "true_positives": 0, "true_negatives": 0,
        "n_samples": 0, "n_positive": 0, "n_negative": 0,
        "prevalence": float("nan"), "threshold": 0.5,
    }
