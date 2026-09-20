"""
Fairness metrics: correctly defined.

  - ΔEOpp  = |TPR_F − TPR_M|           (equality of opportunity)
  - ΔEOdds = max(|ΔTPR|, |ΔFPR|)       (equalized odds)
  - ΔDP    = |PPR_F − PPR_M|           (demographic parity, descriptive only)

All metrics use a single global threshold fixed on validation data.
Demographic parity is reported descriptively when prevalences differ.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import confusion_matrix


def _group_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute TPR, FPR, and positive prediction rate for a group."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    tpr = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    ppr = (tp + fp) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else float("nan")

    return {"tpr": tpr, "fpr": fpr, "ppr": ppr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_equality_of_opportunity_gap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_ids: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute equality of opportunity gap: ΔEOpp = |TPR_Female − TPR_Male|.

    This measures the absolute difference in true positive rates across groups.
    A model satisfies equality of opportunity when ΔEOpp = 0.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_prob: Predicted probabilities.
        group_ids: Integer group identifiers (0=Female, 1=Male).
        threshold: Global threshold fixed on validation data.

    Returns:
        Dict with ΔEOpp, per-group TPRs, and signed gap (Male - Female).
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true, dtype=int)
    group_ids = np.asarray(group_ids, dtype=int)

    rates = {}
    for gid, gname in [(0, "Female"), (1, "Male")]:
        mask = group_ids == gid
        if mask.sum() > 0:
            rates[gname] = _group_rates(y_true[mask], y_pred[mask])
        else:
            rates[gname] = {"tpr": float("nan"), "fpr": float("nan"), "ppr": float("nan")}

    delta_tpr = rates["Male"]["tpr"] - rates["Female"]["tpr"]

    return {
        "delta_eopp": abs(delta_tpr),
        "delta_tpr_signed": delta_tpr,  # Positive means Male has higher TPR
        "tpr_female": rates["Female"]["tpr"],
        "tpr_male": rates["Male"]["tpr"],
    }


def compute_equalized_odds_gap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_ids: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute equalized odds gap: ΔEOdds = max(|ΔTPR|, |ΔFPR|).

    A model satisfies equalized odds when both TPR and FPR are equal
    across groups. The gap captures the worst violation.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        group_ids: Group identifiers (0=Female, 1=Male).
        threshold: Global threshold from validation.

    Returns:
        Dict with ΔEOdds, per-group TPR/FPR, and component gaps.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true, dtype=int)
    group_ids = np.asarray(group_ids, dtype=int)

    rates = {}
    for gid, gname in [(0, "Female"), (1, "Male")]:
        mask = group_ids == gid
        if mask.sum() > 0:
            rates[gname] = _group_rates(y_true[mask], y_pred[mask])
        else:
            rates[gname] = {"tpr": float("nan"), "fpr": float("nan"), "ppr": float("nan")}

    delta_tpr = rates["Male"]["tpr"] - rates["Female"]["tpr"]
    delta_fpr = rates["Male"]["fpr"] - rates["Female"]["fpr"]

    return {
        "delta_eodds": max(abs(delta_tpr), abs(delta_fpr)),
        "delta_tpr": abs(delta_tpr),
        "delta_fpr": abs(delta_fpr),
        "delta_tpr_signed": delta_tpr,
        "delta_fpr_signed": delta_fpr,
        "tpr_female": rates["Female"]["tpr"],
        "tpr_male": rates["Male"]["tpr"],
        "fpr_female": rates["Female"]["fpr"],
        "fpr_male": rates["Male"]["fpr"],
    }


def compute_demographic_parity_gap(
    y_prob: np.ndarray,
    group_ids: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute demographic parity gap: ΔDP = |PPR_Female − PPR_Male|.

    NOTE: Demographic parity is reported as DESCRIPTIVE ONLY when disease
    prevalences differ between groups. It is NOT a fairness criterion in
    this setting.

    Args:
        y_prob: Predicted probabilities.
        group_ids: Group identifiers (0=Female, 1=Male).
        threshold: Global threshold.

    Returns:
        Dict with ΔDP and per-group positive prediction rates.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    group_ids = np.asarray(group_ids, dtype=int)

    pprs = {}
    for gid, gname in [(0, "Female"), (1, "Male")]:
        mask = group_ids == gid
        if mask.sum() > 0:
            pprs[gname] = float(y_pred[mask].mean())
        else:
            pprs[gname] = float("nan")

    delta_dp = abs(pprs["Male"] - pprs["Female"])

    return {
        "delta_dp": delta_dp,
        "ppr_female": pprs["Female"],
        "ppr_male": pprs["Male"],
    }


def compute_all_fairness_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_ids: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all fairness metrics in a single call.

    Returns a flat dict combining ΔEOpp, ΔEOdds, and ΔDP.
    """
    eopp = compute_equality_of_opportunity_gap(y_true, y_prob, group_ids, threshold)
    eodds = compute_equalized_odds_gap(y_true, y_prob, group_ids, threshold)
    dp = compute_demographic_parity_gap(y_prob, group_ids, threshold)

    return {**eopp, **eodds, **dp}
