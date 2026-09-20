"""Weak-calibration diagnostics (Cox 1958 logistic recalibration) without statsmodels.

    P(Y=1 | p) = sigmoid(a + b * logit(p))
    slope b:                    1 = right spread; <1 over-confident; >1 under-confident
    intercept a (joint fit):    systematic shift after accounting for the slope
    intercept_fixed_slope:      a with b := 1 (calibration-in-the-large): the pure
                                over/under-prediction offset. a > 0 means observed
                                outcome rates exceed predicted risk (under-prediction).
    mean_bias:                  mean(y) - mean(p), the same quantity on the probability scale.

Fits are Newton-Raphson maximum likelihood (identical to IRLS / statsmodels GLM Binomial);
confidence intervals are Wald. Degenerate inputs return NaN rather than raising.
"""
import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

EPS = 1e-6
KEYS = ['slope', 'slope_ci_low', 'slope_ci_high', 'intercept', 'intercept_ci_low', 'intercept_ci_high',
        'intercept_fixed_slope', 'intercept_fixed_slope_ci_low', 'intercept_fixed_slope_ci_high', 'mean_bias', 'n', 'n_pos']


def _newton(y, X, offset, ridge=1e-10, iters=100):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = expit(offset + X @ beta)
        w = np.clip(mu * (1 - mu), 1e-12, None)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = expit(offset + X @ beta)
    w = np.clip(mu * (1 - mu), 1e-12, None)
    cov = np.linalg.inv(X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1]))
    return beta, np.sqrt(np.diag(cov))


def cox_calibration(y, p, alpha=.05, eps=EPS):
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    keep = np.isfinite(y) & np.isfinite(p)
    y, p = y[keep], p[keep]
    out = {k: float('nan') for k in KEYS}
    out['n'], out['n_pos'] = int(y.size), int(y.sum())
    if y.size < 2 or out['n_pos'] in (0, y.size):
        return out
    out['mean_bias'] = float(y.mean() - p.mean())
    lp = logit(np.clip(p, eps, 1 - eps))
    z = norm.ppf(1 - alpha / 2)
    if np.ptp(lp) > 0:
        beta, se = _newton(y, np.column_stack([np.ones_like(lp), lp]), np.zeros_like(lp))
        out['intercept'], out['slope'] = float(beta[0]), float(beta[1])
        out['intercept_ci_low'], out['intercept_ci_high'] = float(beta[0] - z * se[0]), float(beta[0] + z * se[0])
        out['slope_ci_low'], out['slope_ci_high'] = float(beta[1] - z * se[1]), float(beta[1] + z * se[1])
    beta, se = _newton(y, np.ones((y.size, 1)), lp)
    out['intercept_fixed_slope'] = float(beta[0])
    out['intercept_fixed_slope_ci_low'], out['intercept_fixed_slope_ci_high'] = float(beta[0] - z * se[0]), float(beta[0] + z * se[0])
    return out


def group_calibration(y, p, a):
    """Overall and per-group Cox diagnostics plus between-group gaps (group1 - group0)."""
    y, p, a = np.asarray(y, float), np.asarray(p, float), np.asarray(a)
    rows = {'': cox_calibration(y, p)}
    for g in (0, 1):
        rows[f'group{g}_'] = cox_calibration(y[a == g], p[a == g])
    flat = {f'{prefix}cal_{k}': v for prefix, r in rows.items() for k, v in r.items()}
    for k in ['slope', 'intercept', 'intercept_fixed_slope', 'mean_bias']:
        flat[f'cal_{k}_gap'] = flat[f'group1_cal_{k}'] - flat[f'group0_cal_{k}']
    return flat
