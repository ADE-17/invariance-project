"""
Distribution distance metrics between group representations.

  - MMD (Maximum Mean Discrepancy) with Gaussian RBF kernel + median heuristic
  - HSIC (Hilbert-Schmidt Independence Criterion) between representations and
    group indicator

These measure distributional alignment in the representation space,
complementing the probe-based decodability measurements.
"""

from typing import Optional

import numpy as np


def compute_mmd(
    features_a: np.ndarray,
    features_b: np.ndarray,
    bandwidth: Optional[float] = None,
    n_subsample: int = 2000,
    seed: int = 42,
) -> float:
    """
    Estimate MMD^2 between two sets of representations using Gaussian RBF kernel.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]

    where k is the RBF kernel and x~P, y~Q are representations from two groups.

    Args:
        features_a: (N_a, D) representations from group A.
        features_b: (N_b, D) representations from group B.
        bandwidth: RBF kernel bandwidth σ. If None, uses median heuristic.
        n_subsample: Max samples per group (for computational efficiency).
        seed: Random seed for subsampling.

    Returns:
        Estimated MMD^2 value (≥ 0, lower means more similar distributions).
    """
    rng = np.random.RandomState(seed)
    a = np.asarray(features_a, dtype=np.float64)
    b = np.asarray(features_b, dtype=np.float64)

    if len(a) == 0 or len(b) == 0:
        return float("nan")

    # Subsample for efficiency
    if len(a) > n_subsample:
        idx = rng.choice(len(a), size=n_subsample, replace=False)
        a = a[idx]
    if len(b) > n_subsample:
        idx = rng.choice(len(b), size=n_subsample, replace=False)
        b = b[idx]

    # Median heuristic for bandwidth
    if bandwidth is None:
        combined = np.concatenate([a, b], axis=0)
        # Use pairwise distances on a subsample for efficiency
        sub = combined[:min(500, len(combined))]
        dists = _pairwise_sq_distances(sub, sub)
        median_dist = np.sqrt(np.median(dists[dists > 0]))
        bandwidth = median_dist if median_dist > 0 else 1.0

    gamma = 1.0 / (2.0 * bandwidth ** 2)

    # Compute kernel matrices
    kaa = np.exp(-gamma * _pairwise_sq_distances(a, a))
    kbb = np.exp(-gamma * _pairwise_sq_distances(b, b))
    kab = np.exp(-gamma * _pairwise_sq_distances(a, b))

    # Unbiased estimator
    n_a = len(a)
    n_b = len(b)

    # Zero out diagonal for unbiased estimate
    np.fill_diagonal(kaa, 0)
    np.fill_diagonal(kbb, 0)

    term1 = kaa.sum() / (n_a * (n_a - 1)) if n_a > 1 else 0.0
    term2 = kbb.sum() / (n_b * (n_b - 1)) if n_b > 1 else 0.0
    term3 = kab.sum() / (n_a * n_b)

    mmd_sq = term1 + term2 - 2.0 * term3
    return float(max(mmd_sq, 0.0))  # Clamp numerical noise


def compute_hsic(
    features: np.ndarray,
    group_ids: np.ndarray,
    bandwidth: Optional[float] = None,
    n_subsample: int = 2000,
    seed: int = 42,
) -> float:
    """
    Estimate HSIC between representations and binary group indicator.

    HSIC measures statistical dependence between the representation Z and
    the demographic attribute A in a kernel RKHS. HSIC=0 iff Z⊥A
    (in the limit of sufficient kernel richness).

    Uses biased HSIC estimator: HSIC = (1/n^2) tr(KHLH)
    where K is the feature kernel, L is the group kernel, H = I - 11'/n.

    Args:
        features: (N, D) representations.
        group_ids: (N,) binary group labels (0 or 1).
        bandwidth: RBF bandwidth for feature kernel. Median heuristic if None.
        n_subsample: Max samples for efficiency.
        seed: Random seed.

    Returns:
        Estimated HSIC value (≥ 0, lower means more independent).
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(features, dtype=np.float64)
    a = np.asarray(group_ids, dtype=np.float64)

    if len(X) == 0:
        return float("nan")

    # Subsample
    if len(X) > n_subsample:
        idx = rng.choice(len(X), size=n_subsample, replace=False)
        X = X[idx]
        a = a[idx]

    n = len(X)

    # Feature kernel (Gaussian RBF)
    if bandwidth is None:
        sub = X[:min(500, n)]
        dists = _pairwise_sq_distances(sub, sub)
        median_dist = np.sqrt(np.median(dists[dists > 0]))
        bandwidth = median_dist if median_dist > 0 else 1.0

    gamma = 1.0 / (2.0 * bandwidth ** 2)
    K = np.exp(-gamma * _pairwise_sq_distances(X, X))

    # Group kernel (linear on binary labels)
    L = np.outer(a, a) + np.outer(1 - a, 1 - a)  # Delta kernel for binary

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # Biased HSIC
    hsic = np.trace(K @ H @ L @ H) / (n ** 2)
    return float(max(hsic, 0.0))


def _pairwise_sq_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise squared Euclidean distances between rows of X and Y.

    Returns:
        (N_x, N_y) distance matrix.
    """
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    Y_sq = np.sum(Y ** 2, axis=1, keepdims=True)
    dists = X_sq + Y_sq.T - 2.0 * X @ Y.T
    return np.maximum(dists, 0.0)  # Clamp numerical noise
