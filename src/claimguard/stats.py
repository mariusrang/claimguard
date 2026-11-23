"""Statistical tests, so I can say if a difference is real or just noise.

The SciFact test set only has ~300 claims, so a 2-point gap can easily be luck.
- bootstrap_ci: 95% confidence interval of any metric
- mcnemar_test: compare two classifiers on the same examples
- paired_permutation_test: compare two systems on a per-example score (ex: recall@5)
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> dict:
    """Percentile bootstrap: resample the examples with replacement n_boot times."""
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    if n == 0:
        return {"value": 0.0, "low": 0.0, "high": 0.0}
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # same indexes for every array = paired resampling
        values[b] = metric_fn(*(a[idx] for a in arrays))
    return {
        "value": float(metric_fn(*arrays)),
        "low": float(np.quantile(values, alpha / 2)),
        "high": float(np.quantile(values, 1 - alpha / 2)),
    }


def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact McNemar test.

    Only the examples where the two systems disagree matter:
    b = A right and B wrong, c = A wrong and B right.
    Under H0 (same error rate), b follows Binomial(b + c, 0.5).
    """
    a = np.asarray(correct_a, dtype=bool)
    b_arr = np.asarray(correct_b, dtype=bool)
    b = int((a & ~b_arr).sum())
    c = int((~a & b_arr).sum())
    if b + c == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    p = binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
    return {"b": b, "c": c, "p_value": float(p)}


def paired_permutation_test(
    scores_a: np.ndarray, scores_b: np.ndarray, n_perm: int = 10000, seed: int = 13
) -> dict:
    """Paired permutation (sign-flip) test on the mean difference of per-example scores."""
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    observed = float(diff.mean()) if len(diff) else 0.0
    if len(diff) == 0 or np.allclose(diff, 0):
        return {"mean_diff": observed, "p_value": 1.0}
    rng = np.random.default_rng(seed)
    # under H0 the sign of each difference is random, so flip signs and see how
    # often we get a mean as extreme as the one we observed
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    perm_means = (signs * diff).mean(axis=1)
    p = (np.sum(np.abs(perm_means) >= abs(observed) - 1e-12) + 1) / (n_perm + 1)
    return {"mean_diff": observed, "p_value": float(p)}
