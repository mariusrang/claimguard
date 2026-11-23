"""Calibration and abstention.

A model is calibrated if "90% confident" really means "right 9 times out of 10".
Neural nets are usually over-confident, so I fit one scalar T (temperature) on
the dev set and divide the logits by T (Guo et al., 2017). It does not change
the predictions, only how confident they look.

With calibrated confidences the system can abstain ("UNCERTAIN") below a
threshold. The risk-coverage curve shows the trade-off: the fewer claims we
answer (coverage), the fewer mistakes we make on them (risk).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from claimguard import SUPPORTED
from claimguard.utils import softmax


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the gold labels."""
    probs = softmax(np.asarray(logits) / temperature)
    picked = probs[np.arange(len(labels)), np.asarray(labels)]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find T > 0 that minimises the NLL on held-out data (1-D search on log T)."""
    if len(labels) == 0:
        return 1.0
    result = minimize_scalar(
        lambda log_t: nll(logits, labels, float(np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )
    return float(np.exp(result.x))


def reliability_bins(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Split predictions in confidence bins -> mean confidence vs accuracy per bin."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # the last bin includes confidence == 1.0
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence >= lo) & (confidence <= hi)
        if mask.any():
            bins.append(
                {
                    "low": float(lo),
                    "high": float(hi),
                    "count": int(mask.sum()),
                    "confidence": float(confidence[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    return bins


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE = weighted average of |accuracy - confidence| over the bins (0 = perfect)."""
    n = len(confidence)
    if n == 0:
        return 0.0
    bins = reliability_bins(confidence, correct, n_bins)
    return float(sum(b["count"] / n * abs(b["accuracy"] - b["confidence"]) for b in bins))


def risk_coverage_curve(confidence: np.ndarray, correct: np.ndarray) -> dict:
    """Sort by confidence, then for each coverage level compute the error rate (risk).

    AURC = area under this curve, lower is better (errors are concentrated in
    the low-confidence predictions, which is what we want for abstention).
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    n = len(confidence)
    if n == 0:
        return {"coverage": [], "risk": [], "aurc": 0.0}
    order = np.argsort(-confidence, kind="stable")
    errors = 1.0 - correct[order]
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(errors) / np.arange(1, n + 1)
    return {"coverage": coverage.tolist(), "risk": risk.tolist(), "aurc": float(risk.mean())}


def selective_metrics(confidence: np.ndarray, correct: np.ndarray, threshold: float) -> dict:
    """Coverage and accuracy when we only answer if confidence >= threshold."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    answered = confidence >= threshold
    coverage = float(answered.mean()) if len(answered) else 0.0
    acc = float(correct[answered].mean()) if answered.any() else 0.0
    return {"threshold": threshold, "coverage": coverage, "selective_accuracy": acc}


def alert_metrics(pred: np.ndarray, gold: np.ndarray, confidence: np.ndarray, threshold: float = 0.0) -> dict:
    """The metric I would show to a non-technical person.

    An "alert" = the system says a claim is NOT supported (REFUTED or NEI) with
    confidence >= threshold. Alert precision = out of 10 alerts, how many are
    real problems. If it is low, users learn to ignore the tool.
    """
    pred, gold = np.asarray(pred), np.asarray(gold)
    alerts = (pred != SUPPORTED) & (np.asarray(confidence) >= threshold)
    truly_unsupported = gold != SUPPORTED
    tp = int((alerts & truly_unsupported).sum())
    precision = tp / int(alerts.sum()) if alerts.any() else 0.0
    recall = tp / int(truly_unsupported.sum()) if truly_unsupported.any() else 0.0
    return {
        "threshold": threshold,
        "n_alerts": int(alerts.sum()),
        "alert_precision": float(precision),
        "alert_recall": float(recall),
    }
