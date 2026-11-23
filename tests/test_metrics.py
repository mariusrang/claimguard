"""Tests of the metrics, the calibration tools and the statistical tests (hand-made cases)."""

from __future__ import annotations

import numpy as np
import pytest

from claimguard.calibrate import (
    alert_metrics,
    expected_calibration_error,
    fit_temperature,
    nll,
    risk_coverage_curve,
    selective_metrics,
)
from claimguard.metrics import (
    accuracy,
    cohen_kappa,
    confusion_matrix,
    evidence_prf,
    fever_score,
    macro_f1,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from claimguard.stats import bootstrap_ci, mcnemar_test, paired_permutation_test


def test_retrieval_metrics():
    ranked, gold = ["a", "b", "c", "d"], {"b", "d"}
    assert recall_at_k(ranked, gold, 1) == 0.0
    assert recall_at_k(ranked, gold, 2) == 0.5
    assert recall_at_k(ranked, gold, 4) == 1.0
    assert precision_at_k(ranked, gold, 2) == 0.5
    assert reciprocal_rank(ranked, gold) == 0.5
    assert reciprocal_rank(ranked, {"zzz"}) == 0.0
    assert recall_at_k(ranked, set(), 3) == 0.0


def test_accuracy_confusion_and_macro_f1():
    gold = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    assert accuracy(gold, pred) == pytest.approx(4 / 6)
    cm = confusion_matrix(gold, pred)
    assert cm.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    # by hand: class0 P=1/2 R=1/2 F1=0.5 | class1 P=2/3 R=1 F1=0.8 | class2 P=1 R=1/2 F1=2/3
    assert macro_f1(gold, pred) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_macro_f1_perfect_and_empty_class():
    assert macro_f1(np.array([0, 1, 2]), np.array([0, 1, 2])) == 1.0
    # a class that never appears gets F1 = 0 (no crash, no division by zero)
    assert macro_f1(np.array([0, 0]), np.array([0, 0])) == pytest.approx(1 / 3)


def test_fever_score_needs_label_and_evidence():
    label_correct = np.array([True, True, True, False])
    evidence_found = np.array([True, False, False, True])
    gold_is_nei = np.array([False, False, True, False])
    # claim 1 ok, claim 2 right label but no evidence, claim 3 NEI ok, claim 4 wrong label
    assert fever_score(label_correct, evidence_found, gold_is_nei) == 0.5


def test_evidence_prf():
    predicted = [{("d1", 0), ("d1", 1)}, set()]
    gold = [{("d1", 1), ("d1", 2)}, {("d2", 0)}]
    m = evidence_prf(predicted, gold)
    assert m["precision"] == pytest.approx(1 / 2)
    assert m["recall"] == pytest.approx(1 / 3)
    assert m["f1"] == pytest.approx(0.4)


def test_ece_perfect_and_overconfident():
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([1, 1])) == 0.0
    # 4 predictions at 90% confidence but only half are right -> gap of 0.4
    conf, correct = np.full(4, 0.9), np.array([1, 1, 0, 0])
    assert expected_calibration_error(conf, correct) == pytest.approx(0.4)


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=600)
    # logits point to the right class only ~70% of the time but are huge -> overconfident
    logits = rng.normal(size=(600, 3))
    hit = rng.random(600) < 0.7
    logits[np.arange(600), np.where(hit, labels, (labels + 1) % 3)] += 8.0
    t = fit_temperature(logits, labels)
    assert t > 1.5  # must soften the probabilities
    assert nll(logits, labels, t) < nll(logits, labels, 1.0)
    assert fit_temperature(np.zeros((0, 3)), np.zeros(0, dtype=int)) == 1.0


def test_risk_coverage_and_selective():
    conf = np.array([0.9, 0.8, 0.6, 0.5])
    correct = np.array([1, 1, 0, 0])
    curve = risk_coverage_curve(conf, correct)
    assert curve["coverage"] == [0.25, 0.5, 0.75, 1.0]
    assert curve["risk"] == pytest.approx([0.0, 0.0, 1 / 3, 0.5])
    sel = selective_metrics(conf, correct, threshold=0.7)
    assert sel["coverage"] == 0.5 and sel["selective_accuracy"] == 1.0


def test_alert_metrics():
    pred = np.array([1, 2, 0, 1])  # alerts on items 0, 1, 3
    gold = np.array([1, 0, 0, 2])  # item 1 is a false alert
    conf = np.array([0.9, 0.6, 0.9, 0.8])
    m = alert_metrics(pred, gold, conf, threshold=0.0)
    assert m["n_alerts"] == 3 and m["alert_precision"] == pytest.approx(2 / 3) and m["alert_recall"] == 1.0
    strict = alert_metrics(pred, gold, conf, threshold=0.7)  # drops the low-confidence false alert
    assert strict["n_alerts"] == 2 and strict["alert_precision"] == 1.0


def test_bootstrap_ci_contains_the_value():
    gold = np.array([0, 1] * 50)
    pred = np.array([0, 1] * 40 + [1, 0] * 10)
    ci = bootstrap_ci(accuracy, gold, pred, n_boot=300)
    assert ci["value"] == pytest.approx(0.8)
    assert ci["low"] < 0.8 < ci["high"]
    assert ci["high"] - ci["low"] < 0.25


def test_mcnemar():
    a = np.array([1] * 10 + [0] * 0 + [1] * 5, dtype=bool)
    b = np.array([0] * 10 + [1] * 0 + [1] * 5, dtype=bool)
    res = mcnemar_test(a, b)
    assert res["b"] == 10 and res["c"] == 0
    assert res["p_value"] == pytest.approx(2 * 0.5**10)
    assert mcnemar_test(a, a)["p_value"] == 1.0


def test_paired_permutation():
    rng = np.random.default_rng(1)
    base = rng.random(200)
    assert paired_permutation_test(base + 0.2, base, n_perm=2000)["p_value"] < 0.01
    assert paired_permutation_test(base, base)["p_value"] == 1.0


def test_cohen_kappa():
    a = ["x", "x", "y", "y"]
    assert cohen_kappa(a, a) == 1.0
    # observed agreement 0.5, expected 0.5 -> kappa 0 (no better than chance)
    assert cohen_kappa(a, ["x", "y", "x", "y"]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cohen_kappa(["x"], [])


def test_tune_on_dev_picks_the_best_setting():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from evaluate import tune_on_dev

    S, N = [3.0, 0.0, 0.0], [0.0, 0.0, 3.0]
    # a SUPPORTED claim whose only supporting passage is ranked first, then 4 NEI passages:
    # a majority vote over k >= 3 says NEI, so the best setting must use k = 1 or max-confidence
    record = {"gold": "SUPPORTED", "gold_evidence": {"d1": [0]},
              "evidence": [{"source": "d1", "logits": S}] + [{"source": f"x{i}", "logits": N} for i in range(4)]}
    nei = {"gold": "NEI", "gold_evidence": {}, "evidence": [{"source": "y", "logits": N}] * 5}
    choice = tune_on_dev([record, nei], temperature=1.0)
    assert choice["dev_macro_f1"] == pytest.approx(2 / 3)  # S and NEI right, REFUTED absent -> F1 0
    assert choice["top_k"] == 1  # ties are broken towards fewer passages
    assert len(choice["grid"]) == 12
