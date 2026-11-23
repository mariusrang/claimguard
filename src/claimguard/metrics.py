"""All evaluation metrics, written with numpy so I know exactly what they compute."""

from __future__ import annotations

import numpy as np

# ---------- retrieval ----------


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Share of the gold documents that are in the top-k."""
    if not gold:
        return 0.0
    return len(set(ranked[:k]) & gold) / len(gold)


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Share of the top-k that is gold (divided by k, even if fewer docs came back)."""
    if k <= 0:
        return 0.0
    return len(set(ranked[:k]) & gold) / k


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    """1 / rank of the first gold document (0 if none was retrieved)."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


# ---------- classification ----------


def accuracy(gold: np.ndarray, pred: np.ndarray) -> float:
    gold, pred = np.asarray(gold), np.asarray(pred)
    return float((gold == pred).mean()) if len(gold) else 0.0


def confusion_matrix(gold: np.ndarray, pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """cm[i, j] = number of examples with gold class i predicted as class j."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for g, p in zip(np.asarray(gold), np.asarray(pred)):
        cm[int(g), int(p)] += 1
    return cm


def per_class_prf(gold: np.ndarray, pred: np.ndarray, n_classes: int = 3) -> list[dict]:
    """Precision / recall / F1 for each class."""
    cm = confusion_matrix(gold, pred, n_classes)
    out = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out.append({"precision": float(p), "recall": float(r), "f1": float(f1), "support": int(cm[c].sum())})
    return out


def macro_f1(gold: np.ndarray, pred: np.ndarray, n_classes: int = 3) -> float:
    """Mean of the per-class F1 (every class counts the same, even the rare ones)."""
    return float(np.mean([m["f1"] for m in per_class_prf(gold, pred, n_classes)]))


# ---------- end-to-end (retrieval + verification) ----------


def fever_score(label_correct: np.ndarray, evidence_found: np.ndarray, gold_is_nei: np.ndarray) -> float:
    """FEVER-style strict score.

    A claim counts as correct only if the label is right AND (for SUPPORTED/REFUTED
    claims) at least one gold evidence document was retrieved. For NEI claims only
    the label matters since there is no evidence to find.
    """
    label_correct = np.asarray(label_correct, dtype=bool)
    ok = label_correct & (np.asarray(evidence_found, dtype=bool) | np.asarray(gold_is_nei, dtype=bool))
    return float(ok.mean()) if len(ok) else 0.0


def evidence_prf(
    predicted: list[set[tuple[str, int]]], gold: list[set[tuple[str, int]]]
) -> dict:
    """Micro precision/recall/F1 on evidence sentences, each one a (doc_id, sentence_idx)."""
    tp = sum(len(p & g) for p, g in zip(predicted, gold))
    n_pred = sum(len(p) for p in predicted)
    n_gold = sum(len(g) for g in gold)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


# ---------- annotation agreement ----------


def cohen_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's kappa between two annotations of the same items.

    kappa = (observed agreement - agreement expected by chance) / (1 - expected).
    I use it to check that my error categories are consistent when I annotate
    the same errors a second time, one week later.
    """
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("need two annotation lists of the same (non-zero) length")
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    categories = set(labels_a) | set(labels_b)
    expected = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories)
    if expected == 1.0:
        return 1.0  # both annotators used one single category everywhere
    return (observed - expected) / (1.0 - expected)
