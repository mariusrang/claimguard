"""Evaluation on LLM-AggreFact: is a sentence written by an LLM supported by its source document?

This is the "check an LLM answer" extension evaluated on real model outputs
(summaries and RAG answers from several LLMs, annotated by humans).
Dataset on the Hugging Face hub: lytang/LLM-AggreFact (Tang et al., 2024).

Each row = (doc, claim, label) with label 1 = supported, 0 = not supported.
It is a BINARY task, so I map our 3 labels to 2: SUPPORTED vs (REFUTED or NEI).
The benchmark metric is balanced accuracy (mean of the recall of both classes),
because some sub-datasets have very few unsupported claims.

Documents can be long, so I do NOT give the whole document to the verifier (it
would be truncated at 384 tokens). I use the "sources" mode of the pipeline:
the document is cut into chunks, BM25 + reranker pick the best chunks, then the
verifier and the aggregation run as usual. So this also tests the full pipeline.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

DATASET_NAME = "lytang/LLM-AggreFact"


def balanced_accuracy(gold_supported: np.ndarray, pred_supported: np.ndarray) -> float:
    """Mean recall of the 'supported' class and of the 'not supported' class."""
    gold = np.asarray(gold_supported, dtype=bool)
    pred = np.asarray(pred_supported, dtype=bool)
    recalls = [float((pred[gold == c] == c).mean()) for c in (True, False) if (gold == c).any()]
    return float(np.mean(recalls)) if recalls else 0.0


def load_rows(split: str = "test") -> list[dict]:
    """Download the benchmark with the `datasets` library (needs internet, works on Colab)."""
    from datasets import load_dataset

    return [dict(row) for row in load_dataset(DATASET_NAME, split=split)]


def sample_per_dataset(rows: list[dict], per_dataset: int | None, seed: int = 13) -> list[dict]:
    """Keep at most `per_dataset` rows of each sub-dataset (the full benchmark is too big for Colab)."""
    by_name: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_name[row["dataset"]].append(row)
    rng = random.Random(seed)
    out = []
    for name in sorted(by_name):
        group = list(by_name[name])
        rng.shuffle(group)
        out.extend(group if per_dataset is None else group[:per_dataset])
    return out


def evaluate_rows(pipeline, rows: list[dict]) -> dict:
    """Run the pipeline on each (doc, claim) and compute balanced accuracy per sub-dataset."""
    per_name: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    latencies = []
    for n, row in enumerate(rows, start=1):
        result = pipeline.verify_claim(row["claim"], [row["doc"]])
        per_name[row["dataset"]].append((bool(int(row["label"])), result.label == "SUPPORTED"))
        latencies.append(sum(result.timings_ms.values()))
        if n % 200 == 0:
            print(f"  {n}/{len(rows)} claims done")
    by_dataset = {}
    for name, pairs in sorted(per_name.items()):
        gold, pred = np.array([g for g, _ in pairs]), np.array([p for _, p in pairs])
        by_dataset[name] = {"n": len(pairs), "balanced_accuracy": balanced_accuracy(gold, pred)}
    all_pairs = [p for pairs in per_name.values() for p in pairs]
    gold, pred = np.array([g for g, _ in all_pairs]), np.array([p for _, p in all_pairs])
    return {
        "n": len(all_pairs),
        "by_dataset": by_dataset,
        # the benchmark reports the plain average over sub-datasets, so I do the same
        "mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in by_dataset.values()])),
        "pooled_balanced_accuracy": balanced_accuracy(gold, pred),
        "latency_ms_per_claim": {"p50": float(np.percentile(latencies, 50)), "p95": float(np.percentile(latencies, 95))},
    }
