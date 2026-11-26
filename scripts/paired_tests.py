"""Paired tests between ClaimGuard and the baselines on the SciFact test claims.

    python scripts/paired_tests.py      # reads results/predictions*.jsonl, writes results/paired_tests.json

Each system is scored with its own settings chosen on dev (k, aggregation, temperature,
from results/pipeline*.json). Two paired comparisons on the same 300 claims:
- McNemar's exact test on correct / wrong (accuracy),
- a paired bootstrap of the macro-F1 difference (same resampled claims for both systems).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

from claimguard.metrics import macro_f1  # noqa: E402
from claimguard.stats import mcnemar_test  # noqa: E402
from claimguard.utils import read_jsonl, save_json  # noqa: E402

SYSTEMS = {
    "main": "", "seed13": "_seed13", "seed42": "_seed42", "seed2026": "_seed2026",
    "zero_shot_nli": "_zeroshot", "llm_judge": "_llm_judge", "bm25_zero_shot": "_bm25_zeroshot",
    "rules": "_lexical", "plain_encoder": "_plain_encoder",
}
BASELINES = ["zero_shot_nli", "llm_judge", "bm25_zero_shot", "rules", "plain_encoder"]


def predictions(results: Path, tag: str) -> tuple[list, np.ndarray, np.ndarray]:
    """Claim ids, gold and predicted labels of one system, with its dev-chosen settings."""
    records = read_jsonl(results / f"predictions{tag}.jsonl")
    meta = json.loads((results / f"pipeline{tag}.json").read_text())
    chosen = meta["chosen_on_dev"]
    temperature = meta.get("temperature") or 1.0
    s = ev.score(records, chosen["top_k"], chosen["aggregation"], float(temperature))
    return [r["claim_id"] for r in records], s["gold"], s["pred"]


def paired_bootstrap_f1(gold, a, b, n_boot: int = 2000, seed: int = 13) -> dict:
    """Macro-F1(a) - macro-F1(b) with a 95% interval, resampling the same claims for both."""
    rng = np.random.default_rng(seed)
    n = len(gold)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(macro_f1(gold[idx], a[idx]) - macro_f1(gold[idx], b[idx]))
    diffs = np.array(diffs)
    return {"diff": float(macro_f1(gold, a) - macro_f1(gold, b)), "low": float(np.percentile(diffs, 2.5)),
            "high": float(np.percentile(diffs, 97.5)), "share_above_0": float((diffs > 0).mean())}


def main() -> None:
    results = Path("results")
    preds = {name: predictions(results, tag) for name, tag in SYSTEMS.items()}
    ref_ids = preds["main"][0]
    out = {"n_claims": len(ref_ids), "comparisons": {}}
    for system in ("main", "seed13", "seed42", "seed2026"):
        ids, gold, pred = preds[system]
        assert ids == ref_ids, "systems must be scored on the same claims in the same order"
        for base in BASELINES:
            bids, bgold, bpred = preds[base]
            assert bids == ref_ids and (bgold == gold).all()
            key = f"{system}_vs_{base}"
            out["comparisons"][key] = {
                "macro_f1": [float(macro_f1(gold, pred)), float(macro_f1(gold, bpred))],
                "f1_difference": paired_bootstrap_f1(gold, pred, bpred),
                "mcnemar": mcnemar_test(pred == gold, bpred == gold),
            }
    save_json(results / "paired_tests.json", out)
    for key, c in out["comparisons"].items():
        d = c["f1_difference"]
        print(f"{key:32s} dF1 {d['diff']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}]  McNemar p={c['mcnemar']['p_value']:.4f}")


if __name__ == "__main__":
    main()
