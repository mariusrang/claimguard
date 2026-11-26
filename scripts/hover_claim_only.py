"""Claim-only baseline on HoVer: can the label be guessed without reading any article?

    python scripts/hover_claim_only.py --hover-dir path/to/hover/data/hover

TF-IDF (word 1-2 grams) + logistic regression trained on the HoVer train claims, evaluated
on the same balanced dev sample as scripts/run_multihop.py (100 claims per hops x label,
seed 13). It tells how much of the multi-hop accuracy could come from wording artefacts.
Writes results/hover_claim_only.json.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from claimguard.multihop import load_hover
from claimguard.stats import bootstrap_ci
from claimguard.utils import save_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hover-dir", required=True)
    ap.add_argument("--per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="results/hover_claim_only.json")
    args = ap.parse_args()
    hover = Path(args.hover_dir)
    train = load_hover(hover / "hover_train_release_v1.1.json")  # all train claims
    dev = load_hover(hover / "hover_dev_release_v1.1.json", args.per_cell, args.seed)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train = vec.fit_transform([c.claim for c in train])
    y_train = np.array([c.supported for c in train])
    # balanced classes: HoVer train has more NOT_SUPPORTED claims than SUPPORTED ones
    model = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(x_train, y_train)

    y_dev = np.array([c.supported for c in dev])
    pred = model.predict(vec.transform([c.claim for c in dev]))
    correct = (pred == y_dev).astype(float)
    ci = bootstrap_ci(np.mean, correct)
    by_hops = {str(h): float(correct[[c.num_hops == h for c in dev]].mean()) for h in sorted({c.num_hops for c in dev})}
    out = {"n_train": len(train), "n_dev": len(dev), "accuracy": float(correct.mean()), "ci95": ci, "accuracy_by_hops": by_hops,
           "chance": 0.5, "model": "tf-idf (1-2 grams) + logistic regression, claim text only"}
    save_json(Path(args.out), out)
    print(out)


if __name__ == "__main__":
    main()
