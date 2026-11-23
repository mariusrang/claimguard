"""Prepare SciFact (or the toy sample) for the experiments.

    python data/scripts/preprocess.py --raw data/raw/scifact --out data/processed/scifact
    python data/scripts/preprocess.py --raw data/sample      --out data/processed/sample

What it does:
- the official test set has no public labels, so the official DEV set becomes my TEST set
- the official TRAIN set is split into train / dev (stratified on the label)
- writes stats.json (number of claims per label, lengths...) for the README and the report
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from claimguard.data import load_corpus, parse_scifact_claim
from claimguard.text import tokenize
from claimguard.utils import read_jsonl, save_json, write_jsonl


def stratified_split(rows: list[dict], dev_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split so that train and dev have the same proportion of each label."""
    by_label = defaultdict(list)
    for row in rows:
        by_label[parse_scifact_claim(row).label].append(row)
    rng = random.Random(seed)
    train, dev = [], []
    for label in sorted(by_label):
        group = by_label[label]
        rng.shuffle(group)
        n_dev = max(1, round(len(group) * dev_ratio))
        dev.extend(group[:n_dev])
        train.extend(group[n_dev:])
    rng.shuffle(train)
    rng.shuffle(dev)
    return train, dev


def claim_stats(rows: list[dict]) -> dict:
    claims = [parse_scifact_claim(r) for r in rows]
    lengths = [len(tokenize(c.text, remove_stopwords=False)) for c in claims]
    return {
        "n_claims": len(claims),
        "labels": dict(Counter(c.label for c in claims)),
        "avg_claim_tokens": round(sum(lengths) / max(1, len(lengths)), 1),
        "avg_evidence_docs": round(
            sum(len(c.evidence) for c in claims) / max(1, sum(1 for c in claims if c.evidence)), 2
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/scifact")
    parser.add_argument("--out", default="data/processed/scifact")
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(raw / "corpus.jsonl", out / "corpus.jsonl")
    train_rows, dev_rows = stratified_split(read_jsonl(raw / "claims_train.jsonl"), args.dev_ratio, args.seed)
    test_rows = read_jsonl(raw / "claims_dev.jsonl")  # official dev = our test
    write_jsonl(out / "claims_train.jsonl", train_rows)
    write_jsonl(out / "claims_dev.jsonl", dev_rows)
    write_jsonl(out / "claims_test.jsonl", test_rows)

    corpus = load_corpus(out / "corpus.jsonl")
    n_sents = [len(d.sentences) for d in corpus.values()]
    stats = {
        "corpus": {
            "n_documents": len(corpus),
            "avg_sentences_per_doc": round(sum(n_sents) / max(1, len(n_sents)), 1),
        },
        "train": claim_stats(train_rows),
        "dev": claim_stats(dev_rows),
        "test": claim_stats(test_rows),
    }
    save_json(out / "stats.json", stats)
    for split in ("train", "dev", "test"):
        print(split, stats[split])
    print("corpus", stats["corpus"])


if __name__ == "__main__":
    main()
