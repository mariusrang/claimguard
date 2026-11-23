"""Build the contrast set from the SUPPORTED claims of the SciFact test split.

    python data/scripts/build_contrast_set.py --data data/processed/scifact

Output: data/processed/<name>/contrast.jsonl  (edited claims, expected NOT supported)
        data/processed/<name>/contrast_originals.jsonl (the untouched pairs, expected SUPPORTED)
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from claimguard.contrast import build_contrast_set
from claimguard.data import load_scifact
from claimguard.utils import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/processed/scifact")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    corpus, claims = load_scifact(args.data, args.split)
    originals = []
    for claim in claims:
        if claim.label != "SUPPORTED":
            continue
        # evidence text = the gold rationale sentences of the first evidence document
        doc_id, sent_ids = next(iter(claim.evidence.items()))
        if doc_id not in corpus:
            continue
        doc = corpus[doc_id]
        text = " ".join(doc.sentences[i] for i in sent_ids if i < len(doc.sentences))
        originals.append({"id": claim.claim_id, "claim": claim.text, "document": text, "label": "SUPPORTED"})

    contrast = build_contrast_set(originals, seed=args.seed)
    out = Path(args.data)
    write_jsonl(out / "contrast_originals.jsonl", originals)
    write_jsonl(out / "contrast.jsonl", contrast)
    print(f"{len(originals)} supported pairs -> {len(contrast)} contrast examples")
    print(dict(Counter(row["perturbation"] for row in contrast)))


if __name__ == "__main__":
    main()
