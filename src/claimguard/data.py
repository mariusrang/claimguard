"""Dataclasses + loaders for every dataset we use.

Two kinds of data:
- retrieval datasets (SciFact): a corpus of documents + claims with gold evidence
- "given evidence" datasets (Climate-FEVER, our French bank set, the contrast set):
  each claim already comes with its evidence text, so we only test the verifier
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from claimguard.utils import read_jsonl

# how each dataset names its labels -> our names
LABEL_ALIASES = {
    "SUPPORT": "SUPPORTED",
    "SUPPORTS": "SUPPORTED",
    "SUPPORTED": "SUPPORTED",
    "CONTRADICT": "REFUTED",
    "REFUTES": "REFUTED",
    "REFUTED": "REFUTED",
    "NOT_ENOUGH_INFO": "NEI",
    "NOT ENOUGH INFO": "NEI",
    "NEI": "NEI",
}


@dataclass
class Document:
    """One abstract, already split into sentences."""

    doc_id: str
    title: str
    sentences: list[str]

    @property
    def text(self) -> str:
        body = " ".join(self.sentences)
        return f"{self.title}. {body}" if self.title else body


@dataclass
class Claim:
    """A claim with its gold label and gold evidence (doc_id -> sentence indices)."""

    claim_id: str
    text: str
    label: str  # SUPPORTED / REFUTED / NEI
    evidence: dict[str, list[int]] = field(default_factory=dict)
    cited_doc_ids: list[str] = field(default_factory=list)


@dataclass
class EvidenceClaim:
    """A claim that comes with its evidence text (no retrieval needed)."""

    claim_id: str
    text: str
    label: str
    evidence_texts: list[str]
    meta: dict = field(default_factory=dict)


def parse_scifact_claim(row: dict) -> Claim:
    """Convert one line of the original SciFact claims file to our Claim."""
    evidence: dict[str, list[int]] = {}
    labels = set()
    for doc_id, rationales in row.get("evidence", {}).items():
        sent_ids: list[int] = []
        for rationale in rationales:
            sent_ids.extend(rationale["sentences"])
            labels.add(LABEL_ALIASES[rationale["label"]])
        evidence[str(doc_id)] = sorted(set(sent_ids))
    # a claim with no evidence at all is NEI. SciFact never mixes SUPPORT and
    # CONTRADICT for one claim, but if it happened we keep REFUTED to be safe
    if not labels:
        label = "NEI"
    elif "REFUTED" in labels:
        label = "REFUTED"
    else:
        label = "SUPPORTED"
    return Claim(
        claim_id=str(row["id"]),
        text=row["claim"],
        label=label,
        evidence=evidence,
        cited_doc_ids=[str(d) for d in row.get("cited_doc_ids", [])],
    )


def load_corpus(path: str | Path) -> dict[str, Document]:
    """Load a corpus.jsonl (SciFact format) as {doc_id: Document}."""
    corpus = {}
    for row in read_jsonl(path):
        doc = Document(str(row["doc_id"]), row["title"], list(row["abstract"]))
        corpus[doc.doc_id] = doc
    return corpus


def load_claims(path: str | Path) -> list[Claim]:
    """Load a claims file in the original SciFact format."""
    return [parse_scifact_claim(row) for row in read_jsonl(path)]


def load_scifact(data_dir: str | Path, split: str) -> tuple[dict[str, Document], list[Claim]]:
    """Load the processed SciFact folder. split is train / dev / test."""
    data_dir = Path(data_dir)
    corpus = load_corpus(data_dir / "corpus.jsonl")
    claims = load_claims(data_dir / f"claims_{split}.jsonl")
    return corpus, claims


def load_climate_fever(path: str | Path, drop_disputed: bool = True) -> list[EvidenceClaim]:
    """Climate-FEVER: each claim has 5 evidence sentences from Wikipedia.

    DISPUTED claims (some evidence supports, some refutes) are dropped by default
    because they do not fit in a 3-label setup.
    """
    out = []
    for row in read_jsonl(path):
        raw_label = row["claim_label"]
        if raw_label == "DISPUTED":
            if drop_disputed:
                continue
            raw_label = "NOT_ENOUGH_INFO"  # closest thing we have
        out.append(
            EvidenceClaim(
                claim_id=str(row["claim_id"]),
                text=row["claim"],
                label=LABEL_ALIASES[raw_label],
                evidence_texts=[e["evidence"] for e in row["evidences"]],
                meta={"evidence_labels": [e["evidence_label"] for e in row["evidences"]]},
            )
        )
    return out


def load_evidence_claims(path: str | Path) -> list[EvidenceClaim]:
    """Load our own simple format: {id, claim, document, label, ...extra fields}."""
    out = []
    for row in read_jsonl(path):
        extra = {k: v for k, v in row.items() if k not in ("id", "claim", "document", "label")}
        out.append(
            EvidenceClaim(
                claim_id=str(row["id"]),
                text=row["claim"],
                label=LABEL_ALIASES[row["label"]],
                evidence_texts=[row["document"]],
                meta=extra,
            )
        )
    return out


def ensure_processed_sample(root: str | Path = ".") -> Path:
    """Build data/processed/sample from data/sample if it is missing.

    On a fresh clone (or when the Docker volume hides the prepared folder) the offline
    demo would otherwise load an empty corpus. Used by the demo and by the tests."""
    import subprocess
    import sys

    root = Path(root)
    out = root / "data" / "processed" / "sample"
    if not (out / "corpus.jsonl").exists():
        for script, args in (
            ("preprocess.py", ["--raw", str(root / "data" / "sample"), "--out", str(out)]),
            ("build_contrast_set.py", ["--data", str(out)]),
        ):
            subprocess.run([sys.executable, str(root / "data" / "scripts" / script), *args],
                           check=True, cwd=root, stdout=subprocess.DEVNULL)
    return out
