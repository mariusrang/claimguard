"""Multi-hop verification on HoVer (claims that need 2, 3 or 4 Wikipedia articles).

The teacher's topic asks for a reasoning module that aggregates evidence from several
documents and for claims that need multi-hop reasoning. ClaimGuard reads every passage
on its own and then combines the scores (noisy-OR / max). That cannot work when no
single article proves the claim: "the director of film X was born in city Y" needs the
article on X AND the article on the director. So this file adds:

1. a joint reader: all the evidence documents are concatenated into one premise and
   the verifier reads them together (the "reasoning module");
2. an iterative retriever: a second retrieval round uses what the first round found
   (title + first sentence of the best documents) to reach the "bridge" articles that
   share no words with the claim.

HoVer labels are SUPPORTED / NOT_SUPPORTED, so our 3-class verifier is read as binary:
SUPPORTED if SUPPORTED is the most probable class, NOT_SUPPORTED otherwise.
"""

from __future__ import annotations

import gzip
import json
import random
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from claimguard.aggregate import aggregate
from claimguard.retrieval.bm25 import BM25Index
from claimguard.text import split_sentences

SUPPORTED_IDX = 0  # our label order is SUPPORTED, REFUTED, NEI


@dataclass
class HoverClaim:
    uid: str
    claim: str
    supported: bool  # True = SUPPORTED, False = NOT_SUPPORTED
    num_hops: int
    gold_titles: list[str] = field(default_factory=list)  # articles of the supporting facts


def norm_title(title: str) -> str:
    """HoVer and the Wikipedia dump store titles in NFD unicode form."""
    return unicodedata.normalize("NFD", title)


def load_hover(path: str | Path, per_cell: int | None = None, seed: int = 13) -> list[HoverClaim]:
    """Read a HoVer release file. With per_cell, keep a balanced sample:
    per_cell claims for each (number of hops, label) pair."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    claims = []
    for row in raw:
        titles = []
        for title, _ in row["supporting_facts"]:
            if norm_title(title) not in titles:
                titles.append(norm_title(title))
        claims.append(
            HoverClaim(row["uid"], row["claim"], row["label"] == "SUPPORTED", int(row["num_hops"]), titles)
        )
    if per_cell is None:
        return claims
    rng = random.Random(seed)
    sample = []
    for hops in sorted({c.num_hops for c in claims}):
        for label in (True, False):
            cell = [c for c in claims if c.num_hops == hops and c.supported == label]
            rng.shuffle(cell)
            sample.extend(cell[:per_cell])
    return sample


class WikiTexts:
    """title -> intro paragraph of the 2017 Wikipedia used by HotpotQA and HoVer.

    Two possible sources, same text:
    - the SQLite file of the HoVer authors (wiki_wo_links.db, table documents(id, text))
    - the BEIR copy of the HotpotQA corpus (corpus.jsonl.gz, fields title / text)
    Only the titles we need are kept in memory (the full dump has 5M articles)."""

    def __init__(self, texts: dict[str, str]):
        self.texts = texts

    @classmethod
    def from_sqlite(cls, db_path: str | Path, titles: set[str]) -> "WikiTexts":
        conn = sqlite3.connect(str(db_path))
        texts = {}
        for title in titles:
            row = conn.execute("SELECT text FROM documents WHERE id = ?", (title,)).fetchone()
            if row:
                texts[title] = row[0]
        conn.close()
        return cls(texts)

    @classmethod
    def from_beir(cls, corpus_path: str | Path, titles: set[str] | None, extra: int = 0) -> "WikiTexts":
        """extra = also keep the first `extra` other articles (distractors for a closed pool)."""
        texts = {}
        kept_extra = 0
        opener = gzip.open if str(corpus_path).endswith(".gz") else open
        with opener(corpus_path, "rt", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                title = norm_title(row.get("title", ""))
                if titles is None or title in titles:
                    texts[title] = row.get("text", "")
                elif kept_extra < extra and title:
                    texts[title] = row.get("text", "")
                    kept_extra += 1
        return cls(texts)

    def get(self, title: str) -> str:
        return self.texts.get(norm_title(title), "")

    def __contains__(self, title: str) -> bool:
        return norm_title(title) in self.texts

    def __len__(self) -> int:
        return len(self.texts)


def doc_text(title: str, text: str) -> str:
    """How a document is shown to the models: title first, it often carries the entity."""
    return f"{title}. {text}".strip()


def joint_premise(docs: list[tuple[str, str]], max_words: int = 330) -> str:
    """Concatenate several documents into one premise for the verifier.

    The verifier reads at most ~512 tokens, so every document gets the same share of
    words (a long first article must not push the others out of the window)."""
    if not docs:
        return ""
    share = max(20, max_words // len(docs))
    parts = []
    for title, text in docs:
        words = doc_text(title, text).split()
        parts.append(" ".join(words[:share]))
    return " ".join(parts)


def verify_independent(verifier, claim: str, docs: list[tuple[str, str]], strategy: str) -> np.ndarray:
    """ClaimGuard as it is: read each document alone, then aggregate the probabilities."""
    if not docs:
        return np.array([0.0, 0.0, 1.0])
    logits = verifier.predict_logits([(claim, doc_text(t, x)) for t, x in docs])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    return aggregate(probs, strategy)


def verify_joint(verifier, claim: str, docs: list[tuple[str, str]]) -> np.ndarray:
    """Reasoning module: one pass over all documents together."""
    logits = verifier.predict_logits([(claim, joint_premise(docs))])[0]
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def is_supported(probs: np.ndarray) -> bool:
    return int(np.argmax(probs)) == SUPPORTED_IDX


# ----------------------------------------------------------------------------- retrieval


class PoolRetriever:
    """BM25 + cross-encoder over a closed pool of Wikipedia articles.

    The pool is the union of the 100 TF-IDF candidates that the HoVer authors give for
    each claim, plus the gold articles (they are real Wikipedia articles, so they would
    be in a full index too). This keeps the experiment small enough for one GPU session."""

    def __init__(self, wiki: WikiTexts, reranker, first_stage: int = 50):
        self.wiki = wiki
        self.titles = list(wiki.texts)
        self.index = BM25Index().build(self.titles, [doc_text(t, wiki.texts[t]) for t in self.titles])  # noqa: E501
        self.reranker = reranker
        self.first_stage = first_stage

    def _rerank(self, query: str, titles: list[str]) -> list[tuple[str, float]]:
        if not titles:
            return []
        scores = self.reranker.score(query, [doc_text(t, self.wiki.get(t)) for t in titles])
        order = np.argsort(-scores)
        return [(titles[i], float(scores[i])) for i in order]

    def single_shot(self, claim: str, k: int) -> list[str]:
        """One BM25 search with the claim, then rerank."""
        candidates = [t for t, _ in self.index.search(claim, self.first_stage)]
        return [t for t, _ in self._rerank(claim, candidates)[:k]]

    def iterative(self, claim: str, k: int, hop_seeds: int = 2, per_hop: int = 20) -> list[str]:
        """Two rounds. Round 2 searches again with the claim + the title and first
        sentence of the best round-1 articles, to reach the bridge articles."""
        first = [t for t, _ in self.index.search(claim, self.first_stage)]
        ranked = self._rerank(claim, first)
        seeds = [t for t, _ in ranked[:hop_seeds]]
        pool = list(dict.fromkeys(first))
        for seed in seeds:
            sentences = split_sentences(self.wiki.get(seed)) or [""]
            query = f"{claim} {seed} {sentences[0]}"
            for title, _ in self.index.search(query, per_hop):
                if title not in pool:
                    pool.append(title)
        # final score: best of the claim alone and the claim enriched with a seed title,
        # so that a bridge article that only matches through the seed can still rank high
        best: dict[str, float] = {}
        for query in [claim] + [f"{claim} {s}" for s in seeds]:
            for title, score in self._rerank(query, pool):
                best[title] = max(best.get(title, -1e9), score)
        return [t for t, _ in sorted(best.items(), key=lambda kv: -kv[1])[:k]]


def all_gold_found(retrieved: list[str], gold: list[str]) -> bool:
    """Multi-hop retrieval is only useful if EVERY needed article is found."""
    found = {norm_title(t) for t in retrieved}
    return all(norm_title(g) in found for g in gold)


def gold_recall(retrieved: list[str], gold: list[str]) -> float:
    found = {norm_title(t) for t in retrieved}
    return sum(norm_title(g) in found for g in gold) / max(1, len(gold))


# ----------------------------------------------------------------------------- training data


def hover_training_pairs(
    claims: list[HoverClaim], wiki: WikiTexts, not_supported_label: int = 2
) -> list[tuple[str, str, int]]:
    """(claim, joint premise of the gold articles, label) pairs to fine-tune the joint reader.

    NOT_SUPPORTED mixes contradicted claims and claims the articles do not settle; it is
    mapped to NEI by default. At test time only the SUPPORTED vs not decision is used."""
    pairs = []
    for c in claims:
        docs = [(t, wiki.get(t)) for t in c.gold_titles if t in wiki]
        if len(docs) != len(c.gold_titles):
            continue  # an article is missing from the dump: skip rather than train on half
        pairs.append((c.claim, joint_premise(docs), SUPPORTED_IDX if c.supported else not_supported_label))
    return pairs
