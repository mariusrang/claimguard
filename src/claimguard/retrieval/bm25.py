"""BM25 (Okapi) written from scratch with an inverted index.

I wrote it myself instead of using rank_bm25 because (1) it is ~60 lines,
(2) I wanted to really understand the formula, (3) with an inverted index a
query only touches the documents that contain its terms, so it is fast.

score(q, d) = sum over query terms t of
    idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * len(d) / avg_len))
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from claimguard.text import tokenize


class BM25Index:
    """BM25 index over a list of (doc_id, text)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_len = np.zeros(0)
        self.avg_len = 0.0
        self.idf: dict[str, float] = {}
        # term -> (array of doc indexes, array of term frequencies)
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def build(self, doc_ids: list[str], texts: list[str]) -> "BM25Index":
        """Tokenize every document and fill the inverted index."""
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        self.doc_ids = list(doc_ids)
        lengths = []
        tmp: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, text in enumerate(texts):
            counts = Counter(tokenize(text))
            lengths.append(sum(counts.values()))
            for term, tf in counts.items():
                tmp[term].append((idx, tf))
        self.doc_len = np.array(lengths, dtype=np.float64)
        self.avg_len = float(self.doc_len.mean()) if lengths else 0.0
        n_docs = len(doc_ids)
        for term, plist in tmp.items():
            df = len(plist)
            # the "+1 inside the log" version of idf, so it is never negative
            self.idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            idxs = np.array([p[0] for p in plist], dtype=np.int64)
            tfs = np.array([p[1] for p in plist], dtype=np.float64)
            self.postings[term] = (idxs, tfs)
        return self

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for this query (0 if no term matches)."""
        scores = np.zeros(len(self.doc_ids), dtype=np.float64)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue  # word never seen in the corpus
            idxs, tfs = self.postings[term]
            norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[idxs] / self.avg_len)
            scores[idxs] += self.idf[term] * tfs * (self.k1 + 1.0) / (tfs + norm)
        return scores

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k (doc_id, score), best first. Documents with score 0 are not returned."""
        scores = self.scores(query)
        k = min(k, len(scores))
        if k == 0:
            return []
        # argpartition = O(n) instead of sorting everything
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.doc_ids[i], float(scores[i])) for i in top if scores[i] > 0]
