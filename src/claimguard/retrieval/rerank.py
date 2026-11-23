"""Rerankers: score (query, text) pairs jointly.

The cross-encoder reads the claim and the passage together, so it is much more
precise than the retrievers, but too slow to run on the whole corpus. That is
why it only reranks the top-50 of the first stage. I also use it to pick the
best sentences inside each abstract (the "rationale" sentences).
"""

from __future__ import annotations

import numpy as np

from claimguard.text import tokenize


class LexicalReranker:
    """Offline fallback: fraction of the query words found in the text.

    Used in the tests / CI and when no model can be downloaded.
    """

    name = "lexical-overlap"

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return np.zeros(len(texts))
        return np.array([len(q_tokens & set(tokenize(t))) / len(q_tokens) for t in texts])


class CrossEncoderReranker:
    """MS MARCO cross-encoder (one relevance logit per pair)."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.name = model_name

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Relevance score of each text for the query (higher = more relevant)."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                enc = self.tokenizer(
                    [query] * len(batch),
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                # ms-marco models have 1 output; if a model has 2+, take the last one
                out.append(logits[:, -1].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def rerank(reranker, query: str, texts: list[str], top_k: int) -> list[tuple[int, float]]:
    """Return (index_in_texts, score) for the top_k texts, best first."""
    if not texts:
        return []
    scores = reranker.score(query, texts)
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(i), float(scores[i])) for i in order]
