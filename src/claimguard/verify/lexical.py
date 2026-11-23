"""Rule-based verifier. It is my "dumb" baseline and my offline fallback.

Rules (in this order):
1. not enough word overlap between claim and evidence      -> NEI
2. the claim has a number that is not in the evidence
   (and the evidence does contain numbers)                 -> REFUTED
3. one side is negated and the other is not                -> REFUTED
4. otherwise                                               -> SUPPORTED

It needs no model, so the tests, the CI, the Docker image and the MCP server can
all run without downloading anything. It is also a useful sanity check: if the
neural verifier is not clearly better than this, something is wrong.
"""

from __future__ import annotations

import numpy as np

from claimguard.text import extract_numbers, has_negation, tokenize


class LexicalVerifier:
    """Heuristic verifier that returns pseudo-logits in our label order."""

    name = "lexical-rules"

    def __init__(self, overlap_threshold: float = 0.5) -> None:
        self.overlap_threshold = overlap_threshold

    def _one(self, claim: str, evidence: str) -> np.ndarray:
        claim_tokens = set(tokenize(claim))
        evid_tokens = set(tokenize(evidence))
        overlap = len(claim_tokens & evid_tokens) / max(len(claim_tokens), 1)
        if overlap < self.overlap_threshold:
            # logits are [SUPPORTED, REFUTED, NEI]; less overlap = more sure it is NEI
            return np.array([overlap, 0.0, 1.0 + (self.overlap_threshold - overlap) * 4])
        claim_nums = set(extract_numbers(claim))
        evid_nums = set(extract_numbers(evidence))
        number_clash = bool(claim_nums) and bool(evid_nums) and not claim_nums <= evid_nums
        negation_clash = has_negation(claim) != has_negation(evidence)
        if number_clash or negation_clash:
            return np.array([0.5, 1.5 + overlap, 0.3])
        return np.array([1.0 + 2.0 * overlap, 0.3, 0.6])

    def predict_logits(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """pairs = [(claim, evidence), ...] -> array [n, 3]."""
        if not pairs:
            return np.zeros((0, 3))
        return np.stack([self._one(c, e) for c, e in pairs])
