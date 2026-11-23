"""Multi-evidence aggregation: k evidence-level predictions -> one verdict per claim.

Each retrieved document gives probabilities [P(SUPPORTED), P(REFUTED), P(NEI)].
Most retrieved documents are irrelevant (so NEI), and one good document should
be enough to decide. The three strategies handle this differently:

- majority : soft vote = average of the k probability vectors (NEI-heavy, the
             irrelevant documents drown the signal). I average probabilities instead of
             counting hard votes so that the confidence still means something: a hard
             vote with k=1 would always say "100% sure" and kill the abstention.
- max_conf : trust the single most decisive document
- noisy_or : "the claim is supported if at least one document supports it"
             P(S) = 1 - prod(1 - p_s_i), same for R, and P(NEI) = prod(p_nei_i)
"""

from __future__ import annotations

import numpy as np

from claimguard import NEI, REFUTED, SUPPORTED

STRATEGIES = ("majority", "max_conf", "noisy_or")


def aggregate(probs: np.ndarray, strategy: str = "noisy_or") -> np.ndarray:
    """probs: [k, 3] evidence-level probabilities -> [3] claim-level probabilities."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("probs must have shape [k, 3]")
    if len(probs) == 0:
        # no evidence at all: we cannot say anything
        return np.array([0.0, 0.0, 1.0])

    if strategy == "majority":
        return probs.mean(axis=0)

    if strategy == "max_conf":
        # the most decisive document = the one with the highest P(S) or P(R)
        decisive = probs[:, [SUPPORTED, REFUTED]].max(axis=1)
        return probs[int(decisive.argmax())]

    if strategy == "noisy_or":
        p_s = 1.0 - np.prod(1.0 - probs[:, SUPPORTED])
        p_r = 1.0 - np.prod(1.0 - probs[:, REFUTED])
        p_nei = np.prod(probs[:, NEI])
        scores = np.array([p_s, p_r, p_nei])
        return scores / max(scores.sum(), 1e-12)

    raise ValueError(f"unknown aggregation strategy '{strategy}', choose from {STRATEGIES}")
