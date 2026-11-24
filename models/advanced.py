"""Main system of the project (the guidelines ask for models/advanced.py).

The code lives in the package `src/claimguard`; this file only points to it.

- HybridRetriever + DenseIndex: BM25 and BGE-small fused with Reciprocal Rank Fusion
- CrossEncoderReranker: reranking and sentence selection
- NLIVerifier: DeBERTa-v3-base verifier fine-tuned on SciFact
- ClaimGuard / build_pipeline: the whole pipeline (aggregation, temperature, abstention)
"""

from claimguard.pipeline import ClaimGuard, build_pipeline
from claimguard.retrieval.dense import DenseIndex
from claimguard.retrieval.hybrid import HybridRetriever
from claimguard.retrieval.rerank import CrossEncoderReranker
from claimguard.verify.nli import NLIVerifier

__all__ = ["ClaimGuard", "build_pipeline", "DenseIndex", "HybridRetriever", "CrossEncoderReranker", "NLIVerifier"]
