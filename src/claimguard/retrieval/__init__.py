"""Retrievers: BM25 (lexical), dense (bi-encoder), hybrid (RRF) + rerankers."""

from claimguard.retrieval.bm25 import BM25Index
from claimguard.retrieval.dense import DenseIndex, TransformerEncoder
from claimguard.retrieval.hybrid import HybridRetriever, rrf_fuse
from claimguard.retrieval.rerank import CrossEncoderReranker, LexicalReranker

__all__ = [
    "BM25Index",
    "DenseIndex",
    "TransformerEncoder",
    "HybridRetriever",
    "rrf_fuse",
    "CrossEncoderReranker",
    "LexicalReranker",
]
