"""Baselines of the project (the guidelines ask for models/baseline.py).

The code lives in the package `src/claimguard`; this file only points to it so the
repository follows the structure of the course guidelines.

- LexicalVerifier: rule-based verifier (numbers, negation, word overlap)
- BM25Index: lexical retrieval written from scratch
- the zero-shot NLI and LLM-judge baselines are NLIVerifier / LLMJudgeVerifier with other checkpoints
"""

from claimguard.retrieval.bm25 import BM25Index
from claimguard.verify.lexical import LexicalVerifier
from claimguard.verify.llm_judge import LLMJudgeVerifier

__all__ = ["BM25Index", "LexicalVerifier", "LLMJudgeVerifier"]
