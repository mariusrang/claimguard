"""Verifiers: given (claim, evidence) pairs, return logits over SUPPORTED / REFUTED / NEI."""

from claimguard.verify.lexical import LexicalVerifier
from claimguard.verify.nli import NLIVerifier, map_model_labels

__all__ = ["LexicalVerifier", "NLIVerifier", "map_model_labels"]
