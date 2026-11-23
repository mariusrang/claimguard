"""Split an LLM answer into checkable claims.

I keep it simple on purpose: one sentence = one claim, minus the sentences that
are not factual statements (questions, greetings, "I think...", etc.).
Known limit: pronouns are not resolved ("It also reduces..." stays as is), so a
sentence that depends on the previous one can be harder to verify. A small LLM
doing the decomposition is listed as future work in the report.
"""

from __future__ import annotations

import re

from claimguard.text import split_sentences, tokenize

# sentence starts that usually are not factual claims
_NOT_A_CLAIM = re.compile(
    r"^(i think|i believe|in my opinion|here is|here are|sure|of course|let me|hope this|"
    r"please|note that you|feel free|thanks|thank you|as an ai)",
    re.IGNORECASE,
)


def split_into_claims(answer: str, min_tokens: int = 3) -> list[str]:
    """Return the sentences of `answer` that look like factual claims."""
    claims = []
    for sentence in split_sentences(answer):
        sentence = sentence.strip().lstrip("-*• ").strip()
        if sentence.endswith("?"):
            continue  # a question cannot be true or false
        if _NOT_A_CLAIM.match(sentence):
            continue
        if len(tokenize(sentence)) < min_tokens:
            continue  # too short to mean anything ("Yes.", "Great!")
        claims.append(sentence)
    return claims
