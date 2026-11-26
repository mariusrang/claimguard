"""Split an LLM answer into checkable claims, without losing any sentence silently.

One sentence = one claim. Before, sentences that started like "Sure, ..." or
"Please note that ..." were dropped, and a wrong statement hidden behind such an
opening disappeared from the report. Now every sentence gets a status:

- "claim":       checked against the documents (polite openings are removed first)
- "too_short":   a statement too short to check reliably ("Fees: none." is kept, "Yes." is not);
                 it is shown and counted as NOT CHECKED, so the user still sees it
- "ignored":     not a statement at all (a question, "Thanks!", "Hope this helps.");
                 shown in the list of ignored sentences, not counted as an alert

Known limit: pronouns are not resolved ("It also reduces..." stays as is). A small LLM
doing the decomposition is listed as future work in the report.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from claimguard.text import split_sentences, tokenize

# polite or hedging openings: removed, the rest of the sentence is still checked
_LEAD_IN = re.compile(
    r"^(?:(?:sure|of course|certainly|okay|ok|yes|absolutely|great|good question)[,!.:]?\s+"
    r"|(?:please\s+)?note\s+that\s+|please note[,:]?\s+|importantly[,:]?\s+"
    r"|here (?:is|are) (?:the |a )?(?:key point|answer|summary|main point|short answer)s?[:,]?\s+"
    r"|to (?:sum up|summari[sz]e)[,:]?\s+|in short[,:]?\s+"
    r"|i (?:think|believe)(?: that)?\s+|in my opinion[,:]?\s+|as an ai[^,]*,\s+)",
    re.IGNORECASE,
)
# whole sentences that state nothing that could be true or false
_NOT_A_STATEMENT = re.compile(
    r"^(?:thanks|thank you|hope (?:this|that) helps|feel free|let me know|happy to help|here it is|"
    r"here you go|here (?:is|are) (?:the |a |some )?(?:answer|summary|details|information)[.!:]*$|"
    r"great|perfect|good luck|you're welcome)\b"
    # a bare acknowledgement on its own: "Sure!", "Of course.", "OK"
    r"|^(?:sure|of course|certainly|absolutely|okay|ok|yes|no problem)[\s.!,]*$",
    re.IGNORECASE,
)


@dataclass
class Segment:
    """One sentence of the answer and what we do with it."""

    original: str  # the sentence as written in the answer
    text: str  # what is checked (opening removed)
    status: str  # "claim", "too_short" or "ignored"
    reason: str = ""


def _strip_lead_in(sentence: str) -> str:
    """Remove polite openings, possibly several ("Sure, please note that ...")."""
    previous = None
    while previous != sentence:
        previous = sentence
        sentence = _LEAD_IN.sub("", sentence, count=1).strip()
    return sentence[:1].upper() + sentence[1:] if sentence else sentence


def analyse_answer(answer: str, min_tokens: int = 2) -> list[Segment]:
    """Give a status to every sentence of `answer` (nothing is dropped silently)."""
    segments = []
    for raw in split_sentences(answer):
        original = raw.strip().lstrip("-*• ").strip()
        if not original:
            continue
        if original.endswith("?"):
            segments.append(Segment(original, original, "ignored", "a question is not a statement"))
            continue
        text = _strip_lead_in(original)
        if not text or _NOT_A_STATEMENT.match(text):
            segments.append(Segment(original, text, "ignored", "not a factual statement"))
        elif len(tokenize(text)) < min_tokens:
            segments.append(Segment(original, text, "too_short", "too short to check reliably"))
        else:
            segments.append(Segment(original, text, "claim"))
    return segments


def split_into_claims(answer: str, min_tokens: int = 2) -> list[str]:
    """Only the sentences that will be checked (kept for the experiments and tests)."""
    return [s.text for s in analyse_answer(answer, min_tokens) if s.status == "claim"]
