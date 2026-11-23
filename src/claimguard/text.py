"""Text helpers: tokenizer for BM25, sentence splitter, number / negation detection."""

from __future__ import annotations

import re

# a short stopword list is enough for BM25 on abstracts (english + a few french ones
# because the french test set also goes through the lexical baseline)
STOPWORDS = frozenset(
    """a an the of in on at to for from by with and or is are was were be been it its this that
    these those as than which who whom into over under between we our their there here also
    le la les un une des du de et ou est sont dans pour par sur au aux ce cette ces que qui""".split()
)

NEGATION_WORDS = frozenset(
    """not no never none neither nor cannot without n't fails failed lack lacks absence
    ne pas jamais aucun aucune sans ni""".split()
)

_TOKEN_RE = re.compile(r"[a-zà-ÿ0-9]+(?:[.,][0-9]+)*|n't")
_NUMBER_RE = re.compile(r"\d+(?:[.,\s]\d{3})*(?:[.,]\d+)?|\d+")
# things that end with a dot but are not the end of a sentence
_ABBREV = ("e.g.", "i.e.", "et al.", "fig.", "vs.", "dr.", "mr.", "mrs.", "no.", "approx.", "art.")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9\"'(\[])")


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase + regex tokenizer. Good enough for lexical matching."""
    tokens = _TOKEN_RE.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences with a regex (and glue back abbreviations)."""
    text = " ".join(text.split())  # normalise whitespace first
    if not text:
        return []
    parts = _SENT_SPLIT_RE.split(text)
    sentences: list[str] = []
    for part in parts:
        # if the previous chunk ended with an abbreviation, it was a false split
        if sentences and sentences[-1].lower().endswith(_ABBREV):
            sentences[-1] = sentences[-1] + " " + part
        else:
            sentences.append(part)
    return [s.strip() for s in sentences if s.strip()]


def normalize_number(raw: str) -> str:
    """Turn '1,200', '1 200', '4,5' or '4.50' into a comparable string ('1200', '4.5')."""
    s = raw.replace(" ", "").replace(" ", "")
    # a separator followed by exactly 3 digits is a thousands separator
    s = re.sub(r"[.,](?=\d{3}(?:\D|$))", "", s)
    s = s.replace(",", ".")
    try:
        value = float(s)
    except ValueError:
        return s
    return str(int(value)) if value == int(value) else str(value)


def extract_numbers(text: str) -> list[str]:
    """All the numbers of a text, normalised (so 4,5 % and 4.5% match)."""
    return [normalize_number(m) for m in _NUMBER_RE.findall(text)]


def has_negation(text: str) -> bool:
    """True if the text contains an explicit negation word."""
    tokens = _TOKEN_RE.findall(text.lower())
    return any(t in NEGATION_WORDS for t in tokens)
