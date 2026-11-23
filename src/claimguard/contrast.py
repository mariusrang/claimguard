"""Contrast set: small edits of SUPPORTED claims that should break the support.

I take (claim, evidence) pairs that are SUPPORTED and change ONE thing in the claim:
- number  : 30% -> 3%, 12 -> 120 ...
- negation: "X increases Y" -> "X does not increase Y" (or remove an existing "not")
- antonym : increase <-> decrease, higher <-> lower ...

After the edit the evidence should not support the claim any more. A verifier
that still answers SUPPORTED was matching words, not checking the meaning.
These are the errors that hurt the most in practice (wrong amounts, wrong
thresholds), which is why this test is focused on them.

The labels come from rules, so a few edited claims may still be true by
accident. I report the "flip rate" (prediction is no longer SUPPORTED) and I
check a sample by hand (see the report).
"""

from __future__ import annotations

import random
import re

ANTONYMS = [
    ("increase", "decrease"),
    ("increases", "decreases"),
    ("increased", "decreased"),
    ("higher", "lower"),
    ("more", "less"),
    ("high", "low"),
    ("promotes", "inhibits"),
    ("promote", "inhibit"),
    ("activates", "suppresses"),
    ("positive", "negative"),
    ("positively", "negatively"),
    ("reduces", "raises"),
    ("reduced", "elevated"),
    ("upregulated", "downregulated"),
    ("gain", "loss"),
    ("improves", "worsens"),
    ("greater", "smaller"),
    ("before", "after"),
]
_ANTONYM_MAP = {}
for _a, _b in ANTONYMS:
    _ANTONYM_MAP[_a] = _b
    _ANTONYM_MAP[_b] = _a

_NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]*\d)")
_AUX_RE = re.compile(r"\b(is|are|was|were|can|does|do|did|has|have|will|may)\b(?! not)")
_NOT_RE = re.compile(r"\b(is|are|was|were|can|does|do|did|has|have|will|may) not\b")


def perturb_number(claim: str, rng: random.Random) -> str | None:
    """Change one number of the claim (x10, /10 or +50%)."""
    matches = list(_NUMBER_RE.finditer(claim))
    if not matches:
        return None
    m = rng.choice(matches)
    value = float(m.group(1))
    options = [value * 10, value / 10, value * 1.5 + 1]
    # keep the candidates that really differ and are still "nice" numbers
    options = [o for o in options if abs(o - value) > 1e-9]
    new_value = rng.choice(options)
    is_int = "." not in m.group(1) and float(new_value).is_integer()
    new_text = str(int(new_value)) if is_int else f"{new_value:.2f}".rstrip("0").rstrip(".")
    return claim[: m.start(1)] + new_text + claim[m.end(1) :]


def perturb_negation(claim: str, rng: random.Random) -> str | None:
    """Add 'not' after the first auxiliary verb, or remove an existing 'not'."""
    if _NOT_RE.search(claim):
        return _NOT_RE.sub(lambda m: m.group(1), claim, count=1)
    m = _AUX_RE.search(claim)
    if m is None:
        return None
    return claim[: m.end()] + " not" + claim[m.end() :]


def perturb_antonym(claim: str, rng: random.Random) -> str | None:
    """Swap one direction word with its opposite (increase -> decrease...)."""
    words = claim.split(" ")
    positions = [i for i, w in enumerate(words) if w.lower().strip(".,;:") in _ANTONYM_MAP]
    if not positions:
        return None
    i = rng.choice(positions)
    core = words[i].strip(".,;:")
    new = _ANTONYM_MAP[core.lower()]
    if core[0].isupper():
        new = new.capitalize()
    words[i] = words[i].replace(core, new)
    return " ".join(words)


PERTURBATIONS = {"number": perturb_number, "negation": perturb_negation, "antonym": perturb_antonym}


def build_contrast_set(pairs: list[dict], seed: int = 13) -> list[dict]:
    """pairs = [{"id", "claim", "document"}] of SUPPORTED claims -> edited versions.

    Every perturbation that applies to a claim gives one new example.
    """
    rng = random.Random(seed)
    out = []
    for pair in pairs:
        for kind, fn in PERTURBATIONS.items():
            new_claim = fn(pair["claim"], rng)
            if new_claim is None or new_claim == pair["claim"]:
                continue
            out.append(
                {
                    "id": f"{pair['id']}-{kind}",
                    "original_claim": pair["claim"],
                    "claim": new_claim,
                    "document": pair["document"],
                    "label": "REFUTED",  # expected: at least "not supported any more"
                    "perturbation": kind,
                }
            )
    return out
