# French bank test set — data card

**File:** `fr_bank_claims.jsonl` (60 lines)

**What it is.** A small stress test for the verifier, in French, on banking-style
documents. 20 short documents (product sheets, compliance rules, internal policy,
an annual report extract), each with 3 claims: one SUPPORTED, one REFUTED, one NEI.

**Why.** The main benchmark (SciFact) is in English and about biomedical abstracts.
A French bank would ask two things first: does it work in French, and does it catch a
wrong amount, date or threshold. This set tests exactly that, on a small scale.

**How it was built.** The documents are invented. "Banque Horizon" does not exist,
and no number comes from a real bank. For each document:
- the SUPPORTED claim rephrases one sentence of the document;
- the REFUTED claim changes one thing: an amount, a rate, a duration, a threshold, a
  date, a direction (hausse → baisse) or a negation;
- the NEI claim is on the same topic but the document does not say it.

The first draft of the documents and claims was written with the help of an AI
assistant; every item then has to be checked by hand (label and category).

**Review log.**
- 23 Nov 2026, first review (with the coding assistant): `fr-14-r` was wrong ("within
  30 days" is not contradicted by "within 10 days", it is implied by it) and was
  replaced by a real contradiction; `fr-01-s` was reworded because the card is free the
  first year; `fr-18-n` is marked as a world-knowledge trap (true NEI for the document,
  "false" in real life). Threshold claims (`fr-09-r`, `fr-16-r`, `fr-17-r`) read the
  document's threshold as a definition, which is how a bank would read it.
- [ ] second review by me, item by item (to tick before the final run)

**Fields.** `id`, `topic`, `document`, `claim`, `label` (SUPPORTED / REFUTED / NEI),
`category` (paraphrase, amount, rate, duration, threshold, date, direction, negation,
missing_info).

**How it is used.** `python scripts/evaluate.py given --dataset french`. The verifier
gets the whole document as evidence (no retrieval). I compare the English verifier
with a multilingual one (`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, zero-shot).

**Limits.**
- 60 examples: confidence intervals are wide, it shows tendencies only.
- Synthetic and clean: real bank documents are long, with tables and cross-references.
- One author, so the style of the claims is not varied.
- REFUTED claims are mostly numeric edits, so a rule that compares numbers already
  does well on them (the rule-based baseline reaches 0.77 accuracy on this set). The
  interesting part is where the rules fail: paraphrases, negations and NEI claims that
  share many words with the document.
