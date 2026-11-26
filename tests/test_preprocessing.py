"""Tests of the text preprocessing and of the data loading."""

from __future__ import annotations

import sys
from pathlib import Path

from claimguard.data import load_climate_fever, load_evidence_claims, parse_scifact_claim
from claimguard.decompose import split_into_claims
from claimguard.pipeline import chunk_sources
from claimguard.text import extract_numbers, has_negation, normalize_number, split_sentences, tokenize
from claimguard.utils import deep_update, load_config, write_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_tokenize_lowercases_and_removes_stopwords():
    assert tokenize("The Vitamin C of the study") == ["vitamin", "c", "study"]
    assert tokenize("The study", remove_stopwords=False) == ["the", "study"]


def test_tokenize_keeps_decimal_numbers_together():
    assert "3.45" in tokenize("a rate of 3.45 percent")


def test_split_sentences_basic_and_abbreviations():
    text = "Aspirin helps. It was shown by Smith et al. in 2010. Is it safe? Yes."
    sentences = split_sentences(text)
    assert sentences[0] == "Aspirin helps."
    assert "et al. in 2010." in sentences[1]  # no split after "et al."
    assert len(sentences) == 4
    assert split_sentences("   ") == []


def test_normalize_number_formats():
    assert normalize_number("1,200") == "1200"
    assert normalize_number("8 000") == "8000"
    assert normalize_number("4,5") == "4.5"
    assert normalize_number("4.50") == "4.5"
    assert normalize_number("30") == "30"


def test_extract_numbers_french_and_english():
    assert extract_numbers("Le taux est de 3,45 % sur 20 ans") == ["3.45", "20"]
    assert extract_numbers("no number here") == []


def test_has_negation():
    assert has_negation("Aspirin does not help")
    assert has_negation("Le contrat ne peut pas être souscrit")
    assert not has_negation("Aspirin helps")


def test_parse_scifact_claim_labels():
    supported = parse_scifact_claim(
        {"id": 1, "claim": "x", "evidence": {"42": [{"sentences": [2, 0], "label": "SUPPORT"}]}, "cited_doc_ids": [42]}
    )
    assert supported.label == "SUPPORTED"
    assert supported.evidence == {"42": [0, 2]}  # sorted sentence ids, doc id as string
    refuted = parse_scifact_claim(
        {"id": 2, "claim": "x", "evidence": {"7": [{"sentences": [1], "label": "CONTRADICT"}]}}
    )
    assert refuted.label == "REFUTED"
    nei = parse_scifact_claim({"id": 3, "claim": "x", "evidence": {}, "cited_doc_ids": [5]})
    assert nei.label == "NEI" and nei.cited_doc_ids == ["5"]


def test_stratified_split_keeps_every_label():
    sys.path.insert(0, str(ROOT / "data" / "scripts"))
    from preprocess import stratified_split

    from claimguard.utils import read_jsonl

    rows = read_jsonl(ROOT / "data" / "sample" / "claims_train.jsonl")
    train, dev = stratified_split(rows, dev_ratio=0.2, seed=1)
    assert len(train) + len(dev) == len(rows)
    assert {parse_scifact_claim(r).label for r in dev} == {"SUPPORTED", "REFUTED", "NEI"}
    # same seed -> same split (reproducibility)
    assert stratified_split(rows, 0.2, 1)[1] == dev


def test_load_evidence_claims_french_set():
    examples = load_evidence_claims(ROOT / "data" / "french_bank" / "fr_bank_claims.jsonl")
    assert len(examples) == 60
    labels = [e.label for e in examples]
    assert labels.count("SUPPORTED") == labels.count("REFUTED") == labels.count("NEI") == 20
    assert all(e.evidence_texts[0] and "category" in e.meta for e in examples)


def test_load_climate_fever_drops_disputed(tmp_path):
    rows = [
        {"claim_id": 1, "claim": "a", "claim_label": "SUPPORTS", "evidences": [{"evidence": "e1", "evidence_label": "SUPPORTS"}]},
        {"claim_id": 2, "claim": "b", "claim_label": "DISPUTED", "evidences": [{"evidence": "e2", "evidence_label": "REFUTES"}]},
        {"claim_id": 3, "claim": "c", "claim_label": "NOT_ENOUGH_INFO", "evidences": [{"evidence": "e3", "evidence_label": "NOT_ENOUGH_INFO"}]},
    ]
    write_jsonl(tmp_path / "cf.jsonl", rows)
    examples = load_climate_fever(tmp_path / "cf.jsonl")
    assert [e.label for e in examples] == ["SUPPORTED", "NEI"]


def test_chunk_sources_cuts_long_documents():
    source = " ".join(f"Sentence number {i} is here." for i in range(14))
    chunks = chunk_sources([source, "Second doc."], chunk_size=6)
    assert list(chunks) == ["source 1 / chunk 1", "source 1 / chunk 2", "source 1 / chunk 3", "source 2 / chunk 1"]
    assert len(chunks["source 1 / chunk 3"].sentences) == 2


def test_split_into_claims_filters_non_claims():
    answer = "Sure, here it is. Aspirin reduces the risk of heart attack. Is that clear? Great! I think it is safe."
    assert split_into_claims(answer) == ["Aspirin reduces the risk of heart attack."]


def test_config_inheritance():
    cfg = load_config(ROOT / "configs" / "offline.yaml")
    assert cfg["verifier"]["kind"] == "lexical"  # overridden
    assert cfg["retrieval"]["rrf_k"] == 60  # inherited from default.yaml
    assert deep_update({"a": {"b": 1, "c": 2}}, {"a": {"b": 5}}) == {"a": {"b": 5, "c": 2}}


def test_no_sentence_disappears_silently():
    """Bug found in review: a wrong statement after "Please note that" was dropped."""
    from claimguard.decompose import analyse_answer

    answer = ("Please note that early repayment is free. Here is the key point: the rate is 1% for all "
              "clients. Sure, application fees are 900 euros. Fees: none. Is that clear? Hope this helps!")
    segs = analyse_answer(answer)
    claims = [s.text for s in segs if s.status == "claim"]
    assert claims[:3] == ["Early repayment is free.", "The rate is 1% for all clients.",
                          "Application fees are 900 euros."]
    statuses = {s.original: s.status for s in segs}
    assert statuses["Is that clear?"] == "ignored" and statuses["Hope this helps!"] == "ignored"
    bare = {s.original: s.status for s in analyse_answer("Sure! Of course. The rate is 3.45%.")}
    assert bare["Sure!"] == "ignored" and bare["Of course."] == "ignored"  # bare filler, not a claim
    assert statuses["Fees: none."] in ("claim", "too_short")  # never dropped: checked or NOT_CHECKED
    assert len(segs) == 6  # every sentence has a status
