"""Tests of the multi-hop module on a tiny fake HoVer (no download, no GPU)."""

import gzip
import json
import sqlite3

import numpy as np

from claimguard.multihop import (
    PoolRetriever,
    WikiTexts,
    all_gold_found,
    gold_recall,
    hover_training_pairs,
    is_supported,
    joint_premise,
    load_hover,
    verify_independent,
    verify_joint,
)
from claimguard.retrieval.rerank import LexicalReranker
from claimguard.verify.nli import NLIVerifier

ARTICLES = {
    "Inception": "Inception is a 2010 film directed by Christopher Nolan.",
    "Christopher Nolan": "Christopher Nolan is a director born in London in 1970.",
    "London": "London is the capital of England.",
    "Paris": "Paris is the capital of France.",
}


def fake_hover(tmp_path):
    rows = []
    for i in range(8):
        rows.append({
            "uid": f"u{i}",
            "claim": "The director of Inception was born in London.",
            "supporting_facts": [["Inception", 0], ["Christopher Nolan", 0], ["Inception", 0]],
            "label": "SUPPORTED" if i % 2 == 0 else "NOT_SUPPORTED",
            "num_hops": 2 if i < 4 else 3,
        })
    path = tmp_path / "hover.json"
    path.write_text(json.dumps(rows))
    return path


def test_load_hover_balanced_sample_and_unique_titles(tmp_path):
    claims = load_hover(fake_hover(tmp_path), per_cell=1)
    assert len(claims) == 4  # 2 hop counts x 2 labels x 1
    assert {(c.num_hops, c.supported) for c in claims} == {(2, True), (2, False), (3, True), (3, False)}
    assert claims[0].gold_titles == ["Inception", "Christopher Nolan"]  # duplicates removed, order kept


def test_wiki_texts_from_sqlite_and_beir_give_the_same_text(tmp_path):
    db = tmp_path / "wiki.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE documents (id PRIMARY KEY, text)")
    conn.executemany("INSERT INTO documents VALUES (?, ?)", list(ARTICLES.items()))
    conn.commit()
    conn.close()
    beir = tmp_path / "corpus.jsonl.gz"
    with gzip.open(beir, "wt", encoding="utf-8") as f:
        for title, text in ARTICLES.items():
            f.write(json.dumps({"_id": title, "title": title, "text": text}) + "\n")
    wanted = {"Inception", "Christopher Nolan"}
    a, b = WikiTexts.from_sqlite(db, wanted), WikiTexts.from_beir(beir, wanted)
    assert len(a) == len(b) == 2
    assert a.get("Inception") == b.get("Inception") == ARTICLES["Inception"]
    assert "Paris" not in a


def test_joint_premise_gives_every_document_its_share():
    long_doc = ("Long", "word " * 1000)
    premise = joint_premise([long_doc, ("Short", "the bridge fact")], max_words=100)
    assert "bridge fact" in premise  # the long article must not push the second one out
    assert len(premise.split()) <= 110


def test_readers_return_probabilities(tiny_nli_model, tiny_tokenizer):
    verifier = NLIVerifier(model=tiny_nli_model, tokenizer=tiny_tokenizer, device="cpu")
    docs = [("Inception", ARTICLES["Inception"]), ("Christopher Nolan", ARTICLES["Christopher Nolan"])]
    for probs in (
        verify_independent(verifier, "Nolan was born in London.", docs, "noisy_or"),
        verify_independent(verifier, "Nolan was born in London.", docs, "max_conf"),
        verify_joint(verifier, "Nolan was born in London.", docs),
    ):
        assert probs.shape == (3,) and np.isclose(probs.sum(), 1.0, atol=1e-6)
        assert isinstance(is_supported(probs), bool)
    assert not is_supported(verify_independent(verifier, "x", [], "noisy_or"))  # no document = not supported


def test_iterative_retrieval_reaches_the_bridge_article():
    wiki = WikiTexts(dict(ARTICLES))
    retriever = PoolRetriever(wiki, LexicalReranker(), first_stage=2)
    claim = "Inception was directed by someone born in 1970."
    single = retriever.single_shot(claim, 4)
    iterative = retriever.iterative(claim, 4, hop_seeds=1, per_hop=2)
    assert "Inception" in single
    assert all_gold_found(iterative, ["Inception", "Christopher Nolan"])
    assert gold_recall(["Inception"], ["Inception", "Christopher Nolan"]) == 0.5


def test_training_pairs_skip_claims_with_a_missing_article(tmp_path):
    claims = load_hover(fake_hover(tmp_path))
    full = hover_training_pairs(claims, WikiTexts(dict(ARTICLES)))
    assert len(full) == 8 and {y for _, _, y in full} == {0, 2}  # SUPPORTED and NEI
    partial = hover_training_pairs(claims, WikiTexts({"Inception": ARTICLES["Inception"]}))
    assert partial == []


def test_run_multihop_script_end_to_end(tiny_tokenizer, tmp_path, monkeypatch):
    """The whole experiment script on a fake HoVer, with a tiny saved model (smoke test)."""
    import sys

    import torch
    from transformers import BertConfig, BertForSequenceClassification

    torch.manual_seed(0)
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    config = BertConfig(vocab_size=len(tiny_tokenizer), hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=600,
                        num_labels=3, id2label=id2label, label2id={v: k for k, v in id2label.items()})
    model_dir = tmp_path / "tiny"
    BertForSequenceClassification(config).save_pretrained(model_dir)
    tiny_tokenizer.save_pretrained(model_dir)

    hover = tmp_path / "hover"
    hover.mkdir()
    rows = json.loads(fake_hover(tmp_path).read_text())
    for split in ("dev", "train"):
        (hover / f"hover_{split}_release_v1.1.json").write_text(json.dumps(rows))
    tfidf = [{"id": r["uid"], "doc_retrieval_results": [[list(ARTICLES), [1.0] * len(ARTICLES)]]} for r in rows]
    (hover / "dev_tfidf_doc_retrieval_results.json").write_text(json.dumps(tfidf))
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"title": t, "text": x}) + "\n" for t, x in ARTICLES.items()))

    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_multihop

    out = tmp_path / "results"
    monkeypatch.setattr(sys, "argv", [
        "run_multihop.py", "--hover-dir", str(hover), "--wiki", str(corpus), "--verifier", str(model_dir),
        "--zero-shot", str(model_dir), "--reranker", "lexical", "--per-cell", "2", "--train-joint",
        "--train-per-cell", "2", "--epochs", "1", "--out", str(out), "--model-out", str(tmp_path / "reader"),
    ])
    run_multihop.main()
    res = json.loads((out / "multihop.json").read_text())
    assert res["n_dev"] == 8 and res["gold_coverage"] == 1.0
    assert set(res["e1_gold"]) == {"zero_shot", "scifact_finetuned", "hover_joint_trained"}
    assert "joint" in res["e1_gold"]["zero_shot"] and "2_hops" in res["e1_gold"]["zero_shot"]["joint"]
    assert "all_found@10" in res["e2_retrieval"]["iterative"]
    assert res["e2b_end_to_end"]["reader"] == "hover_joint_trained"
