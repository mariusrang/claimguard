"""Tests of the aggregation, the full pipeline, the contrast set, the audit log, the MCP tools and the demo."""

from __future__ import annotations

import random

import numpy as np
import pytest

from claimguard.aggregate import aggregate
from claimguard.contrast import build_contrast_set, perturb_antonym, perturb_negation, perturb_number
from claimguard.pipeline import InvalidInputError, validate_text
from claimguard.utils import read_jsonl

S, R, N = [0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]


def test_aggregate_majority_is_drowned_by_nei():
    probs = np.array([S, N, N, N, N])
    assert aggregate(probs, "majority").argmax() == 2  # 4 NEI votes win
    assert aggregate(probs, "max_conf").argmax() == 0  # the one decisive doc wins
    assert aggregate(probs, "noisy_or").argmax() == 0


def test_aggregate_noisy_or_hand_values():
    probs = np.array([[0.5, 0.1, 0.4], [0.5, 0.1, 0.4]])
    out = aggregate(probs, "noisy_or")
    raw = np.array([1 - 0.25, 1 - 0.81, 0.16])
    assert np.allclose(out, raw / raw.sum())
    assert out.sum() == pytest.approx(1.0)


def test_aggregate_edge_cases():
    assert aggregate(np.zeros((0, 3)), "noisy_or").tolist() == [0.0, 0.0, 1.0]  # no evidence -> NEI
    assert aggregate(np.array([R, N]), "max_conf").argmax() == 1
    with pytest.raises(ValueError):
        aggregate(np.array([S]), "unknown")
    with pytest.raises(ValueError):
        aggregate(np.array([0.5, 0.5]), "majority")


def test_pipeline_corpus_mode(offline_pipeline):
    result = offline_pipeline.verify_claim("Tobacco smoking accounts for about 85 percent of lung cancer cases.")
    assert result.label == "SUPPORTED" and result.decision == "SUPPORTED"
    assert result.evidence[0].source == "8"
    assert result.retrieved[0] == "8"
    assert sum(result.probs) == pytest.approx(1.0)
    assert {"retrieval", "rerank", "verification"} <= set(result.timings_ms)
    refuted = offline_pipeline.verify_claim("Tobacco smoking accounts for about 8 percent of lung cancer cases.")
    assert refuted.label == "REFUTED"


def test_pipeline_sources_mode_and_answer(offline_pipeline):
    source = "The fixed rate of the loan is 3.45% over 20 years. The minimum contribution is 10% of the project."
    answer = "The fixed rate of the loan is 4.35% over 20 years. The minimum contribution is 10% of the project."
    results = offline_pipeline.verify_answer(answer, [source])
    assert [r.label for r in results] == ["REFUTED", "SUPPORTED"]
    assert results[0].evidence[0].source.startswith("source 1")


def test_pipeline_abstains_below_threshold(offline_pipeline):
    offline_pipeline.threshold = 0.99
    result = offline_pipeline.verify_claim("Statins reduce LDL cholesterol by 30 to 40 percent.")
    assert result.decision == "UNCERTAIN" and result.label in ("SUPPORTED", "REFUTED", "NEI")


def test_pipeline_citation_mode(offline_pipeline):
    items = [{"sentence": "Aspirin increases the risk of bleeding.", "cited_text": "Aspirin increases the risk of gastrointestinal bleeding."}]
    assert offline_pipeline.verify_citations(items)[0].label == "SUPPORTED"
    with pytest.raises(InvalidInputError):
        offline_pipeline.verify_citations([{"sentence": "only one field"}])


@pytest.mark.parametrize("bad", ["", "   ", "12345 !!!", None, "x" * 5000])
def test_invalid_inputs_are_rejected(offline_pipeline, bad):
    with pytest.raises(InvalidInputError):
        offline_pipeline.verify_claim(bad)


def test_validate_text_cleans_whitespace():
    assert validate_text("  hello \n world ", 100) == "hello world"


def test_no_corpus_and_no_sources_gives_a_clear_error():
    from claimguard.pipeline import ClaimGuard
    from claimguard.verify.lexical import LexicalVerifier

    with pytest.raises(InvalidInputError, match="No corpus"):
        ClaimGuard(LexicalVerifier()).verify_claim("Aspirin reduces risk.")


def test_audit_log_records_everything(offline_pipeline):
    offline_pipeline.verify_claim("Caffeine has a half-life of about 5 hours in healthy adults.")
    records = read_jsonl(offline_pipeline.audit.path)
    assert len(records) == 1
    record = records[0]
    assert {"timestamp", "model_version", "claim", "label", "confidence", "decision", "evidence"} <= set(record)
    assert "verifier=lexical-rules" in record["model_version"]
    assert record["evidence"][0]["text"]
    assert offline_pipeline.audit.tail(5) == records


def test_contrast_perturbations():
    rng = random.Random(0)
    assert perturb_number("no digits here", rng) is None
    changed = perturb_number("It reduces risk by 30 percent.", rng)
    assert changed != "It reduces risk by 30 percent." and "30 " not in changed
    assert perturb_negation("Aspirin is effective.", rng) == "Aspirin is not effective."
    assert perturb_negation("Aspirin is not effective.", rng) == "Aspirin is effective."
    assert perturb_antonym("Statins increase survival.", rng) == "Statins decrease survival."
    assert perturb_antonym("Higher doses work.", rng) == "Lower doses work."


def test_build_contrast_set():
    pairs = [{"id": "1", "claim": "Aspirin is linked to a higher risk of 30 percent.", "document": "doc"}]
    rows = build_contrast_set(pairs, seed=1)
    assert {r["perturbation"] for r in rows} == {"number", "negation", "antonym"}
    assert all(r["claim"] != r["original_claim"] and r["label"] == "REFUTED" for r in rows)
    assert rows == build_contrast_set(pairs, seed=1)


def test_mcp_tools(offline_pipeline):
    pytest.importorskip("mcp")
    from claimguard import mcp_server

    mcp_server.set_pipeline(offline_pipeline)
    source = "Aspirin increases the risk of gastrointestinal bleeding."
    out = mcp_server.verify_claim("Aspirin does not increase the risk of gastrointestinal bleeding.", [source])
    assert out["decision"] == "REFUTED" and out["evidence"] == source
    answer = mcp_server.verify_answer("Aspirin increases the risk of gastrointestinal bleeding. Is it ok?", [source])
    assert answer["n_claims"] == 1 and answer["n_flagged"] == 0
    cites = mcp_server.verify_citations([{"sentence": source, "cited_text": source}])
    assert cites["citations"][0]["decision"] == "SUPPORTED"
    assert mcp_server.verify_claim("", None) == {"error": "The claim is empty."}
    assert len(mcp_server.get_audit_log(10)["records"]) == 3


def test_demo_handlers(monkeypatch, tmp_path):
    pytest.importorskip("gradio")
    monkeypatch.setenv("CLAIMGUARD_CONFIG", "configs/offline.yaml")
    monkeypatch.chdir(__import__("pathlib").Path(__file__).resolve().parents[1])
    import importlib

    import app

    importlib.reload(app)
    app.PIPELINE.audit = None  # do not write logs from the tests
    card, scores = app.verify_claim_ui(
        "Tobacco smoking accounts for about 85 percent of lung cancer cases.", "", None, 0.0, 5
    )
    assert "Supported by the sources" in card and set(scores) == {"Supported", "Contradicted", "Not in the sources"}
    assert "Inference time" in card and "Model size" in card
    assert "cg-msg" in app.verify_claim_ui("", "", None, 0.0, 5)[0]  # empty claim -> error message, no crash
    assert "at least 3 words" in app.verify_claim_ui("ok", "", None, 0.0, 5)[0]  # too short to be a claim
    bad_file = tmp_path / "x.pdf"
    bad_file.write_bytes(b"%PDF")
    assert "upload a .txt" in app.verify_claim_ui("Aspirin lowers the risk of heart attack.", "", str(bad_file), 0.0, 5)[0]
    summary, page, rows = app.check_answer_ui(app.ANSWER_EXAMPLE, app.SOURCE_EXAMPLE, None, 0.0)
    assert "5 claims checked" in summary and len(rows) == 5
    assert page.count("class='claim ") == 5 and "Model size" in page
    # the contradicted rate must come back as contradicted, with the source passage in the margin
    assert rows[0][2] == "Contradicted" and "3.45%" in page


def test_aggrefact_evaluation_logic(offline_pipeline):
    from claimguard.aggrefact import balanced_accuracy, evaluate_rows, sample_per_dataset

    # half right on each class -> 0.5; all predicted "supported" on a balanced set -> 0.5
    assert balanced_accuracy(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1])) == 0.5
    assert balanced_accuracy(np.array([1, 1, 0, 0]), np.array([1, 1, 1, 1])) == 0.5
    assert balanced_accuracy(np.array([1, 0]), np.array([1, 0])) == 1.0
    doc = "The fixed rate of the loan is 3.45% over 20 years. Application fees are 900 euros."
    rows = [
        {"dataset": "A", "doc": doc, "claim": "The fixed rate of the loan is 3.45% over 20 years.", "label": 1},
        {"dataset": "A", "doc": doc, "claim": "The fixed rate of the loan is 4.35% over 20 years.", "label": 0},
        {"dataset": "B", "doc": doc, "claim": "Application fees are 900 euros.", "label": 1},
    ]
    assert len(sample_per_dataset(rows, per_dataset=1)) == 2
    out = evaluate_rows(offline_pipeline, rows)
    assert out["n"] == 3 and set(out["by_dataset"]) == {"A", "B"}
    assert out["by_dataset"]["A"]["balanced_accuracy"] == 1.0
