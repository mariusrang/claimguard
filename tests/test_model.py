"""Tests of the neural parts with a tiny random model (forward pass, training loop, quantization)."""

from __future__ import annotations

import numpy as np
import pytest

from claimguard.utils import softmax
from claimguard.verify.lexical import LexicalVerifier
from claimguard.verify.nli import map_model_labels

torch = pytest.importorskip("torch")


def test_map_model_labels_handles_different_orders():
    # hub order of the MoritzLaurer checkpoints
    assert map_model_labels({0: "entailment", 1: "neutral", 2: "contradiction"}) == [0, 2, 1]
    # another common order
    assert map_model_labels({0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}) == [2, 0, 1]
    # our own checkpoints
    assert map_model_labels({0: "SUPPORTED", 1: "REFUTED", 2: "NEI"}) == [0, 1, 2]
    with pytest.raises(ValueError):
        map_model_labels({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_nli_verifier_forward_pass(tiny_nli_model, tiny_tokenizer):
    from claimguard.verify.nli import NLIVerifier

    verifier = NLIVerifier(model=tiny_nli_model, tokenizer=tiny_tokenizer, device="cpu", batch_size=2)
    pairs = [("vitamin c does not reduce colds", "vitamin c reduce colds")] * 5
    logits = verifier.predict_logits(pairs)
    assert logits.shape == (5, 3)
    assert np.isfinite(logits).all()
    assert np.allclose(softmax(logits).sum(axis=1), 1.0)
    assert verifier.predict_logits([]).shape == (0, 3)


def test_nli_verifier_reorders_columns(tiny_nli_model, tiny_tokenizer):
    from claimguard.verify.nli import NLIVerifier

    verifier = NLIVerifier(model=tiny_nli_model, tokenizer=tiny_tokenizer, device="cpu")
    claim, evidence = "smoking causes cancer", "smoking causes lung cancer"
    ours = verifier.predict_logits([(claim, evidence)])[0]
    enc = tiny_tokenizer([evidence], [claim], return_tensors="pt")
    with torch.no_grad():
        raw = tiny_nli_model(**enc).logits[0].numpy()
    # model order = entailment, neutral, contradiction -> ours = SUPPORTED, REFUTED, NEI
    assert np.allclose(ours, raw[[0, 2, 1]], atol=1e-5)


def test_training_loop_runs_and_saves(tiny_nli_model, tiny_tokenizer, tmp_path):
    from claimguard.verify.nli import NLIVerifier
    from claimguard.verify.train import train_verifier

    pairs = [
        ("smoking causes cancer", "smoking causes lung cancer", 0),
        ("smoking does not cause cancer", "smoking causes lung cancer", 1),
        ("caffeine increases sleep", "statins lower cholesterol", 2),
    ] * 8
    summary = train_verifier(
        pairs, pairs[:6], tmp_path / "ckpt", epochs=2, batch_size=4, lr=1e-3,
        max_length=32, device="cpu", model=tiny_nli_model, tokenizer=tiny_tokenizer,
    )
    assert len(summary["history"]) == 2
    assert summary["history"][-1]["train_loss"] < summary["history"][0]["train_loss"] + 1e-6
    assert (tmp_path / "ckpt" / "train_log.json").exists()
    # the saved checkpoint can be loaded back like a hub model
    reloaded = NLIVerifier(str(tmp_path / "ckpt"), device="cpu")
    assert reloaded.predict_logits([("smoking causes cancer", "smoking causes lung cancer")]).shape == (1, 3)


def test_build_training_pairs(corpus, claims):
    from claimguard.retrieval.bm25 import BM25Index
    from claimguard.verify.train import build_training_pairs

    bm25 = BM25Index().build(list(corpus), [d.text for d in corpus.values()])
    pairs = build_training_pairs(claims, corpus, retriever=bm25, n_neg=1, seed=3)
    labels = [y for _, _, y in pairs]
    assert set(labels) == {0, 1, 2}
    # every non-NEI claim gets one hard negative, so NEI pairs outnumber the NEI claims
    n_nei_claims = sum(c.label == "NEI" for c in claims)
    assert labels.count(2) > n_nei_claims
    assert pairs == build_training_pairs(claims, corpus, retriever=bm25, n_neg=1, seed=3)  # deterministic


def test_transformer_encoder_and_cross_encoder(tiny_nli_model, tiny_tokenizer, tmp_path):
    from claimguard.retrieval.dense import TransformerEncoder
    from claimguard.retrieval.rerank import CrossEncoderReranker

    tiny_nli_model.save_pretrained(tmp_path / "m")
    tiny_tokenizer.save_pretrained(tmp_path / "m")
    for pooling in ("cls", "mean"):
        encoder = TransformerEncoder(str(tmp_path / "m"), pooling=pooling, device="cpu")
        emb = encoder.encode(["smoking causes cancer", "caffeine", "statins lower cholesterol"])
        assert emb.shape == (3, 32)
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    reranker = CrossEncoderReranker(str(tmp_path / "m"), device="cpu")
    assert reranker.score("smoking", ["smoking causes cancer", "caffeine"]).shape == (2,)


def test_dynamic_quantization_keeps_the_interface(tiny_nli_model, tiny_tokenizer):
    from claimguard.efficiency import benchmark, count_parameters, model_size_mb, quantize_dynamic_int8
    from claimguard.verify.nli import NLIVerifier

    size_fp32 = model_size_mb(tiny_nli_model)
    quantized = quantize_dynamic_int8(tiny_nli_model)
    assert model_size_mb(quantized) < size_fp32
    verifier = NLIVerifier(model=quantized, tokenizer=tiny_tokenizer, device="cpu")
    assert verifier.predict_logits([("smoking causes cancer", "smoking causes lung cancer")]).shape == (1, 3)
    assert count_parameters(tiny_nli_model) > 0
    # partial recipe: only the feed-forward layers are quantized, attention and classifier stay float32
    partial = quantize_dynamic_int8(tiny_nli_model, only_feed_forward=True)
    quantized_names = {n for n, m in partial.named_modules() if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)}
    assert quantized_names == {"bert.encoder.layer.0.intermediate.dense", "bert.encoder.layer.0.output.dense"}
    assert model_size_mb(quantized) < model_size_mb(partial) < size_fp32
    assert isinstance(tiny_nli_model.classifier, torch.nn.Linear)  # the original model was not modified
    stats = benchmark(lambda: None, n_warmup=1, n_runs=5)
    assert stats["n_runs"] == 5 and stats["p95_ms"] >= stats["p50_ms"] >= 0


def test_lexical_verifier_rules():
    v = LexicalVerifier()
    evidence = "Tobacco smoking accounts for about 85 percent of lung cancer cases."
    pairs = [
        ("Tobacco smoking accounts for about 85 percent of lung cancer cases.", evidence),  # same
        ("Tobacco smoking accounts for about 8 percent of lung cancer cases.", evidence),  # number changed
        ("Tobacco smoking does not account for lung cancer cases.", evidence),  # negation added
        ("Coffee improves the memory of students.", evidence),  # unrelated
    ]
    assert v.predict_logits(pairs).argmax(axis=1).tolist() == [0, 1, 1, 2]


def test_llm_judge_reads_the_letter_logits(tiny_tokenizer):
    from transformers import GPT2Config, GPT2LMHeadModel

    from claimguard.verify.llm_judge import LLMJudgeVerifier

    torch.manual_seed(0)
    lm = GPT2LMHeadModel(GPT2Config(vocab_size=len(tiny_tokenizer), n_embd=32, n_layer=1, n_head=2, n_positions=256))
    judge = LLMJudgeVerifier(model=lm, tokenizer=tiny_tokenizer, device="cpu", max_length=200)
    assert len(set(judge.letter_ids)) == 3  # A, B and C are three different tokens
    logits = judge.predict_logits([("smoking causes cancer", "smoking causes lung cancer")] * 2)
    assert logits.shape == (2, 3) and np.isfinite(logits).all()
    assert np.allclose(logits[0], logits[1])  # same input -> same scores
    assert judge.predict_logits([]).shape == (0, 3)


def test_half_precision_checkpoint_is_converted_to_float32(tiny_nli_model, tiny_tokenizer):
    # this is the Kaggle bug: a checkpoint stored in float16 gave loss=nan in training
    from claimguard.efficiency import quantize_dynamic_int8
    from claimguard.verify.nli import NLIVerifier

    verifier = NLIVerifier(model=tiny_nli_model.half(), tokenizer=tiny_tokenizer, device="cpu")
    assert next(verifier.model.parameters()).dtype == torch.float32
    assert np.isfinite(verifier.predict_logits([("smoking causes cancer", "smoking causes lung cancer")])).all()
    quantize_dynamic_int8(verifier.model)  # failed with "expected Float but found Half" before the fix


def test_training_from_a_half_model_gives_finite_loss(tiny_nli_model, tiny_tokenizer, tmp_path):
    from claimguard.verify.train import train_verifier

    pairs = [("smoking causes cancer", "smoking causes lung cancer", 0), ("caffeine increases sleep", "statins lower cholesterol", 2)] * 6
    summary = train_verifier(pairs, pairs, tmp_path / "ckpt", epochs=1, batch_size=4, lr=1e-3, max_length=32,
                             device="cpu", model=tiny_nli_model.half(), tokenizer=tiny_tokenizer)
    assert np.isfinite(summary["history"][0]["train_loss"])


def test_training_stops_on_nan_loss(tiny_nli_model, tiny_tokenizer, tmp_path):
    from claimguard.verify.train import train_verifier

    with torch.no_grad():
        for p in tiny_nli_model.classifier.parameters():
            p.fill_(float("nan"))  # broken weights -> NaN logits -> NaN loss
    pairs = [("smoking causes cancer", "smoking causes lung cancer", 0)] * 4
    with pytest.raises(RuntimeError, match="loss is nan"):
        train_verifier(pairs, pairs, tmp_path / "ckpt", epochs=1, batch_size=2, lr=1e-3, max_length=32,
                       device="cpu", model=tiny_nli_model, tokenizer=tiny_tokenizer)
