"""Shared fixtures. Nothing here needs the internet: the 'models' are tiny random ones."""

from __future__ import annotations

from pathlib import Path

import pytest

from claimguard.data import load_claims, load_corpus
from claimguard.pipeline import ClaimGuard
from claimguard.retrieval.bm25 import BM25Index
from claimguard.retrieval.rerank import LexicalReranker
from claimguard.verify.lexical import LexicalVerifier

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample"

WORDS = (
    "vitamin c does not reduce the incidence of colds aspirin risk heart attack statins lower "
    "cholesterol smoking causes lung cancer caffeine sleep increases decreases percent 85 8 30 A B C"
).split()


@pytest.fixture(scope="session")
def corpus():
    return load_corpus(SAMPLE / "corpus.jsonl")


@pytest.fixture(scope="session")
def claims():
    return load_claims(SAMPLE / "claims_train.jsonl")


@pytest.fixture()
def offline_pipeline(corpus, tmp_path):
    """BM25 + rule-based verifier on the toy corpus, with an audit log in a temp folder."""
    from claimguard.audit import AuditLogger

    bm25 = BM25Index().build(list(corpus), [d.text for d in corpus.values()])
    return ClaimGuard(
        verifier=LexicalVerifier(),
        reranker=LexicalReranker(),
        retriever=bm25,
        corpus=corpus,
        audit=AuditLogger(tmp_path / "audit.jsonl"),
    )


@pytest.fixture(scope="session")
def tiny_tokenizer():
    """Word-level tokenizer built on the fly (so the tests never download a vocab)."""
    pytest.importorskip("torch")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for word in WORDS:
        vocab.setdefault(word, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]"
    )


@pytest.fixture()
def tiny_nli_model(tiny_tokenizer):
    """1-layer BERT with random weights and an NLI-style head (labels in the hub order)."""
    import torch
    from transformers import BertConfig, BertForSequenceClassification

    torch.manual_seed(0)
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    config = BertConfig(
        vocab_size=len(tiny_tokenizer),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        num_labels=3,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
    )
    return BertForSequenceClassification(config)
