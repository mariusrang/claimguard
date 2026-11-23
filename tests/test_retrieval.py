"""Tests of BM25, dense search, RRF and the rerankers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from claimguard.retrieval.bm25 import BM25Index
from claimguard.retrieval.dense import DenseIndex
from claimguard.retrieval.hybrid import HybridRetriever, rrf_fuse
from claimguard.retrieval.rerank import LexicalReranker, rerank


def test_bm25_score_matches_hand_computation():
    # 2 documents, query term "cat" appears once in doc A only
    index = BM25Index(k1=1.5, b=0.75).build(["A", "B"], ["cat dog", "dog dog dog bird"])
    # by hand: N=2, df=1 -> idf = ln(1 + (2-1+0.5)/(1+0.5)) = ln(2)
    idf = math.log(2.0)
    avg_len = (2 + 4) / 2
    norm = 1.5 * (1 - 0.75 + 0.75 * 2 / avg_len)
    expected = idf * 1 * (1.5 + 1) / (1 + norm)
    scores = index.scores("cat")
    assert scores[0] == pytest.approx(expected)
    assert scores[1] == 0.0


def test_bm25_search_ranks_the_relevant_doc_first(corpus):
    index = BM25Index().build(list(corpus), [d.text for d in corpus.values()])
    top = index.search("Does caffeine block adenosine receptors?", k=3)
    assert top[0][0] == "7"
    assert all(score > 0 for _, score in top)
    assert index.search("zzzz qqqq", k=3) == []  # unknown words -> nothing


def test_bm25_rejects_bad_input():
    with pytest.raises(ValueError):
        BM25Index().build(["a"], ["x", "y"])


def test_rrf_fuse_hand_values():
    fused = dict(rrf_fuse([["a", "b"], ["b", "c"]], k=60))
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)  # rank 2 then rank 1
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)
    # b is in both lists so it wins
    assert rrf_fuse([["a", "b"], ["b", "c"]])[0][0] == "b"


class FakeEncoder:
    """Bag-of-letters 'embedding', just to test the index logic without a model."""

    model_name = "fake"

    def encode(self, texts):
        out = np.zeros((len(texts), 26), dtype=np.float32)
        for i, text in enumerate(texts):
            for ch in text.lower():
                if "a" <= ch <= "z":
                    out[i, ord(ch) - 97] += 1
        return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)


def test_dense_index_search_and_cache(tmp_path):
    docs = {"d1": "aaaa bbbb", "d2": "zzzz yyyy", "d3": "aabb"}
    index = DenseIndex(FakeEncoder(), cache_dir=str(tmp_path)).build(list(docs), list(docs.values()))
    top = index.search("aaa bbb", k=2)
    assert {d for d, _ in top} == {"d1", "d3"}
    assert top[0][1] == pytest.approx(1.0, abs=1e-5)  # same direction -> cosine 1
    assert len(list(tmp_path.glob("dense_*.npy"))) == 1
    # second build reads the cache and gives the same matrix
    again = DenseIndex(FakeEncoder(), cache_dir=str(tmp_path)).build(list(docs), list(docs.values()))
    assert np.allclose(again.matrix, index.matrix)


def test_hybrid_combines_both_retrievers(corpus):
    ids, texts = list(corpus), [d.text for d in corpus.values()]
    bm25 = BM25Index().build(ids, texts)
    dense = DenseIndex(FakeEncoder()).build(ids, texts)
    hybrid = HybridRetriever([bm25, dense], rrf_k=60, depth=12)
    top = [d for d, _ in hybrid.search("statins lower LDL cholesterol", k=5)]
    assert "4" in top and len(top) == 5
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_lexical_reranker_and_rerank_helper():
    reranker = LexicalReranker()
    texts = ["the weather is nice", "aspirin reduces heart attack risk", "aspirin is a drug"]
    order = rerank(reranker, "aspirin heart attack", texts, top_k=2)
    assert [i for i, _ in order] == [1, 2]
    assert rerank(reranker, "aspirin", [], 3) == []
