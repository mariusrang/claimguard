"""The full ClaimGuard pipeline: retrieve -> rerank -> select sentences -> verify -> aggregate.

Two ways to use it:
- corpus mode : the evidence is searched in an indexed corpus (SciFact)
- sources mode: the user gives the documents (the RAG / "is this answer faithful
                to my documents?" use case). Sources are cut into small chunks so
                long documents also work.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from claimguard import LABELS, __version__
from claimguard.aggregate import aggregate
from claimguard.audit import AuditLogger
from claimguard.data import Document, load_corpus
from claimguard.decompose import analyse_answer
from claimguard.retrieval.bm25 import BM25Index
from claimguard.retrieval.rerank import LexicalReranker, rerank
from claimguard.text import split_sentences
from claimguard.utils import load_json, softmax, timer

MAX_CLAIM_CHARS = 1000
MAX_ANSWER_CHARS = 20000


class InvalidInputError(ValueError):
    """Raised when the user input cannot be processed (empty, too long...)."""


def validate_text(text: object, max_chars: int, what: str = "text") -> str:
    """Clean and check a user input, raise InvalidInputError with a clear message."""
    if not isinstance(text, str):
        raise InvalidInputError(f"The {what} must be text.")
    text = " ".join(text.split())
    if not text:
        raise InvalidInputError(f"The {what} is empty.")
    if len(text) > max_chars:
        raise InvalidInputError(f"The {what} is too long ({len(text)} characters, max {max_chars}).")
    if not any(ch.isalpha() for ch in text):
        raise InvalidInputError(f"The {what} does not contain any word.")
    return text


@dataclass
class Evidence:
    """One piece of evidence and what the verifier thinks of it."""

    source: str  # doc id, or "source 1 / chunk 2" in sources mode
    title: str
    sentence_ids: list[int]
    text: str
    rerank_score: float
    logits: list[float]
    probs: list[float]  # [SUPPORTED, REFUTED, NEI] after temperature scaling


@dataclass
class ClaimResult:
    """Final verdict for one claim."""

    claim: str
    label: str  # SUPPORTED / REFUTED / NEI (the model's best guess)
    confidence: float
    decision: str  # same as label, or UNCERTAIN if confidence < threshold
    probs: list[float]
    threshold: float
    evidence: list[Evidence] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)  # first-stage ranking (for metrics)
    timings_ms: dict = field(default_factory=dict)
    top_k: int = 0  # number of documents read for this verdict
    model_version: str = ""  # setup that produced this verdict (also written to the audit log)
    note: str = ""  # why a sentence was not checked (decision NOT_CHECKED)

    def to_dict(self) -> dict:
        """JSON-ready view of the result (used by the MCP server and the audit log)."""
        return asdict(self)


def chunk_sources(sources: list[str], chunk_size: int = 6) -> dict[str, Document]:
    """Cut user documents into chunks of `chunk_size` sentences."""
    chunks: dict[str, Document] = {}
    for s_idx, source in enumerate(sources, start=1):
        sentences = split_sentences(source)
        for c_idx, start in enumerate(range(0, len(sentences), chunk_size), start=1):
            doc_id = f"source {s_idx} / chunk {c_idx}"
            chunks[doc_id] = Document(doc_id, "", sentences[start : start + chunk_size])
    return chunks


class ClaimGuard:
    """Glue object that holds the components and runs them in order."""

    def __init__(
        self,
        verifier,
        reranker=None,
        retriever=None,
        corpus: dict[str, Document] | None = None,
        aggregation: str = "noisy_or",
        temperature: float = 1.0,
        threshold: float = 0.0,
        top_k: int = 5,
        n_sentences: int = 3,
        depth: int = 50,
        audit: AuditLogger | None = None,
    ) -> None:
        self.verifier = verifier
        self.reranker = reranker  # can be None (ablation "no reranker")
        self.sentence_selector = reranker or LexicalReranker()
        self.retriever = retriever
        self.corpus = corpus or {}
        self.aggregation = aggregation
        self.temperature = temperature
        self.threshold = threshold
        self.top_k = top_k
        self.n_sentences = n_sentences
        self.depth = depth
        self.audit = audit

    @property
    def model_version(self) -> str:
        """Setup with the default threshold and top_k (see version_for for one verdict)."""
        return self.version_for(self.threshold, self.top_k)

    def version_for(self, threshold: float, top_k: int) -> str:
        """String stored with each verdict so it can be traced to the exact setup.

        The weights fingerprint (sha256 of the checkpoint file) changes if the model
        files change even when the folder name stays the same."""
        name = getattr(self.verifier, "name", type(self.verifier).__name__)
        weights = getattr(self.verifier, "weights_sha", "")
        return (
            f"claimguard={__version__};verifier={name}{'@' + weights if weights else ''};"
            f"agg={self.aggregation};T={self.temperature:.3f};threshold={threshold:.2f};top_k={top_k}"
        )

    # ---- step 1 + 2: find candidate documents ----

    def _rank_documents(
        self, claim: str, corpus: dict[str, Document], retriever, timings: dict, top_k: int
    ) -> tuple[list[str], list[tuple[str, float]]]:
        """Return (first-stage ranking, [(doc_id, rerank score)] for the top_k docs)."""
        with timer(timings, "retrieval"):
            first_stage = [d for d, _ in retriever.search(claim, self.depth) if d in corpus]
        with timer(timings, "rerank"):
            if self.reranker is None:
                # no reranker: keep the retriever order, fake decreasing scores
                top = [(d, 1.0 / (i + 1)) for i, d in enumerate(first_stage[:top_k])]
            else:
                texts = [corpus[d].text for d in first_stage]
                top = [(first_stage[i], s) for i, s in rerank(self.reranker, claim, texts, top_k)]
        return first_stage, top

    # ---- step 3 + 4: pick sentences and verify ----

    def _verify_documents(
        self, claim: str, corpus: dict[str, Document], top: list[tuple[str, float]], timings: dict
    ) -> list[Evidence]:
        """Select the best sentences of each document, run the verifier, aggregate and decide."""
        selected = []
        with timer(timings, "rerank"):
            for doc_id, score in top:
                doc = corpus[doc_id]
                picked = sorted(i for i, _ in rerank(self.sentence_selector, claim, doc.sentences, self.n_sentences))
                text = " ".join(doc.sentences[i] for i in picked)
                selected.append((doc_id, doc.title, picked, text, score))
        with timer(timings, "verification"):
            logits = self.verifier.predict_logits([(claim, s[3]) for s in selected])
        probs = softmax(logits / self.temperature) if len(selected) else logits
        return [
            Evidence(doc_id, title, picked, text, float(score), logits[i].tolist(), probs[i].tolist())
            for i, (doc_id, title, picked, text, score) in enumerate(selected)
        ]

    def _finish(
        self, claim: str, evidence: list[Evidence], retrieved: list[str], timings: dict, mode: str,
        threshold: float | None = None, top_k: int = 0,
    ) -> ClaimResult:
        """Step 5: aggregate, apply the abstention threshold, write the audit log."""
        threshold = self.threshold if threshold is None else threshold
        probs = aggregate(np.array([e.probs for e in evidence]).reshape(-1, 3), self.aggregation)
        idx = int(probs.argmax())
        confidence = float(probs[idx])
        decision = LABELS[idx] if confidence >= threshold else "UNCERTAIN"
        result = ClaimResult(
            claim=claim,
            label=LABELS[idx],
            confidence=confidence,
            decision=decision,
            probs=probs.tolist(),
            threshold=threshold,
            evidence=evidence,
            retrieved=retrieved,
            timings_ms={k: round(v, 2) for k, v in timings.items()},
            top_k=top_k or len(evidence),
            model_version=self.version_for(threshold, top_k or len(evidence)),
        )
        if self.audit is not None:
            self.audit.log(result.to_dict(), mode=mode)
        return result

    def _not_checked(self, sentence: str, reason: str, threshold: float | None) -> ClaimResult:
        """A sentence of an answer that was not verified: reported, never dropped silently."""
        threshold = self.threshold if threshold is None else threshold
        result = ClaimResult(
            claim=sentence, label="NOT_CHECKED", confidence=0.0, decision="NOT_CHECKED",
            probs=[0.0, 0.0, 0.0], threshold=threshold, note=reason,
            model_version=self.version_for(threshold, 0),
        )
        if self.audit is not None:
            self.audit.log(result.to_dict(), mode="not_checked")
        return result

    # ---- public API ----

    def verify_claim(
        self, claim: str, sources: list[str] | None = None, threshold: float | None = None,
        top_k: int | None = None, min_words: int = 3,
    ) -> ClaimResult:
        """Verify one claim against the indexed corpus, or against `sources` if given.

        threshold and top_k can be set per call (the demo has one slider per user, so they
        must not change the shared pipeline object)."""
        claim = validate_text(claim, MAX_CLAIM_CHARS, "claim")
        if len(claim.split()) < min_words:
            raise InvalidInputError(f"Type a full factual sentence (at least {min_words} words).")
        top_k = self.top_k if top_k is None else int(top_k)
        timings: dict = {}
        if sources:
            corpus = chunk_sources([s for s in sources if isinstance(s, str) and s.strip()])
            if not corpus:
                raise InvalidInputError("The sources are empty.")
            # small on-the-fly BM25 index over the chunks of the user documents
            retriever = BM25Index().build(list(corpus), [d.text for d in corpus.values()])
            mode = "verify_claim_sources"
        else:
            if self.retriever is None or not self.corpus:
                raise InvalidInputError("No corpus is loaded: please give source documents.")
            corpus, retriever, mode = self.corpus, self.retriever, "verify_claim_corpus"
        first_stage, top = self._rank_documents(claim, corpus, retriever, timings, top_k)
        if sources and not top:
            # BM25 found no shared word at all: still look at the first chunks
            top = [(d, 0.0) for d in list(corpus)[:top_k]]
        evidence = self._verify_documents(claim, corpus, top, timings)
        return self._finish(claim, evidence, first_stage, timings, mode, threshold, top_k)

    def verify_with_evidence(self, claim: str, evidence_texts: list[str], mode: str = "given_evidence") -> ClaimResult:
        """Skip retrieval: the evidence is given (oracle experiments, citation checking)."""
        claim = validate_text(claim, MAX_CLAIM_CHARS, "claim")
        texts = [t for t in evidence_texts if isinstance(t, str) and t.strip()]
        timings: dict = {}
        with timer(timings, "verification"):
            logits = self.verifier.predict_logits([(claim, t) for t in texts])
        probs = softmax(logits / self.temperature) if texts else logits
        evidence = [
            Evidence(f"evidence {i + 1}", "", [], t, 0.0, logits[i].tolist(), probs[i].tolist())
            for i, t in enumerate(texts)
        ]
        return self._finish(claim, evidence, [], timings, mode)

    def verify_answer(
        self, answer: str, sources: list[str] | None = None, threshold: float | None = None,
        top_k: int | None = None,
    ) -> list[ClaimResult]:
        """Split an LLM answer into sentences and verify each factual one.

        Every statement comes back: a sentence too short to check gets the decision
        NOT_CHECKED (it counts as something to review). Questions and courtesy sentences
        ("Hope this helps!") are skipped; ignored_sentences() lists them."""
        answer = validate_text(answer, MAX_ANSWER_CHARS, "answer")
        segments = analyse_answer(answer)
        if not any(s.status != "ignored" for s in segments):
            raise InvalidInputError("No factual statement was found in this text.")
        results = []
        for seg in segments:
            if seg.status == "claim":
                results.append(self.verify_claim(seg.text, sources, threshold, top_k, min_words=1))
            elif seg.status == "too_short":
                results.append(self._not_checked(seg.original, seg.reason, threshold))
        return results

    @staticmethod
    def ignored_sentences(answer: str) -> list[dict]:
        """Sentences of an answer that are not statements (questions, courtesy)."""
        return [{"sentence": s.original, "reason": s.reason} for s in analyse_answer(answer) if s.status == "ignored"]

    def verify_citations(self, items: list[dict]) -> list[ClaimResult]:
        """Check that each cited passage really supports its sentence.

        items = [{"sentence": "...", "cited_text": "..."}]. Useful when the LLM
        already gives citations: no need to search again, just check them.
        """
        if not items:
            raise InvalidInputError("No citation to check.")
        results = []
        for item in items:
            if not isinstance(item, dict) or "sentence" not in item or "cited_text" not in item:
                raise InvalidInputError("Each item needs a 'sentence' and a 'cited_text'.")
            results.append(self.verify_with_evidence(item["sentence"], [item["cited_text"]], mode="verify_citation"))
        return results


def build_retriever(cfg: dict, corpus: dict[str, Document]):
    """Create the first-stage retriever described in the config."""
    kind = cfg.get("kind", "bm25")
    doc_ids = list(corpus)
    texts = [corpus[d].text for d in doc_ids]
    if kind == "none" or not corpus:
        return None
    bm25_cfg = cfg.get("bm25", {})
    bm25 = BM25Index(bm25_cfg.get("k1", 1.5), bm25_cfg.get("b", 0.75)).build(doc_ids, texts)
    if kind == "bm25":
        return bm25
    # dense and hybrid need the neural encoder (imported here so offline mode stays light)
    from claimguard.retrieval.dense import DenseIndex, TransformerEncoder
    from claimguard.retrieval.hybrid import HybridRetriever

    dense_cfg = cfg.get("dense", {})
    encoder = TransformerEncoder(dense_cfg.get("model", "BAAI/bge-small-en-v1.5"), dense_cfg.get("pooling", "cls"))
    dense = DenseIndex(encoder, dense_cfg.get("query_prefix", ""), cfg.get("cache_dir")).build(doc_ids, texts)
    if kind == "dense":
        return dense
    if kind == "hybrid":
        return HybridRetriever([bm25, dense], cfg.get("rrf_k", 60), cfg.get("depth", 50))
    raise ValueError(f"unknown retrieval kind '{kind}'")


def build_reranker(cfg: dict):
    """Cross-encoder, lexical reranker or none, as asked by the config."""
    kind = cfg.get("kind", "lexical")
    if kind == "none":
        return None
    if kind == "lexical":
        return LexicalReranker()
    if kind == "cross_encoder":
        from claimguard.retrieval.rerank import CrossEncoderReranker

        return CrossEncoderReranker(cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    raise ValueError(f"unknown rerank kind '{kind}'")


def weights_fingerprint(model: str) -> str:
    """First 12 hex digits of the sha256 of a local checkpoint ('' for a hub name)."""
    path = Path(model) / "model.safetensors"
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:12]


def build_verifier(cfg: dict):
    """NLI verifier (fine-tuned folder or hub checkpoint), LLM judge or rule-based baseline."""
    kind = cfg.get("kind", "lexical")
    if kind == "lexical":
        from claimguard.verify.lexical import LexicalVerifier

        return LexicalVerifier()
    if kind == "nli":
        from claimguard.verify.nli import NLIVerifier

        model = cfg.get("model", "")
        fallback = cfg.get("fallback_model")
        # if the fine-tuned folder does not exist yet, fall back to the public checkpoint
        looks_local = model.startswith(("models/", "./", "/"))
        fallback_from = ""
        if fallback and (not model or (looks_local and not Path(model).exists())):
            # stderr, not stdout: the MCP server speaks JSON on stdout
            print(f"[claimguard] '{model}' not found, using the public checkpoint {fallback}", file=sys.stderr)
            fallback_from, model = model, fallback
        verifier = NLIVerifier(model, cfg.get("max_length", 384), cfg.get("batch_size", 32))
        verifier.fallback_from = fallback_from  # shown as a warning in the demo
        verifier.weights_sha = weights_fingerprint(model)
        return verifier
    if kind == "llm":
        from claimguard.verify.llm_judge import LLMJudgeVerifier

        return LLMJudgeVerifier(cfg.get("model", "Qwen/Qwen2.5-1.5B-Instruct"), cfg.get("load_in_4bit", False))
    raise ValueError(f"unknown verifier kind '{kind}'")


def build_pipeline(cfg: dict, with_corpus: bool = True) -> ClaimGuard:
    """Build the whole pipeline from a config dict (see configs/default.yaml)."""
    corpus: dict[str, Document] = {}
    corpus_path = Path(cfg.get("data", {}).get("dir", "")) / "corpus.jsonl"
    if with_corpus and corpus_path.exists():
        corpus = load_corpus(corpus_path)
    retrieval_cfg = cfg.get("retrieval", {})
    rerank_cfg = cfg.get("rerank", {})
    calib = dict(cfg.get("calibration", {}))
    calib_file = calib.get("file")
    if calib_file and Path(calib_file).exists():
        calib.update(load_json(calib_file))  # values fitted by `evaluate.py calibration`
    audit_path = cfg.get("audit", {}).get("path")
    return ClaimGuard(
        verifier=build_verifier(cfg.get("verifier", {})),
        reranker=build_reranker(rerank_cfg),
        retriever=build_retriever(retrieval_cfg, corpus) if corpus else None,
        corpus=corpus,
        # k and the aggregation chosen on dev (calibration file) win over the config defaults
        aggregation=calib.get("aggregation", cfg.get("aggregation", "noisy_or")),
        temperature=float(calib.get("temperature", 1.0)),
        threshold=float(calib.get("threshold", 0.0)),
        top_k=int(calib.get("top_k", rerank_cfg.get("top_k", 5))),
        n_sentences=rerank_cfg.get("n_sentences", 3),
        depth=retrieval_cfg.get("depth", 50),
        audit=AuditLogger(audit_path) if audit_path else None,
    )
