"""Dense retrieval with a bi-encoder (BGE-small by default).

The corpus has ~5k abstracts so I do exact search with one matrix product
(5k x 384 floats is tiny). FAISS would only make sense for a much bigger corpus.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Protocol

import numpy as np


class Encoder(Protocol):
    """Anything that turns texts into vectors (lets me plug a fake one in tests)."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class TransformerEncoder:
    """Sentence encoder built directly on transformers (CLS or mean pooling)."""

    def __init__(
        self,
        model_name: str,
        pooling: str = "cls",
        max_length: int = 512,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        from claimguard.utils import ensure_float32

        self.model = ensure_float32(AutoModel.from_pretrained(model_name)).to(self.device).eval()
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.model_name = model_name

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings, shape [len(texts), dim]."""
        torch = self.torch
        chunks = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                if self.pooling == "cls":
                    emb = hidden[:, 0]
                else:
                    # mean over real tokens only (mask out the padding)
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                emb = torch.nn.functional.normalize(emb, dim=-1)
                chunks.append(emb.cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 1), dtype=np.float32)


class DenseIndex:
    """Exact cosine-similarity search over document embeddings."""

    def __init__(self, encoder: Encoder, query_prefix: str = "", cache_dir: str | None = None) -> None:
        self.encoder = encoder
        self.query_prefix = query_prefix  # BGE wants an instruction before the query
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.doc_ids: list[str] = []
        self.matrix = np.zeros((0, 1), dtype=np.float32)

    def build(self, doc_ids: list[str], texts: list[str]) -> "DenseIndex":
        """Embed the corpus (or load the embeddings from disk if already done)."""
        self.doc_ids = list(doc_ids)
        cache_file = None
        if self.cache_dir is not None:
            # the cache key depends on the model and on the exact texts
            name = getattr(self.encoder, "model_name", "encoder")
            digest = hashlib.md5((name + "".join(texts)).encode("utf-8")).hexdigest()[:16]
            cache_file = self.cache_dir / f"dense_{digest}.npy"
            if cache_file.exists():
                self.matrix = np.load(cache_file)
                return self
        # first run only: on a laptop CPU this takes several minutes, so say it
        print(f"Embedding {len(texts)} documents for the dense index (first run only, cached after)...",
              file=sys.stderr, flush=True)  # stderr: stdout belongs to the MCP protocol
        self.matrix = self.encoder.encode(texts).astype(np.float32)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, self.matrix)
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        """Top-k (doc_id, cosine similarity)."""
        if len(self.doc_ids) == 0:
            return []
        q = self.encoder.encode([self.query_prefix + query])[0]
        scores = self.matrix @ q  # vectors are normalised so dot product = cosine
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.doc_ids[i], float(scores[i])) for i in top]
