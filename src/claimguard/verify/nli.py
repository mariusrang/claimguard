"""Neural verifier: a sequence-classification model used as an NLI model.

premise = evidence, hypothesis = claim
entailment -> SUPPORTED, contradiction -> REFUTED, neutral -> NEI

Different checkpoints order their labels differently, so I never hardcode the
indexes: I read the names in model.config.id2label and map them to our order.
"""

from __future__ import annotations

import numpy as np

from claimguard import LABELS

# names that models on the hub use for each of our labels
_ALIASES = {
    "SUPPORTED": {"entailment", "entail", "supported", "support", "supports"},
    "REFUTED": {"contradiction", "contradict", "refuted", "refutes"},
    "NEI": {"neutral", "nei", "not_enough_info", "not enough info"},
}


def map_model_labels(id2label: dict) -> list[int]:
    """Return [idx_supported, idx_refuted, idx_nei] in the model's own indexing."""
    names = {int(i): str(name).lower() for i, name in id2label.items()}
    mapping = []
    for label in LABELS:
        found = [i for i, name in names.items() if name in _ALIASES[label]]
        if len(found) != 1:
            raise ValueError(
                f"cannot find a unique '{label}' class in the model labels {id2label}. "
                "The verifier needs a 3-way NLI model (entailment / neutral / contradiction)."
            )
        mapping.append(found[0])
    return mapping


class NLIVerifier:
    """Wraps a HF model and returns logits in OUR label order."""

    def __init__(
        self,
        model_name: str | None = None,
        max_length: int = 384,
        batch_size: int = 32,
        device: str | None = None,
        model=None,
        tokenizer=None,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            # normal case: load from the hub or from a local folder
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()
        self.index_map = map_model_labels(self.model.config.id2label)
        self.max_length = max_length
        self.batch_size = batch_size
        self.name = model_name or "in-memory-model"

    def predict_logits(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """pairs = [(claim, evidence), ...] -> logits [n, 3] as SUPPORTED/REFUTED/NEI."""
        if not pairs:
            return np.zeros((0, 3))
        torch = self.torch
        out = []
        with torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                enc = self.tokenizer(
                    [evidence for _, evidence in batch],  # premise first
                    [claim for claim, _ in batch],  # then the hypothesis
                    padding=True,
                    truncation="only_first",  # if too long, cut the evidence, never the claim
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits.float().cpu().numpy()
                out.append(logits[:, self.index_map])  # reorder the columns
        return np.concatenate(out, axis=0)
