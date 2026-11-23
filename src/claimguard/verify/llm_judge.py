"""Baseline: an open-source LLM used as a zero-shot judge.

The idea is to compare my small fine-tuned verifier with "just ask a big LLM".
To keep it fair and cheap I do not let the LLM generate text: I ask for one
letter (A/B/C) and read the logits of these three tokens. So I get scores that
I can treat exactly like the logits of the other verifiers (same metrics, same
calibration code).
"""

from __future__ import annotations

import numpy as np

PROMPT = """You are a careful fact-checker. Read the evidence and the claim.

Evidence: {evidence}

Claim: {claim}

Does the evidence support the claim, refute it, or not give enough information?
A) The evidence supports the claim
B) The evidence refutes the claim
C) Not enough information

Answer with a single letter (A, B or C)."""


class LLMJudgeVerifier:
    """Zero-shot judge on top of a causal LM (ex: Qwen2.5-7B-Instruct in 4-bit)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        load_in_4bit: bool = False,
        max_length: int = 1024,
        device: str | None = None,
        model=None,
        tokenizer=None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            kwargs: dict = {}
            if load_in_4bit:
                # needs bitsandbytes + a GPU, this is how a 7B model fits on a T4
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
                kwargs["device_map"] = "auto"
            elif self.device == "cuda":
                kwargs["torch_dtype"] = torch.float16
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            if not load_in_4bit:
                model.to(self.device)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.name = model_name
        # id of the token for each answer letter (last sub-token, to be safe)
        self.letter_ids = [tokenizer.encode(x, add_special_tokens=False)[-1] for x in ("A", "B", "C")]

    def _format(self, claim: str, evidence: str) -> str:
        # cut very long evidence here, otherwise the tokenizer truncation would
        # remove the END of the prompt (the question itself)
        text = PROMPT.format(claim=claim, evidence=evidence[:6000])
        if getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": text}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return text + "\nAnswer:"

    def predict_logits(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """One forward pass per pair, logits of the tokens A / B / C."""
        torch = self.torch
        out = []
        with torch.no_grad():
            for claim, evidence in pairs:
                enc = self.tokenizer(
                    self._format(claim, evidence),
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                ).to(self.model.device)
                next_token_logits = self.model(**enc).logits[0, -1]
                out.append(next_token_logits[self.letter_ids].float().cpu().numpy())
        return np.stack(out) if out else np.zeros((0, 3))
