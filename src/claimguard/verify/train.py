"""Fine-tuning of the verifier with my own training loop (no Trainer).

Steps:
1. build (claim, evidence, label) pairs from SciFact
2. train with AdamW + linear warmup/decay, evaluate on dev after each epoch
3. keep the best epoch (macro-F1 on dev) and save it
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from claimguard import LABELS, NEI
from claimguard.data import Claim, Document
from claimguard.metrics import accuracy, macro_f1
from claimguard.retrieval.rerank import LexicalReranker, rerank
from claimguard.utils import ensure_float32, save_json, set_seed
from claimguard.verify.nli import NLIVerifier, map_model_labels

Pair = tuple[str, str, int]  # (claim, evidence text, label index in our order)


def select_sentences(reranker, claim: str, doc: Document, n: int) -> list[int]:
    """Indexes of the n sentences of `doc` most relevant to the claim (in reading order)."""
    top = rerank(reranker, claim, doc.sentences, n)
    return sorted(i for i, _ in top)


def build_training_pairs(
    claims: list[Claim],
    corpus: dict[str, Document],
    retriever=None,
    n_sentences: int = 3,
    n_neg: int = 1,
    augment_with_selected: bool = True,
    seed: int = 13,
) -> list[Pair]:
    """Turn SciFact claims into (claim, evidence, label) training pairs.

    - gold evidence doc  -> its gold rationale sentences, label = claim label
    - NEI claim          -> top sentences of its cited docs, label = NEI
    - hard negatives     -> a retrieved doc that is NOT gold evidence, label = NEI
      (this teaches the model that "same topic" does not mean "supports")
    """
    rng = random.Random(seed)
    selector = LexicalReranker()  # cheap sentence picker, good enough for training data
    label_to_idx = {name: i for i, name in enumerate(LABELS)}
    pairs: list[Pair] = []
    for claim in claims:
        y = label_to_idx[claim.label]
        for doc_id, sent_ids in claim.evidence.items():
            if doc_id not in corpus:
                continue
            doc = corpus[doc_id]
            gold = [doc.sentences[i] for i in sent_ids if i < len(doc.sentences)]
            if gold:
                pairs.append((claim.text, " ".join(gold), y))
            if augment_with_selected:
                # also show the model what evidence looks like at test time
                # (sentences picked automatically), but only if a gold sentence is in it
                picked = select_sentences(selector, claim.text, doc, n_sentences)
                if set(picked) & set(sent_ids) and picked != sent_ids:
                    pairs.append((claim.text, " ".join(doc.sentences[i] for i in picked), y))
        if claim.label == "NEI":
            for doc_id in claim.cited_doc_ids:
                if doc_id in corpus:
                    doc = corpus[doc_id]
                    picked = select_sentences(selector, claim.text, doc, n_sentences)
                    pairs.append((claim.text, " ".join(doc.sentences[i] for i in picked), NEI))
        elif retriever is not None and n_neg > 0:
            forbidden = set(claim.evidence) | set(claim.cited_doc_ids)
            candidates = [d for d, _ in retriever.search(claim.text, 10) if d not in forbidden]
            for doc_id in rng.sample(candidates, min(n_neg, len(candidates))):
                doc = corpus[doc_id]
                picked = select_sentences(selector, claim.text, doc, n_sentences)
                pairs.append((claim.text, " ".join(doc.sentences[i] for i in picked), NEI))
    rng.shuffle(pairs)
    return pairs


def load_model_for_training(base_model: str):
    """Load tokenizer + model. Keep the NLI head if the checkpoint has one."""
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    config = AutoConfig.from_pretrained(base_model)
    try:
        map_model_labels(config.id2label)  # works only for NLI checkpoints
        model = AutoModelForSequenceClassification.from_pretrained(base_model)
    except ValueError:
        # plain encoder (ex: microsoft/deberta-v3-base): create a fresh 3-class head
        id2label = dict(enumerate(LABELS))
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=3,
            id2label=id2label,
            label2id={v: k for k, v in id2label.items()},
            ignore_mismatched_sizes=True,
        )
    return tokenizer, ensure_float32(model)


def evaluate_pairs(verifier, pairs: list[Pair]) -> dict:
    """Accuracy and macro-F1 of a verifier on labelled pairs."""
    logits = verifier.predict_logits([(c, e) for c, e, _ in pairs])
    preds = logits.argmax(axis=1)
    gold = np.array([y for _, _, y in pairs])
    return {"accuracy": accuracy(gold, preds), "macro_f1": macro_f1(gold, preds), "n": len(pairs)}


def train_verifier(
    train_pairs: list[Pair],
    dev_pairs: list[Pair],
    out_dir: str | Path,
    base_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    lr: float = 2e-5,
    epochs: int = 5,
    batch_size: int = 16,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_length: int = 384,
    fp16: bool = False,
    seed: int = 13,
    device: str | None = None,
    model=None,
    tokenizer=None,
) -> dict:
    """Fine-tune and save the best checkpoint. Returns the training history."""
    import torch

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        tokenizer, model = load_model_for_training(base_model)
    model = ensure_float32(model).to(device)  # never train in float16 (NaN losses)
    index_map = map_model_labels(model.config.id2label)  # our label idx -> model idx

    # no weight decay on biases and LayerNorm weights (usual trick for transformers)
    no_decay = ("bias", "LayerNorm.weight", "layer_norm")
    params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(params, lr=lr)
    steps_per_epoch = max(1, (len(train_pairs) + batch_size - 1) // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(warmup_ratio * total_steps)

    def lr_lambda(step: int) -> float:
        # goes 0 -> 1 during warmup, then linearly back to 0
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = fp16 and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    history: list[dict] = []
    best_f1 = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_pairs)))
        rng.shuffle(order)
        running_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch = [train_pairs[i] for i in order[start : start + batch_size]]
            enc = tokenizer(
                [e for _, e, _ in batch],
                [c for c, _, _ in batch],
                padding=True,
                truncation="only_first",
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            targets = torch.tensor([index_map[y] for _, _, y in batch], device=device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(**enc).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), targets)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # avoid exploding gradients
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if not torch.isfinite(loss):
                # stop at once instead of wasting GPU time on a broken run
                raise RuntimeError(
                    f"loss is {loss.item()} at epoch {epoch}, batch {start // batch_size}: "
                    f"model dtype is {next(model.parameters()).dtype}, lr={lr}. Training stopped."
                )
            running_loss += loss.item() * len(batch)

        # end of epoch: check how we do on dev
        verifier = NLIVerifier(model=model, tokenizer=tokenizer, max_length=max_length, device=device)
        dev_metrics = evaluate_pairs(verifier, dev_pairs)
        row = {"epoch": epoch, "train_loss": running_loss / max(1, len(train_pairs)), **dev_metrics}
        history.append(row)
        print(f"epoch {epoch}: loss={row['train_loss']:.4f} dev_f1={row['macro_f1']:.4f}")
        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)

    summary = {
        "base_model": base_model,
        "seed": seed,
        "lr": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "n_train_pairs": len(train_pairs),
        "best_dev_macro_f1": best_f1,
        "history": history,
    }
    save_json(out_dir / "train_log.json", summary)
    return summary
