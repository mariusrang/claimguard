"""Fine-tune the verifier on SciFact.

    python scripts/train_verifier.py --config configs/train_verifier.yaml              # 1 seed
    python scripts/train_verifier.py --config configs/train_verifier.yaml --all-seeds  # 3 seeds
    python scripts/train_verifier.py --config configs/train_verifier.yaml --sweep      # grid search
    python scripts/train_verifier.py --config ... --base-model microsoft/deberta-v3-base --out-dir models/verifier-plain
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from claimguard.data import load_scifact
from claimguard.retrieval.bm25 import BM25Index
from claimguard.utils import load_config, save_json, set_seed
from claimguard.verify.train import build_training_pairs, train_verifier


def make_pairs(cfg: dict, seed: int):
    """Build the train and dev pairs (BM25 gives the hard negatives)."""
    data_dir = cfg["data"]["dir"]
    corpus, train_claims = load_scifact(data_dir, "train")
    _, dev_claims = load_scifact(data_dir, "dev")
    bm25 = BM25Index().build(list(corpus), [d.text for d in corpus.values()])
    t = cfg["train"]
    kwargs = dict(
        retriever=bm25,
        n_sentences=t.get("n_sentences", 3),
        n_neg=t.get("n_neg", 1),
        augment_with_selected=t.get("augment_with_selected", True),
    )
    train_pairs = build_training_pairs(train_claims, corpus, seed=seed, **kwargs)
    dev_pairs = build_training_pairs(dev_claims, corpus, seed=seed, **kwargs)
    return train_pairs, dev_pairs


def run_one(cfg: dict, seed: int, out_dir: str, **overrides) -> dict:
    t = {**cfg["train"], **overrides}
    set_seed(seed)
    train_pairs, dev_pairs = make_pairs(cfg, seed)
    print(f"seed {seed}: {len(train_pairs)} train pairs, {len(dev_pairs)} dev pairs -> {out_dir}")
    return train_verifier(
        train_pairs,
        dev_pairs,
        out_dir,
        base_model=t["base_model"],
        lr=float(t["lr"]),
        epochs=int(t["epochs"]),
        batch_size=int(t["batch_size"]),
        warmup_ratio=float(t.get("warmup_ratio", 0.1)),
        weight_decay=float(t.get("weight_decay", 0.01)),
        max_length=int(t.get("max_length", 384)),
        fp16=bool(t.get("fp16", False)),
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_verifier.yaml")
    parser.add_argument("--all-seeds", action="store_true", help="train one model per seed of the config")
    parser.add_argument("--sweep", action="store_true", help="grid search on lr / batch size / epochs")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.base_model:
        cfg["train"]["base_model"] = args.base_model
    out_dir = args.out_dir or cfg["train"]["out_dir"]
    seeds = cfg["train"].get("seeds", [cfg.get("seed", 13)])
    results_dir = Path(args.results_dir)

    if args.sweep:
        grid = cfg["sweep"]
        keys = sorted(grid)
        rows = []
        for values in itertools.product(*(grid[k] for k in keys)):
            params = dict(zip(keys, values))
            # every combination is trained with the same seed so they are comparable
            summary = run_one(cfg, seeds[0], "models/sweep_tmp", **params)
            rows.append({**params, "dev_macro_f1": summary["best_dev_macro_f1"]})
            save_json(results_dir / "sweep.json", rows)  # save after each run in case Colab dies
        best = max(rows, key=lambda r: r["dev_macro_f1"])
        print("best hyperparameters:", best)
        return

    if args.all_seeds:
        summaries = []
        for i, seed in enumerate(seeds):
            # the first seed goes to the main folder, the others next to it
            folder = out_dir if i == 0 else f"{out_dir}-seed{seed}"
            summaries.append({"seed": seed, "dir": folder, **run_one(cfg, seed, folder)})
        f1s = [s["best_dev_macro_f1"] for s in summaries]
        print(f"dev macro-F1 over {len(seeds)} seeds: {np.mean(f1s):.4f} +- {np.std(f1s, ddof=1):.4f} (sample std)")
        save_json(results_dir / "train_seeds.json", summaries)
        return

    summary = run_one(cfg, seeds[0], out_dir)
    save_json(results_dir / "train_single.json", summary)


if __name__ == "__main__":
    main()
