"""Multi-hop experiment on HoVer (Section 4.9 of the report).

    python scripts/run_multihop.py --hover-dir data/raw/hover --wiki data/raw/hover/wiki_wo_links.db \
        --verifier models/verifier --train-joint --out results

E1  gold articles given: read them one by one (ClaimGuard, noisy-OR / max) vs together
    (joint reader), by number of hops; also with 3 distractor articles added.
E2  retrieval in a closed pool: one BM25+rerank round vs two rounds (iterative),
    share of claims for which ALL the needed articles are found, by number of hops.
E2b end to end: iterative retrieval + the best reader, compared with ClaimGuard.
E3  (--train-joint) fine-tune the joint reader on HoVer train claims (never on dev).

HoVer has no public test labels, so the dev split is our test set: nothing is tuned on it.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np

from claimguard.metrics import macro_f1
from claimguard.multihop import (
    PoolRetriever,
    WikiTexts,
    all_gold_found,
    gold_recall,
    hover_training_pairs,
    is_supported,
    load_hover,
    norm_title,
    verify_independent,
    verify_joint,
)
from claimguard.stats import bootstrap_ci, mcnemar_test
from claimguard.utils import save_json, set_seed

ZERO_SHOT = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


def binary_scores(gold: np.ndarray, pred: np.ndarray) -> dict:
    """Accuracy (= balanced accuracy, the sample is balanced) and binary macro-F1, with CIs."""
    acc = bootstrap_ci(lambda g, p: float((g == p).mean()), gold, pred)
    f1 = bootstrap_ci(lambda g, p: macro_f1(g.astype(int), p.astype(int), n_classes=2), gold, pred)
    return {"accuracy": acc, "macro_f1": f1, "n": int(len(gold))}


def by_hops(claims, gold, pred) -> dict:
    hops = np.array([c.num_hops for c in claims])
    out = {"all": binary_scores(gold, pred)}
    for h in sorted(set(hops.tolist())):
        mask = hops == h
        out[f"{h}_hops"] = binary_scores(gold[mask], pred[mask])
    return out


def load_tfidf(path: Path, uids: set[str]) -> dict[str, list[str]]:
    """uid -> the 100 articles found by the TF-IDF retriever of the HoVer authors."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["id"]: [norm_title(t) for t in row["doc_retrieval_results"][0][0]] for row in data if row["id"] in uids}


def load_wiki(args, titles: set[str]) -> WikiTexts:
    if args.wiki.endswith(".db"):
        return WikiTexts.from_sqlite(args.wiki, titles)
    return WikiTexts.from_beir(args.wiki, titles)


def run_readers(verifier, claims, docs_of, tag: str) -> dict:
    """E1 for one verifier: the three ways of reading the same documents."""
    gold = np.array([c.supported for c in claims])
    preds = {"independent_noisy_or": [], "independent_max": [], "joint": []}
    start = time.perf_counter()
    for c in claims:
        docs = docs_of(c)
        preds["independent_noisy_or"].append(is_supported(verify_independent(verifier, c.claim, docs, "noisy_or")))
        preds["independent_max"].append(is_supported(verify_independent(verifier, c.claim, docs, "max_conf")))
        preds["joint"].append(is_supported(verify_joint(verifier, c.claim, docs)))
    out = {"seconds": time.perf_counter() - start}
    for name, p in preds.items():
        p = np.array(p)
        out[name] = by_hops(claims, gold, p)
        out[name]["correct"] = (p == gold).astype(int).tolist()
    for other in ("independent_noisy_or", "independent_max"):
        a = np.array(out["joint"]["correct"])
        b = np.array(out[other]["correct"])
        out[f"mcnemar_joint_vs_{other}"] = mcnemar_test(a, b)
    print(tag, {k: round(v["all"]["accuracy"]["value"], 3) for k, v in out.items() if isinstance(v, dict) and "all" in v})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hover-dir", default="data/raw/hover")
    ap.add_argument("--wiki", required=True, help="wiki_wo_links.db or BEIR hotpotqa corpus.jsonl(.gz)")
    ap.add_argument("--verifier", default="models/verifier", help="my SciFact-fine-tuned verifier")
    ap.add_argument("--zero-shot", default=ZERO_SHOT, help="NLI checkpoint used without fine-tuning")
    ap.add_argument("--reranker", choices=["cross", "lexical"], default="cross", help="lexical = for tests")
    ap.add_argument("--per-cell", type=int, default=100, help="dev claims per (hops, label): 100 -> 600 claims")
    ap.add_argument("--train-joint", action="store_true", help="E3: fine-tune the joint reader on HoVer train")
    ap.add_argument("--train-per-cell", type=int, default=700)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--fallback-distractors", type=int, default=50000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--model-out", default="models/hover_joint_reader", help="where E3 saves its reader")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    set_seed(args.seed)
    hover_dir, out_dir = Path(args.hover_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = load_hover(hover_dir / "hover_dev_release_v1.1.json", args.per_cell, args.seed)
    train = load_hover(hover_dir / "hover_train_release_v1.1.json", args.train_per_cell, args.seed) if args.train_joint else []
    tfidf_file = hover_dir / "dev_tfidf_doc_retrieval_results.json"
    gold_needed = {t for c in dev + train for t in c.gold_titles}
    if tfidf_file.exists():
        # pool = the 100 TF-IDF candidates of the HoVer authors for each claim + the gold articles
        tfidf = load_tfidf(tfidf_file, {c.uid for c in dev})
        wiki = load_wiki(args, gold_needed | {t for ts in tfidf.values() for t in ts})
        pool_kind = "tfidf_top100_plus_gold"
    else:
        # fallback if the UNC server is down: gold articles + the first N articles of the dump
        # as distractors, and our own BM25 top-100 plays the role of the TF-IDF candidates
        wiki = WikiTexts.from_beir(args.wiki, gold_needed, extra=args.fallback_distractors)
        index = PoolRetriever(wiki, reranker=None).index
        tfidf = {c.uid: [t for t, _ in index.search(c.claim, 100)] for c in dev}
        pool_kind = f"gold_plus_{args.fallback_distractors}_first_articles"
    coverage = sum(all(t in wiki for t in c.gold_titles) for c in dev) / len(dev)
    print(f"{len(dev)} dev claims, {len(train)} train claims, {len(wiki)} articles loaded, gold coverage {coverage:.3f}")
    dev = [c for c in dev if all(t in wiki for t in c.gold_titles)]  # drop the few claims with a missing article

    import torch

    from claimguard.retrieval.rerank import CrossEncoderReranker
    from claimguard.verify.nli import NLIVerifier

    results: dict = {
        "n_dev": len(dev),
        "gold_coverage": coverage,
        "pool_kind": pool_kind,
        "per_cell": args.per_cell,
        "hops": {str(h): sum(c.num_hops == h for c in dev) for h in (2, 3, 4)},
    }
    verifiers = {"zero_shot": args.zero_shot, "scifact_finetuned": args.verifier}

    # E3 first, so that the trained reader joins the comparison
    if args.train_joint:
        from claimguard.verify.train import train_verifier

        pairs = hover_training_pairs(train, wiki)
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(pairs))
        n_val = max(200, len(pairs) // 10)  # validation taken from TRAIN claims, not from dev
        val, tr = [pairs[i] for i in idx[:n_val]], [pairs[i] for i in idx[n_val:]]
        t0 = time.perf_counter()
        history = train_verifier(
            tr, val, args.model_out, base_model=args.verifier, lr=2e-5,
            epochs=args.epochs, batch_size=8, max_length=512, fp16=torch.cuda.is_available(), seed=args.seed,
        )
        results["e3_training"] = {"n_train": len(tr), "n_val": len(val), "minutes": (time.perf_counter() - t0) / 60,
                                  "history": history}
        verifiers["hover_joint_trained"] = args.model_out

    # E1: gold articles, then gold + 3 distractors
    gold_docs = lambda c: [(t, wiki.get(t)) for t in c.gold_titles]  # noqa: E731

    def with_distractors(c):
        extra = [t for t in tfidf.get(c.uid, []) if t not in c.gold_titles and t in wiki][:3]
        docs = gold_docs(c) + [(t, wiki.get(t)) for t in extra]
        np.random.default_rng(zlib.crc32(c.uid.encode())).shuffle(docs)  # gold must not always come first
        return docs

    results["e1_gold"], results["e1_distractors"] = {}, {}
    for name, path in verifiers.items():
        verifier = NLIVerifier(path, max_length=512, batch_size=16)
        results["e1_gold"][name] = run_readers(verifier, dev, gold_docs, f"E1 gold {name}")
        results["e1_distractors"][name] = run_readers(verifier, dev, with_distractors, f"E1 distractors {name}")
        del verifier
        torch.cuda.empty_cache()
    save_json(out_dir / "multihop.json", results)  # save early, retrieval is the slow part

    # E2: retrieval in the closed pool
    if args.reranker == "cross":
        reranker = CrossEncoderReranker()
    else:
        from claimguard.retrieval.rerank import LexicalReranker

        reranker = LexicalReranker()
    retriever = PoolRetriever(wiki, reranker)
    results["pool_size"] = len(wiki)
    hops = np.array([c.num_hops for c in dev])
    retrieved = {"single_shot": [], "iterative": []}
    t0 = time.perf_counter()
    for c in dev:
        retrieved["single_shot"].append(retriever.single_shot(c.claim, 10))
        retrieved["iterative"].append(retriever.iterative(c.claim, 10))
    e2 = {"seconds": time.perf_counter() - t0}
    for mode, lists in retrieved.items():
        e2[mode] = {}
        for k in (5, 10):
            found = np.array([all_gold_found(r[:k], c.gold_titles) for r, c in zip(lists, dev)], dtype=float)
            rec = np.array([gold_recall(r[:k], c.gold_titles) for r, c in zip(lists, dev)])
            e2[mode][f"all_found@{k}"] = {"all": bootstrap_ci(np.mean, found)} | {
                f"{h}_hops": bootstrap_ci(np.mean, found[hops == h]) for h in sorted(set(hops.tolist()))
            }
            e2[mode][f"doc_recall@{k}"] = float(rec.mean())
    a = np.array([all_gold_found(r[:10], c.gold_titles) for r, c in zip(retrieved["iterative"], dev)], dtype=int)
    b = np.array([all_gold_found(r[:10], c.gold_titles) for r, c in zip(retrieved["single_shot"], dev)], dtype=int)
    e2["mcnemar_iterative_vs_single@10"] = mcnemar_test(a, b)
    results["e2_retrieval"] = e2

    # E2b: end to end with the best reader on dev? No: the reader is fixed in advance
    # (the HoVer-trained one if E3 ran, else my SciFact verifier), to keep dev untouched.
    best = "hover_joint_trained" if "hover_joint_trained" in verifiers else "scifact_finetuned"
    verifier = NLIVerifier(verifiers[best], max_length=512, batch_size=16)
    gold = np.array([c.supported for c in dev])
    e2b = {"reader": best}
    for mode in ("single_shot", "iterative"):
        for reading in ("independent_noisy_or", "joint"):
            preds = []
            for r, c in zip(retrieved[mode], dev):
                docs = [(t, wiki.get(t)) for t in r[:5]]
                probs = (verify_joint(verifier, c.claim, docs) if reading == "joint"
                         else verify_independent(verifier, c.claim, docs, "noisy_or"))
                preds.append(is_supported(probs))
            e2b[f"{mode}+{reading}"] = by_hops(dev, gold, np.array(preds))
    results["e2b_end_to_end"] = e2b
    save_json(out_dir / "multihop.json", results)
    plot(results, out_dir / "figures" / "multihop.png")
    print("saved", out_dir / "multihop.json")


def plot(results: dict, path: Path) -> None:
    """Left: accuracy by number of hops, read one by one vs together. Right: retrieval."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6))
    some = next(iter(results["e1_gold"].values()))["joint"]
    hops = [h for h in ("2_hops", "3_hops", "4_hops") if h in some]  # the ones present in the sample
    xs = [int(h[0]) for h in hops]
    names = {"zero_shot": "zero-shot NLI", "scifact_finetuned": "SciFact-tuned", "hover_joint_trained": "HoVer-tuned"}
    for key, label in names.items():
        if key not in results["e1_gold"]:
            continue
        for reading, style in (("independent_noisy_or", "--"), ("joint", "-")):
            ys = [results["e1_gold"][key][reading][h]["accuracy"]["value"] for h in hops]
            ax1.plot(xs, ys, style, marker="o", label=f"{label}, {'one by one' if style == '--' else 'together'}")
    ax1.set(xlabel="number of hops", ylabel="accuracy (gold articles)", xticks=xs, ylim=(0.4, 1.0))
    ax1.axhline(0.5, color="grey", lw=0.8)
    ax1.legend(fontsize=7)
    for mode, style in (("single_shot", "--"), ("iterative", "-")):
        block = results["e2_retrieval"][mode]["all_found@10"]
        ax2.plot(xs, [block[h]["value"] for h in hops], style, marker="o", label=mode.replace("_", " "))
    ax2.set(xlabel="number of hops", ylabel="all needed articles in top 10", xticks=xs, ylim=(0, 1))
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


if __name__ == "__main__":
    main()
