"""All the experiments of the report. Each sub-command writes a JSON file in results/.

    python scripts/evaluate.py retrieval   --config configs/default.yaml   # A1 + A2 (retrieval side)
    python scripts/evaluate.py pipeline    --config configs/default.yaml   # main table + A2, A3, A4, A6
    python scripts/evaluate.py calibration --config configs/default.yaml   # temperature, ECE, abstention
    python scripts/evaluate.py given --dataset french|climate|contrast     # verifier-only test sets
    python scripts/evaluate.py baselines   --config configs/default.yaml   # majority + claim-only
    python scripts/evaluate.py efficiency  --config configs/default.yaml   # size / latency / INT8
    python scripts/evaluate.py errors      --config configs/default.yaml   # CSV to annotate by hand
    python scripts/evaluate.py aggrefact   --config configs/default.yaml   # LLM answers (extension)

Useful options: --verifier-model X --verifier-kind nli|lexical|llm --tag NAME --limit N
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from claimguard import LABELS, NEI
from claimguard.aggregate import STRATEGIES, aggregate
from claimguard.calibrate import (
    alert_metrics,
    expected_calibration_error,
    reliability_bins,
    risk_coverage_curve,
    selective_metrics,
)
from claimguard.data import (
    Claim,
    EvidenceClaim,
    load_climate_fever,
    load_evidence_claims,
    load_scifact,
)
from claimguard.metrics import (
    accuracy,
    confusion_matrix,
    evidence_prf,
    fever_score,
    macro_f1,
    per_class_prf,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from claimguard.pipeline import ClaimGuard, build_pipeline, build_reranker, build_retriever
from claimguard.retrieval.rerank import rerank
from claimguard.stats import bootstrap_ci, mcnemar_test, paired_permutation_test
from claimguard.text import extract_numbers, has_negation, tokenize
from claimguard.utils import load_config, load_json, read_jsonl, save_json, set_seed, softmax, write_jsonl

LABEL_TO_IDX = {name: i for i, name in enumerate(LABELS)}


# --------------------------------------------------------------------------- helpers


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Command line options that change the verifier of the config."""
    if getattr(args, "verifier_kind", None):
        cfg["verifier"]["kind"] = args.verifier_kind
    if getattr(args, "verifier_model", None):
        cfg["verifier"]["model"] = args.verifier_model
        cfg["verifier"].pop("fallback_model", None)
    if getattr(args, "retrieval_kind", None):
        cfg["retrieval"]["kind"] = args.retrieval_kind
    if getattr(args, "rerank_kind", None):
        cfg["rerank"]["kind"] = args.rerank_kind
    if getattr(args, "load_in_4bit", False):
        cfg["verifier"]["load_in_4bit"] = True
    if getattr(args, "verifier_kind", None) or getattr(args, "verifier_model", None):
        # the saved temperature was fitted for the main verifier, it means nothing for another model
        cfg["calibration"] = {"temperature": 1.0, "threshold": 0.0}
    cfg["audit"] = {}  # no audit log during experiments, it would just fill the disk
    return cfg


def out_path(args: argparse.Namespace, name: str) -> Path:
    tag = f"_{args.tag}" if args.tag else ""
    return Path(args.out) / f"{name}{tag}.json"


def collect(pipeline: ClaimGuard, claims: list[Claim]) -> list[dict]:
    """Run the pipeline on every claim and keep everything needed for the metrics."""
    records = []
    for n, claim in enumerate(claims, start=1):
        result = pipeline.verify_claim(claim.text)
        records.append(
            {
                "claim_id": claim.claim_id,
                "claim": claim.text,
                "gold": claim.label,
                "gold_evidence": claim.evidence,
                "retrieved": result.retrieved,
                "evidence": [
                    {"source": e.source, "sentence_ids": e.sentence_ids, "text": e.text, "logits": e.logits}
                    for e in result.evidence
                ],
                "timings_ms": result.timings_ms,
            }
        )
        if n % 50 == 0:
            print(f"  {n}/{len(claims)} claims done")
    return records


def with_oracle_evidence(pipeline: ClaimGuard, claims: list[Claim], records: list[dict]) -> list[dict]:
    """Same records, but the retrieved evidence is replaced by the GOLD evidence sentences.

    Only for claims that have gold evidence (NEI claims keep their retrieved documents,
    there is nothing better to give them). Comparing both versions tells how many errors
    come from retrieval and how many from the verifier itself.
    """
    out = []
    for claim, record in zip(claims, records):
        gold_docs = [(d, ids) for d, ids in claim.evidence.items() if d in pipeline.corpus]
        if not gold_docs:
            out.append(record)
            continue
        texts = [" ".join(pipeline.corpus[d].sentences[i] for i in ids) for d, ids in gold_docs]
        logits = pipeline.verifier.predict_logits([(claim.text, t) for t in texts])
        evidence = [
            {"source": d, "sentence_ids": ids, "text": t, "logits": logit.tolist()}
            for (d, ids), t, logit in zip(gold_docs, texts, logits)
        ]
        out.append({**record, "evidence": evidence})
    return out


def claim_probs(record: dict, k: int, strategy: str, temperature: float) -> np.ndarray:
    """Claim-level probabilities from the first k evidence documents of a record."""
    logits = np.array([e["logits"] for e in record["evidence"][:k]], dtype=float).reshape(-1, 3)
    probs = softmax(logits / temperature) if len(logits) else logits
    return aggregate(probs, strategy)


def score(records: list[dict], k: int, strategy: str, temperature: float = 1.0) -> dict:
    """Predictions + arrays used by every table."""
    probs = np.array([claim_probs(r, k, strategy, temperature) for r in records])
    pred = probs.argmax(axis=1)
    gold = np.array([LABEL_TO_IDX[r["gold"]] for r in records])
    found = np.array(
        [bool(set(e["source"] for e in r["evidence"][:k]) & set(r["gold_evidence"])) for r in records]
    )
    return {"probs": probs, "pred": pred, "gold": gold, "conf": probs.max(axis=1), "evidence_found": found}


def main_metrics(records: list[dict], k: int, strategy: str, temperature: float = 1.0) -> dict:
    s = score(records, k, strategy, temperature)
    pred, gold = s["pred"], s["gold"]
    # evidence sentences we "claim": sentences of the docs that the verifier found decisive
    pred_ev, gold_ev = [], []
    for r, p in zip(records, pred):
        chosen = set()
        if p != NEI:
            for e in r["evidence"][:k]:
                if int(np.argmax(e["logits"])) != NEI:
                    chosen |= {(e["source"], i) for i in e["sentence_ids"]}
        pred_ev.append(chosen)
        gold_ev.append({(d, i) for d, ids in r["gold_evidence"].items() for i in ids})
    return {
        "k": k,
        "aggregation": strategy,
        "n": len(records),
        "accuracy": bootstrap_ci(accuracy, gold, pred),
        "macro_f1": bootstrap_ci(macro_f1, gold, pred),
        "fever_score": fever_score(pred == gold, s["evidence_found"], gold == NEI),
        "evidence_sentences": evidence_prf(pred_ev, gold_ev),
        "per_class": dict(zip(LABELS, per_class_prf(gold, pred))),
        "confusion_matrix": confusion_matrix(gold, pred).tolist(),
    }


# --------------------------------------------------------------------------- choices made on dev

K_GRID = (1, 3, 5, 10)


def claim_nll(records: list[dict], k: int, strategy: str, temperature: float) -> float:
    """Negative log-likelihood of the gold labels, on the CLAIM-level probabilities."""
    probs = np.array([claim_probs(r, k, strategy, temperature) for r in records])
    gold = np.array([LABEL_TO_IDX[r["gold"]] for r in records])
    return float(-np.log(np.clip(probs[np.arange(len(gold)), gold], 1e-12, 1.0)).mean())


def fit_claim_temperature(records: list[dict], k: int, strategy: str) -> float:
    """Temperature that makes the confidence the USER sees (claim level) as honest as possible.

    T divides the passage logits, then the passages are aggregated. I first fitted T on
    the passage pairs, but on SciFact that made the claim-level ECE worse, because the
    aggregation changes the probabilities. So T is fitted on what is shown in the end.
    """
    from scipy.optimize import minimize_scalar

    if not records:
        return 1.0
    res = minimize_scalar(
        lambda log_t: claim_nll(records, k, strategy, float(np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )
    return float(np.exp(res.x))


def tune_on_dev(dev_records: list[dict], temperature: float | None = None) -> dict:
    """Choose k, the aggregation (and the temperature if None) on DEV, by macro-F1.

    The test set is only used to report results, never to pick a setting. Every
    system (baselines included) goes through this, so the comparison is fair.
    """
    grid = []
    for k in K_GRID:
        for strategy in STRATEGIES:
            t = fit_claim_temperature(dev_records, k, strategy) if temperature is None else temperature
            s = score(dev_records, k, strategy, t)
            grid.append({"k": k, "aggregation": strategy, "temperature": t,
                         "dev_macro_f1": macro_f1(s["gold"], s["pred"])})
    # if two settings tie, keep the one that reads fewer passages (it is faster)
    best = max(grid, key=lambda r: (round(r["dev_macro_f1"], 6), -r["k"]))
    return {"top_k": best["k"], "aggregation": best["aggregation"], "temperature": best["temperature"],
            "dev_macro_f1": best["dev_macro_f1"], "grid": grid}


def load_calibration_file(cfg: dict) -> dict:
    """Values written by the `calibration` command (empty dict if none)."""
    calib_file = cfg.get("calibration", {}).get("file")
    return load_json(calib_file) if calib_file and Path(calib_file).exists() else {}


# --------------------------------------------------------------------------- retrieval (A1, A2)


def cmd_retrieval(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    corpus, claims = load_scifact(cfg["data"]["dir"], args.split)
    claims = [c for c in claims if c.evidence][: args.limit]  # NEI claims have nothing to retrieve
    kinds = args.kinds.split(",") if args.kinds else (
        ["bm25"] if cfg["retrieval"]["kind"] == "bm25" else ["bm25", "dense", "hybrid"]
    )
    reranker = build_reranker(cfg["rerank"])
    depth = cfg["retrieval"].get("depth", 50)
    ks = [1, 3, 5, 10, 20]
    results, per_claim = {}, {}
    for kind in kinds:
        retriever = build_retriever({**cfg["retrieval"], "kind": kind}, corpus)
        for use_rerank in (False, True):
            if use_rerank and reranker is None:
                continue
            name = f"{kind}+rerank" if use_rerank else kind
            rows = defaultdict(list)
            for claim in claims:
                ranked = [d for d, _ in retriever.search(claim.text, depth)]
                if use_rerank:
                    order = rerank(reranker, claim.text, [corpus[d].text for d in ranked], len(ranked))
                    ranked = [ranked[i] for i, _ in order]
                gold = set(claim.evidence)
                for k in ks:
                    rows[f"recall@{k}"].append(recall_at_k(ranked, gold, k))
                    rows[f"precision@{k}"].append(precision_at_k(ranked, gold, k))
                rows["mrr"].append(reciprocal_rank(ranked, gold))
            results[name] = {m: float(np.mean(v)) for m, v in rows.items()}
            results[name]["recall@5_ci"] = bootstrap_ci(lambda x: float(np.mean(x)), np.array(rows["recall@5"]))
            per_claim[name] = np.array(rows["recall@5"])
            print(f"{name:16s} R@5={results[name]['recall@5']:.3f} R@20={results[name]['recall@20']:.3f} MRR={results[name]['mrr']:.3f}")
    # is each system really different from plain BM25? (paired test on recall@5)
    tests = {
        name: paired_permutation_test(scores, per_claim["bm25"])
        for name, scores in per_claim.items()
        if name != "bm25" and "bm25" in per_claim
    }
    save_json(out_path(args, "retrieval"), {"n_claims": len(claims), "systems": results, "vs_bm25_recall@5": tests})


# --------------------------------------------------------------------------- pipeline (main, A2, A3, A4, A6)


def cmd_pipeline(args: argparse.Namespace) -> None:
    cfg = apply_overrides(load_config(args.config), args)
    set_seed(cfg.get("seed", 13))
    pipeline = build_pipeline(cfg)
    _, claims = load_scifact(cfg["data"]["dir"], args.split)
    claims = claims[: args.limit]
    temp = pipeline.temperature
    pipeline.top_k = max(K_GRID)  # collect 10 docs once, then slice for the k ablation

    # k and aggregation: from the calibration file if the calibration step already
    # chose them on dev, otherwise choose them on dev now (baselines, other seeds...)
    calib = load_calibration_file(cfg)
    if "top_k" in calib and "aggregation" in calib:
        dev_choice = {"top_k": calib["top_k"], "aggregation": calib["aggregation"], "source": "calibration file"}
    else:
        print("choosing k and aggregation on dev...")
        _, dev_claims = load_scifact(cfg["data"]["dir"], "dev")
        dev_records = collect(pipeline, dev_claims[: args.limit])
        dev_choice = {**tune_on_dev(dev_records, temperature=temp), "source": "tuned in this run"}
    main_k, main_strategy = dev_choice["top_k"], dev_choice["aggregation"]
    print(f"setting chosen on dev: k={main_k}, aggregation={main_strategy}")

    print("full pipeline...")
    records = collect(pipeline, claims)
    print("oracle evidence...")
    oracle_records = with_oracle_evidence(pipeline, claims, records)
    print("no reranker...")
    no_rerank = ClaimGuard(
        pipeline.verifier, None, pipeline.retriever, pipeline.corpus,
        top_k=pipeline.top_k, n_sentences=pipeline.n_sentences, depth=pipeline.depth,
    )
    no_rerank_records = collect(no_rerank, claims)

    out = {
        "verifier": getattr(pipeline.verifier, "name", "?"),
        "temperature": temp,
        "chosen_on_dev": dev_choice,
        "main": main_metrics(records, main_k, main_strategy, temp),
        "A2_no_reranker": main_metrics(no_rerank_records, main_k, main_strategy, temp),
        "A3_number_of_evidence": [main_metrics(records, k, main_strategy, temp) for k in (1, 3, 5, 10)],
        "A4_aggregation": [main_metrics(records, main_k, s, temp) for s in STRATEGIES],
        "A6_oracle_evidence": main_metrics(oracle_records, main_k, main_strategy, temp),
    }
    # significance: main system vs each variant, on the same test claims
    ref = score(records, main_k, main_strategy, temp)
    ref_correct = ref["pred"] == ref["gold"]
    tests = {}
    for s in STRATEGIES:
        if s != main_strategy:
            other = score(records, main_k, s, temp)
            tests[f"{main_strategy}_vs_{s}"] = mcnemar_test(ref_correct, other["pred"] == other["gold"])
    nr = score(no_rerank_records, main_k, main_strategy, temp)
    tests["rerank_vs_no_rerank"] = mcnemar_test(ref_correct, nr["pred"] == nr["gold"])
    orc = score(oracle_records, main_k, main_strategy, temp)
    tests["oracle_vs_retrieved"] = mcnemar_test(orc["pred"] == orc["gold"], ref_correct)
    out["mcnemar"] = tests
    # share of the errors that disappear with perfect evidence = errors caused by retrieval
    n_err = int((~ref_correct).sum())
    fixed = int((~ref_correct & (orc["pred"] == orc["gold"])).sum())
    out["errors_fixed_by_oracle"] = {"n_errors": n_err, "fixed": fixed, "share": fixed / n_err if n_err else 0.0}
    lat = np.array([sum(r["timings_ms"].values()) for r in records])
    out["latency_ms_per_claim"] = {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}

    save_json(out_path(args, "pipeline"), out)
    tag = f"_{args.tag}" if args.tag else ""
    write_jsonl(Path(args.out) / f"predictions{tag}.jsonl", records)
    m = out["main"]
    print(f"main: acc={m['accuracy']['value']:.3f} macroF1={m['macro_f1']['value']:.3f} fever={m['fever_score']:.3f}")
    print(f"oracle: macroF1={out['A6_oracle_evidence']['macro_f1']['value']:.3f}")


# --------------------------------------------------------------------------- calibration


def cmd_calibration(args: argparse.Namespace) -> None:
    """Every choice made on DEV for the main system: temperature, k, aggregation, threshold.

    Then the calibration is reported on TEST, before and after temperature scaling.
    """
    cfg = apply_overrides(load_config(args.config), args)
    cfg["calibration"] = {"temperature": 1.0, "threshold": 0.0}  # start from the raw model
    pipeline = build_pipeline(cfg)
    pipeline.top_k = max(K_GRID)  # collect 10 passages, the k choice is done afterwards
    _, dev_claims = load_scifact(cfg["data"]["dir"], "dev")
    _, test_claims = load_scifact(cfg["data"]["dir"], args.split)
    dev_records = collect(pipeline, dev_claims[: args.limit])
    test_records = collect(pipeline, test_claims[: args.limit])

    # 1) choose k and the aggregation on dev, each with its own claim-level temperature
    choice = tune_on_dev(dev_records)
    k, strategy, temperature = choice["top_k"], choice["aggregation"], choice["temperature"]
    print(f"dev choice: k={k}, aggregation={strategy}, T={temperature:.3f} (dev macro-F1 {choice['dev_macro_f1']:.3f})")

    # 3) pick the abstention threshold on dev: the lowest one that reaches the target alert precision
    dev = score(dev_records, k, strategy, temperature)
    threshold = 0.0
    for t in np.arange(0.40, 0.96, 0.05):
        m = alert_metrics(dev["pred"], dev["gold"], dev["conf"], float(t))
        if m["n_alerts"] > 0 and m["alert_precision"] >= args.target_precision:
            threshold = round(float(t), 2)
            break

    # 4) report everything on test, before and after temperature scaling
    out = {
        "temperature": temperature,
        "threshold": threshold,
        "target_alert_precision": args.target_precision,
        "chosen_on_dev": choice,
    }
    for name, temp in (("before", 1.0), ("after", temperature)):
        s = score(test_records, k, strategy, temp)
        correct = s["pred"] == s["gold"]
        curve = risk_coverage_curve(s["conf"], correct)
        out[name] = {
            "ece": expected_calibration_error(s["conf"], correct),
            "aurc": curve["aurc"],
            "reliability_bins": reliability_bins(s["conf"], correct),
            "risk_coverage": {"coverage": curve["coverage"], "risk": curve["risk"]},
            "selective": [selective_metrics(s["conf"], correct, t) for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)],
            "alerts": [alert_metrics(s["pred"], s["gold"], s["conf"], t) for t in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)],
        }
        print(f"{name}: ECE={out[name]['ece']:.4f} AURC={out[name]['aurc']:.4f}")
    save_json(out_path(args, "calibration"), out)
    calib_file = load_config(args.config).get("calibration", {}).get("file")
    if calib_file and not args.tag:
        save_json(calib_file, {"temperature": temperature, "threshold": threshold, "top_k": k, "aggregation": strategy})
        print(f"saved T={temperature:.3f}, threshold={threshold}, k={k}, aggregation={strategy} to {calib_file}")


# --------------------------------------------------------------------------- given-evidence test sets


def load_given(args: argparse.Namespace, cfg: dict) -> tuple[list[EvidenceClaim], str]:
    if args.dataset == "french":
        return load_evidence_claims(args.path or "data/french_bank/fr_bank_claims.jsonl"), "category"
    if args.dataset == "climate":
        return load_climate_fever(args.path or "data/raw/climate_fever/climate-fever.jsonl"), ""
    if args.dataset == "contrast":
        return load_evidence_claims(args.path or str(Path(cfg["data"]["dir"]) / "contrast.jsonl")), "perturbation"
    return load_evidence_claims(args.path), ""


def cmd_given(args: argparse.Namespace) -> None:
    cfg = apply_overrides(load_config(args.config), args)
    pipeline = build_pipeline(cfg, with_corpus=False)
    examples, group_key = load_given(args, cfg)
    examples = examples[: args.limit]
    results = [pipeline.verify_with_evidence(ex.text, ex.evidence_texts) for ex in examples]
    pred = np.array([LABEL_TO_IDX[r.label] for r in results])
    gold = np.array([LABEL_TO_IDX[ex.label] for ex in examples])
    conf = np.array([r.confidence for r in results])
    # one line per example, so errors can be analysed later without running the model again
    tag = f"_{args.tag}" if args.tag else ""
    write_jsonl(
        Path(args.out) / f"predictions_given_{args.dataset}{tag}.jsonl",
        [
            {
                "claim_id": ex.claim_id,
                "claim": ex.text,
                "gold": ex.label,
                "pred": r.label,
                "confidence": round(r.confidence, 4),
                "top_evidence": max(r.evidence, key=lambda e: max(e.probs[:2])).text if r.evidence else "",
                **{k: v for k, v in ex.meta.items() if isinstance(v, (str, int, float))},
            }
            for ex, r in zip(examples, results)
        ],
    )
    out = {
        "dataset": args.dataset,
        "verifier": getattr(pipeline.verifier, "name", "?"),
        "n": len(examples),
        "accuracy": bootstrap_ci(accuracy, gold, pred),
        "macro_f1": bootstrap_ci(macro_f1, gold, pred),
        "ece": expected_calibration_error(conf, pred == gold),
        "per_class": dict(zip(LABELS, per_class_prf(gold, pred))),
        "confusion_matrix": confusion_matrix(gold, pred).tolist(),
        "alerts": alert_metrics(pred, gold, conf, pipeline.threshold),
    }
    if group_key:
        groups = defaultdict(list)
        for ex, p, g in zip(examples, pred, gold):
            groups[ex.meta.get(group_key, "?")].append((p, g))
        out[f"by_{group_key}"] = {
            name: {
                "n": len(v),
                "accuracy": float(np.mean([p == g for p, g in v])),
                # for the contrast set what matters is: did the model stop saying SUPPORTED?
                "not_supported_rate": float(np.mean([p != 0 for p, _ in v])),
            }
            for name, v in sorted(groups.items())
        }
    if args.dataset == "contrast":
        # sanity check on the untouched pairs: they should all be SUPPORTED
        originals = load_evidence_claims(Path(cfg["data"]["dir"]) / "contrast_originals.jsonl")
        orig_pred = {
            ex.claim_id: pipeline.verify_with_evidence(ex.text, ex.evidence_texts).label for ex in originals
        }
        out["originals_supported_rate"] = float(np.mean([v == "SUPPORTED" for v in orig_pred.values()]))
        # consistent = right on the original AND notices the edit
        consistent = [
            orig_pred.get(ex.claim_id.rsplit("-", 1)[0]) == "SUPPORTED" and r.label != "SUPPORTED"
            for ex, r in zip(examples, results)
        ]
        out["consistency"] = float(np.mean(consistent)) if consistent else 0.0
    save_json(out_path(args, f"given_{args.dataset}"), out)
    print(f"{args.dataset}: acc={out['accuracy']['value']:.3f} macroF1={out['macro_f1']['value']:.3f} (n={len(examples)})")


# --------------------------------------------------------------------------- baselines


def cmd_baselines(args: argparse.Namespace) -> None:
    """Baselines that do not look at any evidence (they show the bias of the dataset)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    cfg = load_config(args.config)
    _, train = load_scifact(cfg["data"]["dir"], "train")
    _, test = load_scifact(cfg["data"]["dir"], args.split)
    y_train = np.array([LABEL_TO_IDX[c.label] for c in train])
    gold = np.array([LABEL_TO_IDX[c.label] for c in test])

    majority = np.full(len(gold), np.bincount(y_train, minlength=3).argmax())
    # claim-only: guess the label from the words of the claim, without any evidence
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.get("seed", 13))
    clf.fit(vectorizer.fit_transform([c.text for c in train]), y_train)
    claim_only = clf.predict(vectorizer.transform([c.text for c in test]))

    out = {}
    for name, pred in (("majority_class", majority), ("claim_only_tfidf_lr", claim_only)):
        out[name] = {
            "accuracy": bootstrap_ci(accuracy, gold, pred),
            "macro_f1": bootstrap_ci(macro_f1, gold, pred),
        }
        print(f"{name:22s} acc={out[name]['accuracy']['value']:.3f} macroF1={out[name]['macro_f1']['value']:.3f}")
    save_json(out_path(args, "baselines"), out)


# --------------------------------------------------------------------------- efficiency


def cmd_efficiency(args: argparse.Namespace) -> None:
    from claimguard.efficiency import benchmark, count_parameters, model_size_mb, quantize_dynamic_int8
    from claimguard.verify.nli import NLIVerifier

    cfg = apply_overrides(load_config(args.config), args)
    pipeline = build_pipeline(cfg)
    verifier = pipeline.verifier
    if not isinstance(verifier, NLIVerifier):
        raise SystemExit("the efficiency benchmark needs a neural verifier (verifier.kind: nli)")
    corpus, claims = load_scifact(cfg["data"]["dir"], args.split)
    # gold-evidence pairs: the accuracy here only depends on the verifier
    pairs, gold = [], []
    for c in claims[: args.limit]:
        for doc_id, ids in c.evidence.items():
            if doc_id in corpus:
                pairs.append((c.text, " ".join(corpus[doc_id].sentences[i] for i in ids)))
                gold.append(LABEL_TO_IDX[c.label])
    gold = np.array(gold)
    batch = pairs[:5]  # one claim = about 5 evidence passages

    out = {}
    fp32 = NLIVerifier(model=verifier.model, tokenizer=verifier.tokenizer, device="cpu")
    int8 = NLIVerifier(model=quantize_dynamic_int8(verifier.model), tokenizer=verifier.tokenizer, device="cpu")
    int8_ffn = NLIVerifier(
        model=quantize_dynamic_int8(verifier.model, only_feed_forward=True), tokenizer=verifier.tokenizer, device="cpu"
    )
    variants = (("fp32_cpu", fp32), ("int8_cpu", int8), ("int8_ffn_cpu", int8_ffn))
    for name, v in variants:
        pred = v.predict_logits(pairs).argmax(axis=1)
        per_class = per_class_prf(gold, pred)
        out[name] = {
            "size_mb": model_size_mb(v.model),
            "accuracy_gold_evidence": accuracy(gold, pred),
            # gold-evidence pairs are only SUPPORTED or REFUTED (NEI claims have no gold evidence),
            # so a 3-class macro-F1 would count an empty class: average S and R only
            "macro_f1_sr_gold_evidence": float(np.mean([per_class[0]["f1"], per_class[1]["f1"]])),
            "latency_5_pairs": benchmark(lambda v=v: v.predict_logits(batch), n_runs=args.runs),
        }
        print(name, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in out[name].items() if k != "latency_5_pairs"},
              f"p50={out[name]['latency_5_pairs']['p50_ms']:.1f}ms")
    out["n_parameters"] = count_parameters(fp32.model)
    out["n_pairs"] = len(pairs)
    # seconds to verify 1,000 claims (5 passages each) -> multiply by the price per second of a machine
    for name, _ in variants:
        out[name]["seconds_per_1000_claims"] = out[name]["latency_5_pairs"]["mean_ms"]
    save_json(out_path(args, "efficiency"), out)


# --------------------------------------------------------------------------- error analysis export


def cmd_errors(args: argparse.Namespace) -> None:
    """Export the wrong predictions to a CSV, with automatic hints, for manual annotation."""
    tag = f"_{args.tag}" if args.tag else ""
    records = read_jsonl(Path(args.out) / f"predictions{tag}.jsonl")
    # same setting as the run that produced the predictions (chosen on dev)
    run = load_json(Path(args.out) / f"pipeline{tag}.json")
    k, strategy = run["main"]["k"], run["main"]["aggregation"]
    temperature = run["temperature"]
    s = score(records, k, strategy, temperature)
    rows = []
    for r, pred, gold, conf, found in zip(records, s["pred"], s["gold"], s["conf"], s["evidence_found"]):
        if pred == gold:
            continue
        # which stage failed? if the claim has gold evidence and none of it reached
        # the verifier, the verifier had no chance: it is a retrieval error
        stage = "retrieval_miss" if r["gold_evidence"] and not found else "verification"
        best = max(r["evidence"][:k], key=lambda e: max(e["logits"][:2]), default={"text": "", "source": ""})
        rows.append(
            {
                "claim_id": r["claim_id"],
                "claim": r["claim"],
                "gold": LABELS[gold],
                "pred": LABELS[pred],
                "confidence": round(float(conf), 3),
                "stage": stage,
                "has_number": bool(extract_numbers(r["claim"])),
                "has_negation": has_negation(r["claim"]),
                "claim_tokens": len(tokenize(r["claim"], remove_stopwords=False)),
                "top_evidence_source": best["source"],
                "top_evidence": best["text"],
                "category": "",  # <- filled by hand (see notebooks/02_error_analysis.ipynb)
                "note": "",
            }
        )
    csv_path = Path(args.out) / f"errors_to_annotate{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["claim_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} errors out of {len(records)} claims -> {csv_path}")


# --------------------------------------------------------------------------- LLM-AggreFact


def cmd_aggrefact(args: argparse.Namespace) -> None:
    """Extension: are sentences written by LLMs supported by their source document?"""
    from claimguard.aggrefact import evaluate_rows, load_rows, sample_per_dataset

    cfg = apply_overrides(load_config(args.config), args)
    pipeline = build_pipeline(cfg, with_corpus=False)
    rows = sample_per_dataset(load_rows("test"), args.per_dataset, cfg.get("seed", 13))
    print(f"{len(rows)} claims from {len({r['dataset'] for r in rows})} sub-datasets")
    out = {"verifier": getattr(pipeline.verifier, "name", "?"), **evaluate_rows(pipeline, rows)}
    save_json(out_path(args, "aggrefact"), out)
    for name, m in out["by_dataset"].items():
        print(f"  {name:20s} BAcc={m['balanced_accuracy']:.3f} (n={m['n']})")
    print(f"mean balanced accuracy: {out['mean_balanced_accuracy']:.3f}")


# --------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "retrieval": cmd_retrieval,
        "pipeline": cmd_pipeline,
        "calibration": cmd_calibration,
        "given": cmd_given,
        "baselines": cmd_baselines,
        "efficiency": cmd_efficiency,
        "errors": cmd_errors,
        "aggrefact": cmd_aggrefact,
    }
    for name in commands:
        p = sub.add_parser(name)
        p.add_argument("--config", default="configs/default.yaml")
        p.add_argument("--out", default="results")
        p.add_argument("--split", default="test")
        p.add_argument("--limit", type=int, default=None, help="only the first N examples (quick tests)")
        p.add_argument("--tag", default="", help="suffix added to the result file name")
        p.add_argument("--verifier-kind", default=None)
        p.add_argument("--verifier-model", default=None)
        p.add_argument("--retrieval-kind", default=None, help="bm25 | dense | hybrid")
        p.add_argument("--rerank-kind", default=None, help="cross_encoder | lexical | none")
        p.add_argument("--load-in-4bit", action="store_true", help="LLM judge in 4-bit (GPU + bitsandbytes)")
        if name == "retrieval":
            p.add_argument("--kinds", default=None, help="ex: bm25,dense,hybrid")
        if name == "calibration":
            p.add_argument("--target-precision", type=float, default=0.9)
        if name == "given":
            p.add_argument("--dataset", required=True, choices=["french", "climate", "contrast", "custom"])
            p.add_argument("--path", default=None)
        if name == "aggrefact":
            p.add_argument("--per-dataset", type=int, default=100, help="claims kept per sub-dataset")
        if name == "efficiency":
            p.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    commands[args.command](args)


if __name__ == "__main__":
    main()
