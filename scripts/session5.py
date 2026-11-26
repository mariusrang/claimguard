"""Kaggle session 5: the checks asked after the review of the first report draft.

    python scripts/session5.py oracle  --config configs/default.yaml   # GPU, ~5 min
    python scripts/session5.py latency --config configs/default.yaml   # GPU (+ CPU with --device cpu)

oracle   Reads results/predictions.jsonl (the main run, retrieved evidence) and re-runs ONLY the
         verifier on two kinds of perfect evidence, for the claims that have gold evidence:
           gold_doc_system_sentences : the gold abstracts, but our own sentence selection (3 best)
           gold_sentences            : the gold rationale sentences (the "oracle" of the first draft)
         NEI claims have no gold evidence, so nothing can change for them: the comparison is made
         on the claims WITH evidence only. Writes results/oracle.json and results/oracle_predictions.jsonl.
latency  Time per claim of each system with its OWN number of passages (k chosen on dev),
         20 warm-up-free test claims after 3 warm-up claims. Writes results/latency_<device>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402

from claimguard import LABELS  # noqa: E402
from claimguard.data import load_scifact  # noqa: E402
from claimguard.metrics import macro_f1  # noqa: E402
from claimguard.pipeline import build_pipeline  # noqa: E402
from claimguard.stats import bootstrap_ci, mcnemar_test  # noqa: E402
from claimguard.utils import load_config, read_jsonl, save_json, set_seed, write_jsonl  # noqa: E402

ZS = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
# tag -> overrides of the config, exactly as in the Kaggle runs that produced results/pipeline_<tag>.json
SYSTEMS = {
    "": {},
    "zeroshot": {"verifier_model": ZS},
    "bm25_zeroshot": {"verifier_model": ZS, "retrieval_kind": "bm25", "rerank_kind": "none"},
    "llm_judge": {"verifier_kind": "llm", "verifier_model": "Qwen/Qwen2.5-7B-Instruct", "load_in_4bit": True},
}


def gold_doc_records(pipeline, claims, records) -> list[dict]:
    """Gold abstracts as the only documents, sentences picked by our own sentence selector."""
    out = []
    for claim, record in zip(claims, records):
        gold = [d for d in claim.evidence if d in pipeline.corpus]
        if not gold:
            out.append(record)
            continue
        evidence = pipeline._verify_documents(claim.text, pipeline.corpus, [(d, 1.0) for d in gold], {})
        out.append({**record, "evidence": [
            {"source": e.source, "sentence_ids": e.sentence_ids, "text": e.text, "logits": e.logits} for e in evidence
        ]})
    return out


def cmd_oracle(args) -> None:
    cfg = ev.apply_overrides(load_config(args.config), args)
    set_seed(cfg.get("seed", 13))
    pipeline = build_pipeline(cfg)
    _, claims = load_scifact(cfg["data"]["dir"], "test")
    records = read_jsonl(Path(args.out) / "predictions.jsonl")
    claims = claims[: len(records)]  # a --limit run keeps the first claims
    assert [c.claim_id for c in claims] == [r["claim_id"] for r in records], "predictions.jsonl is not the test run"
    meta = json.loads((Path(args.out) / "pipeline.json").read_text())
    k, strategy, temp = meta["chosen_on_dev"]["top_k"], meta["chosen_on_dev"]["aggregation"], meta["temperature"]

    variants = {
        "retrieved": records,
        "gold_doc_system_sentences": gold_doc_records(pipeline, claims, records),
        "gold_sentences": ev.with_oracle_evidence(pipeline, claims, records),
    }
    scored = {name: ev.score(recs, k, strategy, temp) for name, recs in variants.items()}
    has_ev = np.array([bool(c.evidence) for c in claims])
    gold = scored["retrieved"]["gold"]
    out = {"setting": {"k": k, "aggregation": strategy, "temperature": temp},
           "n_claims": int(len(claims)), "n_with_evidence": int(has_ev.sum()), "variants": {}}
    ref_ok = scored["retrieved"]["pred"] == gold
    for name, s in scored.items():
        ok = s["pred"] == gold
        sub = has_ev
        out["variants"][name] = {
            "accuracy_with_evidence": bootstrap_ci(lambda a: float(a.mean()), ok[sub].astype(float)),
            "macro_f1_all_claims": float(macro_f1(gold, s["pred"])),
            "errors_with_evidence": int((~ok[sub]).sum()),
            "errors_without_evidence": int((~ok[~sub]).sum()),
        }
        if name != "retrieved":
            out["variants"][name]["mcnemar_vs_retrieved_with_evidence"] = mcnemar_test(ok[sub], ref_ok[sub])
            out["variants"][name]["errors_fixed"] = int((~ref_ok[sub] & ok[sub]).sum())
            out["variants"][name]["errors_created"] = int((ref_ok[sub] & ~ok[sub]).sum())
    save_json(Path(args.out) / "oracle.json", out)
    write_jsonl(Path(args.out) / "oracle_predictions.jsonl", [
        {"claim_id": r["claim_id"], "gold": r["gold"], "has_evidence": bool(h),
         **{name: LABELS[int(scored[name]["pred"][i])] for name in scored}}
        for i, (r, h) in enumerate(zip(records, has_ev))
    ])
    print(json.dumps(out, indent=1))


def cmd_latency(args) -> None:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, claims = load_scifact(load_config(args.config)["data"]["dir"], "test")
    systems = SYSTEMS if device == "cuda" else {"": {}}  # on CPU only the main system
    if args.only:
        systems = {t: o for t, o in systems.items() if (t or "main") in args.only}
    out = {"device": device, "torch_threads": torch.get_num_threads(), "n_claims": args.n, "systems": {}}
    for tag, overrides in systems.items():
        ns = argparse.Namespace(**{"verifier_kind": None, "verifier_model": None, "retrieval_kind": None,
                                   "rerank_kind": None, "load_in_4bit": False, **overrides})
        cfg = ev.apply_overrides(load_config(args.config), ns)
        meta = json.loads((Path(args.out) / f"pipeline{'_' + tag if tag else ''}.json").read_text())
        try:
            pipeline = build_pipeline(cfg)
        except Exception as err:  # e.g. bitsandbytes missing for the 4-bit LLM: keep the other systems
            out["systems"][tag or "main"] = {"error": f"{type(err).__name__}: {err}"}
            print(tag or "main", "SKIPPED:", type(err).__name__, err)
            continue
        pipeline.top_k = meta["chosen_on_dev"]["top_k"]
        for c in claims[:3]:  # warm-up: first calls build caches and CUDA kernels
            pipeline.verify_claim(c.text)
        totals, stages = [], {}
        for c in claims[3 : 3 + args.n]:
            start = time.perf_counter()
            result = pipeline.verify_claim(c.text)
            totals.append((time.perf_counter() - start) * 1000)
            for stage, ms in result.timings_ms.items():
                stages.setdefault(stage, []).append(ms)
        out["systems"][tag or "main"] = {
            "k": pipeline.top_k, "p50_ms": float(np.percentile(totals, 50)), "p95_ms": float(np.percentile(totals, 95)),
            "stages_p50_ms": {s: float(np.percentile(v, 50)) for s, v in stages.items()},
        }
        print(tag or "main", out["systems"][tag or "main"])
        save_json(Path(args.out) / f"latency_{device}.json", out)  # saved after each system
        del pipeline
        if device == "cuda":
            torch.cuda.empty_cache()
    save_json(Path(args.out) / f"latency_{device}.json", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["oracle", "latency"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=20, help="timed claims per system (latency)")
    ap.add_argument("--only", nargs="+", help="latency: only these systems (main, zeroshot, bm25_zeroshot, llm_judge)")
    args = ap.parse_args()
    for name in ("verifier_kind", "verifier_model", "retrieval_kind", "rerank_kind"):
        setattr(args, name, None)
    args.load_in_4bit = False
    {"oracle": cmd_oracle, "latency": cmd_latency}[args.command](args)


if __name__ == "__main__":
    main()
