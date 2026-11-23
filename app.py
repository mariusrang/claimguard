"""ClaimGuard demo (Gradio).

    python app.py                                   # full neural pipeline (configs/default.yaml)
    CLAIMGUARD_CONFIG=configs/offline.yaml python app.py   # no model download, rule-based verifier

Tabs:
1. Verify a claim      - one claim, against the SciFact corpus or against documents you paste
2. Check an LLM answer - splits the answer into claims and colours each one
3. Audit log & model   - last verdicts (exportable) + model card numbers
"""

from __future__ import annotations

import html
import os
import time
from pathlib import Path

import gradio as gr

from claimguard import LABELS
from claimguard.pipeline import ClaimGuard, ClaimResult, InvalidInputError, build_pipeline
from claimguard.utils import load_config

CONFIG_PATH = os.environ.get("CLAIMGUARD_CONFIG", "configs/default.yaml")
MAX_UPLOAD_BYTES = 2_000_000
COLORS = {"SUPPORTED": "#d7f5dd", "REFUTED": "#fbd9d9", "NEI": "#eceae4", "UNCERTAIN": "#fdf0c8"}
ICONS = {"SUPPORTED": "✅", "REFUTED": "❌", "NEI": "❔", "UNCERTAIN": "⚠️"}


def load_pipeline() -> tuple[ClaimGuard, str]:
    """Build the pipeline. If the models cannot be loaded, fall back to the offline config."""
    try:
        return build_pipeline(load_config(CONFIG_PATH)), ""
    except Exception as err:  # no internet, missing checkpoint... the demo should still start
        note = (
            f"⚠️ Could not load the neural pipeline ({type(err).__name__}). "
            "Running the **offline rule-based baseline** instead."
        )
        print(note, err)
        return build_pipeline(load_config("configs/offline.yaml")), note


PIPELINE, STARTUP_NOTE = load_pipeline()


def model_info() -> str:
    """Model name, number of parameters and size (asked by the project guidelines)."""
    verifier = PIPELINE.verifier
    lines = [f"**Verifier:** `{getattr(verifier, 'name', type(verifier).__name__)}`"]
    if hasattr(verifier, "model"):
        from claimguard.efficiency import count_parameters, model_size_mb

        lines.append(f"**Parameters:** {count_parameters(verifier.model) / 1e6:.1f} M")
        lines.append(f"**Model size:** {model_size_mb(verifier.model):.0f} MB")
    else:
        lines.append("**Model size:** no neural model (rule-based baseline)")
    lines.append(f"**Aggregation:** {PIPELINE.aggregation} · **Temperature:** {PIPELINE.temperature:.2f}")
    lines.append(f"**Corpus:** {len(PIPELINE.corpus)} documents indexed")
    return "  \n".join(lines)


MODEL_INFO = model_info()


def read_sources(text: str | None, file_path: str | None) -> list[str] | None:
    """Sources can be pasted (one document per blank-line-separated block) or uploaded as .txt."""
    sources = []
    if text and text.strip():
        sources.extend(block.strip() for block in text.split("\n\n") if block.strip())
    if file_path:
        path = Path(file_path)
        if path.suffix.lower() not in (".txt", ".md"):
            raise InvalidInputError("Please upload a .txt or .md file.")
        if path.stat().st_size > MAX_UPLOAD_BYTES:
            raise InvalidInputError("The file is too big (max 2 MB).")
        try:
            sources.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as err:
            raise InvalidInputError("The file is not valid UTF-8 text.") from err
    return sources or None


def badge(result: ClaimResult) -> str:
    return f"{ICONS[result.decision]} **{result.decision}** (confidence {result.confidence:.0%})"


def evidence_html(result: ClaimResult) -> str:
    """Evidence passages with their own probabilities; the most decisive one is highlighted."""
    if not result.evidence:
        return "<p><i>No evidence found.</i></p>"
    best = max(range(len(result.evidence)), key=lambda i: max(result.evidence[i].probs[:2]))
    blocks = []
    for i, e in enumerate(result.evidence):
        style = "background:#fff3b0;" if i == best else ""
        probs = " · ".join(f"{name} {p:.0%}" for name, p in zip(LABELS, e.probs))
        title = f" — {html.escape(e.title)}" if e.title else ""
        blocks.append(
            f"<div style='margin:8px 0;padding:8px;border:1px solid #ddd;border-radius:6px;color:#111;{style}'>"
            f"<b>{html.escape(e.source)}</b>{title}<br>{html.escape(e.text)}"
            f"<br><small>{probs}</small></div>"
        )
    return "".join(blocks)


def perf_text(result_timings: dict, total_ms: float) -> str:
    parts = " · ".join(f"{k} {v:.0f} ms" for k, v in result_timings.items())
    return f"**Inference time:** {total_ms:.0f} ms ({parts})  \n{MODEL_INFO}"


def verify_claim_ui(claim: str, sources_text: str, sources_file: str | None, threshold: float, top_k: int):
    """Handler of tab 1."""
    try:
        sources = read_sources(sources_text, sources_file)
        PIPELINE.threshold, PIPELINE.top_k = float(threshold), int(top_k)
        start = time.perf_counter()
        result = PIPELINE.verify_claim(claim, sources)
        total_ms = (time.perf_counter() - start) * 1000
    except InvalidInputError as err:
        return f"⚠️ {err}", {}, "", ""
    except Exception as err:  # never show a stack trace to the user
        return f"⚠️ Something went wrong: {type(err).__name__}", {}, "", ""
    scores = {name: float(p) for name, p in zip(LABELS, result.probs)}
    return badge(result), scores, evidence_html(result), perf_text(result.timings_ms, total_ms)


def check_answer_ui(answer: str, sources_text: str, sources_file: str | None, threshold: float):
    """Handler of tab 2."""
    try:
        sources = read_sources(sources_text, sources_file)
        PIPELINE.threshold = float(threshold)
        start = time.perf_counter()
        results = PIPELINE.verify_answer(answer, sources)
        total_ms = (time.perf_counter() - start) * 1000
    except InvalidInputError as err:
        return f"⚠️ {err}", "", [], ""
    except Exception as err:
        return f"⚠️ Something went wrong: {type(err).__name__}", "", [], ""
    spans, rows = [], []
    for r in results:
        best = max(r.evidence, key=lambda e: max(e.probs[:2]), default=None)
        tip = html.escape(best.text) if best else "no evidence"
        spans.append(
            f"<span title='{tip}' style='background:{COLORS[r.decision]};color:#111;padding:2px 4px;"
            f"border-radius:4px;line-height:2'>{ICONS[r.decision]} {html.escape(r.claim)}</span>"
        )
        rows.append([r.claim, r.decision, f"{r.confidence:.0%}", best.text if best else ""])
    n_flagged = sum(r.decision != "SUPPORTED" for r in results)
    summary = (
        f"**{len(results)} claims checked · {n_flagged} flagged "
        f"({n_flagged / len(results):.0%} not confirmed by the sources)**"
    )
    timings: dict = {}
    for r in results:
        for k, v in r.timings_ms.items():
            timings[k] = timings.get(k, 0.0) + v
    return summary, " ".join(spans), rows, perf_text(timings, total_ms)


def audit_ui():
    """Handler of tab 3: last records of the audit log + path of the file to download."""
    if PIPELINE.audit is None:
        return [], None
    records = PIPELINE.audit.tail(50)
    rows = [
        [r["timestamp"], r["mode"], r["claim"][:120], r["decision"], r["confidence"], r["model_version"]]
        for r in reversed(records)
    ]
    path = str(PIPELINE.audit.path) if PIPELINE.audit.path.exists() else None
    return rows, path


def corpus_examples() -> list[list[str]]:
    """One SUPPORTED, one REFUTED and one NEI claim from the test split of the loaded dataset."""
    try:
        from claimguard.data import load_claims

        data_dir = Path(load_config(CONFIG_PATH)["data"]["dir"])
        if not PIPELINE.corpus:
            data_dir = Path("data/processed/sample")
        claims = load_claims(data_dir / "claims_test.jsonl")
    except Exception:
        return []
    picked = {}
    for claim in claims:
        picked.setdefault(claim.label, claim.text)
    return [[picked[label], ""] for label in LABELS if label in picked]


SOURCE_EXAMPLE = (
    "The fixed rate of the Horizon Home loan is 3.45% over 20 years. The minimum personal "
    "contribution is 10% of the project amount. Application fees are 900 euros."
)
ANSWER_EXAMPLE = (
    "The Horizon Home loan has a fixed rate of 4.35% over 20 years. The minimum personal "
    "contribution is 10% of the project amount. It can be repaid early without any penalty."
)

with gr.Blocks(title="ClaimGuard") as demo:
    gr.Markdown(
        "# ClaimGuard\nChecks if a claim is **supported by documents**, with evidence and a calibrated "
        "confidence. It measures faithfulness to the sources, not truth in the real world."
    )
    if STARTUP_NOTE:
        gr.Markdown(STARTUP_NOTE)

    with gr.Tab("Verify a claim"):
        with gr.Row():
            with gr.Column():
                claim_in = gr.Textbox(label="Claim", lines=2, placeholder="Type one factual sentence")
                src_in = gr.Textbox(
                    label="Source documents (optional)",
                    lines=5,
                    placeholder="Paste documents here (blank line between two documents). "
                    "Leave empty to search the indexed corpus.",
                )
                src_file = gr.File(label="...or upload a .txt file", file_types=[".txt", ".md"], type="filepath")
                thr = gr.Slider(0.0, 0.95, value=PIPELINE.threshold, step=0.05, label="Abstention threshold")
                topk = gr.Slider(1, 10, value=PIPELINE.top_k, step=1, label="Evidence passages (k)")
                btn = gr.Button("Verify", variant="primary")
            with gr.Column():
                verdict = gr.Markdown(label="Verdict")
                probs = gr.Label(label="Probabilities", num_top_classes=3)
                evidence = gr.HTML(label="Evidence")
                perf = gr.Markdown()
        btn.click(verify_claim_ui, [claim_in, src_in, src_file, thr, topk], [verdict, probs, evidence, perf])
        gr.Examples(
            examples=corpus_examples()
            + [
                ["The Horizon Home loan has a fixed rate of 4.35% over 20 years.", SOURCE_EXAMPLE],
                ["The minimum personal contribution is 10% of the project amount.", SOURCE_EXAMPLE],
            ],
            inputs=[claim_in, src_in],
            label="Examples (supported / refuted / not enough info from the corpus, then two with a pasted source)",
        )

    with gr.Tab("Check an LLM answer"):
        with gr.Row():
            with gr.Column():
                answer_in = gr.Textbox(label="Answer written by an LLM", lines=6)
                src2_in = gr.Textbox(label="Source documents", lines=6)
                src2_file = gr.File(label="...or upload a .txt file", file_types=[".txt", ".md"], type="filepath")
                thr2 = gr.Slider(0.0, 0.95, value=PIPELINE.threshold, step=0.05, label="Abstention threshold")
                btn2 = gr.Button("Check the answer", variant="primary")
            with gr.Column():
                summary = gr.Markdown()
                colored = gr.HTML(label="Answer, claim by claim (hover to see the evidence)")
                table = gr.Dataframe(headers=["claim", "decision", "confidence", "best evidence"], wrap=True)
                perf2 = gr.Markdown()
        btn2.click(check_answer_ui, [answer_in, src2_in, src2_file, thr2], [summary, colored, table, perf2])
        gr.Examples(examples=[[ANSWER_EXAMPLE, SOURCE_EXAMPLE]], inputs=[answer_in, src2_in])

    with gr.Tab("Audit log & model"):
        gr.Markdown(MODEL_INFO)
        gr.Markdown(
            "Every verdict is logged with its evidence, score, decision, date and model version "
            "(`logs/audit.jsonl`), so a decision can be traced later."
        )
        refresh = gr.Button("Refresh")
        log_table = gr.Dataframe(
            headers=["timestamp", "mode", "claim", "decision", "confidence", "model version"], wrap=True
        )
        log_file = gr.File(label="Download the full log (JSONL)")
        refresh.click(audit_ui, [], [log_table, log_file])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=bool(os.environ.get("GRADIO_SHARE")),  # public link, handy on Colab
    )
