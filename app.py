"""ClaimGuard demo (Gradio).

    python app.py                                          # full neural pipeline (configs/default.yaml)
    CLAIMGUARD_CONFIG=configs/offline.yaml python app.py   # no model download, rule-based verifier

Design idea: a proofreader's page. The answer is shown like a text under review:
each claim is underlined (a different line style per verdict, so the meaning does not
rely on colour alone) and numbered, and the margin holds the proofreader's note:
verdict, confidence and the passage the decision is based on.

Tabs:
1. Check an answer - splits an LLM answer into claims and annotates each one
2. Check one claim - one claim, against the indexed corpus or documents you paste
3. Audit log       - every verdict with its model version, downloadable
"""

from __future__ import annotations

import html
import inspect
import os
import time
from pathlib import Path

import gradio as gr

from claimguard import LABELS
from claimguard.pipeline import ClaimGuard, ClaimResult, InvalidInputError, build_pipeline
from claimguard.text import split_sentences, tokenize
from claimguard.utils import load_config

CONFIG_PATH = os.environ.get("CLAIMGUARD_CONFIG", "configs/default.yaml")
MAX_UPLOAD_BYTES = 2_000_000

# user-facing words for the model labels (the model says REFUTED, a person says "contradicted")
VERDICT_TEXT = {
    "SUPPORTED": "Supported by the sources",
    "REFUTED": "Contradicted by the sources",
    "NEI": "Not in the sources",
    "UNCERTAIN": "Unsure, check it yourself",
}
SHORT = {"SUPPORTED": "Supported", "REFUTED": "Contradicted", "NEI": "Not in the sources", "UNCERTAIN": "Unsure"}
CSS_CLASS = {"SUPPORTED": "ok", "REFUTED": "no", "NEI": "na", "UNCERTAIN": "un"}


def load_pipeline() -> tuple[ClaimGuard, str]:
    """Build the pipeline. If the models cannot be loaded, fall back to the offline config."""
    try:
        return build_pipeline(load_config(CONFIG_PATH)), ""
    except Exception as err:  # no internet, missing checkpoint... the demo should still start
        note = (
            f"The neural verifier could not be loaded ({type(err).__name__}), "
            "so this page runs the rule-based baseline instead."
        )
        print(note, err)
        return build_pipeline(load_config("configs/offline.yaml")), note


PIPELINE, STARTUP_NOTE = load_pipeline()


def model_facts() -> dict:
    """Name, parameters and size of the verifier (shown on every result, as the guidelines ask)."""
    verifier = PIPELINE.verifier
    facts = {"name": getattr(verifier, "name", type(verifier).__name__), "params_m": None, "size_mb": None}
    if hasattr(verifier, "model"):
        from claimguard.efficiency import count_parameters, model_size_mb

        facts["params_m"] = count_parameters(verifier.model) / 1e6
        facts["size_mb"] = model_size_mb(verifier.model)
    return facts


FACTS = model_facts()


def model_sentence() -> str:
    if FACTS["params_m"] is None:
        return f"Verifier: {FACTS['name']} (rule-based, no neural model)."
    return f"Verifier: {FACTS['name']}, {FACTS['params_m']:.0f}M parameters, {FACTS['size_mb']:.0f} MB."


def footer(total_ms: float, timings: dict) -> str:
    steps = ", ".join(f"{k} {v:.0f} ms" for k, v in timings.items())
    return (
        f"<p class='cg-foot'>Checked in {total_ms:.0f} ms ({steps}). Inference time includes retrieval. "
        f"Model size: {html.escape(model_sentence())}</p>"
    )


def read_sources(text: str | None, file_path: str | None) -> list[str] | None:
    """Sources can be pasted (blank line between two documents) or uploaded as .txt / .md."""
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


def best_evidence(result: ClaimResult):
    """The passage that weighed most in the decision (highest support or contradiction)."""
    return max(result.evidence, key=lambda e: max(e.probs[0], e.probs[1]), default=None)


def mark_shared_words(evidence: str, claim: str) -> str:
    """Escape the evidence and lightly mark the words it shares with the claim."""
    claim_words = set(tokenize(claim))
    out = []
    for word in evidence.split(" "):
        core = tokenize(word)
        safe = html.escape(word)
        out.append(f"<mark>{safe}</mark>" if core and core[0] in claim_words else safe)
    return " ".join(out)


def closest_sentence(evidence: str, claim: str) -> tuple[str, float]:
    """The sentence of a passage that shares the most words with the claim (and that share)."""
    claim_words = set(tokenize(claim))
    best, best_overlap = evidence, 0.0
    for sent in split_sentences(evidence) or [evidence]:
        overlap = len(claim_words & set(tokenize(sent))) / max(len(claim_words), 1)
        if overlap > best_overlap:
            best, best_overlap = sent, overlap
    return best, best_overlap


def word_diff(claim: str, source: str) -> str:
    """Two-line comparison (answer vs source) where the content words that differ are marked.

    A plain sequence diff is useless here because a paraphrase changes the word order
    ("the rate of the loan is" vs "the loan has a rate of"), so we only mark the content
    words (numbers, or words of 4+ letters) that appear on one side and not on the other."""
    norm = lambda w: w.lower().strip(".,;:!?()")  # noqa: E731

    def content(w: str) -> bool:
        return any(c.isdigit() for c in w) or (len(norm(w)) >= 4 and bool(tokenize(w)))

    a_words, b_words = claim.split(), source.split()
    a_set, b_set = {norm(w) for w in a_words}, {norm(w) for w in b_words}

    def render(words, other, tag):
        return " ".join(
            f"<{tag}>{html.escape(w)}</{tag}>" if content(w) and norm(w) not in other else html.escape(w)
            for w in words
        )

    return (
        "<div class='diff'>"
        f"<div class='row minus'><span class='sign'>answer</span><span>{render(a_words, b_set, 'del')}</span></div>"
        f"<div class='row plus'><span class='sign'>source</span><span>{render(b_words, a_set, 'ins')}</span></div>"
        "</div>"
    )


def message(text: str, kind: str = "error") -> str:
    """Error or empty state, in the interface's voice."""
    return f"<div class='cg-msg cg-msg-{kind}'>{html.escape(text)}</div>"


# ---------------------------------------------------------------- tab 1: answer


def summary_bar(results: list[ClaimResult]) -> str:
    """One sentence + a thin bar with the share of each verdict."""
    counts = {k: sum(r.decision == k for r in results) for k in VERDICT_TEXT}
    n = len(results)
    flagged = n - counts["SUPPORTED"]
    parts = []
    if counts["REFUTED"]:
        parts.append(f"{counts['REFUTED']} contradicted")
    if counts["NEI"]:
        parts.append(f"{counts['NEI']} not in the sources")
    if counts["UNCERTAIN"]:
        parts.append(f"{counts['UNCERTAIN']} unsure")
    if flagged == 0:
        sentence = f"All {n} claims are supported by your sources."
    else:
        sentence = f"{n} claims checked, {flagged} to review: " + ", ".join(parts) + "."
    segments = "".join(
        f"<span class='seg {CSS_CLASS[k]}' style='flex:{c}' title='{SHORT[k]}: {c}'></span>"
        for k, c in counts.items()
        if c
    )
    return f"<div class='cg-summary'><p>{html.escape(sentence)}</p><div class='cg-bar'>{segments}</div></div>"


def annotated_page(answer_results: list[ClaimResult]) -> str:
    """The proofreader's page: underlined, numbered claims on the left, notes in the margin."""
    text, notes = [], []
    for i, r in enumerate(answer_results, start=1):
        cls = CSS_CLASS[r.decision]
        text.append(
            f"<span class='claim {cls}'>{html.escape(r.claim)}"
            f"<sup>{i}</sup></span>"
        )
        ev = best_evidence(r)
        if ev is None:
            quote = "<blockquote>No passage of the sources is related to this claim.</blockquote>"
        else:
            sent, overlap = closest_sentence(ev.text, r.claim)
            if r.decision == "REFUTED" and overlap >= 0.3:
                body = word_diff(r.claim, sent)  # show exactly what the answer changed
            elif r.decision == "NEI":
                body = f"<blockquote class='faint'>Closest passage: {mark_shared_words(sent, r.claim)}</blockquote>"
            else:
                body = f"<blockquote>{mark_shared_words(sent, r.claim)}</blockquote>"
            quote = f"{body}<cite>{html.escape(ev.source)}</cite>"
        notes.append(
            f"<li class='{cls}'><div class='note-head'><b>{i}</b> {SHORT[r.decision]}"
            f"<span class='conf'>{r.confidence:.0%} confidence</span></div>{quote}</li>"
        )
    return (
        "<div class='cg-page'>"
        f"<div class='cg-text'>{' '.join(text)}</div>"
        f"<ol class='cg-margin'>{''.join(notes)}</ol>"
        "</div>"
    )


def check_answer_ui(answer: str, sources_text: str, sources_file: str | None, threshold: float):
    """Handler of tab 1: summary + annotated page + table (for copy / export) + footer."""
    try:
        sources = read_sources(sources_text, sources_file)
        if not sources:
            raise InvalidInputError("Paste or upload the documents the answer should be based on.")
        PIPELINE.threshold = float(threshold)
        start = time.perf_counter()
        results = PIPELINE.verify_answer(answer, sources)
        total_ms = (time.perf_counter() - start) * 1000
    except InvalidInputError as err:
        return message(str(err)), "", []
    except Exception as err:  # never show a stack trace to the user
        return message(f"The check failed ({type(err).__name__}). Try a shorter text."), "", []
    timings: dict = {}
    for r in results:
        for k, v in r.timings_ms.items():
            timings[k] = timings.get(k, 0.0) + v
    rows = [[i, r.claim, SHORT[r.decision], f"{r.confidence:.0%}"] for i, r in enumerate(results, start=1)]
    return summary_bar(results), annotated_page(results) + footer(total_ms, timings), rows


# ---------------------------------------------------------------- tab 2: one claim


def probability_bars(result: ClaimResult) -> str:
    rows = []
    for label, p in zip(LABELS, result.probs):
        rows.append(
            f"<div class='prob {CSS_CLASS[label]}'><span class='name'>{SHORT[label]}</span>"
            f"<span class='track'><span class='fill' style='width:{p * 100:.1f}%'></span></span>"
            f"<span class='val'>{p:.0%}</span></div>"
        )
    return f"<div class='cg-probs'>{''.join(rows)}</div>"


def claim_card(result: ClaimResult) -> str:
    cls = CSS_CLASS[result.decision]
    best = best_evidence(result)
    passages = []
    for e in result.evidence:
        is_best = e is best
        title = f" {html.escape(e.title)}" if e.title else ""
        tag = "<span class='tag'>weighed most</span>" if is_best else ""  # no backslash inside f-strings (py < 3.12)
        passages.append(
            f"<li class='{'best ' + cls if is_best else ''}'>"
            f"<div class='src'>{html.escape(e.source)}{title}{tag}</div>"
            f"<p>{mark_shared_words(e.text, result.claim)}</p></li>"
        )
    abstain = (
        f"<p class='cg-note'>Confidence is below the {result.threshold:.0%} threshold, "
        f"so ClaimGuard does not decide. Its best guess was: {SHORT[result.label].lower()}.</p>"
        if result.decision == "UNCERTAIN"
        else ""
    )
    return (
        f"<div class='cg-card'><p class='cg-claim'><span class='claim {cls}'>{html.escape(result.claim)}</span></p>"
        f"<p class='cg-verdict {cls}'>{VERDICT_TEXT[result.decision]}, {result.confidence:.0%} confidence</p>"
        f"{abstain}{probability_bars(result)}"
        f"<h3>Evidence read by the model</h3><ol class='cg-passages'>{''.join(passages)}</ol></div>"
    )


def verify_claim_ui(claim: str, sources_text: str, sources_file: str | None, threshold: float, top_k: int):
    """Handler of tab 2: the verdict card, the probabilities (for the Label) and the footer."""
    try:
        sources = read_sources(sources_text, sources_file)
        PIPELINE.threshold, PIPELINE.top_k = float(threshold), int(top_k)
        start = time.perf_counter()
        result = PIPELINE.verify_claim(claim, sources)
        total_ms = (time.perf_counter() - start) * 1000
    except InvalidInputError as err:
        return message(str(err)), {}
    except Exception as err:
        return message(f"The check failed ({type(err).__name__})."), {}
    scores = {SHORT[name]: float(p) for name, p in zip(LABELS, result.probs)}
    return claim_card(result) + footer(total_ms, result.timings_ms), scores


# ---------------------------------------------------------------- tab 3: audit log


def audit_ui():
    """Last records of the audit log + path of the file to download."""
    if PIPELINE.audit is None:
        return [], None
    records = PIPELINE.audit.tail(50)
    rows = [
        [r["timestamp"].replace("T", " ")[:16], r["claim"][:140], SHORT.get(r["decision"], r["decision"]),
         f"{r['confidence']:.0%}", r["model_version"]]
        for r in reversed(records)
    ]
    path = str(PIPELINE.audit.path) if PIPELINE.audit.path.exists() else None
    return rows, path


# ---------------------------------------------------------------- examples


def corpus_examples() -> list[list[str]]:
    """One supported, one refuted and one NEI claim from the test split of the loaded dataset."""
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
    "contribution is 10% of the project amount. Application fees are 900 euros. Early "
    "repayment is charged 3% of the remaining capital, capped at six months of interest."
)
ANSWER_EXAMPLE = (
    "The Horizon Home loan has a fixed rate of 4.35% over 20 years. You need a personal "
    "contribution of at least 10% of the project. Application fees are 900 euros. "
    "It can be repaid early without any penalty. It is the cheapest mortgage on the market."
)

EXAMPLE_CLAIM = "The Horizon Home loan has a fixed rate of 4.35% over 20 years."

# ---------------------------------------------------------------- look and feel

INK, PAPER, RULE = "#1B2333", "#F3F5F8", "#D8DEE7"
CSS = """
:root { --ink:#1B2333; --paper:#F3F5F8; --rule:#D8DEE7; --soft:#5B6577;
        --ok:#2F8F6B; --no:#C8413B; --na:#7A8599; --un:#C98A1C; }
.gradio-container { background: var(--paper) !important; color: var(--ink); max-width: 1180px !important; }
.cg-head { padding: 28px 4px 10px; }
.cg-head h1 { font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 40px;
              letter-spacing: -0.5px; margin: 0; color: var(--ink); }
.cg-head p { margin: 6px 0 0; font-size: 16px; color: var(--soft); max-width: 62ch; line-height: 1.5; }
.cg-startup { margin: 10px 4px 0; padding: 10px 14px; border-left: 3px solid var(--un); background: #fff; }

/* the claim underline: the signature of the page. Line style changes with the verdict */
.claim { text-decoration-line: underline; text-decoration-thickness: 3px; text-underline-offset: 5px;
         text-decoration-skip-ink: none; padding-bottom: 1px;
         }
.claim.ok { text-decoration-color: var(--ok); text-decoration-style: solid; }
.claim.no { text-decoration-color: var(--no); text-decoration-style: double; text-decoration-thickness: 2px; }
.claim.na { text-decoration-color: var(--na); text-decoration-style: dashed; }
.claim.un { text-decoration-color: var(--un); text-decoration-style: dotted; text-decoration-thickness: 4px; }
.claim sup { font-family: 'Public Sans', sans-serif; font-size: 11px; font-weight: 700; margin-left: 2px; color: var(--soft); }

.cg-summary p { font-size: 17px; margin: 4px 0 10px; color: var(--ink); }
.cg-bar { display: flex; height: 6px; border-radius: 3px; overflow: hidden; gap: 2px; background: var(--rule); }
.cg-bar .seg.ok { background: var(--ok); } .cg-bar .seg.no { background: var(--no); }
.cg-bar .seg.na { background: var(--na); } .cg-bar .seg.un { background: var(--un); }

.cg-page { display: grid; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); gap: 32px;
           background: #fff; border: 1px solid var(--rule); border-radius: 2px; padding: 30px 32px; }
.cg-text { font-family: 'Source Serif 4', Georgia, serif; font-size: 20px; line-height: 2.05; color: var(--ink); }
.cg-margin { list-style: none; margin: 0; padding: 0 0 0 24px; border-left: 1px solid var(--rule); }
.cg-margin li { margin: 0 0 20px; }
.note-head { font-size: 13px; flex-wrap: wrap; font-weight: 600; color: var(--ink); display: flex; gap: 8px; align-items: baseline; }
.note-head b { display: inline-grid; place-items: center; width: 20px; height: 20px; border-radius: 50%;
               color: #fff; font-size: 11px; }
.cg-margin li.ok .note-head b { background: var(--ok); } .cg-margin li.no .note-head b { background: var(--no); }
.cg-margin li.na .note-head b { background: var(--na); } .cg-margin li.un .note-head b { background: var(--un); }
.note-head .conf { margin-left: auto; white-space: nowrap; font-weight: 400; color: var(--soft); }
.cg-margin blockquote, .cg-passages p { font-family: 'Source Serif 4', Georgia, serif; font-size: 14.5px; line-height: 1.55;
               color: #39424F; margin: 6px 0 2px; padding: 0; border: 0; }
.diff { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 12.5px; line-height: 1.7; margin: 8px 0 4px;
        border: 1px solid var(--rule); border-radius: 2px; overflow: hidden; }
.diff .row { display: grid; grid-template-columns: 58px 1fr; padding: 5px 10px 5px 0; }
.diff .sign { font-size: 11px; color: var(--soft); text-align: right; padding-right: 10px; }
.diff .minus { background: #FBEDEC; } .diff .plus { background: #EAF5EF; }
.diff del { text-decoration: none; background: #F2B8B4; color: #7A1E19; border-radius: 2px; padding: 0 1px; }
.diff ins { text-decoration: none; background: #A9DcC2; color: #12583D; border-radius: 2px; padding: 0 1px; }
blockquote.faint { color: var(--soft) !important; }
.cg-margin cite { font-style: normal; font-size: 12px; color: var(--soft); }
mark { background: #E6EBF2; color: inherit; padding: 0 1px; border-radius: 2px; }

.cg-card { background: #fff; border: 1px solid var(--rule); border-radius: 2px; padding: 28px 30px; }
.cg-claim { font-family: 'Source Serif 4', Georgia, serif; font-size: 23px; line-height: 1.6; margin: 0 0 14px; color: var(--ink); }
.cg-verdict { font-size: 17px; font-weight: 600; margin: 0 0 16px; }
.cg-verdict.ok { color: var(--ok); } .cg-verdict.no { color: var(--no); }
.cg-verdict.na { color: var(--na); } .cg-verdict.un { color: var(--un); }
.cg-note { color: var(--soft); margin: -8px 0 14px; }
.cg-probs { display: grid; gap: 7px; margin-bottom: 22px; max-width: 460px; }
.prob { display: grid; grid-template-columns: 140px 1fr 44px; align-items: center; gap: 10px; font-size: 13px; color: var(--soft); }
.prob .track { height: 6px; background: var(--paper); border-radius: 3px; overflow: hidden; }
.prob .fill { display: block; height: 100%; }
.prob.ok .fill { background: var(--ok); } .prob.no .fill { background: var(--no); }
.prob.na .fill { background: var(--na); } .prob.un .fill { background: var(--un); }
.prob .val { text-align: right; color: var(--ink); font-variant-numeric: tabular-nums; }
.cg-card h3 { font-size: 14px; font-weight: 600; color: var(--soft); margin: 0 0 8px; }
.cg-passages { margin: 0; padding-left: 18px; }
.cg-passages li { padding: 8px 0 8px 12px; border-left: 3px solid transparent; margin-bottom: 6px; color: var(--soft); }
.cg-passages li.best.ok { border-color: var(--ok); } .cg-passages li.best.no { border-color: var(--no); }
.cg-passages li.best.na { border-color: var(--na); } .cg-passages li.best.un { border-color: var(--un); }
.cg-passages .src { font-size: 12px; color: var(--soft); }
.cg-passages .tag { margin-left: 8px; color: var(--ink); font-weight: 600; }
.cg-foot { font-size: 12.5px; color: var(--soft); margin: 14px 2px 0; line-height: 1.5; }
.cg-msg { padding: 14px 16px; background: #fff; border-left: 3px solid var(--no); color: var(--ink); }
.cg-empty { color: var(--soft); font-family: 'Source Serif 4', Georgia, serif; font-size: 18px; padding: 40px 8px; line-height: 1.6; }
.cg-legend { display: flex; flex-wrap: wrap; gap: 18px; font-size: 13px; color: var(--soft); margin: 8px 2px 0; }
.tabs button.selected, button[role='tab'][aria-selected='true'] { color: var(--ink) !important; font-weight: 600; }
.tabs button.selected::after, button[role='tab'][aria-selected='true']::after { background: var(--ink) !important; }
footer { display: none !important; }
.cg-inputs label span, .cg-inputs .label-wrap span { color: var(--ink); }
.cg-legend .claim { color: var(--ink); }
@media (max-width: 860px) {
  .cg-page { grid-template-columns: 1fr; padding: 20px; }
  .cg-margin { padding-left: 0; border-left: 0; border-top: 1px solid var(--rule); padding-top: 16px; }
  .cg-head h1 { font-size: 32px; }
}
"""

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("Public Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
).set(
    body_background_fill=PAPER,
    body_text_color=INK,
    block_background_fill="#FFFFFF",
    block_border_color=RULE,
    block_radius="2px",
    block_shadow="none",
    input_radius="2px",
    input_border_color=RULE,
    input_background_fill="#FFFFFF",
    button_primary_background_fill=INK,
    button_primary_background_fill_hover="#2C3850",
    button_primary_text_color="#FFFFFF",
    button_large_radius="2px",
    button_small_radius="2px",
    slider_color=INK,
)
HEAD = (
    "<link rel='preconnect' href='https://fonts.googleapis.com'>"
    "<link href='https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600"
    "&family=Public+Sans:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap' rel='stylesheet'>"
)
LEGEND = (
    "<div class='cg-legend'>"
    "<span><span class='claim ok'>solid</span> supported</span>"
    "<span><span class='claim no'>double</span> contradicted</span>"
    "<span><span class='claim na'>dashed</span> not in the sources</span>"
    "<span><span class='claim un'>dotted</span> unsure</span></div>"
)
EMPTY_ANSWER = (
    "<div class='cg-empty'>Paste an answer written by an AI and the documents it should rely on. "
    "Each sentence comes back underlined, with a note in the margin.</div>"
)
EMPTY_CLAIM = "<div class='cg-empty'>Type one factual sentence to see which passages support or contradict it.</div>"

# Gradio 6 moved theme / css / head from Blocks() to launch(); support both
FORCE_LIGHT = """() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}"""
_STYLE = {"theme": THEME, "css": CSS, "head": HEAD, "js": FORCE_LIGHT}
_BLOCKS_STYLE = {k: v for k, v in _STYLE.items() if k in inspect.signature(gr.Blocks.__init__).parameters}
_LAUNCH_STYLE = {k: v for k, v in _STYLE.items() if k not in _BLOCKS_STYLE}

with gr.Blocks(title="ClaimGuard", **_BLOCKS_STYLE) as demo:
    gr.HTML(
        "<div class='cg-head'><h1>ClaimGuard</h1>"
        "<p>Checks each sentence of an AI answer against your documents. It tells you whether your "
        "sources support a sentence, not whether the sentence is true.</p></div>"
    )
    if STARTUP_NOTE:
        gr.HTML(f"<div class='cg-startup'>{html.escape(STARTUP_NOTE)}</div>")

    with gr.Tabs(selected=os.environ.get("CLAIMGUARD_PREVIEW", "answer")):
        with gr.Tab("Check an answer", id="answer"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=300, elem_classes="cg-inputs"):
                    answer_in = gr.Textbox(label="Answer written by an AI", lines=6,
                                           placeholder="Paste the answer of a chatbot here")
                    src2_in = gr.Textbox(label="Source documents", lines=6,
                                         placeholder="Paste the documents the answer should rely on. "
                                                     "Leave a blank line between two documents.")
                    with gr.Row():
                        ex2 = gr.Button("Load an example", variant="secondary")
                        btn2 = gr.Button("Check the answer", variant="primary")
                    with gr.Accordion("More options", open=False):
                        src2_file = gr.File(label="Sources as a .txt file", file_types=[".txt", ".md"],
                                            type="filepath", height=90)
                        thr2 = gr.Slider(0.0, 0.95, value=PIPELINE.threshold, step=0.05,
                                         label="Say \"unsure\" below this confidence")
                with gr.Column(scale=7, min_width=380):
                    summary = gr.HTML()
                    page = gr.HTML(EMPTY_ANSWER)
                    gr.HTML(LEGEND)
                    with gr.Accordion("Claims as a table", open=False):
                        table = gr.Dataframe(headers=["#", "claim", "verdict", "confidence"], wrap=True)
            ex2.click(lambda: (ANSWER_EXAMPLE, SOURCE_EXAMPLE), None, [answer_in, src2_in])
            btn2.click(check_answer_ui, [answer_in, src2_in, src2_file, thr2], [summary, page, table])

        with gr.Tab("Check one claim", id="claim"):
            examples = {text[:90] + ("..." if len(text) > 90 else ""): (text, src)
                        for text, src in corpus_examples() + [[EXAMPLE_CLAIM, SOURCE_EXAMPLE]]}
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=300, elem_classes="cg-inputs"):
                    claim_in = gr.Textbox(label="Claim", lines=2, placeholder="One factual sentence")
                    src_in = gr.Textbox(label="Source documents (optional)", lines=5,
                                        placeholder="Leave empty to search the 5,183 scientific abstracts "
                                                    "of SciFact.")
                    pick = gr.Dropdown(list(examples), label="Or pick an example", value=None)
                    btn = gr.Button("Check the claim", variant="primary")
                    with gr.Accordion("More options", open=False):
                        src_file = gr.File(label="Sources as a .txt file", file_types=[".txt", ".md"],
                                           type="filepath", height=90)
                        thr = gr.Slider(0.0, 0.95, value=PIPELINE.threshold, step=0.05,
                                        label="Say \"unsure\" below this confidence")
                        topk = gr.Slider(1, 10, value=PIPELINE.top_k, step=1, label="Passages to read")
                with gr.Column(scale=7, min_width=380):
                    card = gr.HTML(EMPTY_CLAIM)
                    probs = gr.State()  # the probabilities are already drawn in the card
            pick.change(lambda key: examples.get(key, ("", "")), pick, [claim_in, src_in])
            btn.click(verify_claim_ui, [claim_in, src_in, src_file, thr, topk], [card, probs])

        with gr.Tab("Audit log", id="audit"):
            gr.HTML(
                "<p class='cg-foot' style='font-size:14px'>Every verdict is written to <code>logs/audit.jsonl</code> "
                "with the claim, the evidence, the score, the decision, the date and the model version, "
                f"so any decision can be traced later. {html.escape(model_sentence())}</p>"
            )
            refresh = gr.Button("Show the latest verdicts")
            log_table = gr.Dataframe(headers=["time (UTC)", "claim", "verdict", "confidence", "model version"], wrap=True)
            log_file = gr.File(label="Full log (JSONL)")
            refresh.click(audit_ui, [], [log_table, log_file])

    # used to take screenshots of a filled page (CLAIMGUARD_PREVIEW=answer or claim)
    if os.environ.get("CLAIMGUARD_PREVIEW") == "answer":
        demo.load(lambda: (ANSWER_EXAMPLE, SOURCE_EXAMPLE, *check_answer_ui(ANSWER_EXAMPLE, SOURCE_EXAMPLE, None, 0.0)),
                  None, [answer_in, src2_in, summary, page, table])
    elif os.environ.get("CLAIMGUARD_PREVIEW") == "claim":
        demo.load(lambda: (EXAMPLE_CLAIM, SOURCE_EXAMPLE, *verify_claim_ui(EXAMPLE_CLAIM, SOURCE_EXAMPLE, None, 0.0, 3)),
                  None, [claim_in, src_in, card, probs])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=bool(os.environ.get("GRADIO_SHARE")),  # public link, handy on Colab
        **_LAUNCH_STYLE,
    )
