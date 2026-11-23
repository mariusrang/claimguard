"""MCP server: lets Claude (Desktop or Code) call ClaimGuard as a tool.

Run it with:   python -m claimguard.mcp_server
Config file:   env variable CLAIMGUARD_CONFIG (default: configs/default.yaml)

Tools:
- verify_claim      one claim against given sources (or the indexed corpus)
- verify_answer     a whole answer, split into claims
- verify_citations  check that cited passages really support their sentences
- get_audit_log     last verdicts, for traceability

Important: ClaimGuard checks if a claim is supported BY THE DOCUMENTS, not if
it is true in the real world. The tool descriptions say it so the assistant
does not oversell the result.
"""

from __future__ import annotations

import os

from claimguard.pipeline import ClaimGuard, ClaimResult, InvalidInputError, build_pipeline
from claimguard.utils import load_config

try:  # mcp >= 2.0 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

server = _Server("claimguard")
_pipeline: ClaimGuard | None = None


def get_pipeline() -> ClaimGuard:
    """Load the pipeline once, the first time a tool is called (models are slow to load)."""
    global _pipeline
    if _pipeline is None:
        cfg = load_config(os.environ.get("CLAIMGUARD_CONFIG", "configs/default.yaml"))
        _pipeline = build_pipeline(cfg)
    return _pipeline


def set_pipeline(pipeline: ClaimGuard) -> None:
    """Used by the tests to plug a small offline pipeline."""
    global _pipeline
    _pipeline = pipeline


def _short(result: ClaimResult) -> dict:
    """Keep the fields that are useful for an assistant (the full record is in the audit log)."""
    best = max(result.evidence, key=lambda e: max(e.probs[0], e.probs[1]), default=None)
    return {
        "claim": result.claim,
        "decision": result.decision,
        "label": result.label,
        "confidence": round(result.confidence, 3),
        "evidence": best.text if best else None,
        "evidence_source": best.source if best else None,
    }


@server.tool()
def verify_claim(claim: str, sources: list[str] | None = None) -> dict:
    """Check if ONE claim is supported by documents.

    Give the documents in `sources` (list of texts). Without sources, the claim is
    checked against the indexed corpus. Returns SUPPORTED, REFUTED, NEI (not enough
    information) or UNCERTAIN (confidence too low), with the best evidence passage.
    This checks faithfulness to the documents, not truth in the real world.
    """
    try:
        return _short(get_pipeline().verify_claim(claim, sources))
    except InvalidInputError as err:
        return {"error": str(err)}


@server.tool()
def verify_answer(answer: str, sources: list[str] | None = None) -> dict:
    """Check every factual sentence of an answer against documents.

    Use it after writing an answer based on documents: it returns one verdict per
    claim plus a summary. Claims that are not SUPPORTED should be fixed or flagged.
    """
    try:
        results = get_pipeline().verify_answer(answer, sources)
    except InvalidInputError as err:
        return {"error": str(err)}
    verdicts = [_short(r) for r in results]
    n_ok = sum(v["decision"] == "SUPPORTED" for v in verdicts)
    return {
        "n_claims": len(verdicts),
        "n_supported": n_ok,
        "n_flagged": len(verdicts) - n_ok,
        "claims": verdicts,
    }


@server.tool()
def verify_citations(items: list[dict]) -> dict:
    """Check that each cited passage really supports its sentence.

    items = [{"sentence": "...", "cited_text": "..."}]. A sentence can have a
    citation and still misread it, this tool catches that.
    """
    try:
        results = get_pipeline().verify_citations(items)
    except InvalidInputError as err:
        return {"error": str(err)}
    return {"citations": [_short(r) for r in results]}


@server.tool()
def get_audit_log(last_n: int = 20) -> dict:
    """Return the last verdicts of the audit log (claim, evidence, score, decision, model version)."""
    pipeline = get_pipeline()
    if pipeline.audit is None:
        return {"error": "Audit logging is disabled in the config."}
    return {"records": pipeline.audit.tail(max(1, min(int(last_n), 200)))}


def main() -> None:
    server.run()  # stdio transport, this is what Claude Desktop / Claude Code expect


if __name__ == "__main__":
    main()
