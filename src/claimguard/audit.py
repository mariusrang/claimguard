"""Audit log: every verdict is appended to a JSONL file.

In a regulated company (a bank for example) "the tool said it was fine" is not
enough: they want to know which claim, which evidence, which score, which
decision, when, and with which model version. One JSON object per line makes
it easy to export or to load in pandas later.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from claimguard.utils import read_jsonl


class AuditLogger:
    """Append-only JSONL logger (thread-safe because Gradio and MCP can be concurrent)."""

    def __init__(self, path: str | Path, model_version: str = "") -> None:
        self.path = Path(path)
        self.model_version = model_version
        self._lock = threading.Lock()

    def log(self, result: dict, mode: str = "verify_claim") -> dict:
        """Write one verdict. `result` is ClaimResult.to_dict()."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": mode,
            "model_version": result.get("model_version") or self.model_version,
            "claim": result["claim"],
            "label": result["label"],
            "confidence": round(float(result["confidence"]), 4),
            "decision": result["decision"],
            "threshold": result.get("threshold", 0.0),
            "top_k": result.get("top_k", 0),
            "note": result.get("note", ""),
            "evidence": [
                {
                    "source": e["source"],
                    "text": e["text"],
                    "probs": [round(float(p), 4) for p in e["probs"]],
                }
                for e in result.get("evidence", [])
            ],
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def tail(self, n: int = 20) -> list[dict]:
        """Last n records (empty list if nothing was logged yet)."""
        if not self.path.exists():
            return []
        return read_jsonl(self.path)[-n:]
