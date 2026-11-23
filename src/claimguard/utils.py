"""Small helpers used everywhere: seeds, config loading, json I/O, timing."""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Fix every random generator we use so runs can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional (the offline/lexical mode does not need it)
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def deep_update(base: dict, new: dict) -> dict:
    """Merge `new` into `base` recursively (new values win)."""
    out = dict(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Read a YAML config. A config can say `base: other.yaml` to inherit from it."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        # the base path is relative to the config file itself
        parent = load_config(path.parent / cfg.pop("base"))
        cfg = deep_update(parent, cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a .jsonl file into a list of dicts (skips empty lines)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """Write a list of dicts as .jsonl (creates the folder if needed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str | Path, obj: Any) -> None:
    """Save any json-able object, converting numpy types on the way."""

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"cannot serialise {type(o)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def timer(store: dict, key: str) -> Iterator[None]:
    """`with timer(d, "retrieval"):` adds the elapsed milliseconds to d["retrieval"]."""
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = store.get(key, 0.0) + (time.perf_counter() - start) * 1000.0


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax in numpy."""
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
