"""Speed and size measurements + INT8 dynamic quantization.

Why it matters: a verifier that runs after every LLM answer has to be cheap.
Dynamic quantization stores the Linear layers in int8 instead of float32, so the
model is smaller and faster on CPU, usually for a small loss in accuracy.
I measure all three (size, latency, accuracy) instead of assuming.
"""

from __future__ import annotations

import io
import time
from collections.abc import Callable

import numpy as np


def count_parameters(model) -> int:
    """Number of parameters of a torch model."""
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model) -> float:
    """Size of the saved weights in MB (works for quantized models too)."""
    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes / 1e6


def quantize_dynamic_int8(model):
    """Return an INT8 copy of the model (Linear layers only, CPU only)."""
    import torch

    # the function moved between torch versions, so look in both places
    quantize = getattr(getattr(torch, "ao", torch), "quantization", torch.quantization).quantize_dynamic
    return quantize(model.cpu(), {torch.nn.Linear}, dtype=torch.qint8)


def benchmark(fn: Callable[[], object], n_warmup: int = 3, n_runs: int = 30) -> dict:
    """Call fn() many times and report the latency percentiles in milliseconds."""
    for _ in range(n_warmup):
        fn()  # first calls are slower (caches, lazy init), don't count them
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000.0)
    arr = np.array(times)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "mean_ms": float(arr.mean()),
        "calls_per_second": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
        "n_runs": n_runs,
    }
