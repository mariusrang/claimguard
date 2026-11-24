"""Speed and size measurements + INT8 dynamic quantization.

Why it matters: a verifier that runs after every LLM answer has to be cheap.
Dynamic quantization stores the Linear layers in int8 instead of float32, so the
model is smaller and faster on CPU, usually for a small loss in accuracy.
I measure all three (size, latency, accuracy) instead of assuming.
"""

from __future__ import annotations

import io
import re
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


# feed-forward layers of a transformer block: "...layer.3.intermediate.dense" and "...layer.3.output.dense"
# (NOT "...layer.3.attention.output.dense", which belongs to the attention)
_FFN_LAYER = re.compile(r"\.layer\.\d+\.(intermediate|output)\.dense$")


def quantize_dynamic_int8(model, only_feed_forward: bool = False):
    """Return an INT8 copy of the model (Linear layers, CPU only).

    only_feed_forward=False: every Linear layer (the classic recipe).
    only_feed_forward=True : only the feed-forward layers of each block. On Kaggle the
    full recipe dropped the accuracy of the verifier from 0.856 to 0.407 (gold evidence):
    DeBERTa's attention projections are too sensitive. The feed-forward layers hold
    about 2/3 of the encoder weights, so most of the gain should stay.
    """
    import copy

    import torch

    # the function moved between torch versions, so look in both places
    quant = getattr(getattr(torch, "ao", torch), "quantization", torch.quantization)
    model = copy.deepcopy(model).cpu()  # never modify the model we were given
    if not only_feed_forward:
        return quant.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    names = {n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and _FFN_LAYER.search(n)}
    if not names:
        raise ValueError("no feed-forward layer found (unexpected model architecture)")
    spec = {name: quant.default_dynamic_qconfig for name in names}
    return quant.quantize_dynamic(model, spec, dtype=torch.qint8)


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
