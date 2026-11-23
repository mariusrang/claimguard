"""Figures of the report, made from the JSON results.

    python scripts/make_figures.py --results results

Creates (when the needed file exists) in results/figures/:
- reliability.png    reliability diagram before / after temperature scaling
- risk_coverage.png  risk-coverage curve (abstention)
- ablation_k.png     macro-F1 against the number of evidence passages
- confusion.png      confusion matrix of the main system
- errors.png         error categories (from the annotated CSV)
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no screen needed (Colab / CI)
import matplotlib.pyplot as plt
import numpy as np

from claimguard import LABELS
from claimguard.utils import load_json

BLUE, ORANGE, GREY = "#2a6fbb", "#e07b39", "#888888"


def save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("saved", path)


def fig_calibration(calib: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), sharey=True)
    for ax, name in zip(axes, ("before", "after")):
        bins = calib[name]["reliability_bins"]
        ax.plot([0, 1], [0, 1], "--", color=GREY, linewidth=1, label="perfect calibration")
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", color=BLUE, label="model")
        ax.set_title(f"{name} temperature scaling (ECE = {calib[name]['ece']:.3f})", fontsize=10)
        ax.set_xlabel("confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, out / "reliability.png")

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for name, color in (("before", GREY), ("after", BLUE)):
        curve = calib[name]["risk_coverage"]
        ax.plot(curve["coverage"], curve["risk"], color=color, label=f"{name} (AURC = {calib[name]['aurc']:.3f})")
    ax.set_xlabel("coverage (share of claims answered)")
    ax.set_ylabel("risk (error rate on answered claims)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, out / "risk_coverage.png")


def fig_pipeline(pipe: dict, out: Path) -> None:
    rows = pipe["A3_number_of_evidence"]
    ks = [m["k"] for m in rows]
    f1 = np.array([m["macro_f1"]["value"] for m in rows])
    low = np.array([m["macro_f1"]["low"] for m in rows])
    high = np.array([m["macro_f1"]["high"] for m in rows])
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.errorbar(ks, f1, yerr=[f1 - low, high - f1], fmt="o-", color=BLUE, capsize=3)
    ax.set_xticks(ks)
    ax.set_xlabel("number of evidence passages (k)")
    ax.set_ylabel("macro-F1 (95% CI)")
    save(fig, out / "ablation_k.png")

    cm = np.array(pipe["main"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    save(fig, out / "confusion.png")


def fig_errors(csv_path: Path, out: Path) -> None:
    with open(csv_path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("category")]
    if not rows:
        print("no annotated category yet in", csv_path)
        return
    counts = Counter(r["category"] for r in rows).most_common()
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(counts) + 1.2))
    names = [c for c, _ in counts][::-1]
    values = [n for _, n in counts][::-1]
    ax.barh(names, values, color=ORANGE)
    for y, v in enumerate(values):
        ax.text(v + 0.3, y, f"{v} ({v / len(rows):.0%})", va="center", fontsize=8)
    ax.set_xlabel(f"number of errors (n = {len(rows)})")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, out / "errors.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    res = Path(args.results)
    out = res / "figures"
    out.mkdir(parents=True, exist_ok=True)
    if (res / "calibration.json").exists():
        fig_calibration(load_json(res / "calibration.json"), out)
    if (res / "pipeline.json").exists():
        fig_pipeline(load_json(res / "pipeline.json"), out)
    annotated = res / "errors_annotated.csv"
    if annotated.exists():
        fig_errors(annotated, out)


if __name__ == "__main__":
    main()
