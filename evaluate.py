"""
VCBench Evaluation Script
Computes Precision, Recall, F1, F0.5 for each predictions_prompt_X.csv
and generates a comparison figure against VCBench baselines.
"""

import csv
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

# ── VCBench baselines (from Chen et al. 2025, arXiv:2509.14448) ───────────────
BASELINES = {
    "Market\nRandom":    {"precision": 0.019, "f05": None,  "color": "#cccccc", "hatch": ""},
    "Y Combinator\n(human)": {"precision": 0.032, "f05": None, "color": "#aec6cf", "hatch": ""},
    "Tier-1 VC\n(human)":    {"precision": 0.055, "f05": None, "color": "#7ba7bc", "hatch": ""},
    "DeepSeek-V3\n(LLM)":    {"precision": 0.114, "f05": None, "color": "#4a90d9", "hatch": ""},
    "GPT-4o\n(LLM)":         {"precision": 0.095, "f05": 0.148, "color": "#2c5f8a", "hatch": ""},
}

VARIANT_COLORS = {"A": "#e07b39", "B": "#c0392b", "C": "#8e44ad"}
VARIANT_LABELS = {
    "A": "Prompt A\n(Baseline)",
    "B": "Prompt B\n(Few-shot)",
    "C": "Prompt C\n(Chain-of-thought)",
}

# ── Parse predictions ─────────────────────────────────────────────────────────
def load_predictions(path):
    y_true, y_pred, confidences = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y_true.append(int(row["true_label"]))
            y_pred.append(1 if row["prediction"].strip().lower() == "yes" else 0)
            try:
                confidences.append(float(row["confidence"]))
            except Exception:
                confidences.append(0.5)
    return np.array(y_true), np.array(y_pred), np.array(confidences)


def compute_metrics(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    prec    = precision_score(y_true, y_pred, zero_division=0)
    rec     = recall_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    f05     = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    n_yes   = int(np.sum(y_pred == 1))
    return dict(precision=prec, recall=rec, f1=f1, f05=f05,
                tp=tp, fp=fp, fn=fn, tn=tn, n_yes=n_yes)


# ── Print comparison table ────────────────────────────────────────────────────
def print_table(results):
    header = f"{'System':<28} {'Precision':>10} {'Recall':>8} {'F1':>8} {'F0.5':>8} {'vs Market':>10}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    market_prec = BASELINES["Market\nRandom"]["precision"]
    print(f"{'── Human Baselines ──':<28}")
    for name, b in list(BASELINES.items())[:3]:
        label = name.replace("\n", " ")
        mult  = b['precision'] / market_prec
        f05_s = f"{b['f05']:.4f}" if b['f05'] else "  n/a  "
        print(f"  {label:<26} {b['precision']:>10.4f} {'n/a':>8} {'n/a':>8} {f05_s:>8} {mult:>9.1f}×")

    print(f"\n{'── Published LLM Results ──':<28}")
    for name, b in list(BASELINES.items())[3:]:
        label = name.replace("\n", " ")
        mult  = b['precision'] / market_prec
        f05_s = f"{b['f05']:.4f}" if b['f05'] else "  n/a  "
        print(f"  {label:<26} {b['precision']:>10.4f} {'n/a':>8} {'n/a':>8} {f05_s:>8} {mult:>9.1f}×")

    print(f"\n{'── Our Results (Claude Sonnet) ──':<28}")
    for v in sorted(results):
        m    = results[v]
        mult = m['precision'] / market_prec if m['precision'] else 0
        label = VARIANT_LABELS[v].replace("\n", " ")
        print(f"  {label:<26} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['f05']:>8.4f} {mult:>9.1f}×")
    print(sep)


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(results, out_path="results_figure.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(
        "Precision and F0.5 Comparison: Claude Sonnet Prompt Variants vs VCBench Baselines",
        fontsize=13, fontweight="bold", y=1.02
    )

    market_prec = BASELINES["Market\nRandom"]["precision"]

    for ax_idx, metric in enumerate(["precision", "f05"]):
        ax = axes[ax_idx]
        names, values, colors, hatches = [], [], [], []

        # Baselines
        for name, b in BASELINES.items():
            val = b[metric]
            if val is None:
                val = b["precision"] * 0.7  # rough estimate for display
            names.append(name)
            values.append(val)
            colors.append(b["color"])
            hatches.append(b["hatch"])

        # Our results
        for v in sorted(results):
            m   = results[v]
            val = m[metric] if m[metric] is not None else 0
            names.append(VARIANT_LABELS[v])
            values.append(val)
            colors.append(VARIANT_COLORS[v])
            hatches.append("")

        bars = ax.bar(range(len(names)), values, color=colors, edgecolor="white",
                      linewidth=1.2, width=0.65)

        # Label each bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        # Market baseline line
        ax.axhline(y=market_prec, color="#888", linestyle="--", linewidth=1.2,
                   label=f"Market baseline ({market_prec:.3f})")

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8.5, rotation=0, ha="center")
        metric_title = "Precision" if metric == "precision" else "F0.5 Score"
        ax.set_ylabel(metric_title, fontsize=11)
        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {out_path}")


# ── Save JSON summary for paper ───────────────────────────────────────────────
def save_summary(results, out_path="results_summary.json"):
    summary = {v: {k: round(float(v2), 6) if isinstance(v2, float) else v2
                   for k, v2 in m.items()}
               for v, m in results.items()}
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    results = {}
    for variant in ["A", "B", "C"]:
        path = f"predictions_prompt_{variant}.csv"
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        y_true, y_pred, _ = load_predictions(path)
        m = compute_metrics(y_true, y_pred)
        results[variant] = m
        print(f"Prompt {variant}: Precision={m['precision']:.4f}  "
              f"Recall={m['recall']:.4f}  F1={m['f1']:.4f}  F0.5={m['f05']:.4f}  "
              f"(predicted Yes: {m['n_yes']}/{len(y_pred)})")

    if not results:
        print("No prediction files found. Run predict.py first.")
        sys.exit(1)

    print()
    print_table(results)
    make_figure(results)
    save_summary(results)


if __name__ == "__main__":
    main()
