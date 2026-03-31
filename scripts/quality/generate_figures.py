#!/usr/bin/env python3
"""
generate_figures.py - Publication-ready figures for TraceGen paper.

Generates:
  - Figure 1: Quality score distribution (histogram + tier bars)
  - Figure 2: Per-repo radar chart of quality dimensions
  - Figure 3: Fix intent quality heatmap
  - Figure 4: Failure mode distribution

Usage:
    python scripts/quality/generate_figures.py \
        --metrics-json ../tracegen-outputs/quality_metrics.json \
        --output-dir ../tracegen-outputs/figures/
"""

import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Paper-quality defaults
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

REPO_LABELS = {
    "django/django": "Django",
    "pytest-dev/pytest": "Pytest",
    "scikit-learn/scikit-learn": "Scikit-learn",
    "sympy/sympy": "SymPy",
}

REPO_COLORS = {
    "django/django": "#1f77b4",
    "pytest-dev/pytest": "#ff7f0e",
    "scikit-learn/scikit-learn": "#2ca02c",
    "sympy/sympy": "#d62728",
}

TIER_COLORS = {"A": "#2ecc71", "B": "#3498db", "C": "#f39c12", "D": "#e74c3c"}


def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0


def fig1_quality_distribution(metrics, output_dir):
    """Figure 1: Composite quality score distribution with tier coloring."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [2, 1]})

    # Left: histogram colored by tier
    scores = [m["composite_score"] for m in metrics]
    bins = np.arange(0.3, 0.95, 0.025)

    for tier, color in TIER_COLORS.items():
        tier_scores = [s for s, m in zip(scores, metrics) if m["quality_tier"] == tier]
        if tier_scores:
            ax1.hist(tier_scores, bins=bins, color=color, alpha=0.8,
                     label=f"Tier {tier} (n={len(tier_scores)})", edgecolor="white", linewidth=0.5)

    ax1.set_xlabel("Composite Quality Score")
    ax1.set_ylabel("Count")
    ax1.set_title("(a) Quality Score Distribution")
    ax1.legend(loc="upper left")
    ax1.axvline(x=mean(scores), color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.text(mean(scores) + 0.01, ax1.get_ylim()[1] * 0.9,
             f"mean={mean(scores):.3f}", fontsize=9)

    # Right: per-repo box plot
    repos = sorted(set(m["repo"] for m in metrics))
    repo_scores = [[m["composite_score"] for m in metrics if m["repo"] == r] for r in repos]
    repo_labels = [REPO_LABELS.get(r, r.split("/")[-1]) for r in repos]
    repo_colors = [REPO_COLORS.get(r, "#999999") for r in repos]

    bp = ax2.boxplot(repo_scores, tick_labels=repo_labels, patch_artist=True, vert=True)
    for patch, color in zip(bp["boxes"], repo_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel("Composite Quality Score")
    ax2.set_title("(b) Per-Repository")
    ax2.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    out = output_dir / "fig1_quality_distribution.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  Figure 1 → {out}")


def fig2_radar_chart(metrics, output_dir):
    """Figure 2: Per-repo radar chart of quality dimensions."""
    dims = [
        ("Failure Mode", "failure_mode_score"),
        ("Test Impact", "impact_ratio_score"),
        ("Patch Nat.", "patch_naturalness_score"),
        ("PS Info.", "ps_info_score"),
        ("Chain Align.", "chain_alignment"),
    ]
    dim_names = [d[0] for d in dims]
    dim_keys = [d[1] for d in dims]
    N = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    repos = sorted(set(m["repo"] for m in metrics))
    for repo in repos:
        rm = [m for m in metrics if m["repo"] == repo]
        values = [mean([m.get(k, 0.0) for m in rm]) for k in dim_keys]
        values += values[:1]
        label = REPO_LABELS.get(repo, repo.split("/")[-1])
        color = REPO_COLORS.get(repo, "#999999")
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_names)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("Quality Dimensions by Repository", pad=20)

    out = output_dir / "fig2_radar_chart.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  Figure 2 → {out}")


def fig3_failure_mode(metrics, output_dir):
    """Figure 3: Failure mode distribution (horizontal bar)."""
    fm_counts = Counter(m["failure_mode"] for m in metrics)
    # Group rare modes into "Other"
    threshold = 3
    main_modes = {k: v for k, v in fm_counts.items() if v >= threshold}
    other = sum(v for k, v in fm_counts.items() if v < threshold)
    if other > 0:
        main_modes["Other"] = other

    sorted_modes = sorted(main_modes.items(), key=lambda x: x[1])
    labels = [m[0] for m in sorted_modes]
    counts = [m[1] for m in sorted_modes]
    n = len(metrics)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, counts, color="#3498db", edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{count} ({count/n*100:.1f}%)", va="center", fontsize=9)

    ax.set_xlabel("Count")
    ax.set_title("Failure Mode Distribution")
    ax.set_xlim(0, max(counts) * 1.25)

    plt.tight_layout()
    out = output_dir / "fig3_failure_modes.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  Figure 3 → {out}")


def fig4_fix_intent_quality(metrics, output_dir):
    """Figure 4: Fix intent quality comparison (grouped bar)."""
    by_intent = defaultdict(list)
    for m in metrics:
        by_intent[m.get("fix_intent", "Unknown")].append(m)

    intents = sorted(by_intent.keys())
    dims = [
        ("Failure Mode", "failure_mode_score"),
        ("Test Impact", "impact_ratio_score"),
        ("Patch Nat.", "patch_naturalness_score"),
        ("PS Info.", "ps_info_score"),
    ]

    x = np.arange(len(intents))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (name, key) in enumerate(dims):
        vals = [mean([m.get(key, 0.0) for m in by_intent[intent]]) for intent in intents]
        ax.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)

    ax.set_xlabel("Fix Intent")
    ax.set_ylabel("Score")
    ax.set_title("Quality Dimensions by Fix Intent")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([i.replace("_", "\n") for i in intents], fontsize=8)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.15)

    # Add counts
    for i, intent in enumerate(intents):
        n = len(by_intent[intent])
        ax.text(i + width * 1.5, 1.05, f"n={n}", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    out = output_dir / "fig4_fix_intent_quality.pdf"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"))
    plt.close()
    print(f"  Figure 4 → {out}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    output_dir = Path(args.output_dir) if args.output_dir else metrics_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_path) as f:
        metrics = json.load(f)
    print(f"Loaded {len(metrics)} entries")
    print(f"Output dir: {output_dir}")

    fig1_quality_distribution(metrics, output_dir)
    fig2_radar_chart(metrics, output_dir)
    fig3_failure_mode(metrics, output_dir)
    fig4_fix_intent_quality(metrics, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
