#!/usr/bin/env python3
"""
quality_report.py - Analysis and paper-ready tables for TraceGen quality metrics.

Reads quality_metrics.json (produced by compute_metrics.py), correlates with
Stage 4 resolve rate, and outputs paper-ready tables + summary report.

Usage:
    python scripts/quality/quality_report.py \
        --metrics-json ../tracegen-outputs/quality_metrics.json \
        --output ../tracegen-outputs/quality_report.json
"""

import argparse
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return s[n // 2]


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def point_biserial(x: List[float], y: List[int]) -> float:
    """Point-biserial correlation between continuous x and binary y."""
    n = len(x)
    if n < 3:
        return 0.0
    x1 = [x[i] for i in range(n) if y[i] == 1]
    x0 = [x[i] for i in range(n) if y[i] == 0]
    if not x1 or not x0:
        return 0.0
    m1, m0 = mean(x1), mean(x0)
    s = std(x)
    if s == 0:
        return 0.0
    n1, n0 = len(x1), len(x0)
    return (m1 - m0) / s * math.sqrt(n1 * n0 / (n * n))


# ---------------------------------------------------------------------------
# LaTeX table generators
# ---------------------------------------------------------------------------

def generate_main_results_table(metrics: List[Dict], repos: List[str]) -> str:
    """Generate Table 1: Main experiment results (LaTeX)."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{TraceGen synthesis results across 4 repositories.}")
    lines.append(r"\label{tab:main-results}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Repository & Seeds & Valid & Rate & Q$_{\text{avg}}$ & Tier A & Tier B \\")
    lines.append(r"\midrule")

    total_valid = 0
    total_a = 0
    total_b = 0

    for repo in repos:
        rm = [m for m in metrics if m["repo"] == repo]
        n = len(rm)
        total_valid += n
        q_avg = mean([m["composite_score"] for m in rm])
        seeds = len(set(m["seed_id"] for m in rm))
        tiers = Counter(m["quality_tier"] for m in rm)
        a_count = tiers.get("A", 0)
        b_count = tiers.get("B", 0)
        total_a += a_count
        total_b += b_count
        short = repo.split("/")[-1]
        lines.append(f"  {short} & {seeds} & {n} & -- & {q_avg:.3f} & {a_count} & {b_count} \\\\")

    lines.append(r"\midrule")
    total_q = mean([m["composite_score"] for m in metrics])
    total_seeds = len(set(m["seed_id"] for m in metrics))
    lines.append(f"  Total & {total_seeds} & {total_valid} & -- & {total_q:.3f} & {total_a} & {total_b} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_quality_dimension_table(metrics: List[Dict]) -> str:
    """Generate Table: Quality dimensions breakdown (LaTeX)."""
    dims = [
        ("Failure Mode", "failure_mode_score"),
        ("Test Impact", "impact_ratio_score"),
        ("Patch Naturalness", "patch_naturalness_score"),
        ("PS Informativeness", "ps_info_score"),
        ("BM25 Retrievability", "bm25_score"),
        ("Chain Alignment", "chain_alignment"),
    ]

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Quality dimension scores across all 228 valid synthetic bugs.}")
    lines.append(r"\label{tab:quality-dims}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Dimension & Mean & Median & Std \\")
    lines.append(r"\midrule")

    for name, key in dims:
        vals = [m.get(key, 0.0) for m in metrics]
        lines.append(f"  {name} & {mean(vals):.3f} & {median(vals):.3f} & {std(vals):.3f} \\\\")

    lines.append(r"\midrule")
    q_vals = [m["composite_score"] for m in metrics]
    lines.append(f"  Composite Q & {mean(q_vals):.3f} & {median(q_vals):.3f} & {std(q_vals):.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def generate_fix_intent_table(metrics: List[Dict]) -> str:
    """Generate Table: Quality by fix intent (LaTeX)."""
    by_intent = defaultdict(list)
    for m in metrics:
        by_intent[m.get("fix_intent", "Unknown")].append(m)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Quality metrics by fix intent category.}")
    lines.append(r"\label{tab:fix-intent}")
    lines.append(r"\begin{tabular}{lrccc}")
    lines.append(r"\toprule")
    lines.append(r"Fix Intent & $n$ & Q$_{\text{avg}}$ & Nat. & Impact \\")
    lines.append(r"\midrule")

    for intent in sorted(by_intent.keys()):
        entries = by_intent[intent]
        n = len(entries)
        q = mean([e["composite_score"] for e in entries])
        nat = mean([e.get("patch_naturalness_score", 0) for e in entries])
        imp = mean([e.get("impact_ratio_score", 0) for e in entries])
        display = intent.replace("_", " ")
        lines.append(f"  {display} & {n} & {q:.3f} & {nat:.2f} & {imp:.2f} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quality report and analysis")
    parser.add_argument("--metrics-json", required=True, help="quality_metrics.json")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    metrics_path = Path(args.metrics_json)
    output_path = Path(args.output) if args.output else metrics_path.parent / "quality_report.json"

    with open(metrics_path) as f:
        metrics = json.load(f)
    n = len(metrics)
    print(f"Loaded {n} entries from {metrics_path}")

    repos = sorted(set(m["repo"] for m in metrics))
    report = {}

    # 1. Overall stats
    q_scores = [m["composite_score"] for m in metrics]
    tiers = Counter(m["quality_tier"] for m in metrics)
    report["overall"] = {
        "n": n,
        "repos": len(repos),
        "composite_score": {
            "mean": round(mean(q_scores), 4),
            "median": round(median(q_scores), 4),
            "std": round(std(q_scores), 4),
        },
        "tiers": {t: tiers.get(t, 0) for t in "ABCD"},
    }

    # 2. Per-repo
    repo_stats = {}
    for repo in repos:
        rm = [m for m in metrics if m["repo"] == repo]
        rq = [m["composite_score"] for m in rm]
        rt = Counter(m["quality_tier"] for m in rm)
        fm = Counter(m["failure_mode"] for m in rm)
        repo_stats[repo] = {
            "n": len(rm),
            "seeds": len(set(m["seed_id"] for m in rm)),
            "composite_mean": round(mean(rq), 3),
            "tiers": {t: rt.get(t, 0) for t in "ABCD"},
            "top_failure_modes": fm.most_common(3),
        }
    report["per_repo"] = repo_stats

    # 3. Per-dimension stats
    dims = {
        "failure_mode_score": "Failure Mode",
        "impact_ratio_score": "Test Impact",
        "patch_naturalness_score": "Patch Naturalness",
        "ps_info_score": "PS Informativeness",
        "chain_alignment": "Chain Alignment",
    }
    dim_stats = {}
    for key, name in dims.items():
        vals = [m.get(key, 0.0) for m in metrics]
        dim_stats[name] = {
            "mean": round(mean(vals), 3),
            "median": round(median(vals), 3),
            "std": round(std(vals), 3),
        }
    report["dimensions"] = dim_stats

    # 4. Fix Intent breakdown
    by_intent = defaultdict(list)
    for m in metrics:
        by_intent[m.get("fix_intent", "Unknown")].append(m)
    intent_stats = {}
    for intent, entries in sorted(by_intent.items()):
        eq = [e["composite_score"] for e in entries]
        intent_stats[intent] = {
            "n": len(entries),
            "q_mean": round(mean(eq), 3),
            "q_std": round(std(eq), 3),
        }
    report["fix_intent"] = intent_stats

    # 5. Failure mode distribution
    fm_all = Counter(m["failure_mode"] for m in metrics)
    report["failure_modes"] = {fm: cnt for fm, cnt in fm_all.most_common()}

    # 7. LaTeX tables
    latex_tables = {
        "main_results": generate_main_results_table(metrics, repos),
        "quality_dimensions": generate_quality_dimension_table(metrics),
        "fix_intent": generate_fix_intent_table(metrics),
    }
    report["latex_tables"] = latex_tables

    # Write output
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Quality Report: {n} bugs across {len(repos)} repos")
    print(f"{'='*60}")

    print(f"\nComposite Q: mean={report['overall']['composite_score']['mean']:.3f}, "
          f"median={report['overall']['composite_score']['median']:.3f}, "
          f"std={report['overall']['composite_score']['std']:.3f}")

    print(f"\nTier Distribution:")
    for tier in "ABCD":
        cnt = tiers.get(tier, 0)
        print(f"  Tier {tier}: {cnt:>3d} ({cnt/n*100:5.1f}%)")

    print(f"\nPer-Repo:")
    for repo in repos:
        rs = repo_stats[repo]
        short = repo.split("/")[-1]
        print(f"  {short:<15s} n={rs['n']:>3d}  seeds={rs['seeds']:>3d}  "
              f"Q={rs['composite_mean']:.3f}  "
              f"A={rs['tiers']['A']} B={rs['tiers']['B']} C={rs['tiers']['C']}")

    print(f"\nDimension Scores:")
    for name, stats in dim_stats.items():
        print(f"  {name:<22s} mean={stats['mean']:.3f}  std={stats['std']:.3f}")

    print(f"\nFix Intent Quality:")
    for intent in sorted(intent_stats.keys(), key=lambda k: -intent_stats[k]["q_mean"]):
        s = intent_stats[intent]
        print(f"  {intent:<25s} n={s['n']:>3d}  Q={s['q_mean']:.3f}")

    print(f"\nFailure Mode Distribution:")
    for fm, cnt in fm_all.most_common():
        print(f"  {fm:<25s} {cnt:>3d} ({cnt/n*100:5.1f}%)")

    # Print LaTeX tables
    print(f"\n{'='*60}")
    print("LaTeX Tables (copy to paper)")
    print(f"{'='*60}")
    for name, table in latex_tables.items():
        print(f"\n% --- {name} ---")
        print(table)

    print(f"\n→ Report saved to {output_path}")


if __name__ == "__main__":
    main()
