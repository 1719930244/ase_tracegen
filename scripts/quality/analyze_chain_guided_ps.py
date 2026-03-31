#!/usr/bin/env python3
"""
Analyze chain-guided PS results for paper tables.

Produces:
1. Three-level PS metrics comparison (Table: PS difficulty control)
2. Correlation analysis (chain signal → resolve rate)
3. Per-fix-intent breakdown
4. Comparison with original PS and Saving SWE-Bench baseline
"""

import json
import math
from collections import Counter
from pathlib import Path


def spearman_rank(x, y):
    """Simple Spearman rank correlation (no scipy dependency)."""
    n = len(x)
    if n < 3:
        return 0.0

    def _ranks(vals):
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = _ranks(x), _ranks(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1 - (6 * d_sq) / (n * (n * n - 1))


def main():
    outputs_root = Path("../tracegen-outputs")

    # Load data
    with open(outputs_root / "chain_guided_ps.json") as f:
        data = json.load(f)

    with open(outputs_root / "info_theory_pilot.json") as f:
        pilot = json.load(f)
    pilot_map = {p["instance_id"]: p for p in pilot}

    n = len(data)
    resolved = [d for d in data if d["stage4_resolved"]]
    unresolved = [d for d in data if not d["stage4_resolved"]]
    resolved_binary = [1 if d["stage4_resolved"] else 0 for d in data]

    def avg(lst, key):
        vals = [d[key] for d in lst if key in d]
        return sum(vals) / len(vals) if vals else 0

    # =========================================================================
    # Table 1: PS Information Level Comparison
    # =========================================================================
    print("=" * 80)
    print("TABLE 1: Chain-Guided PS Information Levels (Django 76 valid bugs)")
    print("=" * 80)
    print()
    print(f"{'Level':<12s} {'Tokens':>8s} {'ID Dens':>8s} {'H(PS)':>8s} "
          f"{'ChainSig':>9s} {'ρ(Res)':>8s} {'ρ_chain':>8s}")
    print("-" * 70)

    for lvl in ("L1", "L2", "L3"):
        tokens = avg(data, f"ps_{lvl}_tokens")
        id_d = avg(data, f"ps_{lvl}_identifier_density")
        h = avg(data, f"ps_{lvl}_h_ps")
        cs = avg(data, f"ps_{lvl}_chain_signal")

        vals_id = [d[f"ps_{lvl}_identifier_density"] for d in data]
        vals_cs = [d[f"ps_{lvl}_chain_signal"] for d in data]
        rho_id = spearman_rank(vals_id, resolved_binary)
        rho_cs = spearman_rank(vals_cs, resolved_binary)

        print(f"{lvl:<12s} {tokens:8.1f} {id_d:8.3f} {h:8.3f} "
              f"{cs:9.3f} {rho_id:+8.3f} {rho_cs:+8.3f}")

    # Original PS
    orig_tokens = avg(data, "original_ps_tokens")
    orig_id = [pilot_map.get(d["instance_id"], {}).get("ps_identifier_density", 0) for d in data]
    orig_h = [pilot_map.get(d["instance_id"], {}).get("h_ps", 0) for d in data]
    rho_orig = spearman_rank(orig_id, resolved_binary)

    print(f"{'Original':<12s} {orig_tokens:8.1f} "
          f"{sum(orig_id)/len(orig_id):8.3f} "
          f"{sum(orig_h)/len(orig_h):8.3f} "
          f"{'---':>9s} {rho_orig:+8.3f} {'---':>8s}")

    # =========================================================================
    # Table 2: Resolved vs Unresolved by Level
    # =========================================================================
    print()
    print("=" * 80)
    print("TABLE 2: Resolved vs Unresolved (identifier density by level)")
    print("=" * 80)
    print()
    print(f"{'Level':<12s} {'Resolved':>10s} {'Unresolved':>12s} {'Δ':>8s} {'Effect':>8s}")
    print("-" * 55)

    for lvl in ("L1", "L2", "L3"):
        key = f"ps_{lvl}_identifier_density"
        r = avg(resolved, key)
        u = avg(unresolved, key)
        delta = r - u
        # Cohen's d approximation
        vals_r = [d[key] for d in resolved]
        vals_u = [d[key] for d in unresolved]
        all_vals = vals_r + vals_u
        sd = (sum((v - sum(all_vals) / len(all_vals)) ** 2 for v in all_vals) / len(all_vals)) ** 0.5
        cohen_d = delta / sd if sd > 0 else 0
        print(f"{lvl:<12s} {r:10.3f} {u:12.3f} {delta:+8.3f} {cohen_d:+8.2f}d")

    r_orig = sum(pilot_map.get(d["instance_id"], {}).get("ps_identifier_density", 0)
                 for d in resolved) / len(resolved)
    u_orig = sum(pilot_map.get(d["instance_id"], {}).get("ps_identifier_density", 0)
                 for d in unresolved) / len(unresolved)
    print(f"{'Original':<12s} {r_orig:10.3f} {u_orig:12.3f} {r_orig - u_orig:+8.3f}")

    # =========================================================================
    # Table 3: Per Fix Intent Breakdown
    # =========================================================================
    print()
    print("=" * 80)
    print("TABLE 3: Per Fix Intent (L2 identifier density)")
    print("=" * 80)
    print()
    intents = sorted(set(d["fix_intent"] for d in data))
    print(f"{'Fix Intent':<28s} {'N':>3s} {'Res%':>6s} {'L1_id':>7s} {'L2_id':>7s} "
          f"{'L3_id':>7s} {'Orig_id':>8s}")
    print("-" * 72)

    for intent in intents:
        subset = [d for d in data if d["fix_intent"] == intent]
        nn = len(subset)
        res_pct = sum(1 for d in subset if d["stage4_resolved"]) / nn * 100
        l1 = avg(subset, "ps_L1_identifier_density")
        l2 = avg(subset, "ps_L2_identifier_density")
        l3 = avg(subset, "ps_L3_identifier_density")
        orig = sum(pilot_map.get(d["instance_id"], {}).get("ps_identifier_density", 0)
                   for d in subset) / nn
        print(f"{intent:<28s} {nn:3d} {res_pct:5.1f}% {l1:7.3f} {l2:7.3f} "
              f"{l3:7.3f} {orig:8.3f}")

    # =========================================================================
    # Comparison with Saving SWE-Bench baseline (expected impact)
    # =========================================================================
    print()
    print("=" * 80)
    print("COMPARISON: TraceGen Chain-Guided vs Saving SWE-Bench (Garg et al. 2025)")
    print("=" * 80)
    print()
    print("Saving SWE-Bench: random template mutation → -36.5% resolve rate")
    print("TraceGen chain-guided: structured chain-level control")
    print()
    print("Expected resolve rate impact (based on identifier density correlation):")
    print(f"  Original PS (ρ=+0.336): baseline resolve rate = {len(resolved)/n*100:.1f}%")
    print(f"  L3 (easy, ρ=+0.293):    expected ~similar to original")
    print(f"  L2 (medium, ρ=+0.162):  expected moderate decrease")
    print(f"  L1 (hard, ρ=-0.055):    expected significant decrease")
    print()
    print("Key advantage: TraceGen provides PRINCIPLED difficulty control")
    print("  vs. Saving SWE-Bench's RANDOM template mutation")


if __name__ == "__main__":
    main()
