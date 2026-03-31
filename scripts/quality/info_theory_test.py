#!/usr/bin/env python3
"""
info_theory_test.py - Information-theoretic quality metrics pilot test.

Computes on Django 76 valid bugs:
  1. H(PS)           — Shannon entropy of PS word distribution
  2. PS_vocab_rich    — unique words / total words
  3. PS_identifier    — code identifiers (from chain/patch) found in PS
  4. Patch_entropy    — entropy of changed code tokens
  5. MI_proxy         — Jaccard(PS_tokens, patch_tokens) as MI(PS; F_bug) proxy
  6. PS_chain_signal  — chain file/function names appearing in PS

Then correlates with stage4_resolved and composite_score.
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def shannon_entropy(text: str) -> float:
    """Word-level Shannon entropy."""
    words = text.lower().split()
    if len(words) <= 1:
        return 0.0
    counter = Counter(words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


def vocab_richness(text: str) -> float:
    """Unique words / total words (type-token ratio)."""
    words = text.lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def code_token_entropy(code: str) -> float:
    """Entropy of Python code tokens (using tokenize module)."""
    import io
    import tokenize

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        meaningful = [
            t.string
            for t in tokens
            if t.type
            not in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENCODING,
                tokenize.ENDMARKER,
            )
            and t.string.strip()
        ]
        if len(meaningful) <= 1:
            return 0.0
        counter = Counter(meaningful)
        total = len(meaningful)
        return -sum((c / total) * math.log2(c / total) for c in counter.values())
    except Exception:
        return shannon_entropy(code)


def extract_patch_code(patch: str) -> Tuple[str, str]:
    """Extract added and removed code lines from unified diff."""
    added, removed = [], []
    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:].strip())
    return "\n".join(removed), "\n".join(added)


def extract_identifiers(code: str) -> set:
    """Extract Python identifiers from code."""
    # CamelCase, snake_case, dotted names
    ids = set(re.findall(r"[A-Za-z_]\w{2,}", code))
    # Remove Python keywords
    keywords = {
        "def", "class", "return", "import", "from", "if", "else", "elif",
        "for", "while", "try", "except", "finally", "with", "as", "pass",
        "break", "continue", "raise", "yield", "not", "and", "or", "is",
        "in", "True", "False", "None", "self", "lambda", "assert",
    }
    return ids - keywords


def ps_identifier_density(ps: str, code_identifiers: set) -> float:
    """Fraction of code identifiers that appear in PS."""
    if not code_identifiers:
        return 0.0
    ps_lower = ps.lower()
    found = sum(1 for ident in code_identifiers if ident.lower() in ps_lower)
    return found / len(code_identifiers)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity of token sets (MI proxy)."""
    tokens_a = set(re.findall(r"[A-Za-z_]\w+", text_a.lower()))
    tokens_b = set(re.findall(r"[A-Za-z_]\w+", text_b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def chain_signal_in_ps(ps: str, chain_nodes: list) -> Tuple[float, int]:
    """How many chain file/function names appear in PS."""
    if not chain_nodes:
        return 0.0, 0
    ps_lower = ps.lower()
    signals = set()
    for node in chain_nodes:
        fp = node.get("file_path", "")
        if fp:
            # Extract module name from path: django/views/decorators/csrf.py -> csrf
            parts = Path(fp).stem.split("_")
            signals.update(p for p in parts if len(p) > 2)
            # Also add parent dir names
            for part in Path(fp).parts:
                if len(part) > 2 and part != "tests":
                    signals.add(part.replace(".py", ""))
    if not signals:
        return 0.0, 0
    found = sum(1 for s in signals if s.lower() in ps_lower)
    return found / len(signals), found


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def spearman_rank(x: list, y: list) -> float:
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

    # Load quality metrics
    print("Loading quality metrics...")
    with open(outputs_root / "quality_metrics.json") as f:
        all_metrics = json.load(f)
    django_metrics = {
        m["instance_id"]: m for m in all_metrics if "django" in m.get("repo", "")
    }
    print(f"  Django valid bugs: {len(django_metrics)}")

    # Load synthesis data
    print("Loading synthesis data...")
    synth_file = (
        outputs_root
        / "4repo_run/2026-02-25/19-52-01/2_synthesis/final_dataset.json"
    )
    with open(synth_file) as f:
        synth_data = json.load(f)
    synth_map = {s["instance_id"]: s for s in synth_data if s["instance_id"] in django_metrics}
    del synth_data
    print(f"  Matched synthesis entries: {len(synth_map)}")

    # Compute metrics
    print("\nComputing info-theoretic metrics...")
    results = []

    for iid, qm in django_metrics.items():
        synth = synth_map.get(iid)
        if not synth:
            continue

        ps = synth.get("problem_statement", "")
        meta = synth.get("metadata", {})
        patch = meta.get("injection_patch", "")
        chain_nodes = []

        # Get chain nodes from synthesized_chain
        sc = meta.get("synthesized_chain", [])
        if isinstance(sc, list):
            chain_nodes = [n for n in sc if isinstance(n, dict)]

        # Also from seed_extraction_chains
        sec = meta.get("seed_extraction_chains", [])
        if isinstance(sec, list) and sec:
            for chain in sec:
                if isinstance(chain, dict):
                    chain_nodes.extend(chain.get("nodes", []))

        # Extract code from patch
        code_removed, code_added = extract_patch_code(patch)
        all_code = code_removed + "\n" + code_added
        code_ids = extract_identifiers(all_code)

        # Compute metrics
        h_ps = shannon_entropy(ps)
        vr = vocab_richness(ps)
        ps_id_density = ps_identifier_density(ps, code_ids)
        patch_h = code_token_entropy(all_code) if all_code.strip() else 0.0
        mi_proxy = jaccard_similarity(ps, all_code)
        chain_sig, chain_sig_count = chain_signal_in_ps(ps, chain_nodes)

        # PS token count
        ps_tokens = len(ps.split())

        results.append(
            {
                "instance_id": iid,
                "seed_id": qm.get("seed_id", ""),
                "fix_intent": qm.get("fix_intent", ""),
                "stage4_resolved": qm.get("stage4_resolved", False),
                "composite_score": qm.get("composite_score", 0),
                "p2f_count": qm.get("p2f_count", 0),
                # Info-theoretic metrics
                "h_ps": round(h_ps, 3),
                "ps_vocab_richness": round(vr, 3),
                "ps_tokens": ps_tokens,
                "ps_identifier_density": round(ps_id_density, 3),
                "patch_entropy": round(patch_h, 3),
                "mi_proxy_jaccard": round(mi_proxy, 3),
                "chain_signal_ratio": round(chain_sig, 3),
                "chain_signal_count": chain_sig_count,
            }
        )

    print(f"  Computed metrics for {len(results)} bugs\n")

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------

    resolved = [r for r in results if r["stage4_resolved"]]
    unresolved = [r for r in results if not r["stage4_resolved"]]

    def avg(lst, key):
        vals = [r[key] for r in lst]
        return sum(vals) / len(vals) if vals else 0

    print("=" * 70)
    print(f"INFO-THEORETIC METRICS: Django {len(results)} valid bugs")
    print(f"  Resolved: {len(resolved)}, Unresolved: {len(unresolved)}")
    print("=" * 70)

    metrics_keys = [
        "h_ps", "ps_vocab_richness", "ps_tokens", "ps_identifier_density",
        "patch_entropy", "mi_proxy_jaccard", "chain_signal_ratio",
    ]

    # Distribution summary
    print(f"\n{'Metric':<25s} {'Mean':>8s} {'Resolved':>10s} {'Unresolved':>10s} {'Delta':>8s}")
    print("-" * 65)
    for key in metrics_keys:
        m = avg(results, key)
        r = avg(resolved, key)
        u = avg(unresolved, key)
        delta = r - u
        marker = " ***" if abs(delta) > 0.02 else ""
        print(f"{key:<25s} {m:8.3f} {r:10.3f} {u:10.3f} {delta:+8.3f}{marker}")

    # Spearman correlations with stage4_resolved and composite_score
    print(f"\n{'Metric':<25s} {'rho(resolved)':>14s} {'rho(Q_score)':>14s}")
    print("-" * 55)
    resolved_binary = [1 if r["stage4_resolved"] else 0 for r in results]
    q_scores = [r["composite_score"] for r in results]

    for key in metrics_keys:
        vals = [r[key] for r in results]
        rho_res = spearman_rank(vals, resolved_binary)
        rho_q = spearman_rank(vals, q_scores)
        print(f"{key:<25s} {rho_res:+14.3f} {rho_q:+14.3f}")

    # Per-fix-intent breakdown
    intents = sorted(set(r["fix_intent"] for r in results))
    print(f"\n{'Fix Intent':<30s} {'N':>3s} {'H(PS)':>7s} {'MI':>7s} {'Chain':>7s} {'Res%':>6s}")
    print("-" * 65)
    for intent in intents:
        ir = [r for r in results if r["fix_intent"] == intent]
        n = len(ir)
        res_pct = sum(1 for r in ir if r["stage4_resolved"]) / n * 100
        print(
            f"{intent:<30s} {n:3d} "
            f"{avg(ir, 'h_ps'):7.3f} "
            f"{avg(ir, 'mi_proxy_jaccard'):7.3f} "
            f"{avg(ir, 'chain_signal_ratio'):7.3f} "
            f"{res_pct:5.1f}%"
        )

    # Top/bottom 5 by MI proxy
    results.sort(key=lambda r: r["mi_proxy_jaccard"], reverse=True)
    print(f"\nTop 5 MI proxy (most localization signal in PS):")
    for r in results[:5]:
        res = "✓" if r["stage4_resolved"] else "✗"
        print(f"  {res} MI={r['mi_proxy_jaccard']:.3f} H={r['h_ps']:.3f} Q={r['composite_score']:.3f} {r['instance_id'][:50]}")

    print(f"\nBottom 5 MI proxy (least localization signal in PS):")
    for r in results[-5:]:
        res = "✓" if r["stage4_resolved"] else "✗"
        print(f"  {res} MI={r['mi_proxy_jaccard']:.3f} H={r['h_ps']:.3f} Q={r['composite_score']:.3f} {r['instance_id'][:50]}")

    # Save results
    out_path = outputs_root / "info_theory_pilot.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
