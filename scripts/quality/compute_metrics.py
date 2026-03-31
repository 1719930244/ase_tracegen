#!/usr/bin/env python3
"""
compute_metrics.py - Intrinsic quality metrics for TraceGen synthetic bugs.

Computes 5+1 quality dimensions from existing synthesis + validation data,
without requiring LLM calls or Docker containers.

Metrics:
  1. Failure Mode     - exception type quality (AssertionError > TypeError > SyntaxError)
  2. Test Impact      - P2F/(P2F+P2P) ratio, sweet spot [0.5%-15%]
  3. Patch Naturalness - changed lines, hunks, sabotage detection
  4. PS Informativeness- behavioral description, leak detection, fallback check
  5. Chain Alignment   - from validation data (structure_similarity / trace_coverage)
  6. Composite Quality - weighted combination → tier A/B/C/D

Usage (multi-directory mode, recommended):
    python scripts/quality/compute_metrics.py \
        --aggregated ../tracegen-outputs/aggregated_results.json \
        --output ../tracegen-outputs/quality_metrics.json

Usage (single-directory mode, legacy):
    python scripts/quality/compute_metrics.py \
        --synth-dir ../tracegen-outputs/4repo_run/2026-02-25/19-52-01 \
        --output quality_metrics.json
"""

import argparse
import json
import re
import math
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Failure Mode Classification
# ---------------------------------------------------------------------------

# Exception types ordered by quality (logic bugs > type errors > crashes)
FAILURE_MODE_SCORES = {
    "AssertionError": 1.0,
    "AssertionError": 1.0,  # alias
    "ValueError": 0.8,
    "KeyError": 0.8,
    "IndexError": 0.8,
    "AttributeError": 0.7,
    "TypeError": 0.6,
    "RuntimeError": 0.6,
    "NotImplementedError": 0.5,
    "PermissionError": 0.5,
    "FileNotFoundError": 0.5,
    "OSError": 0.5,
    "NameError": 0.3,
    "ImportError": 0.2,
    "ModuleNotFoundError": 0.2,
    "SyntaxError": 0.1,
    "IndentationError": 0.1,
}

# Match exception names: at line start, in tracebacks, or standalone
_EXCEPTION_RE = re.compile(
    r"^([A-Z]\w*(?:Error|Exception))\s*(?:[:(\[]|$)", re.MULTILINE
)
# pytest format: "E   TypeError: ..."
_PYTEST_E_RE = re.compile(
    r"^E\s+([A-Z]\w*(?:Error|Exception))\s*[:(\[]", re.MULTILINE
)
# Also match "raise ExceptionType" patterns
_RAISE_RE = re.compile(
    r"raise\s+([A-Z]\w*(?:Error|Exception))", re.MULTILINE
)


def classify_failure_mode(test_output_path: Path) -> Tuple[str, float]:
    """Parse test output to classify the primary exception type."""
    if not test_output_path.exists():
        return "unknown", 0.5

    try:
        content = test_output_path.read_text(errors="replace")
    except Exception:
        return "unknown", 0.5

    # Check for broken test runs (pytest not found, etc.)
    if "command not found" in content and len(content) < 2000:
        return "test_broken", 0.0

    # Find all exception types
    exceptions = _EXCEPTION_RE.findall(content)
    exceptions.extend(_PYTEST_E_RE.findall(content))
    exceptions.extend(_RAISE_RE.findall(content))
    if not exceptions:
        return "unknown", 0.5

    # Filter out generic Exception (too broad)
    specific = [e for e in exceptions if e != "Exception"]
    if not specific:
        specific = exceptions

    counter = Counter(specific)
    primary = counter.most_common(1)[0][0]

    score = FAILURE_MODE_SCORES.get(primary, 0.5)
    return primary, score


# ---------------------------------------------------------------------------
# 2. Test Impact Ratio
# ---------------------------------------------------------------------------

def compute_impact_ratio(p2f_count: int, p2p_count: int) -> Tuple[float, float]:
    """Compute test impact ratio and its quality score."""
    total = p2f_count + p2p_count
    if total == 0:
        return 0.0, 0.0

    ratio = p2f_count / total

    # Score: Gaussian-like around sweet spot [0.005, 0.15]
    # Peak at 0.03, std ~0.05
    if ratio < 0.001:
        score = 0.1  # barely detectable
    elif ratio <= 0.005:
        score = 0.5
    elif ratio <= 0.03:
        score = 0.9
    elif ratio <= 0.10:
        score = 1.0  # sweet spot
    elif ratio <= 0.15:
        score = 0.9
    elif ratio <= 0.30:
        score = 0.6
    elif ratio <= 0.50:
        score = 0.3
    else:
        score = 0.1  # catastrophic

    return ratio, score


# ---------------------------------------------------------------------------
# 3. Patch Naturalness
# ---------------------------------------------------------------------------

SABOTAGE_PATTERNS = [
    re.compile(r"raise\s+Exception\s*\("),
    re.compile(r"return\s+None\s*$", re.MULTILINE),
    re.compile(r"return\s+object\(\)"),
    re.compile(r'= "SABOTAGE"'),
    re.compile(r"= None\s*#"),
    re.compile(r"pass\s*$", re.MULTILINE),
]


def analyze_patch(patch: str) -> Dict[str, Any]:
    """Analyze patch naturalness."""
    if not patch:
        return {
            "changed_lines": 0,
            "added_lines": 0,
            "removed_lines": 0,
            "hunk_count": 0,
            "is_single_hunk": False,
            "sabotage_detected": False,
            "naturalness_score": 0.0,
        }

    lines = patch.split("\n")
    added = [l[1:] for l in lines if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in lines if l.startswith("-") and not l.startswith("---")]
    hunk_count = sum(1 for l in lines if l.startswith("@@"))

    changed = len(added) + len(removed)

    # Sabotage detection on added lines
    added_text = "\n".join(added)
    sabotage = any(p.search(added_text) for p in SABOTAGE_PATTERNS)

    # Score
    if changed == 0:
        score = 0.0
    elif sabotage:
        score = 0.2
    elif 1 <= changed <= 5 and hunk_count == 1:
        score = 1.0
    elif 1 <= changed <= 10 and hunk_count == 1:
        score = 0.9
    elif 1 <= changed <= 10:
        score = 0.7
    elif changed <= 20:
        score = 0.5
    else:
        score = 0.3

    return {
        "changed_lines": changed,
        "added_lines": len(added),
        "removed_lines": len(removed),
        "hunk_count": hunk_count,
        "is_single_hunk": hunk_count == 1,
        "sabotage_detected": sabotage,
        "naturalness_score": score,
    }


# ---------------------------------------------------------------------------
# 4. Problem Statement Informativeness
# ---------------------------------------------------------------------------

_BEHAVIORAL_PATTERNS = [
    re.compile(r"\b(when|instead of|expected|returns|produces|should|fails to)\b", re.I),
]
_ERROR_DESC_PATTERNS = [
    re.compile(r"\b(Error|Exception|TypeError|ValueError|AssertionError|raises|fails with|traceback)\b", re.I),
]
_LEAK_PATTERNS = [
    re.compile(r"\b(the fix is|fixed by|should be changed to|replace .+ with)\b", re.I),
    re.compile(r"\b(use .+ instead of|change .+ to|switching from .+ to)\b", re.I),
    re.compile(r"\b(needs to be|should use)\b", re.I),
]
_FALLBACK_PATTERNS = [
    "Synthetic defect",
    "synthetic defect",
    "No description",
    "Bug injection",
]


def analyze_problem_statement(ps: str) -> Dict[str, Any]:
    """Analyze problem statement informativeness."""
    if not ps or not ps.strip():
        return {
            "token_count": 0,
            "has_behavioral_desc": False,
            "has_error_desc": False,
            "is_fallback": True,
            "leak_count": 0,
            "info_score": 0.0,
        }

    tokens = ps.split()
    token_count = len(tokens)
    is_fallback = any(fp in ps for fp in _FALLBACK_PATTERNS) or token_count < 10

    has_behavioral = any(p.search(ps) for p in _BEHAVIORAL_PATTERNS)
    has_error = any(p.search(ps) for p in _ERROR_DESC_PATTERNS)
    leak_count = sum(1 for p in _LEAK_PATTERNS if p.search(ps))

    # Score
    if is_fallback:
        score = 0.0
    else:
        score = 0.3  # base for non-fallback
        if has_behavioral:
            score += 0.3
        if has_error:
            score += 0.2
        if 20 <= token_count <= 200:
            score += 0.2
        elif token_count > 200:
            score += 0.1
        # Penalize leaks
        score -= 0.15 * leak_count
        score = max(0.0, min(1.0, score))

    return {
        "token_count": token_count,
        "has_behavioral_desc": has_behavioral,
        "has_error_desc": has_error,
        "is_fallback": is_fallback,
        "leak_count": leak_count,
        "info_score": round(score, 3),
    }


# ---------------------------------------------------------------------------
# 5. Chain Alignment (read from existing data)
# ---------------------------------------------------------------------------

def get_chain_alignment(val_data: Dict) -> float:
    """Extract chain alignment score from validation result.

    Prefers structural alignment (Eq.3: seed vs synthetic chain) if available,
    falls back to traceback-based overall_score.
    """
    ca = val_data.get("chain_alignment_score", {})
    if not isinstance(ca, dict):
        return 0.0
    # New schema: structural_alignment is nested
    sa = ca.get("structural_alignment", {})
    if isinstance(sa, dict) and "overall_score" in sa:
        return sa["overall_score"]
    # Legacy schema: overall_score at top level
    return ca.get("overall_score", 0.0)


# ---------------------------------------------------------------------------
# 6. Composite Quality Score
# ---------------------------------------------------------------------------

def compute_composite_score(
    fm_score: float,
    ir_score: float,
    nat_score: float,
    ps_score: float,
    ca_score: float,
) -> Tuple[float, str]:
    """Compute weighted composite quality score and tier.

    Returns (score, tier) where tier is A/B/C/D.
    """
    q = (
        0.15 * fm_score
        + 0.15 * ir_score
        + 0.15 * nat_score
        + 0.15 * ps_score
        + 0.20 * ca_score
        # Remaining 0.20 reserved for BM25 (default 0.5 if unavailable)
        + 0.20 * 0.5
    )
    q = round(min(1.0, max(0.0, q)), 3)

    if q >= 0.70:
        tier = "A"
    elif q >= 0.50:
        tier = "B"
    elif q >= 0.30:
        tier = "C"
    else:
        tier = "D"

    return q, tier


# ---------------------------------------------------------------------------
# File finders
# ---------------------------------------------------------------------------

def find_test_output(val_logs_dirs: List[Path], instance_id: str) -> Optional[Path]:
    """Find test_output.txt across multiple validation log directories."""
    for logs_dir in val_logs_dirs:
        if not logs_dir.exists():
            continue
        for p in logs_dir.rglob(f"{instance_id}/test_output.txt"):
            return p
    return None


def find_validation_json(val_dirs: List[Path], instance_id: str) -> Optional[Path]:
    """Find validation JSON across multiple directories."""
    fname = f"{instance_id}_validation.json"
    for vd in val_dirs:
        p = vd / fname
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Main: multi-directory mode
# ---------------------------------------------------------------------------

# Data directory registry for 4-repo experiment
_DATA_DIRS = {
    "django": {
        "synth": "4repo_run/2026-02-25/19-52-01",
        "val": "4repo_run/2026-02-25/19-52-01",
    },
    "pytest": {
        "synth": "4repo_run/stage2_full",
        "val": "4repo_run/stage2_full_salvage_validation_full_20260303_w12",
    },
    "sklearn": {
        "synth": "4repo_run/stage2_full",
        "val": "4repo_run/stage2_full_salvage_validation_full_20260303_w12",
    },
    "sympy": {
        "synth": "4repo_run/stage2_full",
        "val": "4repo_run/revalidation_sympy_full",
    },
}


_REPO_KEY_MAP = {
    "scikit-learn/scikit-learn": "sklearn",
    "scikit-learn": "sklearn",
}


def _repo_key(repo: str) -> str:
    """Extract short repo key from 'org/repo' format."""
    if repo in _REPO_KEY_MAP:
        return _REPO_KEY_MAP[repo]
    return repo.split("/")[-1] if "/" in repo else repo


def main():
    parser = argparse.ArgumentParser(description="Compute intrinsic quality metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--aggregated", help="Path to aggregated_results.json (multi-repo)")
    group.add_argument("--synth-dir", help="Single stage output directory (legacy mode)")
    parser.add_argument("--output-base", help="Base dir for tracegen-outputs (default: auto-detect)")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--repo-filter", default=None, help="Filter by repo")
    args = parser.parse_args()

    if args.synth_dir:
        return _main_single_dir(args)

    # --- Multi-directory mode ---
    agg_path = Path(args.aggregated)
    output_base = Path(args.output_base) if args.output_base else agg_path.parent
    output_path = Path(args.output) if args.output else output_base / "quality_metrics.json"

    print(f"Loading aggregated results from {agg_path}")
    with open(agg_path) as f:
        agg = json.load(f)
    valid_bugs = agg["valid_bugs"]
    print(f"  {len(valid_bugs)} valid bugs across {len(agg['summary']['repos'])} repos")

    # Build validation + synthesis lookup dirs per repo
    val_dirs = {}  # repo_key -> list of val dirs
    val_logs_dirs = {}  # repo_key -> list of log dirs
    synth_maps = {}  # repo_key -> {instance_id: synth_entry}

    for rkey, dirs in _DATA_DIRS.items():
        # Validation dirs
        vd = output_base / dirs["val"] / "3_validation"
        val_dirs.setdefault(rkey, []).append(vd)
        val_logs_dirs.setdefault(rkey, []).append(vd / "logs")

    # Also add the salvage dir for sympy (fallback)
    val_dirs.setdefault("sympy", []).append(
        output_base / "4repo_run/stage2_full_salvage_validation_full_20260303_w12/3_validation"
    )
    val_logs_dirs.setdefault("sympy", []).append(
        output_base / "4repo_run/stage2_full_salvage_validation_full_20260303_w12/3_validation/logs"
    )

    # Load synthesis data (need PS text + patch)
    print("Loading synthesis data...")
    synth_files = set()
    for rkey, dirs in _DATA_DIRS.items():
        sf = output_base / dirs["synth"] / "2_synthesis" / "final_dataset.json"
        synth_files.add(sf)

    for sf in synth_files:
        if not sf.exists():
            print(f"  WARNING: {sf} not found, skipping")
            continue
        print(f"  Loading {sf} ({sf.stat().st_size / 1e6:.0f} MB)...")
        with open(sf) as f:
            sdata = json.load(f)
        for entry in sdata:
            iid = entry.get("instance_id", "")
            rkey = _repo_key(entry.get("repo", ""))
            synth_maps[iid] = entry
        del sdata  # free memory
    print(f"  Loaded synthesis data for {len(synth_maps)} instances")

    # Process each valid bug
    results = []
    fm_found = 0
    patch_found = 0
    ps_found = 0

    for bug in valid_bugs:
        iid = bug["instance_id"]
        repo = bug["repo"]
        rkey = _repo_key(repo)

        if args.repo_filter and repo != args.repo_filter:
            continue

        synth = synth_maps.get(iid, {})
        meta = synth.get("metadata", {})

        # 1. Failure Mode - from test_output.txt
        logs = val_logs_dirs.get(rkey, [])
        test_out_path = find_test_output(logs, iid)
        if test_out_path:
            failure_mode, fm_score = classify_failure_mode(test_out_path)
            fm_found += 1
        else:
            failure_mode, fm_score = "unknown", 0.5

        # 2. Test Impact Ratio - from aggregated data
        p2f_count = bug.get("val_p2f", 0)
        p2p_count = bug.get("val_p2p", 0)
        impact_ratio, ir_score = compute_impact_ratio(p2f_count, p2p_count)

        # 3. Patch Naturalness - from synthesis metadata
        patch = meta.get("injection_patch", "")
        patch_metrics = analyze_patch(patch)
        if patch:
            patch_found += 1

        # 4. PS Informativeness - from synthesis data
        ps = synth.get("problem_statement", "")
        ps_metrics = analyze_problem_statement(ps)
        if ps and ps != "Synthetic defect":
            ps_found += 1

        # 5. Chain Alignment - from aggregated data
        chain_alignment = bug.get("chain_alignment", 0.0)

        # 6. Composite
        composite, tier = compute_composite_score(
            fm_score, ir_score, patch_metrics["naturalness_score"],
            ps_metrics["info_score"], chain_alignment,
        )

        # Fix intent from aggregated data (more reliable)
        fix_intent = bug.get("fix_intent", meta.get("seed_fix_intent", "Unknown"))

        result = {
            "instance_id": iid,
            "repo": repo,
            "seed_id": bug.get("seed_id", ""),
            "fix_intent": fix_intent,
            "chain_depth": bug.get("chain_depth", 0),
            # Metric 1: Failure Mode
            "failure_mode": failure_mode,
            "failure_mode_score": fm_score,
            # Metric 2: Test Impact
            "p2f_count": p2f_count,
            "p2p_count": p2p_count,
            "test_impact_ratio": round(impact_ratio, 6),
            "impact_ratio_score": ir_score,
            # Metric 3: Patch Naturalness
            **{f"patch_{k}": v for k, v in patch_metrics.items()},
            # Metric 4: PS Informativeness
            **{f"ps_{k}": v for k, v in ps_metrics.items()},
            # Metric 5: Chain Alignment
            "chain_alignment": round(chain_alignment, 4),
            # Metric 6: Composite
            "composite_score": composite,
            "quality_tier": tier,
            # Stage 4 label (for correlation analysis)
            "stage4_resolved": bug.get("stage4_resolved", None),
        }
        results.append(result)

    results.sort(key=lambda x: x["instance_id"])

    # Write output
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    n = len(results)
    print(f"\nProcessed {n} valid bugs → {output_path}")
    print(f"  test_output found: {fm_found}/{n}")
    print(f"  patch found: {patch_found}/{n}")
    print(f"  PS found: {ps_found}/{n}")

    _print_summary(results)


def _main_single_dir(args):
    """Legacy single-directory mode."""
    synth_dir = Path(args.synth_dir)
    synth_file = synth_dir / "2_synthesis" / "final_dataset.json"
    val_dir = synth_dir / "3_validation"
    val_logs_dir = val_dir / "logs"
    output_path = Path(args.output) if args.output else Path("quality_metrics.json")

    print(f"Loading synthesis data from {synth_file}")
    with open(synth_file) as f:
        synth_data = json.load(f)
    synth_map = {d["instance_id"]: d for d in synth_data}
    print(f"  Total synthesis entries: {len(synth_data)}")

    val_files = sorted(val_dir.glob("synthetic_*_validation.json"))
    print(f"  Total validation files: {len(val_files)}")

    results = []
    for vf in val_files:
        with open(vf) as f:
            val_data = json.load(f)
        if val_data.get("status") != "valid":
            continue

        iid = val_data["instance_id"]
        synth = synth_map.get(iid, {})
        repo = synth.get("repo", "")
        if args.repo_filter and repo != args.repo_filter:
            continue

        meta = synth.get("metadata", {})
        test_output = find_test_output([val_logs_dir], iid)
        failure_mode, fm_score = classify_failure_mode(test_output) if test_output else ("unknown", 0.5)
        p2f = val_data.get("PASS_TO_FAIL", [])
        p2p = val_data.get("PASS_TO_PASS", [])
        impact_ratio, ir_score = compute_impact_ratio(len(p2f), len(p2p))
        patch_metrics = analyze_patch(meta.get("injection_patch", ""))
        ps_metrics = analyze_problem_statement(synth.get("problem_statement", ""))
        chain_alignment = get_chain_alignment(val_data)
        composite, tier = compute_composite_score(
            fm_score, ir_score, patch_metrics["naturalness_score"],
            ps_metrics["info_score"], chain_alignment,
        )
        fix_intent = meta.get("injection_strategy", meta.get("seed_fix_intent", "Unknown"))

        result = {
            "instance_id": iid, "repo": repo,
            "seed_id": meta.get("seed_id", meta.get("seed_instance_id", "")),
            "fix_intent": fix_intent, "chain_depth": meta.get("chain_depth", 0),
            "failure_mode": failure_mode, "failure_mode_score": fm_score,
            "p2f_count": len(p2f), "p2p_count": len(p2p),
            "test_impact_ratio": round(impact_ratio, 6), "impact_ratio_score": ir_score,
            **{f"patch_{k}": v for k, v in patch_metrics.items()},
            **{f"ps_{k}": v for k, v in ps_metrics.items()},
            "chain_alignment": round(chain_alignment, 4),
            "composite_score": composite, "quality_tier": tier,
        }
        results.append(result)

    results.sort(key=lambda x: x["instance_id"])
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nProcessed {len(results)} valid bugs → {output_path}")
    _print_summary(results)


def _print_summary(results: List[Dict]):
    """Print summary statistics."""
    if not results:
        return

    n = len(results)

    # Per-repo breakdown
    repos = sorted(set(r["repo"] for r in results))
    print(f"\n{'='*60}")
    print(f"Quality Metrics Summary ({n} bugs, {len(repos)} repos)")
    print(f"{'='*60}")

    # Failure Mode
    fm_dist = Counter(r["failure_mode"] for r in results)
    print(f"\nFailure Mode Distribution:")
    for fm, cnt in fm_dist.most_common(10):
        print(f"  {fm:<25s} {cnt:>3d} ({cnt/n*100:5.1f}%)")

    # Impact Ratio
    ratios = [r["test_impact_ratio"] for r in results]
    print(f"\nTest Impact Ratio:")
    print(f"  mean={sum(ratios)/n:.4f}, median={sorted(ratios)[n//2]:.4f}")
    sweet = sum(1 for r in ratios if 0.005 <= r <= 0.15)
    print(f"  sweet spot [0.5%-15%]: {sweet}/{n} ({sweet/n*100:.1f}%)")

    # Patch Naturalness
    nat = [r["patch_naturalness_score"] for r in results]
    sab = sum(1 for r in results if r.get("patch_sabotage_detected", False))
    print(f"\nPatch Naturalness: mean={sum(nat)/n:.2f}")
    print(f"  sabotage detected: {sab}/{n}")

    # PS Informativeness
    psi = [r["ps_info_score"] for r in results]
    fb = sum(1 for r in results if r.get("ps_is_fallback", False))
    print(f"\nPS Informativeness: mean={sum(psi)/n:.2f}")
    print(f"  fallback PS: {fb}/{n}")

    # Chain Alignment
    ca = [r["chain_alignment"] for r in results]
    ca_nonzero = [c for c in ca if c > 0]
    print(f"\nChain Alignment: mean={sum(ca)/n:.3f}, "
          f"median={sorted(ca)[n//2]:.3f}")
    if ca_nonzero:
        print(f"  non-zero only ({len(ca_nonzero)}): mean={sum(ca_nonzero)/len(ca_nonzero):.3f}")

    # Composite Score + Tiers
    scores = [r["composite_score"] for r in results]
    tiers = Counter(r["quality_tier"] for r in results)
    print(f"\nComposite Quality Score: mean={sum(scores)/n:.3f}, "
          f"median={sorted(scores)[n//2]:.3f}")
    print(f"  Tier A (≥0.70): {tiers.get('A',0):>3d} ({tiers.get('A',0)/n*100:5.1f}%)")
    print(f"  Tier B (0.50-0.70): {tiers.get('B',0):>3d} ({tiers.get('B',0)/n*100:5.1f}%)")
    print(f"  Tier C (0.30-0.50): {tiers.get('C',0):>3d} ({tiers.get('C',0)/n*100:5.1f}%)")
    print(f"  Tier D (<0.30): {tiers.get('D',0):>3d} ({tiers.get('D',0)/n*100:5.1f}%)")

    # Per-repo composite
    print(f"\nPer-Repo Quality:")
    for repo in repos:
        repo_results = [r for r in results if r["repo"] == repo]
        rn = len(repo_results)
        rscores = [r["composite_score"] for r in repo_results]
        rtiers = Counter(r["quality_tier"] for r in repo_results)
        print(f"  {repo:<25s} n={rn:>3d}  Q={sum(rscores)/rn:.3f}  "
              f"A={rtiers.get('A',0)} B={rtiers.get('B',0)} "
              f"C={rtiers.get('C',0)} D={rtiers.get('D',0)}")

    # Correlation with Stage 4 (Django only)
    resolved_bugs = [r for r in results if r.get("stage4_resolved") is not None]
    if resolved_bugs:
        resolved = [r for r in resolved_bugs if r["stage4_resolved"]]
        unresolved = [r for r in resolved_bugs if not r["stage4_resolved"]]
        if resolved and unresolved:
            res_q = sum(r["composite_score"] for r in resolved) / len(resolved)
            unres_q = sum(r["composite_score"] for r in unresolved) / len(unresolved)
            print(f"\nStage 4 Correlation ({len(resolved_bugs)} bugs with labels):")
            print(f"  Resolved ({len(resolved)}): mean Q={res_q:.3f}")
            print(f"  Unresolved ({len(unresolved)}): mean Q={unres_q:.3f}")

    # Fix Intent quality distribution
    print(f"\nQuality by Fix Intent:")
    intents = sorted(set(r["fix_intent"] for r in results))
    for intent in intents:
        ir = [r for r in results if r["fix_intent"] == intent]
        iq = sum(r["composite_score"] for r in ir) / len(ir)
        it = Counter(r["quality_tier"] for r in ir)
        print(f"  {intent:<30s} n={len(ir):>3d}  Q={iq:.3f}  "
              f"A={it.get('A',0)} B={it.get('B',0)}")


if __name__ == "__main__":
    main()
