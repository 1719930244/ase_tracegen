"""
Chain-Guided Problem Statement Generator (Stage 3.5).

Generates problem statements at three information levels (L1/L2/L3)
controlled by the DefectChain structure. This is TraceGen's key
differentiator vs. template-based mutation (Saving SWE-Bench):
we simultaneously control both bug synthesis AND PS information density
using the same underlying chain structure.

Information Levels (from chain node exposure):
  L1 (hard):   symptom-only → traceback exception type + affected module
  L2 (medium): + intermediate nodes → module/class names + call scenario
  L3 (easy):   full chain (minus root_cause line) → function signatures + expected-vs-actual

Theoretical grounding:
  - PS identifier density is the strongest predictor of resolve rate
    (Spearman rho=+0.336, pilot on Django 76 valid bugs)
  - Saving SWE-Bench (Garg et al., NeurIPS 2025): PS mutation alone
    causes -36.5% resolve rate drop
  - Code naturalness hypothesis (Hindle et al., ICSE 2012;
    Ray et al., ICSE 2016): entropy-delta constrains patch plausibility

Usage (standalone batch):
    python -m src.modules.synthesis.ps_chain_guided \\
        --synth-file final_dataset.json \\
        --val-dir 3_validation/ \\
        --quality-metrics quality_metrics.json \\
        --output chain_guided_ps.json

Usage (from code):
    gen = ChainGuidedPSGenerator()
    result = gen.generate(
        level="L2",
        chain_nodes=..., injection_patch=...,
        test_output=..., seed_ps=...,
    )
"""

import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chain entity extraction
# ---------------------------------------------------------------------------

def extract_chain_entities(
    chain_nodes: List[Dict],
    level: str = "L2",
) -> Dict[str, Set[str]]:
    """Extract entities from chain nodes based on information level.

    Args:
        chain_nodes: List of chain node dicts with node_id, node_type, file_path.
        level: L1 (symptom only), L2 (+intermediate), L3 (full chain).

    Returns:
        Dict with keys: required (entities to include), forbidden (to exclude),
        file_modules, function_names, class_names.
    """
    result = {
        "required": set(),     # Entities that MUST appear in PS
        "forbidden": set(),    # Entities that MUST NOT appear (root cause details)
        "file_modules": set(),
        "function_names": set(),
        "class_names": set(),
        "exposed_nodes": [],   # Chain nodes exposed at this level
    }

    # Classify nodes by type
    symptom_nodes = []
    intermediate_nodes = []
    root_cause_nodes = []

    for node in chain_nodes:
        nt = node.get("node_type", "").lower()
        if nt == "symptom":
            symptom_nodes.append(node)
        elif nt == "intermediate":
            intermediate_nodes.append(node)
        elif nt == "root_cause":
            root_cause_nodes.append(node)

    # --- L1: symptom only ---
    for node in symptom_nodes:
        _collect_node_entities(node, result)
    result["exposed_nodes"].extend(symptom_nodes)

    # --- L2: + intermediate ---
    if level in ("L2", "L3"):
        for node in intermediate_nodes:
            _collect_node_entities(node, result)
        result["exposed_nodes"].extend(intermediate_nodes)

    # --- L3: full chain minus root cause line number ---
    if level == "L3":
        for node in root_cause_nodes:
            # Include file/module but NOT exact line
            fp = node.get("file_path", "")
            if fp:
                mod = _path_to_module(fp)
                result["file_modules"].add(mod)
                result["required"].add(mod)
            # Include function name from node_id
            fn = _extract_function_from_node_id(node.get("node_id", ""))
            if fn:
                result["function_names"].add(fn)
                result["required"].add(fn)
        result["exposed_nodes"].extend(root_cause_nodes)

    # Root cause exact line info is always forbidden
    for node in root_cause_nodes:
        nid = node.get("node_id", "")
        # Forbid exact "file:line" pattern
        if ":" in nid:
            result["forbidden"].add(nid)

    return result


def _collect_node_entities(node: Dict, result: Dict) -> None:
    """Collect entities from a single chain node into result."""
    fp = node.get("file_path", "")
    nid = node.get("node_id", "")

    if fp:
        mod = _path_to_module(fp)
        result["file_modules"].add(mod)
        result["required"].add(mod)

    fn = _extract_function_from_node_id(nid)
    if fn:
        result["function_names"].add(fn)
        result["required"].add(fn)

    # Extract class name if present (e.g., "module.py:ClassName.method")
    cls = _extract_class_from_node_id(nid)
    if cls:
        result["class_names"].add(cls)
        result["required"].add(cls)


def _path_to_module(file_path: str) -> str:
    """Convert file path to readable module name.

    e.g. 'django/utils/text.py' → 'text',
         'django.utils.text.py' → 'text'
    """
    # Handle both '/' and '.' separated paths
    fp = file_path.replace(".", "/").rstrip("/")
    if fp.endswith("/py"):
        fp = fp[:-3]
    stem = Path(fp).stem if "/" in fp else fp.split(".")[-1]
    # Remove common non-informative stems
    if stem in ("__init__", "utils", "base", "models", "views", "tests"):
        # Use parent directory instead
        parts = fp.replace("\\", "/").split("/")
        for p in reversed(parts[:-1]):
            if p and p not in ("src", "lib", "django", "tests", "__pycache__"):
                return p
    return stem


def _extract_function_from_node_id(node_id: str) -> str:
    """Extract function name from node_id like 'module.py:funcname' or 'module.py:Class.method'."""
    if ":" not in node_id:
        return ""
    after_colon = node_id.split(":")[-1]
    # Handle Class.method
    parts = after_colon.split(".")
    return parts[-1] if parts[-1] else ""


def _extract_class_from_node_id(node_id: str) -> str:
    """Extract class name from node_id like 'module.py:ClassName.method'."""
    if ":" not in node_id:
        return ""
    after_colon = node_id.split(":")[-1]
    parts = after_colon.split(".")
    if len(parts) >= 2 and parts[0][0:1].isupper():
        return parts[0]
    return ""


# ---------------------------------------------------------------------------
# Traceback / test output parsing
# ---------------------------------------------------------------------------

_TRACEBACK_RE = re.compile(
    r"(Traceback \(most recent call last\):.*?^([A-Z]\w*(?:Error|Exception)[^\n]*))",
    re.MULTILINE | re.DOTALL,
)

_ASSERTION_MSG_RE = re.compile(
    r"^(Assertion(?:Error)?:?\s*.+?)$", re.MULTILINE
)


def extract_traceback_summary(test_output: str) -> Dict:
    """Extract structured failure info from test output.

    Returns dict with: exception_type, exception_message, assertion_messages,
    affected_modules.
    """
    result = {
        "exception_type": "",
        "exception_message": "",
        "assertion_messages": [],
        "affected_modules": set(),
        "has_traceback": False,
    }

    if not test_output:
        return result

    # Extract tracebacks
    for m in _TRACEBACK_RE.finditer(test_output):
        exc_line = m.group(2).strip()
        exc_parts = exc_line.split(":", 1)
        result["exception_type"] = exc_parts[0].strip()
        result["exception_message"] = exc_parts[1].strip() if len(exc_parts) > 1 else ""
        result["has_traceback"] = True
        break  # Take only the first traceback

    # Extract assertion messages
    for m in _ASSERTION_MSG_RE.finditer(test_output):
        msg = m.group(1).strip()
        if len(msg) > 300:
            msg = msg[:300] + "..."
        result["assertion_messages"].append(msg)

    # Extract affected module names from FAIL/ERROR headers
    fail_re = re.compile(r"^(?:FAIL|ERROR): \w+ \((\S+)\)", re.MULTILINE)
    for m in fail_re.finditer(test_output):
        mod = m.group(1).rsplit(".", 1)[0] if "." in m.group(1) else m.group(1)
        result["affected_modules"].add(mod)

    result["affected_modules"] = sorted(result["affected_modules"])
    return result


# ---------------------------------------------------------------------------
# Information-theoretic quality metrics
# ---------------------------------------------------------------------------

def compute_ps_metrics(ps: str, patch: str = "", chain_nodes: List[Dict] = None) -> Dict:
    """Compute info-theoretic quality metrics for a generated PS.

    Returns dict with: h_ps, vocab_richness, identifier_density,
    token_count, chain_signal_ratio.
    """
    words = ps.lower().split()
    n_words = len(words)

    # Shannon entropy
    if n_words <= 1:
        h_ps = 0.0
    else:
        counter = Counter(words)
        total = len(words)
        h_ps = -sum((c / total) * math.log2(c / total) for c in counter.values())

    # Vocab richness (type-token ratio)
    vocab_richness = len(set(words)) / n_words if n_words else 0.0

    # Identifier density (code identifiers from patch found in PS)
    code_ids = set()
    if patch:
        keywords = {
            "def", "class", "return", "import", "from", "if", "else", "elif",
            "for", "while", "try", "except", "finally", "with", "as", "pass",
            "break", "continue", "raise", "yield", "not", "and", "or", "is",
            "in", "True", "False", "None", "self", "lambda", "assert",
        }
        code_ids = set(re.findall(r"[A-Za-z_]\w{2,}", patch)) - keywords

    id_density = 0.0
    if code_ids:
        ps_lower = ps.lower()
        found = sum(1 for ident in code_ids if ident.lower() in ps_lower)
        id_density = found / len(code_ids)

    # Chain signal ratio (aligned with info_theory_test.py::chain_signal_in_ps)
    chain_sig = 0.0
    if chain_nodes:
        signals = set()
        for node in chain_nodes:
            fp = node.get("file_path", "")
            if fp:
                # Split stem by underscore (e.g., "csrf_exempt" → {"csrf", "exempt"})
                stem_parts = Path(fp).stem.split("_")
                signals.update(p for p in stem_parts if len(p) > 2)
                # Add all path component names
                for part in Path(fp).parts:
                    if len(part) > 2 and part != "tests":
                        signals.add(part.replace(".py", ""))
        if signals:
            ps_lower = ps.lower()
            found = sum(1 for s in signals if s.lower() in ps_lower)
            chain_sig = found / len(signals)

    return {
        "h_ps": round(h_ps, 3),
        "vocab_richness": round(vocab_richness, 3),
        "identifier_density": round(id_density, 3),
        "token_count": n_words,
        "chain_signal_ratio": round(chain_sig, 3),
    }


# ---------------------------------------------------------------------------
# PS generation templates (no LLM needed)
# ---------------------------------------------------------------------------

def _build_l1_ps(tb_summary: Dict, chain_entities: Dict) -> str:
    """L1 (hard): symptom-only PS.

    Contains: exception type + affected module/functionality.
    Deliberately omits: specific function names, file paths, expected-vs-actual.
    """
    parts = []

    # Affected area (from symptom nodes)
    modules = sorted(chain_entities.get("file_modules", set()))
    if modules:
        area = modules[0]
        parts.append(f"An issue has been identified in the {area} module.")

    # Exception type only (no message details)
    if tb_summary.get("exception_type"):
        exc = tb_summary["exception_type"]
        if "Assertion" in exc:
            parts.append("Certain operations produce incorrect results.")
        elif "Type" in exc:
            parts.append("A type-related error occurs during processing.")
        elif "Attribute" in exc:
            parts.append("An attribute access fails unexpectedly.")
        elif "Value" in exc:
            parts.append("An operation receives or produces an invalid value.")
        elif "Key" in exc:
            parts.append("A dictionary lookup fails for an expected key.")
        else:
            parts.append(f"The operation raises a {exc}.")
    elif tb_summary.get("assertion_messages"):
        parts.append("Certain operations produce incorrect results.")
    else:
        parts.append("Unexpected behavior has been observed.")

    if not parts:
        return "Tests in the project are failing unexpectedly."

    return " ".join(parts)


def _build_l2_ps(
    tb_summary: Dict,
    chain_entities: Dict,
    seed_ps: str = "",
) -> str:
    """L2 (medium): symptom + intermediate context.

    Contains: module/class names, call scenario, exception type + brief message.
    Deliberately omits: exact function signatures, traceback, expected-vs-actual values.
    """
    parts = []

    # Module and class context
    modules = sorted(chain_entities.get("file_modules", set()))
    classes = sorted(chain_entities.get("class_names", set()))
    functions = sorted(chain_entities.get("function_names", set()))

    if modules:
        mod_str = ", ".join(modules[:3])
        parts.append(f"An issue has been identified in the {mod_str} module{'s' if len(modules) > 1 else ''}.")

    if classes:
        cls_str = ", ".join(classes[:2])
        parts.append(f"The issue involves the {cls_str} class{'es' if len(classes) > 1 else ''}.")

    # Exception with brief message
    if tb_summary.get("exception_type"):
        exc = tb_summary["exception_type"]
        msg = tb_summary.get("exception_message", "")
        if msg and len(msg) < 100:
            parts.append(f"The operation raises {exc}: {msg}")
        else:
            parts.append(f"A {exc} is raised during processing.")

    # Add affected test modules for context
    affected = tb_summary.get("affected_modules", [])
    if affected:
        parts.append(
            f"This affects functionality tested in {affected[0]}."
        )

    if not parts:
        return _build_l1_ps(tb_summary, chain_entities)

    return " ".join(parts)


def _build_l3_ps(
    tb_summary: Dict,
    chain_entities: Dict,
    seed_ps: str = "",
) -> str:
    """L3 (easy): full chain context minus root cause line.

    Contains: function signatures, expected vs actual behavior, concrete examples.
    Deliberately omits: exact buggy code line, fix strategy.
    """
    parts = []

    # Full module + function context
    modules = sorted(chain_entities.get("file_modules", set()))
    functions = sorted(chain_entities.get("function_names", set()))
    classes = sorted(chain_entities.get("class_names", set()))

    if functions and modules:
        fn_str = ", ".join(f"`{f}`" for f in functions[:3])
        mod_str = ", ".join(modules[:3])
        parts.append(
            f"The {fn_str} function{'s' if len(functions) > 1 else ''} "
            f"in the {mod_str} module{'s' if len(modules) > 1 else ''} "
            f"{'exhibit' if len(functions) > 1 else 'exhibits'} incorrect behavior."
        )
    elif modules:
        parts.append(f"An issue has been identified in the {', '.join(modules[:3])} module.")

    if classes:
        parts.append(f"The issue manifests through the {', '.join(classes[:2])} class.")

    # Exception with full message
    if tb_summary.get("exception_type"):
        exc = tb_summary["exception_type"]
        msg = tb_summary.get("exception_message", "")
        if msg:
            parts.append(f"The operation raises `{exc}: {msg}`.")
        else:
            parts.append(f"A `{exc}` is raised.")

    # Assertion messages (expected vs actual)
    for amsg in tb_summary.get("assertion_messages", [])[:2]:
        cleaned = re.sub(r"^Assertion(?:Error)?:?\s*", "", amsg).strip()
        if cleaned and len(cleaned) < 200:
            parts.append(f"Observed: {cleaned}")

    # Affected test areas
    affected = tb_summary.get("affected_modules", [])
    if affected:
        parts.append(
            f"This can be reproduced through tests in "
            f"{', '.join(affected[:3])}."
        )

    if not parts:
        return _build_l2_ps(tb_summary, chain_entities, seed_ps)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

@dataclass
class ChainGuidedPSResult:
    """Result of chain-guided PS generation."""
    instance_id: str
    level: str  # L1, L2, L3
    problem_statement: str
    metrics: Dict = field(default_factory=dict)
    chain_entities_used: Dict = field(default_factory=dict)
    original_ps: str = ""

    def to_dict(self) -> Dict:
        return {
            "instance_id": self.instance_id,
            "level": self.level,
            "problem_statement": self.problem_statement,
            "metrics": self.metrics,
            "chain_entities_used": {
                k: sorted(v) if isinstance(v, set) else v
                for k, v in self.chain_entities_used.items()
                if k != "exposed_nodes"
            },
            "original_ps": self.original_ps,
        }


class ChainGuidedPSGenerator:
    """Generate problem statements guided by DefectChain structure.

    The key insight: the chain controls WHAT information appears in PS.
    - Chain depth → how much structural context is exposed
    - Chain node types (symptom/intermediate/root_cause) → information granularity
    - The root cause location is NEVER fully revealed (no fix leakage)

    This provides principled difficulty control:
    - L1 uses only symptom nodes → hard to localize
    - L2 adds intermediate nodes → moderate localizability
    - L3 exposes full chain (minus root_cause line) → easy to localize
    """

    def generate(
        self,
        level: str,
        chain_nodes: List[Dict],
        injection_patch: str = "",
        test_output: str = "",
        seed_ps: str = "",
        instance_id: str = "",
    ) -> ChainGuidedPSResult:
        """Generate a chain-guided problem statement.

        Args:
            level: "L1" (hard), "L2" (medium), or "L3" (easy).
            chain_nodes: The synthesized_chain nodes.
            injection_patch: The unified diff of the injected bug.
            test_output: Raw test_output.txt content from Stage 3.
            seed_ps: Original seed problem statement (for style reference).
            instance_id: Instance identifier.

        Returns:
            ChainGuidedPSResult with generated PS and quality metrics.
        """
        if level not in ("L1", "L2", "L3"):
            logger.warning(f"Unknown level '{level}', defaulting to L2")
            level = "L2"

        # Step 1: Extract chain entities at the specified level
        entities = extract_chain_entities(chain_nodes, level)

        # Step 2: Parse test output for failure symptoms
        tb_summary = extract_traceback_summary(test_output)

        # Step 3: Build PS based on level
        if level == "L1":
            ps = _build_l1_ps(tb_summary, entities)
        elif level == "L2":
            ps = _build_l2_ps(tb_summary, entities, seed_ps)
        else:
            ps = _build_l3_ps(tb_summary, entities, seed_ps)

        # Step 4: Compute quality metrics
        metrics = compute_ps_metrics(ps, injection_patch, chain_nodes)

        return ChainGuidedPSResult(
            instance_id=instance_id,
            level=level,
            problem_statement=ps,
            metrics=metrics,
            chain_entities_used={
                "required": entities["required"],
                "forbidden": entities["forbidden"],
                "file_modules": entities["file_modules"],
                "function_names": entities["function_names"],
                "class_names": entities["class_names"],
            },
            original_ps=seed_ps,
        )

    def generate_all_levels(
        self,
        chain_nodes: List[Dict],
        injection_patch: str = "",
        test_output: str = "",
        seed_ps: str = "",
        instance_id: str = "",
    ) -> Dict[str, ChainGuidedPSResult]:
        """Generate PS at all three levels for comparison."""
        return {
            lvl: self.generate(
                level=lvl,
                chain_nodes=chain_nodes,
                injection_patch=injection_patch,
                test_output=test_output,
                seed_ps=seed_ps,
                instance_id=instance_id,
            )
            for lvl in ("L1", "L2", "L3")
        }


# ---------------------------------------------------------------------------
# Batch processing (standalone script)
# ---------------------------------------------------------------------------

def batch_generate(
    synth_file: str,
    val_dir: str,
    quality_file: str,
    output_file: str,
    repo_filter: str = "django",
) -> None:
    """Batch generate chain-guided PS for all valid bugs.

    For each valid bug, generates L1/L2/L3 variants and computes metrics.
    """
    # Load synthesis data
    with open(synth_file) as f:
        synth_data = json.load(f)
    synth_map = {s["instance_id"]: s for s in synth_data}

    # Load quality metrics to identify valid bugs
    with open(quality_file) as f:
        quality_data = json.load(f)
    valid_bugs = [
        m for m in quality_data
        if repo_filter in m.get("repo", "")
        and m.get("stage4_resolved") is not None
    ]

    val_dir = Path(val_dir)
    generator = ChainGuidedPSGenerator()
    results = []

    logger.info(f"Processing {len(valid_bugs)} valid {repo_filter} bugs")

    for qm in valid_bugs:
        iid = qm["instance_id"]
        synth = synth_map.get(iid)
        if not synth:
            logger.warning(f"No synthesis data for {iid}")
            continue

        meta = synth.get("metadata", {})
        chain_nodes = meta.get("synthesized_chain", [])
        if not isinstance(chain_nodes, list):
            chain_nodes = []
        injection_patch = meta.get("injection_patch", "")
        original_ps = synth.get("problem_statement", "")

        # Find test output from validation logs
        test_output = ""
        # Try common log directory patterns
        for subdir in ["logs/django/django", "logs/django", "logs"]:
            test_out_path = val_dir / subdir / iid / "test_output.txt"
            if test_out_path.exists():
                test_output = test_out_path.read_text(errors="replace")
                break

        # Generate all three levels
        level_results = generator.generate_all_levels(
            chain_nodes=chain_nodes,
            injection_patch=injection_patch,
            test_output=test_output,
            seed_ps=original_ps,
            instance_id=iid,
        )

        entry = {
            "instance_id": iid,
            "seed_id": qm.get("seed_id", ""),
            "fix_intent": qm.get("fix_intent", ""),
            "stage4_resolved": qm.get("stage4_resolved", False),
            "composite_score": qm.get("composite_score", 0),
            "original_ps": original_ps,
            "original_ps_tokens": len(original_ps.split()),
            "chain_node_count": len(chain_nodes),
            "has_test_output": bool(test_output),
        }

        for lvl in ("L1", "L2", "L3"):
            r = level_results[lvl]
            entry[f"ps_{lvl}"] = r.problem_statement
            entry[f"ps_{lvl}_tokens"] = r.metrics.get("token_count", 0)
            entry[f"ps_{lvl}_identifier_density"] = r.metrics.get("identifier_density", 0)
            entry[f"ps_{lvl}_h_ps"] = r.metrics.get("h_ps", 0)
            entry[f"ps_{lvl}_chain_signal"] = r.metrics.get("chain_signal_ratio", 0)

        results.append(entry)

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    n = len(results)
    resolved = [r for r in results if r["stage4_resolved"]]
    print(f"\n{'='*70}")
    print(f"Chain-Guided PS Generation: {n} valid bugs")
    print(f"  Resolved: {len(resolved)}, Unresolved: {n - len(resolved)}")
    print(f"  With test output: {sum(1 for r in results if r['has_test_output'])}")
    print(f"{'='*70}")

    # Compare metrics across levels
    def avg(lst, key):
        vals = [r[key] for r in lst if key in r]
        return sum(vals) / len(vals) if vals else 0

    print(f"\n{'Level':<8s} {'Tokens':>8s} {'ID Density':>12s} {'H(PS)':>8s} {'ChainSig':>10s}")
    print("-" * 50)
    for lvl in ("L1", "L2", "L3"):
        print(
            f"{lvl:<8s} "
            f"{avg(results, f'ps_{lvl}_tokens'):8.1f} "
            f"{avg(results, f'ps_{lvl}_identifier_density'):12.3f} "
            f"{avg(results, f'ps_{lvl}_h_ps'):8.3f} "
            f"{avg(results, f'ps_{lvl}_chain_signal'):10.3f}"
        )

    print(f"\nOriginal PS avg tokens: {avg(results, 'original_ps_tokens'):.1f}")
    print(f"\nResults saved to {output_file}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Chain-Guided PS Generation (Stage 3.5)"
    )
    parser.add_argument(
        "--synth-file", required=True,
        help="Path to final_dataset.json from Stage 2",
    )
    parser.add_argument(
        "--val-dir", required=True,
        help="Path to Stage 3 validation output directory",
    )
    parser.add_argument(
        "--quality-metrics", required=True,
        help="Path to quality_metrics.json",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--repo", default="django",
        help="Repository filter (default: django)",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    batch_generate(
        synth_file=args.synth_file,
        val_dir=args.val_dir,
        quality_file=args.quality_metrics,
        output_file=args.output,
        repo_filter=args.repo,
    )


if __name__ == "__main__":
    main()
