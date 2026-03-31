#!/usr/bin/env python
"""
Django comparison: Stage 0 (our lightweight localizer) vs LocAgent original.

Picks N django instances that already have LocAgent raw_output_loc,
runs Stage 0 on them (Layer 1 + Layer 2 only, no LLM), and compares:
  - found_files coverage
  - found_entities match rate
  - graph skeleton quality

Usage:
    python scripts/compare_stage0_vs_locagent.py [--num 10]
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.localization.localizer import (
    FaultLocalizer,
    LocalizationResult,
    _parse_patch_files,
    _parse_patch_entities,
)


def parse_locagent_entities(raw_output_loc: list[str]) -> list[str]:
    """Extract file:entity pairs from LocAgent's raw_output_loc text."""
    entities = []
    current_file = None
    for text in raw_output_loc:
        for line in text.split("\n"):
            line = line.strip()
            # File path line (ends with .py, no prefix keyword)
            if line.endswith(".py") and not line.startswith(("line:", "function:", "class:", "Description:")):
                current_file = line
            elif current_file:
                m = re.match(r"function:\s*(.+)", line)
                if m:
                    func = m.group(1).strip()
                    entities.append(f"{current_file}:{func}")
    return entities


def compare_one(instance: dict) -> dict:
    """Compare Stage 0 vs LocAgent for a single instance."""
    iid = instance["instance_id"]
    patch = instance["patch"]
    test_patch = instance.get("test_patch", "")

    # --- LocAgent original ---
    loc_files = instance.get("found_files", []) or []
    loc_modules = instance.get("found_modules", []) or []
    loc_entities = instance.get("found_entities", []) or []
    loc_raw = instance.get("raw_output_loc", []) or []

    # Parse entities from LocAgent raw output if found_entities is empty
    if not loc_entities and loc_raw:
        loc_entities = parse_locagent_entities(loc_raw)

    # --- Stage 0 (Layer 1 + Layer 2 without graph) ---
    localizer = FaultLocalizer(llm_client=None, config={})
    result = localizer.localize(
        instance_id=iid,
        repo=instance["repo"],
        problem_statement=instance["problem_statement"],
        patch=patch,
        test_patch=test_patch,
        graph=None,  # No graph for quick comparison
    )

    # --- Ground truth: files actually modified in patch ---
    gt_files = _parse_patch_files(patch)
    gt_entities = _parse_patch_entities(patch)

    # --- Metrics ---
    def coverage(found: list, ground_truth: list) -> float:
        if not ground_truth:
            return 1.0
        return sum(1 for g in ground_truth if g in found) / len(ground_truth)

    def entity_overlap(a: list, b: list) -> float:
        """Fuzzy entity overlap: match by file + suffix of entity name."""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        matched = 0
        for ea in a:
            fa, na = (ea.split(":", 1) + [""])[:2]
            for eb in b:
                fb, nb = (eb.split(":", 1) + [""])[:2]
                if fa == fb and (na.endswith(nb) or nb.endswith(na) or na == nb):
                    matched += 1
                    break
        return matched / max(len(a), len(b))

    return {
        "instance_id": iid,
        "gt_files": gt_files,
        "gt_entities": gt_entities,
        # LocAgent
        "locagent_files": loc_files,
        "locagent_entities": loc_entities,
        "locagent_file_coverage": coverage(loc_files, gt_files),
        "locagent_entity_overlap": entity_overlap(loc_entities, gt_entities),
        # Stage 0
        "stage0_files": result.found_files,
        "stage0_entities": result.found_entities,
        "stage0_file_coverage": coverage(result.found_files, gt_files),
        "stage0_entity_overlap": entity_overlap(result.found_entities, gt_entities),
        "stage0_quality": result.quality,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=10, help="Number of instances to compare")
    parser.add_argument("--data", default="data/swebench_4repo.json")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    django_with_loc = [
        d for d in data
        if "django" in d.get("repo", "") and d.get("raw_output_loc")
    ]
    selected = django_with_loc[: args.num]
    print(f"Comparing {len(selected)} django instances\n")

    results = []
    for inst in selected:
        r = compare_one(inst)
        results.append(r)
        loc_fc = r["locagent_file_coverage"]
        s0_fc = r["stage0_file_coverage"]
        loc_eo = r["locagent_entity_overlap"]
        s0_eo = r["stage0_entity_overlap"]
        print(
            f"{r['instance_id']:40s} | "
            f"file_cov: LocAgent={loc_fc:.0%} Stage0={s0_fc:.0%} | "
            f"entity_overlap: LocAgent={loc_eo:.0%} Stage0={s0_eo:.0%}"
        )

    # Summary
    print("\n" + "=" * 80)
    n = len(results)
    avg_loc_fc = sum(r["locagent_file_coverage"] for r in results) / n
    avg_s0_fc = sum(r["stage0_file_coverage"] for r in results) / n
    avg_loc_eo = sum(r["locagent_entity_overlap"] for r in results) / n
    avg_s0_eo = sum(r["stage0_entity_overlap"] for r in results) / n
    print(f"Average file coverage:    LocAgent={avg_loc_fc:.1%}  Stage0={avg_s0_fc:.1%}")
    print(f"Average entity overlap:   LocAgent={avg_loc_eo:.1%}  Stage0={avg_s0_eo:.1%}")
    print(f"\nNote: Stage 0 sees the patch (by design), so file coverage should be ~100%.")
    print(f"LocAgent does NOT see the patch, so lower coverage is expected.")


if __name__ == "__main__":
    main()
