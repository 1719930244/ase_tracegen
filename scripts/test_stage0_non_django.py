#!/usr/bin/env python
"""
Test Stage 0 with actual LLM calls on non-Django instances (pytest, sklearn, sympy).

These instances have NO LocAgent raw_output_loc, so Stage 0 is the only source.

Usage:
    DASHSCOPE_API_KEY=xxx .venv/bin/python scripts/test_stage0_non_django.py [--num 3] [--repo pytest]
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modules.localization.localizer import FaultLocalizer, _parse_patch_files
from src.modules.llm_client import create_llm_client


def count_loc_entries(text: str) -> dict:
    """Count structured entries in raw_output_loc text."""
    files = set(re.findall(r"^(\S+\.py)\s*$", text, re.MULTILINE))
    functions = re.findall(r"^function:\s*(.+)", text, re.MULTILINE)
    lines = re.findall(r"^line:\s*(.+)", text, re.MULTILINE)
    descriptions = re.findall(r"^Description:\s*(.+)", text, re.MULTILINE)
    return {
        "chars": len(text),
        "files": len(files),
        "functions": len(functions),
        "lines": len(lines),
        "descriptions": len(descriptions),
        "file_list": sorted(files),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=3)
    parser.add_argument("--repo", default="pytest", help="Filter by repo substring")
    parser.add_argument("--data", default="data/swebench_4repo.json")
    args = parser.parse_args()

    llm_config = {
        "provider": "openai",
        "model": "qwen3-coder-plus",
        "temperature": 0.5,
        "max_tokens": 8192,
        "timeout": 60,
        "max_retries": 3,
        "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/",
    }
    llm = create_llm_client(llm_config)
    localizer = FaultLocalizer(llm_client=llm, config={"max_depth": 6})

    with open(args.data) as f:
        data = json.load(f)

    # Filter non-django instances by repo substring
    candidates = [
        d for d in data
        if args.repo.lower() in d.get("repo", "").lower()
        and not d.get("raw_output_loc")  # No existing LocAgent data
    ]
    selected = candidates[: args.num]

    print(f"Testing Stage 0 with LLM on {len(selected)} {args.repo} instances\n")

    success = 0
    total_files = 0
    total_functions = 0
    gt_covered = 0

    for d in selected:
        iid = d["instance_id"]
        print(f"{'=' * 70}")
        print(f"Instance: {iid}")
        print(f"  Repo: {d['repo']}")
        print(f"{'=' * 70}")

        result = localizer.localize(
            instance_id=iid,
            repo=d["repo"],
            problem_statement=d["problem_statement"],
            patch=d["patch"],
            test_patch=d.get("test_patch", ""),
            graph=None,
        )
        s0_combined = "\n".join(result.raw_output_loc)
        stats = count_loc_entries(s0_combined)

        gt_files = set(_parse_patch_files(d["patch"]))
        s0_files = set(stats["file_list"])
        covers_gt = gt_files.issubset(s0_files)

        print(f"\n  Stage 0:  {stats['chars']:4d} chars, {stats['files']} files, "
              f"{stats['functions']} functions, {stats['descriptions']} descriptions")
        print(f"  Quality:  {result.quality.get('quality_score', 0):.2f}")
        print(f"  GT files:       {sorted(gt_files)}")
        print(f"  Stage 0 files:  {stats['file_list']}")
        print(f"  Covers GT: {covers_gt}")

        # Show first 600 chars
        print(f"\n  --- raw_output_loc (first 600 chars) ---")
        print("  " + s0_combined[:600].replace("\n", "\n  "))
        print()

        if stats["files"] > 0:
            success += 1
        total_files += stats["files"]
        total_functions += stats["functions"]
        if covers_gt:
            gt_covered += 1

    # Summary
    n = len(selected)
    print("=" * 70)
    print(f"Summary ({n} instances):")
    print(f"  Success (has files):  {success}/{n}")
    print(f"  Avg files:            {total_files/n:.1f}")
    print(f"  Avg functions:        {total_functions/n:.1f}")
    print(f"  GT coverage:          {gt_covered}/{n}")
    print("=" * 70)


if __name__ == "__main__":
    main()
