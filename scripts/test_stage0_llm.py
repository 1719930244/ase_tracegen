#!/usr/bin/env python
"""
Test Stage 0 with actual LLM calls on Django instances.

Compares Stage 0 LLM output vs LocAgent original raw_output_loc.

Usage:
    DASHSCOPE_API_KEY=xxx .venv/bin/python scripts/test_stage0_llm.py [--num 3]
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
    parser.add_argument("--data", default="data/swebench_4repo.json")
    args = parser.parse_args()

    # Init LLM client (same config as analyzer_llm)
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

    django_with_loc = [
        d for d in data
        if "django" in d.get("repo", "") and d.get("raw_output_loc")
    ]
    selected = django_with_loc[: args.num]

    print(f"Testing Stage 0 with LLM on {len(selected)} Django instances\n")

    for d in selected:
        iid = d["instance_id"]
        print(f"{'=' * 70}")
        print(f"Instance: {iid}")
        print(f"{'=' * 70}")

        # LocAgent original
        loc_combined = "\n".join(d["raw_output_loc"])
        loc_stats = count_loc_entries(loc_combined)

        # Stage 0 with LLM
        result = localizer.localize(
            instance_id=iid,
            repo=d["repo"],
            problem_statement=d["problem_statement"],
            patch=d["patch"],
            test_patch=d.get("test_patch", ""),
            graph=None,  # No graph for this test
        )
        s0_combined = "\n".join(result.raw_output_loc)
        s0_stats = count_loc_entries(s0_combined)

        print(f"\n  LocAgent:  {loc_stats['chars']:4d} chars, {loc_stats['files']} files, "
              f"{loc_stats['functions']} functions, {loc_stats['descriptions']} descriptions")
        print(f"  Stage 0:   {s0_stats['chars']:4d} chars, {s0_stats['files']} files, "
              f"{s0_stats['functions']} functions, {s0_stats['descriptions']} descriptions")
        print(f"  Quality:   {result.quality.get('quality_score', 0):.2f}")
        print(f"\n  LocAgent files: {loc_stats['file_list']}")
        print(f"  Stage 0 files:  {s0_stats['file_list']}")

        # Check overlap
        loc_files = set(loc_stats["file_list"])
        s0_files = set(s0_stats["file_list"])
        gt_files = set(_parse_patch_files(d["patch"]))
        print(f"  GT files:       {sorted(gt_files)}")
        print(f"  Stage0 covers GT: {gt_files.issubset(s0_files)}")
        print()

        # Show first 500 chars of Stage 0 output
        print(f"  --- Stage 0 raw_output_loc (first 500 chars) ---")
        print("  " + s0_combined[:500].replace("\n", "\n  "))
        print()


if __name__ == "__main__":
    main()
