#!/usr/bin/env python3
"""
Convert TraceGen valid synthetic bugs to SWE-agent expert instances format.

Each synthetic bug needs:
1. The seed's SWE-bench Docker image (environment is set up for that commit)
2. injection_patch applied via post_startup_commands
3. problem_statement for the agent to solve

Usage:
    python scripts/sweagent/convert_to_sweagent.py \
        --dataset ../tracegen-outputs/4repo_run/2026-02-25/19-52-01/2_synthesis/final_dataset.json \
        --val-dir ../tracegen-outputs/4repo_run/2026-02-25/19-52-01/3_validation \
        --output scripts/sweagent/instances.json \
        --repo-filter django \
        --limit 3
"""

import argparse
import json
import glob
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


def seed_id_to_image(seed_id: str) -> str:
    """Convert seed instance_id to SWE-bench Docker image name.

    e.g. django__django-11099 -> swebench/sweb.eval.x86_64.django_1776_django-11099:latest
    """
    docker_id = seed_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{docker_id}:latest".lower()


def seed_id_to_repo_path(seed_id: str) -> str:
    """Convert seed instance_id to repo path inside the container.

    e.g. django__django-11099 -> /testbed (SWE-bench standard)
    """
    return "testbed"


def load_valid_ids(val_dir: Path) -> set:
    """Load valid instance IDs from validation results.

    In TraceGen validation, PASS_TO_FAIL = tests broken by injection (= SWE-bench FAIL_TO_PASS).
    A valid bug has status=='valid' or non-empty PASS_TO_FAIL.
    """
    valid_ids = set()
    for vf in sorted(val_dir.glob("*_validation.json")):
        with open(vf) as fh:
            vd = json.load(fh)
        if vd.get("status") == "valid":
            valid_ids.add(vd["instance_id"])
    return valid_ids


def make_apply_patch_script(injection_patch: str) -> str:
    """Create a shell command to apply injection_patch inside the container."""
    # Escape single quotes for shell embedding
    escaped = injection_patch.replace("'", "'\\''")
    return f"cd /testbed && echo '{escaped}' | git apply"


def convert_instance(entry: Dict[str, Any], val_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single TraceGen instance to SWE-agent expert format."""
    meta = entry.get("metadata", {})
    seed_id = meta.get("seed_instance_id", "")
    injection_patch = meta.get("injection_patch", "")

    image_name = seed_id_to_image(seed_id)

    # Build post_startup_commands to apply the injection patch
    post_commands = []
    if injection_patch:
        post_commands.append(make_apply_patch_script(injection_patch))

    # SWE-agent expert instance format
    instance = {
        "instance_id": entry["instance_id"],
        "problem_statement": entry.get("problem_statement", ""),
        "image_name": image_name,
        "repo_name": "testbed",
        "base_commit": entry.get("base_commit", "HEAD"),
        # Extra fields for reference (not used by SWE-agent directly)
        "extra_fields": {
            "seed_instance_id": seed_id,
            "repo": entry.get("repo", ""),
            "injection_patch": injection_patch,
            # TraceGen PASS_TO_FAIL = SWE-bench FAIL_TO_PASS (tests broken by the bug)
            "FAIL_TO_PASS": val_data.get("PASS_TO_FAIL", []),
            "PASS_TO_PASS": val_data.get("PASS_TO_PASS", []),
        }
    }

    return instance, post_commands


def main():
    parser = argparse.ArgumentParser(description="Convert TraceGen bugs to SWE-agent format")
    parser.add_argument("--dataset", required=True, help="final_dataset.json path")
    parser.add_argument("--val-dir", required=True, help="Stage 3 validation directory")
    parser.add_argument("--output", default="scripts/sweagent/instances.json", help="Output JSON path")
    parser.add_argument("--repo-filter", default=None, help="Filter by repo (e.g. django)")
    parser.add_argument("--limit", type=int, default=0, help="Max instances (0=all)")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)

    # Load valid IDs
    valid_ids = load_valid_ids(val_dir)
    print(f"Found {len(valid_ids)} valid instances")

    # Load dataset
    with open(args.dataset) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} entries from dataset")

    # Load validation data for F2P/P2P
    val_cache = {}
    for vf in val_dir.glob("*_validation.json"):
        with open(vf) as fh:
            vd = json.load(fh)
        val_cache[vd["instance_id"]] = vd

    # Convert
    instances = []
    patch_commands = {}  # instance_id -> post_startup_commands

    for entry in dataset:
        iid = entry["instance_id"]
        if iid not in valid_ids:
            continue

        repo = entry.get("repo", "")
        if args.repo_filter and args.repo_filter.lower() not in repo.lower():
            continue

        val_data = val_cache.get(iid, {})
        instance, post_cmds = convert_instance(entry, val_data)
        instances.append(instance)
        if post_cmds:
            patch_commands[iid] = post_cmds

        if args.limit and len(instances) >= args.limit:
            break

    print(f"Converted {len(instances)} instances")

    # Write simple instances file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(instances, f, indent=2, ensure_ascii=False)
    print(f"Written to {output_path}")

    # Write patch commands separately (for manual or wrapper use)
    patch_file = output_path.with_name(output_path.stem + "_patches.json")
    with open(patch_file, "w") as f:
        json.dump(patch_commands, f, indent=2, ensure_ascii=False)
    print(f"Patch commands written to {patch_file}")

    # Print summary
    for inst in instances[:3]:
        print(f"\n  {inst['instance_id'][:50]}")
        print(f"    image: {inst['image_name'][:70]}")
        print(f"    seed: {inst['extra_fields']['seed_instance_id']}")
        print(f"    PS: {inst['problem_statement'][:80]}...")


if __name__ == "__main__":
    main()
