#!/usr/bin/env python3
"""
Generate L1/L2/L3 problem statements for TraceGen VALID bugs.

Reads synthesis details + validation results, generates chain-guided PS
at three difficulty levels, outputs SWE-agent instances for each level.

Usage:
    python scripts/sweagent/generate_ps_levels.py \
        --run-dir ../tracegen-outputs/verified_30_fresh/2026-03-18/15-11-31 \
        --output-dir ../tracegen-outputs/verified_30_fresh/sweagent_ps_levels
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root on path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.modules.synthesis.ps_chain_guided import ChainGuidedPSGenerator


def find_test_output(val_dir: Path, instance_id: str) -> str:
    """Find test_output.txt for a validated instance."""
    # Try multiple path patterns
    for pattern in [
        val_dir / "logs" / "**" / instance_id / "test_output.txt",
    ]:
        matches = list(val_dir.glob(f"logs/**/{instance_id}/test_output.txt"))
        if matches:
            return matches[0].read_text(errors="replace")

    # Also check validation JSON for embedded test output
    val_file = val_dir / f"{instance_id}_validation.json"
    if val_file.exists():
        vd = json.loads(val_file.read_text())
        return vd.get("test_output", "") or ""

    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="TraceGen run directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for PS levels")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    details_dir = run_dir / "2_synthesis" / "details"
    val_dir = run_dir / "3_validation"

    # Load validation results - find VALID instances
    valid_ids = set()
    val_cache = {}
    for vf in val_dir.glob("*_validation.json"):
        vd = json.loads(vf.read_text())
        val_cache[vd["instance_id"]] = vd
        if vd.get("status") == "valid":
            valid_ids.add(vd["instance_id"])

    print(f"Found {len(valid_ids)} VALID instances")

    # Load synthesis details for VALID instances
    synth_map = {}
    for df in details_dir.glob("*.json"):
        d = json.loads(df.read_text())
        iid = d.get("instance_id", "")
        if iid in valid_ids:
            synth_map[iid] = d

    print(f"Matched {len(synth_map)} synthesis details")

    generator = ChainGuidedPSGenerator()
    all_results = []

    for iid, synth in sorted(synth_map.items()):
        meta = synth.get("metadata", {})
        chain_nodes = meta.get("proposed_chain", meta.get("synthesized_chain", []))
        if not isinstance(chain_nodes, list):
            chain_nodes = []
        injection_patch = meta.get("injection_patch", synth.get("patch", ""))
        original_ps = synth.get("problem_statement", "")

        # Find test output
        test_output = find_test_output(val_dir, iid)

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
            "repo": synth.get("repo", ""),
            "seed_id": synth.get("seed_id", meta.get("seed_instance_id", "")),
            "base_commit": synth.get("base_commit", ""),
            "original_ps": original_ps,
            "original_ps_tokens": len(original_ps.split()),
            "chain_node_count": len(chain_nodes),
            "has_test_output": bool(test_output),
            "extra_fields": {
                "seed_instance_id": synth.get("seed_id", meta.get("seed_instance_id", "")),
                "repo": synth.get("repo", ""),
                "injection_patch": injection_patch,
                "FAIL_TO_PASS": val_cache.get(iid, {}).get("PASS_TO_FAIL", []),
                "PASS_TO_PASS": val_cache.get(iid, {}).get("PASS_TO_PASS", []),
            },
        }

        for lvl in ("L1", "L2", "L3"):
            r = level_results[lvl]
            entry[f"ps_{lvl}"] = r.problem_statement
            entry[f"ps_{lvl}_tokens"] = r.metrics.get("token_count", 0)
            entry[f"ps_{lvl}_id_density"] = r.metrics.get("identifier_density", 0)

        all_results.append(entry)
        print(f"  {iid[:60]}: L1={entry['ps_L1_tokens']}w L2={entry['ps_L2_tokens']}w L3={entry['ps_L3_tokens']}w")

    # Save full results
    full_path = output_dir / "ps_all_levels.json"
    with open(full_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full results to {full_path}")

    # Generate SWE-agent instance files for each PS level
    for level in ("Original", "L1", "L2", "L3"):
        instances = []
        for r in all_results:
            seed_id = r["extra_fields"]["seed_instance_id"]
            docker_id = seed_id.replace("__", "_1776_")
            image = f"docker.io/swebench/sweb.eval.x86_64.{docker_id}:latest".lower()

            ps = r["original_ps"] if level == "Original" else r.get(f"ps_{level}", r["original_ps"])

            instances.append({
                "instance_id": r["instance_id"],
                "problem_statement": ps,
                "image_name": image,
                "repo_name": "testbed",
                "base_commit": r["base_commit"],
                "extra_fields": r["extra_fields"],
            })

        out_file = output_dir / f"instances_{level.lower()}.json"
        with open(out_file, "w") as f:
            json.dump(instances, f, indent=2, ensure_ascii=False)
        print(f"SWE-agent instances ({level}): {out_file} ({len(instances)} instances)")

    # Summary table
    print(f"\n{'Level':<10s} {'Avg Tokens':>12s} {'Avg ID Density':>15s}")
    print("-" * 40)
    for lvl in ("Original", "L1", "L2", "L3"):
        if lvl == "Original":
            tokens = [r["original_ps_tokens"] for r in all_results]
            ids = [0] * len(all_results)  # no metric for original
        else:
            tokens = [r.get(f"ps_{lvl}_tokens", 0) for r in all_results]
            ids = [r.get(f"ps_{lvl}_id_density", 0) for r in all_results]
        avg_t = sum(tokens) / len(tokens) if tokens else 0
        avg_id = sum(ids) / len(ids) if ids else 0
        print(f"{lvl:<10s} {avg_t:12.1f} {avg_id:15.3f}")


if __name__ == "__main__":
    main()
