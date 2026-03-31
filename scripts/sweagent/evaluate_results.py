#!/usr/bin/env python3
"""
Evaluate SWE-agent results against TraceGen synthetic bugs.

For each (model, level) combination:
1. Load instances file (has injection_patch, FAIL_TO_PASS, image info)
2. Load trajectory files (has agent's submission patch)
3. For each instance with a submission:
   a. Start container from SWE-bench image (fixed state)
   b. Apply injection_patch (make buggy)
   c. Apply agent's patch (attempt fix)
   d. Run tests
   e. Check if FAIL_TO_PASS tests now pass

Usage:
    python scripts/sweagent/evaluate_results.py \
        --results-dir ../tracegen-outputs/verified_all_v2/sweagent_results_174 \
        --instances-dir ../tracegen-outputs/verified_all_v2/sweagent_instances_174 \
        --output-dir ../tracegen-outputs/verified_all_v2/sweagent_eval \
        --models qwen3-coder-plus qwen3-coder-flash qwen3.5-flash \
        --levels l1 l2 l3
"""
import argparse
import json
import glob
import os
import sys
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.modules.validation.docker_utils import (
    create_container,
    start_container,
    cleanup_container,
    exec_command,
    run_test_in_container,
    apply_patch,
)
from src.modules.validation.profiles.python import UnittestProfile, PythonProfile

from loguru import logger


def load_instances(instances_file):
    """Load instances file and return dict keyed by instance_id."""
    with open(instances_file) as f:
        instances = json.load(f)
    return {inst["instance_id"]: inst for inst in instances}


def extract_submission(traj_file):
    """Extract submission patch from trajectory file."""
    try:
        with open(traj_file) as f:
            data = json.load(f)
        info = data.get("info", {})
        return info.get("submission", "")
    except (json.JSONDecodeError, UnicodeDecodeError):
        # traj file may be still being written by SWE-agent
        return ""


def get_test_cmd(instance):
    """Determine the correct test command for an instance.

    Django: uses runtests.py with module-level labels (NOT pytest).
    F2P format: "test_method (module.Class)" → extract module name.
    See CLAUDE.md "SWE-agent 评测容器" and feedback-django-test-parser.md.
    """
    repo = instance.get("extra_fields", {}).get("repo", "")
    fail_to_pass = instance.get("extra_fields", {}).get("FAIL_TO_PASS", [])

    if "django" in repo.lower():
        # Django F2P is "test_method (module.submodule.TestClass)" format
        # runtests.py needs top-level module names only
        modules = set()
        for test in fail_to_pass:
            # Format: "test_method (module.submodule.Class)"
            if " (" in test and test.endswith(")"):
                inner = test.split(" (")[1].rstrip(")")
                # e.g. "migrations.test_autodetector.AutodetectorTests" → "migrations"
                top_module = inner.split(".")[0]
                modules.add(top_module)
            # Format: "tests/foo/bar.py::Class::method" (pytest style)
            elif test.startswith("tests/"):
                parts = test.replace("tests/", "").split("/")
                modules.add(parts[0])
            elif "::" in test:
                parts = test.split("::")[0].replace("tests/", "").split("/")
                modules.add(parts[0])
            # Format: "module.submodule.Class.method" (dotted)
            elif "." in test and "/" not in test and " " not in test:
                modules.add(test.split(".")[0])
        if modules:
            return f"./tests/runtests.py {' '.join(sorted(modules))} --verbosity=2"
        return "./tests/runtests.py --verbosity=2"
    else:
        test_files = set()
        for test in fail_to_pass:
            if "::" in test:
                test_files.add(test.split("::")[0])
            else:
                test_files.add(test)
        if test_files:
            return f"python -m pytest {' '.join(sorted(test_files))} -v --tb=short"
        return "python -m pytest -v --tb=short"


def get_log_parser(repo):
    """Get the appropriate log parser for a repo."""
    if "django" in repo.lower():
        return UnittestProfile().log_parser
    return PythonProfile().log_parser


def evaluate_single(instance, submission_patch):
    """Evaluate a single instance: apply injection + agent patch, run tests."""
    instance_id = instance["instance_id"]
    extra = instance.get("extra_fields", {})
    image_name = instance["image_name"]
    injection_patch = extra.get("injection_patch", "")
    fail_to_pass = extra.get("FAIL_TO_PASS", [])
    repo = extra.get("repo", "")

    result = {
        "instance_id": instance_id,
        "repo": repo,
        "has_submission": bool(submission_patch),
        "resolved": False,
        "error": None,
        "fail_to_pass_expected": fail_to_pass,
        "tests_passed": [],
        "tests_failed": [],
    }

    if not submission_patch or not submission_patch.strip():
        result["error"] = "no_submission"
        return result

    container = None
    try:
        container = create_container(
            image_name=image_name,
            instance_id=f"eval_{instance_id[:60]}",
            platform="linux/x86_64",
            memory_limit="4g",
        )
        start_container(container)

        # Step 1: Apply injection_patch (fixed → buggy)
        if injection_patch:
            try:
                apply_patch(container, injection_patch)
            except Exception as e:
                result["error"] = f"injection_patch_failed: {e}"
                return result

        # Step 2: Apply agent's patch (attempt buggy → fixed)
        # Agent wrote patch against buggy state (after injection). Context lines
        # should match, but git apply can still fail due to whitespace / fuzz.
        # Try multiple strategies before giving up.
        try:
            apply_patch(container, submission_patch, allow_rejects=True)
        except Exception:
            # Fallback: try git apply with increased fuzz via direct exec
            import base64
            patch_b64 = base64.b64encode(submission_patch.encode()).decode()
            write_cmd = f'/bin/bash -c "echo \'{patch_b64}\' | base64 -d > /tmp/agent.patch"'
            exec_command(container, write_cmd)
            # Try git apply --3way (uses merge strategy)
            r1 = exec_command(container, "/bin/bash -c 'cd /testbed && git apply --3way /tmp/agent.patch'")
            if r1.exit_code != 0:
                # Try patch with fuzz
                r2 = exec_command(container, "/bin/bash -c 'cd /testbed && patch --batch --fuzz=10 -p1 -i /tmp/agent.patch'")
                if r2.exit_code != 0:
                    result["error"] = f"agent_patch_failed: all strategies failed"
                    return result

        # Step 3: Run tests
        test_cmd = get_test_cmd(instance)
        test_result = run_test_in_container(container, test_cmd, timeout=300)
        test_output = test_result.output if hasattr(test_result, 'output') else str(test_result)
        result["test_output_tail"] = test_output[-3000:]

        # Step 4: Parse test results
        log_parser = get_log_parser(repo)
        test_status_map = log_parser(test_output)

        passed = [t for t, s in test_status_map.items() if s == "PASSED"]
        failed = [t for t, s in test_status_map.items() if s in ("FAILED", "ERROR")]
        result["tests_passed"] = passed[:100]
        result["tests_failed"] = failed[:100]

        # Step 5: Check FAIL_TO_PASS resolved
        if fail_to_pass:
            all_resolved = True
            for expected in fail_to_pass:
                found = any(expected in p or p in expected for p in passed)
                if not found:
                    all_resolved = False
                    break
            result["resolved"] = all_resolved
        else:
            result["resolved"] = len(failed) == 0

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error evaluating {instance_id}: {e}")
    finally:
        if container:
            try:
                cleanup_container(container)
            except Exception:
                pass

    return result


def evaluate_model_level(model, level, results_dir, instances_dir, eval_dir):
    """Evaluate all instances for a model/level combination."""
    instances_file = os.path.join(instances_dir, f"instances_{level}.json")
    traj_dir = os.path.join(results_dir, model, level)

    if not os.path.exists(instances_file) or not os.path.exists(traj_dir):
        logger.error(f"Missing files for {model}/{level}")
        return {}

    instance_map = load_instances(instances_file)

    # Load submissions
    submissions = {}
    for traj_file in glob.glob(os.path.join(traj_dir, "*/*.traj")):
        instance_id = os.path.basename(os.path.dirname(traj_file))
        sub = extract_submission(traj_file)
        if sub and sub.strip():
            submissions[instance_id] = sub

    logger.info(f"[{model}/{level}] {len(submissions)} submissions / {len(instance_map)} instances")

    # Check existing results - re-evaluate "no_submission" entries that now have submissions
    output_file = os.path.join(eval_dir, model, level, "eval_results.json")
    existing = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = {r["instance_id"]: r for r in json.load(f)}
        # Remove no_submission entries that now have submissions (new traj available)
        reeval = [iid for iid, r in existing.items()
                  if r.get("error") == "no_submission" and iid in submissions]
        for iid in reeval:
            del existing[iid]
        if reeval:
            logger.info(f"[{model}/{level}] Re-evaluating {len(reeval)} previously no_submission (now have traj)")
        logger.info(f"[{model}/{level}] Skipping {len(existing)} already evaluated")

    all_results = list(existing.values())
    to_eval = [(iid, sub) for iid, sub in submissions.items()
                if iid in instance_map and iid not in existing]

    # Also add instances without submissions (no_patch)
    for iid in instance_map:
        if iid not in existing and iid not in submissions:
            all_results.append({
                "instance_id": iid,
                "repo": instance_map[iid].get("extra_fields", {}).get("repo", ""),
                "has_submission": False,
                "resolved": False,
                "error": "no_submission",
            })

    for i, (iid, sub) in enumerate(to_eval):
        logger.info(f"[{model}/{level}] {i+1}/{len(to_eval)}: {iid[:50]}")
        result = evaluate_single(instance_map[iid], sub)
        all_results.append(result)

        # Save incrementally
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        if result["resolved"]:
            logger.info(f"  ✓ RESOLVED")
        elif result["error"]:
            logger.info(f"  ✗ ERROR: {result['error']}")
        else:
            logger.info(f"  ✗ not resolved")

    # Summary
    total = len(instance_map)
    resolved = sum(1 for r in all_results if r.get("resolved"))
    has_sub = sum(1 for r in all_results if r.get("has_submission"))
    errors = sum(1 for r in all_results if r.get("error") and r["error"] != "no_submission")

    summary = {
        "model": model, "level": level,
        "total": total, "has_submission": has_sub,
        "resolved": resolved, "errors": errors,
        "resolve_rate": f"{resolved}/{total} ({resolved/total*100:.1f}%)" if total else "N/A",
    }

    summary_file = os.path.join(eval_dir, model, level, "summary.json")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--instances-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["qwen3-coder-plus", "qwen3-coder-flash", "qwen3.5-flash"])
    parser.add_argument("--levels", nargs="+", default=["l1", "l2", "l3"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summaries = []

    for model in args.models:
        for level in args.levels:
            logger.info(f"\n{'='*60}\nEvaluating {model}/{level}\n{'='*60}")
            s = evaluate_model_level(model, level, args.results_dir, args.instances_dir, args.output_dir)
            if s:
                summaries.append(s)
                logger.info(f"  → {s['resolve_rate']}")

    # Final table
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print("{:<30} {:>6} {:>8} {:>8} {:>12}".format("Model/Level", "Total", "Patched", "Resolved", "Rate"))
    print("-" * 80)
    for s in summaries:
        print("{:<30} {:>6} {:>8} {:>8} {:>12}".format(
            f"{s['model']}/{s['level']}", s["total"], s["has_submission"], s["resolved"], s["resolve_rate"]))

    with open(os.path.join(args.output_dir, "overall_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
