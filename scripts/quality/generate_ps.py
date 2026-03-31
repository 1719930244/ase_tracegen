#!/usr/bin/env python3
"""
generate_ps.py - Generate three-level problem statements for TraceGen synthetic bugs.

Extracts test failure information from test_output.txt and produces problem
statements at three information levels, enabling controlled evaluation of
debugging agent capabilities.

Level 1 (Minimal):  "Tests in module X fail."
Level 2 (Standard): Assertion message + behavioral description (SWE-bench style).
Level 3 (Detailed): Full traceback + failure context.

Usage:
    python scripts/quality/generate_ps.py \
        --synth-dir ../tracegen-outputs/4repo_run/2026-02-25/19-52-01 \
        --output ../tracegen-outputs/problem_statements_django.json
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Match FAIL/ERROR header lines
_FAIL_HEADER_RE = re.compile(
    r"^(FAIL|ERROR): (\S+) \(([^)]+)\)(?:\s*\n(.+))?", re.MULTILINE
)

# Match traceback blocks (from "Traceback" to the exception line)
_TRACEBACK_RE = re.compile(
    r"(Traceback \(most recent call last\):.*?^([A-Z]\w*(?:Error|Exception)[^\n]*))",
    re.MULTILINE | re.DOTALL,
)

# Match assertion error messages specifically
_ASSERTION_MSG_RE = re.compile(
    r"^(Assertion(?:Error)?:?\s*.+?)$", re.MULTILINE
)

# Match test file references in traceback
_TEST_FILE_RE = re.compile(
    r'File "(/testbed/tests/[^"]+)", line (\d+), in (\w+)'
)

# Match source file references in traceback
_SOURCE_FILE_RE = re.compile(
    r'File "(/testbed/(?!tests/)[^"]+)", line (\d+), in (\w+)'
)

# Summary line: "Ran N tests in X.XXXs"
_RAN_RE = re.compile(r"Ran (\d+) tests? in ([\d.]+)s")

# Result line: "FAILED (failures=N, errors=M)"
_RESULT_RE = re.compile(r"FAILED \(([^)]+)\)")


def parse_test_output(content: str) -> Dict:
    """Parse test_output.txt and extract structured failure information."""
    result = {
        "failing_tests": [],
        "tracebacks": [],
        "assertion_messages": [],
        "test_modules": set(),
        "source_files": set(),
        "total_tests": 0,
        "total_failures": 0,
        "total_errors": 0,
    }

    # Parse FAIL/ERROR headers
    for m in _FAIL_HEADER_RE.finditer(content):
        fail_type = m.group(1)  # FAIL or ERROR
        test_method = m.group(2)
        test_class = m.group(3)
        description = m.group(4) or ""
        module = test_class.rsplit(".", 1)[0] if "." in test_class else test_class
        result["failing_tests"].append({
            "type": fail_type,
            "method": test_method,
            "class": test_class,
            "module": module,
            "description": description.strip(),
        })
        result["test_modules"].add(module)

    # Parse tracebacks
    for m in _TRACEBACK_RE.finditer(content):
        full_tb = m.group(1).strip()
        exception_line = m.group(2).strip()
        result["tracebacks"].append({
            "full": full_tb,
            "exception": exception_line,
        })

    # Parse assertion messages
    for m in _ASSERTION_MSG_RE.finditer(content):
        msg = m.group(1).strip()
        if len(msg) > 500:
            msg = msg[:500] + "..."
        result["assertion_messages"].append(msg)

    # Parse source file references
    for m in _SOURCE_FILE_RE.finditer(content):
        filepath = m.group(1).replace("/testbed/", "")
        result["source_files"].add(filepath)

    # Parse summary
    ran_match = _RAN_RE.search(content)
    if ran_match:
        result["total_tests"] = int(ran_match.group(1))

    result_match = _RESULT_RE.search(content)
    if result_match:
        parts = result_match.group(1)
        for part in parts.split(","):
            part = part.strip()
            if part.startswith("failures="):
                result["total_failures"] = int(part.split("=")[1])
            elif part.startswith("errors="):
                result["total_errors"] = int(part.split("=")[1])

    # Convert sets to sorted lists
    result["test_modules"] = sorted(result["test_modules"])
    result["source_files"] = sorted(result["source_files"])

    return result


# ---------------------------------------------------------------------------
# Problem statement generators
# ---------------------------------------------------------------------------

def generate_level1(parsed: Dict, repo: str) -> str:
    """Level 1 (Minimal): Only the symptom, no details."""
    modules = parsed["test_modules"]
    n_fail = parsed["total_failures"] + parsed["total_errors"]

    if not modules:
        return f"Some tests in {repo} are failing."

    if len(modules) == 1:
        return f"{n_fail} test(s) in {modules[0]} are failing."
    else:
        mod_list = ", ".join(modules[:3])
        if len(modules) > 3:
            mod_list += f" and {len(modules) - 3} other module(s)"
        return f"{n_fail} test(s) across {mod_list} are failing."


def generate_level2(parsed: Dict, original_ps: str) -> str:
    """Level 2 (Standard): Assertion message + behavioral description."""
    parts = []

    # Use the original PS as base if it's good
    if original_ps and len(original_ps.split()) >= 10:
        parts.append(original_ps.strip())
    else:
        # Build from test output
        for ft in parsed["failing_tests"][:3]:
            desc = ft["description"]
            if desc:
                parts.append(f"Test `{ft['method']}` ({ft['class']}): {desc}")
            else:
                parts.append(f"Test `{ft['method']}` in `{ft['class']}` fails.")

    # Add assertion details if not already present
    for msg in parsed["assertion_messages"][:2]:
        # Skip if assertion info already in original PS
        if original_ps and msg[:40] in original_ps:
            continue
        parts.append(f"Observed: {msg}")

    return "\n\n".join(parts)


def generate_level3(parsed: Dict, original_ps: str) -> str:
    """Level 3 (Detailed): Full traceback + failure context."""
    parts = []

    # Start with Level 2 content
    if original_ps and len(original_ps.split()) >= 10:
        parts.append(original_ps.strip())

    # Add full tracebacks
    for i, tb in enumerate(parsed["tracebacks"][:3]):
        parts.append(f"```\n{tb['full']}\n```")

    # Add source file hints
    if parsed["source_files"]:
        files = ", ".join(f"`{f}`" for f in parsed["source_files"][:5])
        parts.append(f"Source files involved: {files}")

    # Add test summary
    parts.append(
        f"Total: {parsed['total_tests']} tests, "
        f"{parsed['total_failures']} failures, "
        f"{parsed['total_errors']} errors."
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_test_output(val_logs_dir: Path, instance_id: str) -> Optional[Path]:
    """Find test_output.txt for a given instance."""
    for p in val_logs_dir.rglob(f"{instance_id}/test_output.txt"):
        return p
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate three-level problem statements")
    parser.add_argument("--synth-dir", required=True, help="Stage output directory")
    parser.add_argument("--output", default="../tracegen-outputs/problem_statements_django.json")
    parser.add_argument("--repo-filter", default=None)
    args = parser.parse_args()

    synth_dir = Path(args.synth_dir)
    synth_file = synth_dir / "2_synthesis" / "final_dataset.json"
    val_dir = synth_dir / "3_validation"
    val_logs_dir = val_dir / "logs"

    # Load synthesis data
    print(f"Loading synthesis data from {synth_file}")
    with open(synth_file) as f:
        synth_data = json.load(f)
    synth_map = {d["instance_id"]: d for d in synth_data}

    # Load validation data
    val_files = sorted(val_dir.glob("synthetic_*_validation.json"))
    print(f"  {len(val_files)} validation files")

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

        original_ps = synth.get("problem_statement", "")

        # Find and parse test output
        test_output_path = find_test_output(val_logs_dir, iid)
        if not test_output_path:
            results.append({
                "instance_id": iid,
                "repo": repo,
                "level1": f"Some tests in {repo} are failing.",
                "level2": original_ps,
                "level3": original_ps,
                "parsed_info": {},
            })
            continue

        content = test_output_path.read_text(errors="replace")
        parsed = parse_test_output(content)

        level1 = generate_level1(parsed, repo)
        level2 = generate_level2(parsed, original_ps)
        level3 = generate_level3(parsed, original_ps)

        results.append({
            "instance_id": iid,
            "repo": repo,
            "level1": level1,
            "level2": level2,
            "level3": level3,
            "parsed_info": {
                "failing_tests": len(parsed["failing_tests"]),
                "assertion_messages": parsed["assertion_messages"][:3],
                "source_files": parsed["source_files"][:5],
                "test_modules": parsed["test_modules"],
                "total_tests": parsed["total_tests"],
                "total_failures": parsed["total_failures"],
                "total_errors": parsed["total_errors"],
            },
        })

    results.sort(key=lambda x: x["instance_id"])

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated {len(results)} three-level problem statements → {output_path}")

    # Quick stats
    if results:
        l1_lens = [len(r["level1"].split()) for r in results]
        l2_lens = [len(r["level2"].split()) for r in results]
        l3_lens = [len(r["level3"].split()) for r in results]
        print(f"\nToken counts (mean):")
        print(f"  Level 1: {sum(l1_lens)/len(l1_lens):.0f}")
        print(f"  Level 2: {sum(l2_lens)/len(l2_lens):.0f}")
        print(f"  Level 3: {sum(l3_lens)/len(l3_lens):.0f}")


if __name__ == "__main__":
    main()
