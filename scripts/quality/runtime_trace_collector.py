"""
Runtime Trace Collector for TraceGen Synthetic Bugs.

Collects actual runtime failure propagation chains by:
1. Injecting a sys.settrace()-based tracer into a Docker container
2. Running F2P tests on the buggy code
3. Capturing the call chain from test entry → assertion failure
4. Comparing actual chains with designed (synthetic_chain) chains

Usage:
    python scripts/quality/runtime_trace_collector.py \
        --output /path/to/runtime_traces.json \
        [--num-workers 3] [--limit 10]
"""

import argparse
import json
import os
import sys
import time
import traceback as tb_module
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.modules.validation.docker_utils import (
    create_container, start_container, cleanup_container,
    exec_command, apply_patch, PatchApplicationError,
)

# ── Paths ──────────────────────────────────────────────────────────────────
SYNTH_SUMMARY = "../tracegen-outputs/4repo_run/2026-02-25/19-52-01/2_synthesis/summary.json"
VAL_DIR = "../tracegen-outputs/4repo_run/2026-02-25/19-52-01/3_validation"
IMAGE = "localhost/sweagent/django-swesmith-swerex:latest"
TIMEOUT = 300
MEMORY = "10g"

# ── Trace Collector Script (injected into container) ───────────────────────
TRACE_SCRIPT = r'''
"""
sys.settrace-based runtime call chain collector.
Wraps Django runtests.py to capture function calls through /testbed/ source.
Usage: python trace_collector.py <test_module> [test_module2 ...]
"""
import sys
import json
import os
import threading

# Args: test modules (same as runtests.py args)
test_modules = sys.argv[1:]

# Collect call chain
call_chain = []
seen_calls = set()
TESTBED = "/testbed/"
EXCLUDE = {"/unittest/", "/lib/python", "/site-packages/", "/trace_collector.py",
           "/runtests.py", "/django/test/", "/django/db/"}

def should_trace(filename):
    if not filename or TESTBED not in filename:
        return False
    for exc in EXCLUDE:
        if exc in filename:
            return False
    rel = filename.split(TESTBED, 1)[-1] if TESTBED in filename else filename
    if rel.startswith("tests/"):
        return False
    return True

def trace_func(frame, event, arg):
    filename = frame.f_code.co_filename or ""
    func_name = frame.f_code.co_name

    if event == "call":
        if should_trace(filename):
            rel_path = filename.split(TESTBED, 1)[-1] if TESTBED in filename else filename
            entry = {
                "file": rel_path,
                "function": func_name,
                "line": frame.f_lineno,
                "caller_file": "",
                "caller_func": "",
                "caller_line": 0,
            }
            caller = frame.f_back
            if caller:
                cf = caller.f_code.co_filename or ""
                if TESTBED in cf:
                    entry["caller_file"] = cf.split(TESTBED, 1)[-1]
                else:
                    entry["caller_file"] = os.path.basename(cf)
                entry["caller_func"] = caller.f_code.co_name
                entry["caller_line"] = caller.f_lineno
            key = (rel_path, func_name, entry["caller_file"], entry["caller_func"])
            if key not in seen_calls:
                seen_calls.add(key)
                call_chain.append(entry)
        return trace_func

    elif event == "exception":
        if should_trace(filename):
            exc_type, exc_value, exc_tb = arg
            rel_path = filename.split(TESTBED, 1)[-1] if TESTBED in filename else filename
            key = (rel_path, func_name, "exception", str(exc_type))
            if key not in seen_calls:
                seen_calls.add(key)
                call_chain.append({
                    "file": rel_path,
                    "function": func_name,
                    "line": frame.f_lineno,
                    "event": "exception",
                    "exception_type": exc_type.__name__ if exc_type else "",
                    "exception_msg": str(exc_value)[:200] if exc_value else "",
                })
        return trace_func

    return trace_func

# Use runtests.py machinery
os.chdir("/testbed/tests")
sys.path.insert(0, "/testbed/tests")
sys.path.insert(0, "/testbed")

# Import runtests setup
try:
    # Django runtests.py setup
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "test_sqlite")
    import django
    from django.conf import settings
    django.setup()
    from django.test.utils import get_runner
    TestRunner = get_runner(settings)
    test_runner = TestRunner(verbosity=0, parallel=1, failfast=False)
    suite = test_runner.build_suite(test_modules)

    # Install tracer
    threading.settrace(trace_func)
    sys.settrace(trace_func)

    # Run tests
    result = test_runner.run_suite(suite)

    sys.settrace(None)
    threading.settrace(None)

    failures = len(result.failures) + len(result.errors)
    error_msgs = []
    for _, msg in result.failures[:3]:
        error_msgs.append(msg[:300])
    for _, msg in result.errors[:3]:
        error_msgs.append(msg[:300])

    output = {
        "status": "fail" if failures > 0 else "pass",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "error_message": "\n---\n".join(error_msgs)[:500],
        "call_chain": call_chain,
        "total_calls": len(call_chain),
    }
except Exception as e:
    sys.settrace(None)
    output = {
        "status": "error",
        "error_message": str(e)[:500],
        "call_chain": call_chain,
        "total_calls": len(call_chain),
    }

print("===TRACE_JSON_START===")
print(json.dumps(output))
print("===TRACE_JSON_END===")
'''


def parse_f2p_test(test_str: str):
    """Parse F2P test string into (module, class, method) tuple.

    Django test format examples:
        'test_slugify (utils_tests.test_text.TestUtilsText)'
        'Built-in decorators set certain attributes... (decorators.tests.DecoratorsTest)'
    """
    import re
    m = re.match(r'^(\S+)\s+\(([^)]+)\)$', test_str)
    if not m:
        return None
    method = m.group(1)
    full_path = m.group(2)
    parts = full_path.rsplit('.', 1)
    if len(parts) == 2:
        module, cls = parts
        return (module, cls, method)
    return None


def collect_trace_for_instance(instance_id, bug_info, val_info):
    """Collect runtime trace for one synthetic bug."""
    result = {
        "instance_id": instance_id,
        "designed_chain": bug_info.get("synthetic_chain", []),
        "chain_depth": bug_info.get("synthetic_chain_depth", 0),
        "injection_strategy": bug_info.get("synthetic_injection_strategy", ""),
        "traces": [],
        "error": "",
    }

    injection_patch = bug_info.get("synthetic_injection_patch", "")
    f2p_tests = val_info.get("PASS_TO_FAIL", [])

    if not injection_patch or not f2p_tests:
        result["error"] = "Missing injection patch or F2P tests"
        return result

    container = None
    try:
        container = create_container(
            image_name=IMAGE, instance_id=f"trace_{instance_id[-20:]}",
            memory_limit=MEMORY,
        )
        start_container(container)

        # Apply the bug-introducing patch
        try:
            apply_patch(container, injection_patch)
        except PatchApplicationError as e:
            result["error"] = f"PATCH_FAIL: {e}"
            return result

        # Inject trace collector script
        exec_command(container, ["bash", "-c", f"cat > /testbed/trace_collector.py << 'PYEOF'\n{TRACE_SCRIPT}\nPYEOF"])

        # Extract test modules from F2P tests
        test_modules = set()
        for t in f2p_tests:
            parsed = parse_f2p_test(t)
            if parsed:
                module = parsed[0]
                # Convert dotted module to runtests.py arg format
                # e.g., "utils_tests.test_text" stays as is
                test_modules.add(module.rsplit('.', 1)[0] if '.' in module else module)

        if not test_modules:
            result["error"] = "Cannot determine test modules from F2P tests"
            return result

        # Run trace collector with test modules (same args as runtests.py)
        modules_str = ' '.join(sorted(test_modules))
        cmd = f"cd /testbed/tests && python /testbed/trace_collector.py {modules_str}"

        exec_result = exec_command(container, ["bash", "-c", cmd])
        output = exec_result.output

        # Parse trace JSON from output
        trace_data = None
        if "===TRACE_JSON_START===" in output and "===TRACE_JSON_END===" in output:
            json_str = output.split("===TRACE_JSON_START===")[1].split("===TRACE_JSON_END===")[0].strip()
            try:
                trace_data = json.loads(json_str)
            except json.JSONDecodeError:
                pass

        if trace_data:
            # Extract unique source files from call chain (in order of first appearance)
            source_files = []
            seen = set()
            for call in trace_data.get("call_chain", []):
                f = call.get("file", "")
                if f and f not in seen:
                    seen.add(f)
                    source_files.append(f)

            result["traces"].append({
                "test_modules": sorted(test_modules),
                "f2p_tests": f2p_tests,
                "status": trace_data.get("status"),
                "tests_run": trace_data.get("tests_run", 0),
                "total_calls": trace_data.get("total_calls", 0),
                "call_chain": trace_data.get("call_chain", []),
                "source_files_touched": source_files,
                "error_message": trace_data.get("error_message", "")[:300],
            })
        else:
            result["traces"].append({
                "test_modules": sorted(test_modules),
                "error": f"No trace output. Exit={exec_result.exit_code}. Raw: {output[-500:]}",
            })

        return result

    except Exception as e:
        result["error"] = f"Exception: {e}\n{tb_module.format_exc()}"
        return result
    finally:
        cleanup_container(container)


def compare_chains(designed_chain, actual_trace):
    """Compare designed chain with actual runtime trace."""
    # Extract designed files (normalize path separators)
    designed_files = []
    for step in designed_chain:
        fp = step.get("file_path", "")
        # Convert django.utils.text.py → django/utils/text.py
        fp_normalized = fp.replace(".", "/").replace("/py", ".py")
        designed_files.append({
            "file": fp_normalized,
            "node_type": step.get("node_type", ""),
            "node_id": step.get("node_id", ""),
        })

    # Extract actual files from trace
    actual_files = actual_trace.get("source_files_touched", [])

    # Compute overlap
    designed_set = set(d["file"] for d in designed_files)
    actual_set = set(actual_files)

    overlap = designed_set & actual_set
    designed_only = designed_set - actual_set
    actual_only = actual_set - designed_set

    # Check if root_cause file appears in actual trace
    root_cause_files = [d["file"] for d in designed_files if d["node_type"] == "root_cause"]
    root_hit = any(rc in actual_set for rc in root_cause_files)

    # Check if symptom file appears in actual trace
    symptom_files = [d["file"] for d in designed_files if d["node_type"] == "symptom"]
    symptom_hit = any(sf in actual_set for sf in symptom_files)

    return {
        "designed_files": [d["file"] for d in designed_files],
        "actual_files": actual_files,
        "overlap": sorted(overlap),
        "designed_only": sorted(designed_only),
        "actual_only": sorted(actual_only),
        "overlap_ratio": len(overlap) / max(len(designed_set), 1),
        "root_cause_hit": root_hit,
        "symptom_hit": symptom_hit,
    }


def main():
    parser = argparse.ArgumentParser(description="Runtime trace collector for TraceGen bugs")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of instances (0=all)")
    args = parser.parse_args()

    # Load data
    summary = json.load(open(SYNTH_SUMMARY))
    valid_bugs = {s["synthetic_instance_id"]: s for s in summary if s.get("validation_status") == "valid"}

    val_map = {}
    for vf in Path(VAL_DIR).glob("*_validation.json"):
        v = json.load(open(vf))
        if v.get("status") == "valid":
            val_map[v["instance_id"]] = v

    # Build task list
    tasks = []
    for iid, bug in valid_bugs.items():
        if iid in val_map:
            tasks.append((iid, bug, val_map[iid]))

    if args.limit > 0:
        tasks = tasks[:args.limit]

    print(f"Collecting runtime traces for {len(tasks)} instances ({args.num_workers} workers)")

    # Resume support
    existing = {}
    if os.path.exists(args.output):
        try:
            existing = {r["instance_id"]: r for r in json.load(open(args.output))}
            print(f"Resuming: {len(existing)} already done")
            tasks = [(iid, b, v) for iid, b, v in tasks if iid not in existing]
        except:
            pass

    results = list(existing.values())
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(collect_trace_for_instance, *t): t[0] for t in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            iid = futures[future]
            try:
                r = future.result()
                results.append(r)
                n_traces = len(r.get("traces", []))
                n_calls = sum(t.get("total_calls", 0) for t in r.get("traces", []))
                status = "ERR" if r.get("error") else "OK"
                print(f"[{i}/{len(tasks)}] {status} {iid[-25:]} traces={n_traces} calls={n_calls} {r.get('error','')[:40]}")

                # Save incrementally
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                print(f"[{i}/{len(tasks)}] CRASH {iid}: {e}")

    elapsed = time.time() - start_time

    # Compute chain comparisons
    print(f"\n{'='*60}")
    print(f"Computing chain comparisons...")

    for r in results:
        if r.get("error") or not r.get("traces"):
            r["chain_comparison"] = None
            continue
        # Use first successful trace for comparison
        for trace in r["traces"]:
            if trace.get("source_files_touched"):
                r["chain_comparison"] = compare_chains(
                    r.get("designed_chain", []), trace
                )
                break
        else:
            r["chain_comparison"] = None

    # Save final
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary stats
    has_trace = sum(1 for r in results if any(t.get("total_calls", 0) > 0 for t in r.get("traces", [])))
    has_comparison = sum(1 for r in results if r.get("chain_comparison"))
    root_hits = sum(1 for r in results if r.get("chain_comparison", {}) and r["chain_comparison"].get("root_cause_hit"))
    symptom_hits = sum(1 for r in results if r.get("chain_comparison", {}) and r["chain_comparison"].get("symptom_hit"))
    avg_overlap = 0
    overlaps = [r["chain_comparison"]["overlap_ratio"] for r in results if r.get("chain_comparison")]
    if overlaps:
        avg_overlap = sum(overlaps) / len(overlaps)

    print(f"\nDONE ({elapsed:.0f}s)")
    print(f"Total: {len(results)}")
    print(f"Has runtime trace: {has_trace}")
    print(f"Has chain comparison: {has_comparison}")
    print(f"Root cause file hit: {root_hits}/{has_comparison}")
    print(f"Symptom file hit: {symptom_hits}/{has_comparison}")
    print(f"Avg overlap ratio: {avg_overlap:.2f}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
