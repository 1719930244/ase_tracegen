#!/usr/bin/env python3
"""TraceGen Opus 运行监控脚本 - 持续跟踪进度直到完成"""
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = PROJECT_ROOT.parent / "tracegen-outputs"
OUTPUT_DIR = OUTPUTS_ROOT / "opus_run/2026-02-23/22-26-37"
LOG_FILE = PROJECT_ROOT / "logs/opus_run_20260223_222635.log"
PID = 42679

def check_process_alive():
    try:
        os.kill(PID, 0)
        return True
    except OSError:
        return False

def count_synthesis_results():
    synth_dir = OUTPUT_DIR / "2_synthesis" / "details"
    if not synth_dir.exists():
        return 0
    return len(list(synth_dir.glob("synthetic_*.json")))

def count_validation_results():
    valid_dir = OUTPUT_DIR / "3_validation"
    if not valid_dir.exists():
        return {"valid": 0, "invalid": 0, "timeout": 0, "error": 0, "total": 0}
    results = {"valid": 0, "invalid": 0, "timeout": 0, "error": 0, "total": 0}
    for f in valid_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            status = data.get("validation_status", "unknown")
            results[status] = results.get(status, 0) + 1
            results["total"] += 1
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.debug(f"Failed to read validation result {f}: {e}")
    return results

def get_resource_usage():
    log_file = OUTPUT_DIR / "logs" / "llm_resource_usage.jsonl"
    if not log_file.exists():
        return None
    try:
        total_input = 0
        total_output = 0
        calls = 0
        for line in log_file.read_text().strip().split("\n"):
            if line.strip():
                r = json.loads(line)
                total_input += r.get("input_tokens", 0)
                total_output += r.get("output_tokens", 0)
                calls += 1
        return {"calls": calls, "input_tokens": total_input, "output_tokens": total_output}
    except (json.JSONDecodeError, ValueError, OSError) as e:
        logger.debug(f"Failed to read resource usage: {e}")
        return None

def get_last_log_line():
    try:
        with open(LOG_FILE, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            pos = max(0, size - 2000)
            f.seek(pos)
            lines = f.read().decode("utf-8", errors="replace").strip().split("\n")
            for line in reversed(lines):
                if line.strip() and "INFO" in line:
                    return line.strip()[-120:]
        return ""
    except (OSError, IOError) as e:
        logger.debug(f"Failed to read log file: {e}")
        return ""

def main():
    print(f"=== TraceGen Opus 运行监控 ===")
    print(f"PID: {PID}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    check_interval = 60  # seconds
    while True:
        alive = check_process_alive()
        synth_count = count_synthesis_results()
        valid_results = count_validation_results()
        resource = get_resource_usage()
        last_log = get_last_log_line()

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] Process: {'RUNNING' if alive else 'STOPPED'}")
        print(f"  Synthesis: {synth_count} instances generated")
        print(f"  Validation: valid={valid_results['valid']}, invalid={valid_results['invalid']}, "
              f"timeout={valid_results['timeout']}, error={valid_results['error']}, total={valid_results['total']}")
        if resource:
            cost_input = (resource["input_tokens"] / 1_000_000) * 15
            cost_output = (resource["output_tokens"] / 1_000_000) * 75
            print(f"  Resources: {resource['calls']} calls, "
                  f"input={resource['input_tokens']:,}, output={resource['output_tokens']:,}, "
                  f"est_cost=${cost_input + cost_output:.2f}")
        print(f"  Last: {last_log}")
        print()

        if not alive:
            print("=== Process finished! ===")
            break

        time.sleep(check_interval)

if __name__ == "__main__":
    main()
