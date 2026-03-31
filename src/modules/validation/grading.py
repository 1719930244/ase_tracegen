"""
Test result grading and comparison utilities.

This module provides functions to:
1. Parse test output from various testing frameworks
2. Compare test results before and after patch application
3. Generate validation reports
"""

import re
from pathlib import Path
from typing import Any, Callable

from src.modules.validation.constants import (
    TestStatus,
    FAIL_TO_PASS,
    PASS_TO_PASS,
    FAIL_TO_FAIL,
    PASS_TO_FAIL,
    TEST_OUTPUT_START,
    TEST_OUTPUT_END,
    APPLY_PATCH_FAIL,
    TESTS_TIMEOUT,
    UTF8,
)


# =============================================================================
# Test Output Parsing
# =============================================================================

def read_test_output(filename: str) -> tuple[str | None, bool]:
    """
    Read and extract test output from a log file.

    The function looks for markers TEST_OUTPUT_START and TEST_OUTPUT_END
    to extract only the relevant test output section.

    Args:
        filename: Path to the log file

    Returns:
        tuple: (extracted_content, success_flag)
            - extracted_content: The extracted test output, or None if failed
            - success_flag: True if extraction succeeded, False otherwise
    """
    try:
        content = Path(filename).read_text(errors="replace")
    except FileNotFoundError:
        return None, False

    # Check for failure indicators
    if APPLY_PATCH_FAIL in content:
        return None, False
    if TESTS_TIMEOUT in content:
        return None, False

    # Extract content between markers
    if TEST_OUTPUT_START not in content or TEST_OUTPUT_END not in content:
        return content, False  # Return raw content but flag as incomplete

    start_sep = f"+ : '{TEST_OUTPUT_START}'"
    end_sep = f"+ : '{TEST_OUTPUT_END}'"
    start_idx = content.find(start_sep)
    end_idx = content.find(end_sep)

    if start_idx > end_idx:
        raise ValueError(
            "Invalid test output - Start and end markers are not in correct order"
        )

    extracted = content[start_idx:end_idx][len(start_sep):]
    return extracted, True


def parse_test_output(
    output: str | None,
    log_parser: Callable[[str], dict[str, str]]
) -> dict[str, str]:
    """
    Parse test output using the provided log parser function.

    Args:
        output: Raw test output string
        log_parser: Function that parses test output into status map

    Returns:
        Dictionary mapping test names to their status (PASSED/FAILED/etc)
    """
    if output is None:
        return {}

    return log_parser(output)


# =============================================================================
# Report Generation
# =============================================================================

def get_valid_report(
    pre_gold_output: str,
    post_gold_output: str,
    log_parser: Callable[[str], dict[str, str]] = None,
) -> dict[str, list[str]]:
    """
    Generate a validation report comparing pre and post patch test results.

    This is the core function that determines:
    - FAIL_TO_PASS: Tests that passed before, failed after (BUG CAUGHT THIS)
    - PASS_TO_PASS: Tests that passed before and after (UNAFFECTED)
    - FAIL_TO_FAIL: Tests that failed before and after (BASELINE FAILURES)
    - PASS_TO_FAIL: Tests that failed before, passed after (UNEXPECTED FIX)

    Args:
        pre_gold_output: Test output before applying patch
        post_gold_output: Test output after applying patch
        log_parser: Function to parse test output into status map (default: parse_pytest_log)

    Returns:
        Dictionary with test lists for each category
    """
    # 使用默认解析器如果没有提供
    if log_parser is None:
        log_parser = parse_pytest_log

    pre_gold_map = log_parser(pre_gold_output) if pre_gold_output else {}
    post_gold_map = log_parser(post_gold_output) if post_gold_output else {}

    def _is_pass(status: str) -> bool:
        return status in {
            TestStatus.PASSED.value,
            TestStatus.XFAIL.value,
            TestStatus.SKIPPED.value,
        }

    def _is_fail(status: str) -> bool:
        # Treat ERROR as failure for both unittest + pytest semantics.
        return status in {
            TestStatus.FAILED.value,
            TestStatus.ERROR.value,
        }

    report = {
        FAIL_TO_PASS: [],
        PASS_TO_PASS: [],
        FAIL_TO_FAIL: [],
        PASS_TO_FAIL: [],
    }

    # Compare test results
    for test_case in post_gold_map:
        if test_case not in pre_gold_map:
            # New test case in post-gold (not in pre-gold)
            continue

        pre_status = pre_gold_map[test_case]
        post_status = post_gold_map[test_case]

        # Classify the test result change (SWE-bench semantics):
        # - FAIL_TO_PASS: failing (or error) before, passing after
        # - PASS_TO_FAIL: passing before, failing (or error) after
        if _is_pass(pre_status) and _is_pass(post_status):
            report[PASS_TO_PASS].append(test_case)
        elif _is_fail(pre_status) and _is_pass(post_status):
            report[FAIL_TO_PASS].append(test_case)
        elif _is_fail(pre_status) and _is_fail(post_status):
            report[FAIL_TO_FAIL].append(test_case)
        elif _is_pass(pre_status) and _is_fail(post_status):
            report[PASS_TO_FAIL].append(test_case)

    return report


# =============================================================================
# Built-in Test Parsers
# =============================================================================

def parse_pytest_log(log: str) -> dict[str, str]:
    """
    Parse pytest-style test output.

    Expected format:
        test_file.py::TestClass::test_name PASSED
        test_file.py::TestClass::test_name FAILED

    Args:
        log: Raw test output string

    Returns:
        Dictionary mapping test names to status
    """
    test_status_map = {}

    for line in log.split("\n"):
        line = line.strip()
        for status in TestStatus:
            # Match: test_name STATUS
            pattern = rf"^(\S+)(\s+){status.value}"
            is_match = re.match(pattern, line)
            if is_match:
                test_name = is_match.group(1)
                test_status_map[test_name] = status.value
                break

    return test_status_map


def parse_pytest_verbose(log: str) -> dict[str, str]:
    """
    Parse pytest verbose output with percentage progress.

    Expected format:
        test_file.py::TestClass::test_name PASSED [10%]
        test_file.py::TestClass::test_name FAILED [20%]

    Args:
        log: Raw test output string

    Returns:
        Dictionary mapping test names to status
    """
    test_status_map = {}

    for line in log.split("\n"):
        line = line.strip()
        # Match: test_name STATUS [xx%]
        match = re.match(r"^(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED)\s+\[\d+%\]", line)
        if match:
            test_name = match.group(1)
            status = match.group(2)
            test_status_map[test_name] = status

    return test_status_map


def parse_unittest_log(log: str) -> dict[str, str]:
    """
    Parse Python unittest-style test output.

    Expected format:
        test_name (package.module.Class) ... ok
        test_name (package.module.Class) ... FAIL

    Handles Django's runtests.py interleaving setup messages with test output:
        test_name (module.Class) ... Testing against Django installed in '...'
        [setup messages...]
        ok

    Args:
        log: Raw test output string

    Returns:
        Dictionary mapping test names to status
    """
    _VALID_STATUSES = {
        "ok": TestStatus.PASSED.value,
        "passed": TestStatus.PASSED.value,
        "fail": TestStatus.FAILED.value,
        "failed": TestStatus.FAILED.value,
        "error": TestStatus.ERROR.value,
        "skipped": TestStatus.SKIPPED.value,
    }

    test_status_map = {}
    pending_test: str | None = None

    for raw in log.split("\n"):
        line = raw.strip()

        if " ... " in line:
            parts = line.split(" ... ", 1)
            if len(parts) == 2:
                test_name = parts[0].strip()
                remainder = parts[1].strip()
                first_word = remainder.split()[0].lower() if remainder.split() else ""

                if first_word in _VALID_STATUSES:
                    test_status_map[test_name] = _VALID_STATUSES[first_word]
                    pending_test = None
                else:
                    # Status not on this line (e.g. Django setup message interleaved).
                    # Remember test name and check subsequent lines.
                    pending_test = test_name
            continue

        # Check for deferred status on a standalone line
        if pending_test:
            first_word = line.split()[0].lower() if line.split() else ""
            if first_word in _VALID_STATUSES:
                test_status_map[pending_test] = _VALID_STATUSES[first_word]
                pending_test = None

    return test_status_map


def parse_sympy_log(log: str) -> dict[str, str]:
    """
    Parse sympy bin/test output.

    Expected format (from sympy's own test runner):
        test_name ok
        test_name F
        test_name E
        ___ file.py:test_name ___  (failure header)

    Args:
        log: Raw test output string

    Returns:
        Dictionary mapping test names to status
    """
    test_status_map = {}

    # Pattern 1: failure headers like "___ file.py:test_name ___"
    pattern = r"(_*) (.*)\.py:(.*) (_*)"
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f"{match[1]}.py:{match[2]}"
        test_status_map[test_case] = TestStatus.FAILED.value

    # Pattern 2: line-based status
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.ERROR.value
            elif line.endswith(" F"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.FAILED.value
            elif line.endswith(" ok"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.PASSED.value

    return test_status_map


# =============================================================================
# Utility Functions
# =============================================================================

def test_passed(test_case: str, status_map: dict[str, str]) -> bool:
    """Check if a test case passed."""
    if test_case not in status_map:
        return False
    return status_map[test_case] in [
        TestStatus.PASSED.value,
        TestStatus.XFAIL.value,
        TestStatus.SKIPPED.value,
    ]


def test_failed(test_case: str, status_map: dict[str, str]) -> bool:
    """Check if a test case failed."""
    if test_case not in status_map:
        return True  # Missing test = failed
    return status_map[test_case] in [
        TestStatus.FAILED.value,
        TestStatus.ERROR.value,
    ]


def calculate_resolution_rate(
    eval_status_map: dict[str, str],
    gold_results: dict[str, list[str]]
) -> dict[str, Any]:
    """
    Calculate resolution rate for evaluation.

    Args:
        eval_status_map: Test results from evaluation
        gold_results: Expected results with FAIL_TO_PASS and PASS_TO_PASS

    Returns:
        Dictionary with resolution statistics
    """
    f2p_tests = gold_results.get(FAIL_TO_PASS, [])
    p2p_tests = gold_results.get(PASS_TO_PASS, [])

    f2p_resolved = [t for t in f2p_tests if test_passed(t, eval_status_map)]
    f2p_failed = [t for t in f2p_tests if test_failed(t, eval_status_map)]

    p2p_resolved = [t for t in p2p_tests if test_passed(t, eval_status_map)]
    p2p_failed = [t for t in p2p_tests if test_failed(t, eval_status_map)]

    total = len(f2p_tests) + len(p2p_tests)
    resolved = len(f2p_resolved) + len(p2p_resolved)

    return {
        "total_tests": total,
        "resolved": resolved,
        "resolution_rate": resolved / total if total > 0 else 0,
        "FAIL_TO_PASS": {
            "total": len(f2p_tests),
            "resolved": len(f2p_resolved),
            "failed": len(f2p_failed),
        },
        "PASS_TO_PASS": {
            "total": len(p2p_tests),
            "resolved": len(p2p_resolved),
            "failed": len(p2p_failed),
        },
    }


# =============================================================================
# Default Parser Registry
# =============================================================================

DEFAULT_PARSERS = {
    "pytest": parse_pytest_verbose,
    "python": parse_pytest_log,
    "unittest": parse_unittest_log,
    "sympy_bin_test": parse_sympy_log,
}


def get_parser(language: str) -> Callable[[str], dict[str, str]]:
    """Get default parser for a language."""
    return DEFAULT_PARSERS.get(language, parse_pytest_log)


# Re-exports for backward compatibility
__all__ = [
    "read_test_output",
    "parse_test_output",
    "get_valid_report",
    "parse_pytest_log",
    "parse_pytest_verbose",
    "parse_unittest_log",
    "parse_sympy_log",
    "test_passed",
    "test_failed",
    "calculate_resolution_rate",
    "get_parser",
    "DEFAULT_PARSERS",
]
