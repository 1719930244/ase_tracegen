"""
Python repository profile.

Implements test execution and log parsing for Python projects using
pytest, unittest, or other Python test frameworks.
"""

import re
from dataclasses import dataclass, field

from src.modules.validation.constants import TestStatus, FAIL_TO_PASS, PASS_TO_PASS
from src.modules.validation.profiles.base import BaseProfile


# =============================================================================
# Python Profile
# =============================================================================

@dataclass
class PythonProfile(BaseProfile):
    """
    Profile for Python repositories.

    Default configuration:
    - Uses pytest for testing
    - Parses pytest-style output
    - Supports conda environments
    """

    name: str = "python"
    language: str = "python"
    file_extensions: list[str] = field(default_factory=lambda: [".py"])

    # Default test command (uses absolute path to pytest in conda env)
    test_cmd: str = (
        "/opt/miniconda3/envs/testbed/bin/pytest "
        "--disable-warnings --color=no --tb=no --verbose"
    )

    # Optional configuration
    python_version: str = "3.10"
    env_name: str = "testbed"  # Conda environment name

    def log_parser(self, log: str) -> dict[str, str]:
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

    def get_test_files(self, instance: dict) -> tuple[list[str], list[str]]:
        """
        Extract test file paths from Python test names.

        For Python, test names are formatted as:
            test_module.py::TestClass::test_method

        We extract just the module (file) part.

        Args:
            instance: Instance with FAIL_TO_PASS and PASS_TO_PASS

        Returns:
            tuple: (f2p_files, p2p_files) - unique test module paths
        """
        if FAIL_TO_PASS not in instance or PASS_TO_PASS not in instance:
            return [], []

        def extract_modules(tests: list[str]) -> list[str]:
            """Extract test module paths from fully-qualified test names."""
            modules = set()
            for test in tests:
                # test_module.py::TestClass::test_method -> test_module.py
                parts = test.split("::")
                if parts:
                    modules.add(parts[0])
            return sorted(list(modules))

        return (
            extract_modules(instance[FAIL_TO_PASS]),
            extract_modules(instance[PASS_TO_PASS]),
        )


# =============================================================================
# Unittest profile
# =============================================================================

@dataclass
class UnittestProfile(PythonProfile):
    """Profile for Python unittest framework."""

    name: str = "unittest"
    test_cmd: str = "python -m unittest discover -v"

    def log_parser(self, log: str) -> dict[str, str]:
        """
        Parse unittest-style output.

        Expected format:
            test_name (module.Class) ... ok
            test_name (module.Class) ... FAIL
        """
        test_status_map = {}

        pending_test: str | None = None

        for raw in log.split("\n"):
            line = raw.strip()

            # Django's runtests.py output can interleave environment lines between the
            # "test_x (...) ..." prefix and the final status token, especially with parallelism.
            if " ... " in line:
                parts = line.split(" ... ", 1)
                if len(parts) == 2:
                    test_name = parts[0].strip()
                    # Use the FIRST token after " ... " for status detection.
                    # Using the last token is wrong for "skipped 'reason text'"
                    # where the last word is part of the reason string.
                    remainder = parts[1].strip()
                    tokens = remainder.split()
                    status_token = tokens[0].lower() if tokens else ""

                    if status_token in ["ok", "passed"]:
                        test_status_map[test_name] = TestStatus.PASSED.value
                        pending_test = None
                    elif status_token in ["fail", "failed"]:
                        test_status_map[test_name] = TestStatus.FAILED.value
                        pending_test = None
                    elif status_token == "error":
                        test_status_map[test_name] = TestStatus.ERROR.value
                        pending_test = None
                    elif status_token == "skipped":
                        test_status_map[test_name] = TestStatus.SKIPPED.value
                        pending_test = None
                    else:
                        # No status token on the same line; remember and try to attach
                        # a subsequent standalone "ok/FAIL/ERROR/skipped".
                        pending_test = test_name
                continue

            if pending_test:
                token = line.split()[0].lower() if line.split() else ""
                if token in ["ok", "passed"]:
                    test_status_map[pending_test] = TestStatus.PASSED.value
                    pending_test = None
                elif token in ["fail", "failed"]:
                    test_status_map[pending_test] = TestStatus.FAILED.value
                    pending_test = None
                elif token == "error":
                    test_status_map[pending_test] = TestStatus.ERROR.value
                    pending_test = None
                elif token == "skipped":
                    test_status_map[pending_test] = TestStatus.SKIPPED.value
                    pending_test = None

        return test_status_map


# =============================================================================
# Pre-configured profiles for common Python projects
# =============================================================================

PRECONFIGURED_PROFILES = {
    "addict": {
        "owner": "mewwts",
        "repo": "addict",
        "commit": "75284f9593dfb929cadd900aff9e35e7c7aec54b",
        "image_name": "swebench/swesmith.x86_64.mewwts_1776_addict.75284f95",
        "min_pregold": True,
    },
    "flask": {
        "owner": "pallets",
        "repo": "flask",
        "commit": "bc098406af9537aacc436cb2ea777fbc9ff4c5aa",
        "image_name": "swebench/swesmith.x86_64.pallets_1776_flask.bc098406",
    },
    "requests": {
        "owner": "psf",
        "repo": "requests",
        "commit": "a0b913e96fce8e2f0bea3ba229ba8d8f6df3fe72",
        "image_name": "swebench/swesmith.x86_64.psf_1776_requests.a0b913e9",
    },
    "django": {
        "owner": "django",
        "repo": "django",
        "commit": "e50f6f40e1b3e873364ecfb340c7f9819cf6ee61",
        "image_name": "swebench/swesmith.x86_64.django_1776_django.e50f6f40",
        "test_cmd": "python -m pytest --disable-warnings --color=no --tb=no --verbose",
    },
}


def get_preconfigured_profile(name: str) -> PythonProfile:
    """Get a preconfigured profile for a common Python project."""
    if name not in PRECONFIGURED_PROFILES:
        raise ValueError(f"Unknown preconfigured profile: {name}")

    config = PRECONFIGURED_PROFILES[name]
    return PythonProfile(**config)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "PythonProfile",
    "UnittestProfile",
    "get_preconfigured_profile",
    "PRECONFIGURED_PROFILES",
]
