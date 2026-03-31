"""
Constants for the bug validation framework.

This module defines all shared constants used across the validator,
including test status types, log markers, and configuration defaults.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# =============================================================================
# Test Status Constants (from SWE-bench)
# =============================================================================

class TestStatus(Enum):
    """Test execution status codes."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"  # Expected failure
    XPASS = "XPASS"  # Unexpected pass


# Test result categories for validation reports
FAIL_TO_PASS = "FAIL_TO_PASS"  # Tests that should pass after fix
PASS_TO_PASS = "PASS_TO_PASS"  # Tests that should continue passing
FAIL_TO_FAIL = "FAIL_TO_FAIL"  # Tests already failing (maintain failure)
PASS_TO_FAIL = "PASS_TO_FAIL"  # Tests that broke (regression)


# =============================================================================
# Docker / Container Constants
# =============================================================================

DOCKER_USER = "swesmith"
DOCKER_WORKDIR = "/testbed"
DOCKER_PATCH = "/tmp/patch.diff"
DOCKER_IMAGE_BASE = "swesmith/swesmith.x86_64"

# Container configuration
DEFAULT_TIMEOUT = 90  # seconds
DEFAULT_MEMORY_LIMIT = "10g"
DEFAULT_PLATFORM = "linux/x86_64"

# Encoding
UTF8 = "utf-8"

# Timeout marker
TESTS_TIMEOUT = ">>>>> Tests Timed Out"


# =============================================================================
# Log File Constants
# =============================================================================

LOG_INSTANCE = "run_instance.log"
LOG_TEST_OUTPUT = "test_output.txt"
LOG_TEST_OUTPUT_PRE_GOLD = "test_output_pre_gold.txt"
LOG_REPORT = "report.json"
LOG_PATCH = "patch.diff"
LOG_EVAL_SH = "eval.sh"

# Test output markers for extraction
TEST_OUTPUT_START = ">>>>> Start Test Output"
TEST_OUTPUT_END = ">>>>> End Test Output"

# Patch application markers
APPLY_PATCH_PASS = "APPLY_PATCH_PASS"
APPLY_PATCH_FAIL = "APPLY_PATCH_FAIL"


# =============================================================================
# Instance Dictionary Keys
# =============================================================================

KEY_INSTANCE_ID = "instance_id"
KEY_PATCH = "patch"
KEY_SEED_FIX_PATCH = "seed_fix_patch"  # Seed 的修复补丁 (buggy→fixed)
KEY_INJECTION_PATCH = "injection_patch"  # Bug 注入补丁 (fixed→buggy)
KEY_PREDICTION = "model_patch"  # For evaluation mode
KEY_IMAGE_NAME = "image_name"
KEY_TIMED_OUT = "timed_out"


# =============================================================================
# Git Apply Commands (fallback strategies)
# =============================================================================

GIT_APPLY_CMDS = [
    # `--recount` makes `git apply` robust to edited/mismatched hunk header counts,
    # which frequently occurs in LLM-generated diffs.
    "git apply --verbose --recount",
    "git apply --verbose --recount --reject",
    "patch --batch --fuzz=5 -p1 -i",
]


# =============================================================================
# Validation Status Types
# =============================================================================

class ValidationStatus(Enum):
    """Overall validation result status."""
    VALID = "valid"           # Bug caused expected test failures
    INVALID = "invalid"       # Bug didn't cause any test failures
    MISSING_IMAGE = "missing_image"  # Required Docker image not available; skipped
    TIMEOUT = "timeout"       # Test execution timed out
    ERROR = "error"           # Error during validation


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class ValidationConfig:
    """Configuration for validation runs."""
    mode: str = "fix"  # "fix" or "injection"
    timeout: int = DEFAULT_TIMEOUT
    memory_limit: str = DEFAULT_MEMORY_LIMIT
    platform: str = DEFAULT_PLATFORM
    clean_containers: bool = True
    verbose: bool = False
    # Chain coverage enforcement.
    # Default: shadow mode (log warnings but don't reject).
    # Set enforce_chain_coverage=True to actually reject low-coverage bugs.
    enforce_chain_coverage: bool = False
    min_chain_coverage: float = 0.30
    require_target_node_in_traceback: bool = True

    # Docker configuration
    docker_user: str = DOCKER_USER
    docker_workdir: str = DOCKER_WORKDIR

    # Log directory
    log_dir: Path = field(default_factory=lambda: Path("logs/validation"))


@dataclass
class ValidationResult:
    """Result of a validation run."""
    instance_id: str
    status: ValidationStatus

    # Test results
    FAIL_TO_PASS: list[str] = field(default_factory=list)
    PASS_TO_PASS: list[str] = field(default_factory=list)
    FAIL_TO_FAIL: list[str] = field(default_factory=list)
    PASS_TO_FAIL: list[str] = field(default_factory=list)

    # Metadata
    timed_out: bool = False
    timeout_value: int = 0
    error_message: str = ""

    # Raw outputs
    pre_gold_output: str = ""
    post_gold_output: str = ""

    # P2: 新增调用栈信息 - 从失败测试日志中提取的 traceback
    # 格式: {"test_name": [{"file_path": ..., "line": ..., "function": ..., "code": ...}, ...]}
    traceback_info: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    # P3: 链路对齐评分 - 包含两层:
    # 顶层 (traceback-based): trace_coverage, matched_nodes, target_node_hit, causal_ordering, overall_score
    # 嵌套 structural_alignment (Eq.3): depth_match, structure_similarity, semantic_alignment, overall_score
    chain_alignment_score: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        return {
            KEY_INSTANCE_ID: self.instance_id,
            "status": self.status.value,
            FAIL_TO_PASS: self.FAIL_TO_PASS,
            PASS_TO_PASS: self.PASS_TO_PASS,
            FAIL_TO_FAIL: self.FAIL_TO_FAIL,
            PASS_TO_FAIL: self.PASS_TO_FAIL,
            KEY_TIMED_OUT: self.timed_out,
            "timeout_value": self.timeout_value,
            "error_message": self.error_message,
            "traceback_info": self.traceback_info,  # P2: 包含调用栈信息
            "chain_alignment_score": self.chain_alignment_score,  # P3: 链路对齐评分
        }

    def is_valid_bug(self) -> bool:
        """
        Check if this is a valid bug (causes test failures).

        对于 Bug 注入场景：
        - 我们期望至少一个测试从 PASS 变为 FAIL（PASS_TO_FAIL > 0）
        - 同时至少一个测试仍然保持通过（PASS_TO_PASS > 0），避免“全挂/过于破坏性”的情况
        - fixed baseline（pre-gold）上不应存在失败测试（FAIL_TO_FAIL/FAIL_TO_PASS 应为 0）

        注意：FAIL_TO_PASS 用于验证修复补丁场景（测试从 FAIL 变为 PASS）
        """
        return (
            self.status == ValidationStatus.VALID and
            len(self.PASS_TO_FAIL) > 0 and
            len(self.PASS_TO_PASS) > 0 and
            len(self.FAIL_TO_FAIL) == 0 and
            len(self.FAIL_TO_PASS) == 0
        )

    def summary(self) -> str:
        """Get a human-readable summary."""
        if self.status == ValidationStatus.MISSING_IMAGE:
            return f"Missing image: {self.error_message}".strip()
        if self.status == ValidationStatus.TIMEOUT:
            return f"Timeout after {self.timeout_value}s"
        elif self.status == ValidationStatus.ERROR:
            return f"Error: {self.error_message}"

        # Bug 注入场景：优先显示 P2F（PASS_TO_FAIL）
        return (
            f"Valid Bug: {self.is_valid_bug()} | "
            f"P2F: {len(self.PASS_TO_FAIL)} | "  # Bug 注入的关键指标
            f"P2P: {len(self.PASS_TO_PASS)} | "
            f"F2P: {len(self.FAIL_TO_PASS)} | "
            f"F2F: {len(self.FAIL_TO_FAIL)}"
        )


# =============================================================================
# Instance Format Reference
# =============================================================================

"""
Expected instance format for validation:

{
    "instance_id": "owner__repo.commit.unique_id",
    "repo": "owner__repo.commit",
    "patch": "diff --git a/file.py b/file.py\\n...",  # Bug-inducing patch

    # Optional fields for validation
    "FAIL_TO_PASS": ["test_case_1", "test_case_2", ...],  # Expected failures
    "PASS_TO_PASS": ["test_case_3", "test_case_4", ...],  # Expected passes

    # Optional metadata
    "problem_statement": "Description of the bug",
    "strategy": "procedural_remove_try",
    "version": "1.0",
}

Result format after validation:

{
    "instance_id": "owner__repo.commit.unique_id",
    "status": "valid" | "invalid" | "timeout" | "error",
    "FAIL_TO_PASS": [...],  # Tests that failed due to bug
    "PASS_TO_PASS": [...],  # Tests that still passed
    "FAIL_TO_FAIL": [...],  # Tests already failing
    "PASS_TO_FAIL": [...],  # Tests that regressed
    "timed_out": false,
    "error_message": "",
}
"""
