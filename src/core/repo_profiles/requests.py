"""
RequestsProfile — psf/requests 仓库的 RepoProfile 实现。

requests 的目录结构比较特殊，测试分布在多个位置：
- tests/test_*.py (主测试)
- requests/packages/urllib3/test/test_*.py (urllib3 子包测试)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.core.repo_profile import RepoProfile


@dataclass
class RequestsProfile(RepoProfile):
    """psf/requests 仓库 Profile。"""

    repo: str = "psf/requests"
    test_framework: str = "pytest"
    test_root: str = "tests"
    test_cmd_template: str = "pytest {targets} -v --tb=short"
    source_to_test_mapping: dict[str, list[str]] = field(default_factory=lambda: {
        "requests/models.py": ["test_requests.py"],
        "requests/api.py": ["test_requests.py"],
        "requests/sessions.py": ["test_requests.py"],
        "requests/auth.py": ["test_requests.py"],
        "requests/cookies.py": ["test_requests.py"],
        "requests/utils.py": ["test_requests.py"],
        "requests/structures.py": ["test_requests.py"],
        "requests/packages/urllib3": ["test_requests.py"],
    })

    def _infer_by_convention(self, source_file: str, repo_root: Path | None = None) -> list[str]:
        """
        requests 的约定：
        requests/models.py → tests/test_requests.py (大多数核心模块共享)
        requests/utils.py → tests/test_utils.py
        requests/packages/urllib3/... → tests/test_requests.py
        """
        p = Path(source_file)
        stem = p.stem.lstrip("_")

        if source_file.startswith("tests/") or source_file.startswith("test_"):
            return [source_file]

        # Try direct mapping first
        # Older SWE-bench commits use root-level test_requests.py
        # Newer versions use tests/ directory
        candidates = [
            f"test_{stem}.py",           # root-level (older commits)
            f"tests/test_{stem}.py",     # tests/ dir (newer commits)
            "test_requests.py",          # root-level fallback
            "tests/test_requests.py",    # tests/ dir fallback
        ]

        if repo_root:
            existing = [c for c in candidates if (repo_root / c).exists()]
            if existing:
                return existing

            # Search tests/ dir
            tests_dir = repo_root / "tests"
            if tests_dir.exists():
                found = [
                    str(f.relative_to(repo_root))
                    for f in tests_dir.glob("test_*.py")
                ]
                if found:
                    return found[:5]

            # Also check for test_requests.py at root (older versions)
            if (repo_root / "test_requests.py").exists():
                return ["test_requests.py"]

        return candidates[:1]
