"""
GenericPytestProfile — 通用 pytest 仓库的 fallback Profile。

用于未知仓库或没有专门 Profile 的 pytest 项目。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.core.repo_profile import RepoProfile


@dataclass
class GenericPytestProfile(RepoProfile):
    """通用 pytest 仓库 Profile。使用路径约定推断测试文件。"""

    test_framework: str = "pytest"
    test_cmd_template: str = "pytest {targets} -v --tb=short"

    def _infer_by_convention(self, source_file: str, repo_root: Path | None = None) -> list[str]:
        """通用推断：尝试多种常见的测试文件命名约定。"""
        p = Path(source_file)
        stem = p.stem.lstrip("_")
        parent = str(p.parent)

        candidates = [
            f"{parent}/tests/test_{stem}.py",
            f"{parent}/test_{stem}.py",
            f"tests/test_{stem}.py",
            f"test_{stem}.py",
        ]

        if repo_root:
            existing = [c for c in candidates if (repo_root / c).exists()]
            if existing:
                return existing
            # 如果都不存在，尝试在 tests/ 下递归搜索
            tests_dir = repo_root / "tests"
            if tests_dir.exists():
                found = list(tests_dir.rglob(f"test_{stem}.py"))
                if found:
                    return [str(f.relative_to(repo_root)) for f in found[:5]]

        return candidates[:1]
