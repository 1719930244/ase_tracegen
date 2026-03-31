"""
PylintProfile — pylint-dev/pylint 仓库的 RepoProfile 实现。

pylint 的目录结构：pylint/<pkg>/... → tests/checkers/..., tests/<domain>/...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.repo_profile import RepoProfile


@dataclass
class PylintProfile(RepoProfile):
    """pylint-dev/pylint 仓库 Profile。"""

    repo: str = "pylint-dev/pylint"
    test_framework: str = "pytest"
    test_root: str = "tests"
    test_cmd_template: str = "pytest {targets} -v --tb=short"

    def _infer_by_convention(self, source_file: str, repo_root: Path | None = None) -> list[str]:
        """
        pylint 的约定：
        pylint/checkers/variables.py → tests/checkers/unittest_variables.py
        pylint/pyreverse/inspector.py → tests/unittest_pyreverse_inspector.py
        pylint/reporters/... → tests/reporters/unittest_reporting.py
        """
        p = Path(source_file)
        stem = p.stem.lstrip("_")

        if source_file.startswith("tests/"):
            return [source_file]

        candidates = []

        # pylint/checkers/variables.py → tests/checkers/unittest_variables.py
        if source_file.startswith("pylint/checkers/"):
            candidates.extend([
                f"tests/checkers/unittest_{stem}.py",
                f"tests/checkers/test_{stem}.py",
            ])
        elif source_file.startswith("pylint/pyreverse/"):
            candidates.extend([
                f"tests/unittest_pyreverse_{stem}.py",
                f"tests/pyreverse/test_{stem}.py",
                f"tests/pyreverse/unittest_{stem}.py",
            ])
        elif source_file.startswith("pylint/reporters/"):
            candidates.extend([
                f"tests/reporters/unittest_reporting.py",
                f"tests/reporters/unittest_{stem}.py",
            ])
        else:
            candidates.extend([
                f"tests/unittest_{stem}.py",
                f"tests/test_{stem}.py",
            ])

        if repo_root:
            existing = [c for c in candidates if (repo_root / c).exists()]
            if existing:
                return existing

            # Broad search
            tests_dir = repo_root / "tests"
            if tests_dir.exists():
                found = []
                for pattern in [f"**/unittest_{stem}*.py", f"**/test_{stem}*.py", f"**/*{stem}*.py"]:
                    found.extend([
                        str(f.relative_to(repo_root))
                        for f in tests_dir.glob(pattern)
                        if f.is_file()
                    ])
                if found:
                    return sorted(set(found))[:5]

        return candidates[:1]
