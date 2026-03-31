"""
AstropyProfile — astropy/astropy 仓库的 RepoProfile 实现。

astropy 的目录结构：astropy/<pkg>/... → astropy/<pkg>/tests/test_<mod>.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.repo_profile import RepoProfile


@dataclass
class AstropyProfile(RepoProfile):
    """astropy/astropy 仓库 Profile。"""

    repo: str = "astropy/astropy"
    test_framework: str = "pytest"
    test_root: str = "astropy"
    test_cmd_template: str = "pytest {targets} -v --tb=short"

    def _infer_by_convention(self, source_file: str, repo_root: Path | None = None) -> list[str]:
        """
        astropy 的约定：
        astropy/io/ascii/ecsv.py → astropy/io/ascii/tests/test_ecsv.py
        astropy/utils/decorators.py → astropy/utils/tests/test_decorators.py
        """
        p = Path(source_file)
        stem = p.stem.lstrip("_")
        parent = str(p.parent)

        if "/tests/" in source_file:
            return [source_file]

        # astropy/<pkg>/foo.py → astropy/<pkg>/tests/test_foo.py
        candidates = [
            f"{parent}/tests/test_{stem}.py",
        ]

        if repo_root:
            existing = [c for c in candidates if (repo_root / c).exists()]
            if existing:
                return existing

            # Search tests/ dir for matching files
            tests_dir = repo_root / parent / "tests"
            if tests_dir.exists():
                found = [
                    str(f.relative_to(repo_root))
                    for f in tests_dir.glob(f"test_*{stem}*.py")
                ]
                if found:
                    return found[:5]
                # Also try: astropy/io/ascii/ecsv.py → test might be test_read.py etc.
                # Return all test files in the tests/ dir as fallback
                all_tests = [
                    str(f.relative_to(repo_root))
                    for f in tests_dir.glob("test_*.py")
                ]
                if all_tests:
                    return all_tests[:10]

        return candidates[:1]
