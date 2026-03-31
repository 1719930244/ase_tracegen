"""
SklearnProfile — scikit-learn/scikit-learn 仓库的 RepoProfile 实现。

sklearn 的目录结构：sklearn/<pkg>/tests/test_<mod>.py
源文件常以 _ 开头（如 _forest.py），测试文件不带前缀下划线。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.core.repo_profile import RepoProfile


@dataclass
class SklearnProfile(RepoProfile):
    """scikit-learn/scikit-learn 仓库 Profile。"""

    repo: str = "scikit-learn/scikit-learn"
    test_framework: str = "pytest"
    test_root: str = "sklearn"
    test_cmd_template: str = "pytest {targets} -v --tb=short"

    def _infer_by_convention(self, source_file: str, repo_root: Path | None = None) -> list[str]:
        """sklearn 的约定：sklearn/ensemble/_forest.py → sklearn/ensemble/tests/test_forest.py"""
        p = Path(source_file)
        stem = p.stem.lstrip("_")  # _forest → forest
        parent = str(p.parent)

        if "/tests/" in source_file:
            return [source_file]

        candidates = [
            f"{parent}/tests/test_{stem}.py",
        ]

        if repo_root:
            existing = [c for c in candidates if (repo_root / c).exists()]
            if existing:
                return existing

            # sklearn 有些模块名和测试名不完全对应，搜索 tests/ 下包含 stem 的文件
            tests_dir = repo_root / parent / "tests"
            if tests_dir.exists():
                found = [
                    str(f.relative_to(repo_root))
                    for f in tests_dir.glob("test_*.py")
                    if stem in f.stem
                ]
                if found:
                    return found[:5]

        return candidates[:1]

    def get_test_file_path(self, source_file: str) -> str:
        """合成测试文件放在对应包的 tests/ 下。"""
        p = Path(source_file)
        parent = str(p.parent)
        return f"{parent}/tests/test_synthetic.py"

    def infer_seed_test_cmd(self, seed_info: dict, max_total_labels: int = 6) -> str:
        """从 seed 元数据推断 pytest 测试命令。"""
        if not isinstance(seed_info, dict):
            return ""

        f2p = seed_info.get("FAIL_TO_PASS", [])
        p2p = seed_info.get("PASS_TO_PASS", [])
        if isinstance(f2p, str):
            try:
                import json
                f2p = json.loads(f2p)
            except Exception:
                f2p = [f2p] if f2p.strip() else []
        if isinstance(p2p, str):
            try:
                import json
                p2p = json.loads(p2p)
            except Exception:
                p2p = [p2p] if p2p.strip() else []

        files: list[str] = []
        for label in (f2p + p2p):
            if not isinstance(label, str):
                continue
            parts = label.split("::")
            if parts and parts[0] not in files:
                files.append(parts[0])

        files = files[:max_total_labels]
        if not files:
            return ""
        return f"pytest {' '.join(files)} -v --tb=short"
