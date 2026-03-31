"""
PytestRepoProfile — pytest-dev/pytest 仓库的 RepoProfile 实现。

pytest 的测试在 testing/ 目录下，源码在 src/_pytest/ 下。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.core.repo_profile import RepoProfile


# pytest 源码 → 测试文件的静态映射（高频模块）
# 注意：pytest 的测试分布在 testing/ 的多个子目录中：
#   testing/*.py, testing/python/*.py, testing/logging/, testing/plugins_integration/
PYTEST_SOURCE_TO_TEST_MAPPING = {
    "src/_pytest/config/": ["testing/test_config.py"],
    "src/_pytest/fixtures.py": [
        "testing/python/fixtures.py",
        "testing/python/test_fixtures.py",
        "testing/test_fixtures.py",
    ],
    "src/_pytest/assertion/": [
        "testing/test_assertion.py",
        "testing/python/approx.py",
    ],
    "src/_pytest/logging.py": ["testing/logging/test_fixture.py"],
    "src/_pytest/capture.py": ["testing/test_capture.py"],
    "src/_pytest/cacheprovider.py": ["testing/test_cacheprovider.py"],
    "src/_pytest/python.py": [
        "testing/python/test_python.py",
        "testing/python/metafunc.py",
        "testing/python/collect.py",
        "testing/python/fixtures.py",
        "testing/python/integration.py",
        "testing/python/approx.py",
        "testing/python/raises.py",
        "testing/python/show_fixtures_per_test.py",
    ],
    "src/_pytest/terminal.py": ["testing/test_terminal.py"],
    "src/_pytest/nodes.py": ["testing/test_nodes.py"],
    "src/_pytest/mark/": [
        "testing/test_mark.py",
        "testing/test_mark_expression.py",
    ],
    "src/_pytest/runner.py": ["testing/test_runner.py"],
    "src/_pytest/skipping.py": ["testing/test_skipping.py"],
    "src/_pytest/tmpdir.py": ["testing/test_tmpdir.py"],
    "src/_pytest/pathlib.py": ["testing/test_pathlib.py"],
    "src/_pytest/doctest.py": ["testing/test_doctest.py"],
    "src/_pytest/warnings.py": ["testing/test_warnings.py"],
    "src/_pytest/stepwise.py": ["testing/test_stepwise.py"],
    "src/_pytest/recwarn.py": ["testing/test_recwarn.py"],
    "src/_pytest/monkeypatch.py": ["testing/test_monkeypatch.py"],
    # testing/python/ 子目录的直接映射
    "src/_pytest/outcomes.py": ["testing/python/raises.py"],
    "src/_pytest/approx.py": ["testing/python/approx.py"],
}


@dataclass
class PytestRepoProfile(RepoProfile):
    """pytest-dev/pytest 仓库 Profile。"""

    repo: str = "pytest-dev/pytest"
    test_framework: str = "pytest"
    source_to_test_mapping: dict[str, list[str]] = field(
        default_factory=lambda: dict(PYTEST_SOURCE_TO_TEST_MAPPING)
    )
    test_root: str = "testing"
    test_cmd_template: str = "pytest {targets} -v --tb=short"

    def _infer_by_convention(self, source_file: str, repo_root: Path | None = None) -> list[str]:
        """pytest 的约定：src/_pytest/<mod>.py → testing/ 下的相关测试文件。

        pytest 测试分布在多个子目录：
        - testing/test_<mod>.py（直接映射）
        - testing/python/<mod>.py（无 test_ 前缀，如 metafunc.py, approx.py）
        - testing/python/test_<mod>.py
        - testing/logging/, testing/plugins_integration/ 等
        """
        p = Path(source_file)
        stem = p.stem.lstrip("_")

        # 候选文件：包含有 test_ 前缀和无前缀两种模式
        candidates = [
            f"testing/test_{stem}.py",
            f"testing/python/test_{stem}.py",
            f"testing/python/{stem}.py",
        ]

        if repo_root:
            existing = [c for c in candidates if (repo_root / c).exists()]
            if existing:
                return existing
            # 递归搜索 testing/：同时搜索 test_<stem>.py 和 <stem>.py
            testing_dir = repo_root / "testing"
            if testing_dir.exists():
                found = []
                for pattern in [f"test_{stem}.py", f"{stem}.py"]:
                    found.extend(testing_dir.rglob(pattern))
                if found:
                    # 去重并排序
                    seen = set()
                    unique = []
                    for f in found:
                        rel = str(f.relative_to(repo_root))
                        if rel not in seen:
                            seen.add(rel)
                            unique.append(rel)
                    return unique[:5]

        return candidates[:1]

    def get_test_file_path(self, source_file: str) -> str:
        """合成测试文件放在 testing/ 下。"""
        return "testing/test_synthetic.py"
