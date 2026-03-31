"""
RepoProfile Registry — 仓库 Profile 注册表和工厂函数。

用法:
    from src.core.repo_profiles import get_repo_profile
    profile = get_repo_profile("django/django")
    profile = get_repo_profile("sympy/sympy")
    profile = get_repo_profile("unknown/repo")  # → GenericPytestProfile
"""

from __future__ import annotations

from src.core.repo_profile import RepoProfile
from src.core.repo_profiles.django import DjangoProfile
from src.core.repo_profiles.pytest_repo import PytestRepoProfile
from src.core.repo_profiles.sympy import SympyProfile
from src.core.repo_profiles.sklearn import SklearnProfile
from src.core.repo_profiles.generic import GenericPytestProfile
from src.core.repo_profiles.matplotlib import MatplotlibProfile
from src.core.repo_profiles.astropy import AstropyProfile
from src.core.repo_profiles.xarray import XarrayProfile
from src.core.repo_profiles.sphinx import SphinxProfile
from src.core.repo_profiles.requests import RequestsProfile
from src.core.repo_profiles.pylint import PylintProfile


# 仓库名 → Profile 类的注册表
PROFILE_REGISTRY: dict[str, type[RepoProfile]] = {
    "django/django": DjangoProfile,
    "pytest-dev/pytest": PytestRepoProfile,
    "sympy/sympy": SympyProfile,
    "scikit-learn/scikit-learn": SklearnProfile,
    "matplotlib/matplotlib": MatplotlibProfile,
    "astropy/astropy": AstropyProfile,
    "pydata/xarray": XarrayProfile,
    "sphinx-doc/sphinx": SphinxProfile,
    "psf/requests": RequestsProfile,
    "pylint-dev/pylint": PylintProfile,
}


def get_repo_profile(repo: str) -> RepoProfile:
    """根据仓库名返回对应的 RepoProfile 实例。

    Args:
        repo: 仓库全名，如 "django/django"、"sympy/sympy"

    Returns:
        对应的 RepoProfile 实例。未知仓库返回 GenericPytestProfile。
    """
    repo_clean = repo.strip().lower()

    # 精确匹配
    for key, cls in PROFILE_REGISTRY.items():
        if key.lower() == repo_clean:
            return cls()

    # 模糊匹配（仓库名可能带前缀或后缀）
    for key, cls in PROFILE_REGISTRY.items():
        if key.lower() in repo_clean or repo_clean in key.lower():
            return cls()

    # Fallback
    return GenericPytestProfile(repo=repo)


def detect_repo_from_instance_id(instance_id: str) -> str:
    """从 SWE-bench instance_id 推断仓库名。

    例: "django__django-14787" → "django/django"
         "pytest-dev__pytest-10051" → "pytest-dev/pytest"
    """
    parts = instance_id.split("__")
    if len(parts) >= 2:
        owner = parts[0]
        # repo name 是第二部分去掉 issue number
        repo_part = parts[1].rsplit("-", 1)[0] if "-" in parts[1] else parts[1]
        return f"{owner}/{repo_part}"
    return instance_id


__all__ = [
    "RepoProfile",
    "DjangoProfile",
    "PytestRepoProfile",
    "SympyProfile",
    "SklearnProfile",
    "GenericPytestProfile",
    "MatplotlibProfile",
    "AstropyProfile",
    "XarrayProfile",
    "SphinxProfile",
    "RequestsProfile",
    "PylintProfile",
    "get_repo_profile",
    "detect_repo_from_instance_id",
    "PROFILE_REGISTRY",
]
