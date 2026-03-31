#!/usr/bin/env python
"""
从 HuggingFace SWE-bench 数据集下载并转换指定仓库的实例到 TraceGen 格式。

用法:
    python scripts/prepare_multi_repo.py --repos pytest-dev/pytest sympy/sympy scikit-learn/scikit-learn
    python scripts/prepare_multi_repo.py --repos sympy/sympy --max-instances 50
    python scripts/prepare_multi_repo.py --all-non-django
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger


# SWE-bench 中所有非 Django 仓库
NON_DJANGO_REPOS = [
    "pytest-dev/pytest",
    "sympy/sympy",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "matplotlib/matplotlib",
    "pydata/xarray",
    "astropy/astropy",
    "pylint-dev/pylint",
    "psf/requests",
    "mwaskom/seaborn",
    "pallets/flask",
]

# 优先扩展的仓库
PRIORITY_REPOS = [
    "pytest-dev/pytest",
    "sympy/sympy",
    "scikit-learn/scikit-learn",
]


def parse_patch_files(patch: str) -> list[str]:
    """从 unified diff patch 中提取被修改的文件路径。"""
    files = []
    for line in patch.split("\n"):
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            fp = line[6:].strip()
            if fp and fp != "/dev/null" and fp not in files:
                files.append(fp)
    return files


def parse_patch_entities(patch: str, files: list[str]) -> list[str]:
    """从 patch 中提取被修改的函数/类名（基于 @@ hunk header）。"""
    entities = []
    current_file = None
    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
        elif line.startswith("@@") and current_file:
            # @@ -10,5 +10,7 @@ def some_function(...)
            m = re.search(r"@@.*@@\s+(?:def|class)\s+(\w+)", line)
            if m:
                entity = f"{current_file}:{m.group(1)}"
                if entity not in entities:
                    entities.append(entity)
    return entities


def parse_patch_modules(patch: str, files: list[str]) -> list[str]:
    """从 patch 文件列表中提取模块级实体。"""
    modules = []
    for fp in files:
        # 提取文件中的函数/类
        entities = []
        current_file = None
        for line in patch.split("\n"):
            if line.startswith("+++ b/"):
                current_file = line[6:].strip()
            elif line.startswith("@@") and current_file == fp:
                m = re.search(r"@@.*@@\s+(?:def|class)\s+(\w+)", line)
                if m:
                    entity = f"{fp}:{m.group(1)}"
                    if entity not in entities:
                        entities.append(entity)
        if entities:
            modules.extend(entities)
        else:
            # fallback: 用文件路径本身
            modules.append(fp)
    return modules


def compute_gt_file_changes(patch: str) -> list[dict]:
    """从 patch 计算 gt_file_changes 结构。"""
    changes = []
    current_file = None
    file_entities = {}

    for line in patch.split("\n"):
        if line.startswith("+++ b/"):
            current_file = line[6:].strip()
            if current_file and current_file != "/dev/null":
                file_entities.setdefault(current_file, [])
        elif line.startswith("@@") and current_file:
            m = re.search(r"@@.*@@\s+(?:def|class)\s+(\w+)", line)
            if m:
                entity = f"{current_file}:{m.group(1)}"
                if entity not in file_entities.get(current_file, []):
                    file_entities.setdefault(current_file, []).append(entity)

    for fp, ents in file_entities.items():
        changes.append({
            "file": fp,
            "changes": {
                "edited_entities": ents if ents else [fp],
                "edited_modules": ents if ents else [fp],
            },
        })
    return changes


def convert_instance(inst: dict) -> dict:
    """将 HuggingFace SWE-bench 实例转换为 TraceGen 格式。"""
    patch = inst.get("patch", "")
    found_files = parse_patch_files(patch)
    found_modules = parse_patch_modules(patch, found_files)
    found_entities = parse_patch_entities(patch, found_files)
    gt_file_changes = compute_gt_file_changes(patch)

    return {
        "repo": inst["repo"],
        "instance_id": inst["instance_id"],
        "base_commit": inst["base_commit"],
        "patch": patch,
        "test_patch": inst.get("test_patch", ""),
        "problem_statement": inst.get("problem_statement", ""),
        "hints_text": inst.get("hints_text", ""),
        "created_at": inst.get("created_at", ""),
        "version": inst.get("version", ""),
        "FAIL_TO_PASS": inst.get("FAIL_TO_PASS", []),
        "PASS_TO_PASS": inst.get("PASS_TO_PASS", []),
        "environment_setup_commit": inst.get("environment_setup_commit", ""),
        # TraceGen 扩展字段
        "found_files": found_files,
        "found_modules": found_modules,
        "found_entities": found_entities,
        "raw_output_loc": [],  # 无 LocAgent 输出，留空
        "gt_file_changes": gt_file_changes,
    }


def main():
    parser = argparse.ArgumentParser(description="从 SWE-bench 下载并转换多仓库实例")
    parser.add_argument(
        "--repos", nargs="+", default=PRIORITY_REPOS,
        help="要处理的仓库列表 (默认: 三个优先仓库)",
    )
    parser.add_argument(
        "--all-non-django", action="store_true",
        help="处理所有非 Django 仓库",
    )
    parser.add_argument(
        "--max-instances", type=int, default=0,
        help="每个仓库最大实例数 (0=不限制)",
    )
    parser.add_argument(
        "--output-dir", default="data",
        help="输出目录 (默认: data/)",
    )
    parser.add_argument(
        "--also-generate-targets", action="store_true", default=True,
        help="同时生成 configs/target_instances_<repo>.json",
    )
    args = parser.parse_args()

    repos = NON_DJANGO_REPOS if args.all_non_django else args.repos
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"目标仓库: {repos}")
    logger.info("从 HuggingFace 加载 SWE-bench 数据集...")

    from datasets import load_dataset
    ds = load_dataset(
        "princeton-nlp/SWE-bench", split="test",
        cache_dir=str(Path.home() / ".cache" / "huggingface" / "datasets"),
    )

    for repo in repos:
        repo_clean = repo.replace("/", "_")
        repo_short = repo.split("/")[-1]

        # 过滤该仓库的实例
        instances = [dict(inst) for inst in ds if inst["repo"] == repo]
        logger.info(f"[{repo}] 找到 {len(instances)} 个实例")

        if args.max_instances > 0:
            instances = instances[: args.max_instances]
            logger.info(f"[{repo}] 限制为 {len(instances)} 个实例")

        # 转换
        converted = [convert_instance(inst) for inst in instances]

        # 保存转换后的数据
        out_path = output_dir / f"swebench_{repo_short}.json"
        with open(out_path, "w") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)
        logger.info(f"[{repo}] 已保存 {len(converted)} 个实例到 {out_path}")

        # 生成 target_instances 文件
        if args.also_generate_targets:
            target_ids = [inst["instance_id"] for inst in converted]
            targets_path = Path("configs") / f"target_instances_{repo_short}.json"
            with open(targets_path, "w") as f:
                json.dump(target_ids, f, indent=2)
            logger.info(f"[{repo}] 已保存 target_instances 到 {targets_path}")

        # 统计
        unique_commits = len(set(inst["base_commit"] for inst in converted))
        avg_files = sum(len(inst["found_files"]) for inst in converted) / max(len(converted), 1)
        logger.info(
            f"[{repo}] 统计: {len(converted)} 实例, "
            f"{unique_commits} 个唯一 commit, "
            f"平均 {avg_files:.1f} 个修改文件/实例"
        )

    logger.info("完成!")


if __name__ == "__main__":
    main()
