#!/usr/bin/env python
"""
批量构建图文件的脚本
用于为指定的 Django 实例预先构建 V1 格式的图缓存
"""
import sys
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.graph.builder import CodeGraphBuilder
from src.core.utils import clone_repository, save_pickle, load_pickle

GRAPH_CACHE_VERSION = "v1"


def build_graph_for_instance(instance: dict, cache_dir: Path, graphs_dir: Path, use_incremental: bool = True):
    """为单个实例构建图"""
    repo = instance["repo"]
    commit = instance["base_commit"]
    instance_id = instance["instance_id"]

    repo_clean = repo.replace("/", "_")
    commit_short = commit[:8]

    graph_cache_path = graphs_dir / f"{repo_clean}__{commit_short}_{GRAPH_CACHE_VERSION}.pkl"

    if graph_cache_path.exists():
        logger.info(f"图缓存已存在: {graph_cache_path.name}")
        return graph_cache_path

    logger.info(f"[{instance_id}] 开始构建图: {repo}@{commit_short}")

    # 克隆/切换仓库
    repo_url = f"https://github.com/{repo}.git"
    repo_dir = cache_dir / "repos" / repo_clean

    try:
        clone_repository(repo_url, repo_dir, commit=commit)
    except Exception as e:
        logger.error(f"克隆仓库失败: {e}")
        return None

    # 查找可复用的基准图
    base_graph = None
    if use_incremental:
        candidate_graphs = list(graphs_dir.glob(f"{repo_clean}__*_{GRAPH_CACHE_VERSION}.pkl"))
        if candidate_graphs:
            latest_graph_path = max(candidate_graphs, key=lambda p: p.stat().st_mtime)
            logger.info(f"将基于已有图进行增量构建: {latest_graph_path.name}")
            try:
                base_graph = load_pickle(latest_graph_path)
            except Exception as e:
                logger.warning(f"加载基准图失败: {e}")

    # 构建图
    config = {
        "fuzzy_search": True,
        "global_import": False,
    }
    builder = CodeGraphBuilder(config)

    # 首先尝试增量构建
    if base_graph is not None:
        try:
            graph = builder.build_graph(
                str(repo_dir),
                entry_files=[],
                base_graph=base_graph
            )
            save_pickle(graph, graph_cache_path)
            logger.info(f"✅ 增量构建完成: {graph_cache_path.name} ({graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边)")
            return graph_cache_path
        except Exception as e:
            logger.warning(f"增量构建失败: {e}，尝试全量构建...")
            base_graph = None  # 清空基准图，尝试全量构建

    # 全量构建
    try:
        graph = builder.build_graph(
            str(repo_dir),
            entry_files=[],
            base_graph=None
        )
        save_pickle(graph, graph_cache_path)
        logger.info(f"✅ 全量构建完成: {graph_cache_path.name} ({graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边)")
        return graph_cache_path
    except Exception as e:
        logger.error(f"构建图失败: {e}")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="批量构建代码图")
    parser.add_argument("--data", default="data/swebench_converted.json",
                        help="SWE-bench 数据文件路径")
    parser.add_argument("--targets", default="configs/target_instances.json",
                        help="目标实例 ID 列表路径")
    parser.add_argument("--all", action="store_true",
                        help="处理所有实例 (忽略 targets 文件)")
    parser.add_argument("--full", action="store_true",
                        help="使用 full_run 配置 (114 Django 实例)")
    args = parser.parse_args()

    # 配置路径
    project_root = Path(__file__).parent.parent
    cache_dir = project_root / "data" / "assets"
    graphs_dir = cache_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # 根据参数选择数据文件
    if args.full:
        data_path = project_root / "data" / "swebench_converted_full.json"
        targets_path = project_root / "configs" / "target_instances_full.json"
    else:
        data_path = project_root / args.data
        targets_path = project_root / args.targets

    # 加载实例数据
    logger.info(f"加载数据文件: {data_path}")
    with open(data_path) as f:
        all_instances = json.load(f)

    if args.all:
        target_ids = [inst["instance_id"] for inst in all_instances]
    else:
        logger.info(f"加载目标实例: {targets_path}")
        with open(targets_path) as f:
            target_ids = json.load(f)

    # 过滤目标实例
    target_instances = [inst for inst in all_instances if inst["instance_id"] in target_ids]

    logger.info(f"将为 {len(target_instances)} 个实例构建图")

    # 按 commit 去重，同一个 commit 只需构建一次
    seen_commits = set()
    unique_instances = []
    for inst in target_instances:
        commit_key = f"{inst['repo']}_{inst['base_commit'][:8]}"
        if commit_key not in seen_commits:
            seen_commits.add(commit_key)
            unique_instances.append(inst)

    logger.info(f"去重后需要构建 {len(unique_instances)} 个不同的图")

    # 构建图
    success_count = 0
    for i, inst in enumerate(unique_instances, 1):
        logger.info(f"\n=== [{i}/{len(unique_instances)}] {inst['instance_id']} ===")
        result = build_graph_for_instance(inst, cache_dir, graphs_dir)
        if result:
            success_count += 1

    logger.info(f"\n=== 完成 ===")
    logger.info(f"成功构建: {success_count}/{len(unique_instances)}")


if __name__ == "__main__":
    main()
