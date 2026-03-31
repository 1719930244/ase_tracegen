#!/usr/bin/env python
"""
读取和统计保存的 pkl 图文件

使用示例:
    python scripts/read_graph_stats.py outputs/2026-01-07_19-11-44/graphs/
    python scripts/read_graph_stats.py --graph outputs/2026-01-07_19-11-44/graphs/django__django-12345.pkl
"""
import sys
import argparse
from pathlib import Path
import networkx as nx

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.utils import load_pickle
from loguru import logger
import re


def count_code_tokens(code: str) -> int:
    """
    估算代码的 token 数量 (基于正则表达式匹配单词和符号)
    """
    if not code:
        return 0
    # 匹配单词、数字和各种操作符/符号
    tokens = re.findall(r"\w+|[^\w\s]", code)
    return len(tokens)


def analyze_single_graph(graph_path: Path, include_details: bool = False) -> dict:
    """
    分析单个图文件的统计信息
    
    Args:
        graph_path: pkl 图文件路径
        include_details: 是否包含详细信息 (如节点/边类型分布)
        
    Returns:
        统计信息字典
    """
    try:
        graph = load_pickle(graph_path)
        
        stats = {
            "file": str(graph_path),
            "nodes": len(graph.nodes()),
            "edges": len(graph.edges()),
            "density": nx.density(graph),
            "is_dag": nx.is_directed_acyclic_graph(graph),
        }
        
        # 计算节点类型分布
        node_types = {}
        for _, data in graph.nodes(data=True):
            ntype = data.get("type", "unknown")
            node_types[ntype] = node_types.get(ntype, 0) + 1
        stats["node_type_distribution"] = node_types
        
        # 计算边类型分布
        edge_types = {}
        for _, _, data in graph.edges(data=True):
            etype = data.get("type", "unknown")
            edge_types[etype] = edge_types.get(etype, 0) + 1
        stats["edge_type_distribution"] = edge_types

        if include_details:
            # 获取每种类型的前10个节点
            type_to_nodes = {}
            for node_id, data in graph.nodes(data=True):
                ntype = data.get("type", "unknown")
                if ntype not in type_to_nodes:
                    type_to_nodes[ntype] = []
                if len(type_to_nodes[ntype]) < 10:
                    type_to_nodes[ntype].append(node_id)
            stats["node_samples"] = type_to_nodes
            
            # 获取每种类型的前10条边
            type_to_edges = {}
            for u, v, data in graph.edges(data=True):
                etype = data.get("type", "unknown")
                if etype not in type_to_edges:
                    type_to_edges[etype] = []
                if len(type_to_edges[etype]) < 10:
                    type_to_edges[etype].append(f"{u} -> {v}")
            stats["edge_samples"] = type_to_edges
        
        # 计算连通性
        if nx.is_weakly_connected(graph):
            stats["weakly_connected_components"] = 1
        else:
            stats["weakly_connected_components"] = nx.number_weakly_connected_components(graph)
        
        # 计算平均出度和入度
        if len(graph.nodes()) > 0:
            in_degrees = [d for n, d in graph.in_degree()]
            out_degrees = [d for n, d in graph.out_degree()]
            stats["avg_in_degree"] = sum(in_degrees) / len(in_degrees)
            stats["avg_out_degree"] = sum(out_degrees) / len(out_degrees)
            stats["max_in_degree"] = max(in_degrees)
            stats["max_out_degree"] = max(out_degrees)
            
            # 计算叶子节点个数 (出度为 0 的节点)
            leaf_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]
            stats["leaf_nodes"] = len(leaf_nodes)
            
            # 计算叶子节点的代码总 token 数
            total_leaf_tokens = 0
            for node_id in leaf_nodes:
                node_data = graph.nodes[node_id]
                # 兼容不同的属性名
                code = node_data.get("code_snippet") or node_data.get("code", "")
                total_leaf_tokens += count_code_tokens(code)
            
            stats["total_leaf_tokens"] = total_leaf_tokens
            if stats["leaf_nodes"] > 0:
                stats["avg_leaf_tokens"] = total_leaf_tokens / stats["leaf_nodes"]
            else:
                stats["avg_leaf_tokens"] = 0
            
            # 计算 function 类型节点的平均代码长度 (字符数)
            func_nodes = [n for n, d in graph.nodes(data=True) if d.get("type") == "function"]
            total_func_len = 0
            for n in func_nodes:
                node_data = graph.nodes[n]
                code = node_data.get("code_snippet") or node_data.get("code", "")
                total_func_len += len(code)
            
            stats["function_count"] = len(func_nodes)
            stats["avg_func_len"] = total_func_len / len(func_nodes) if func_nodes else 0
        
        return stats
    except Exception as e:
        logger.error(f"分析图文件失败 {graph_path}: {e}")
        return None


def analyze_graph_directory(dir_path: Path, include_details_all: bool = False) -> list:
    """
    分析目录中所有的 pkl 图文件
    
    Args:
        dir_path: 包含 pkl 文件的目录
        include_details_all: 是否为所有图包含详细信息
        
    Returns:
        统计信息列表
    """
    graph_files = list(dir_path.glob("*.pkl"))
    
    if not graph_files:
        logger.warning(f"目录中未找到 pkl 文件: {dir_path}")
        return []
    
    logger.info(f"找到 {len(graph_files)} 个 pkl 文件，开始分析...")
    
    results = []
    for i, graph_file in enumerate(graph_files):
        # 如果是第一个文件，或者 include_details_all 为 True，则包含详细信息
        include_details = (i == 0) or include_details_all
        stats = analyze_single_graph(graph_file, include_details=include_details)
        if stats:
            results.append(stats)
    
    return results


def print_stats(stats_list: list):
    """
    格式化打印统计信息
    
    Args:
        stats_list: 统计信息列表
    """
    if not stats_list:
        logger.warning("没有可打印的统计信息")
        return
    
    print("\n" + "="*80)
    print("图统计信息汇总")
    print("="*80)
    
    # 总体统计
    total_nodes = sum(s.get("nodes", 0) for s in stats_list)
    total_edges = sum(s.get("edges", 0) for s in stats_list)
    avg_nodes = total_nodes / len(stats_list) if stats_list else 0
    avg_edges = total_edges / len(stats_list) if stats_list else 0
    
    print(f"\n总文件数: {len(stats_list)}")
    print(f"总节点数: {total_nodes}")
    print(f"总边数: {total_edges}")
    print(f"平均节点数/图: {avg_nodes:.1f}")
    print(f"平均边数/图: {avg_edges:.1f}")
    
    # 详细信息
    print("\n" + "-"*80)
    print("详细信息:")
    print("-"*80)
    
    for i, stats in enumerate(stats_list, 1):
        print(f"\n[{i}] {Path(stats['file']).name}")
        print(f"    节点数: {stats.get('nodes', 0)}")
        print(f"    边数: {stats.get('edges', 0)}")
        print(f"    密度: {stats.get('density', 0):.4f}")
        print(f"    是否DAG: {stats.get('is_dag', False)}")
        print(f"    弱连通分量: {stats.get('weakly_connected_components', 0)}")
        print(f"    平均入度: {stats.get('avg_in_degree', 0):.2f}")
        print(f"    平均出度: {stats.get('avg_out_degree', 0):.2f}")
        print(f"    最大入度: {stats.get('max_in_degree', 0)}")
        print(f"    最大出度: {stats.get('max_out_degree', 0)}")
        print(f"    叶子节点个数: {stats.get('leaf_nodes', 0)}")
        print(f"    叶子节点总 Token 数 (估算): {stats.get('total_leaf_tokens', 0)}")
        print(f"    叶子节点平均 Token 数: {stats.get('avg_leaf_tokens', 0):.1f}")
        print(f"    Function 平均代码字符长度: {stats.get('avg_func_len', 0):.1f}")
        
        # 打印类型分布
        if "node_type_distribution" in stats:
            print(f"    节点类型分布: {stats['node_type_distribution']}")
        if "edge_type_distribution" in stats:
            print(f"    边类型分布: {stats['edge_type_distribution']}")
            
        # 打印详细样本 (如果存在)
        if "node_samples" in stats:
            print("\n    节点采样 (每种类型前10个):")
            for ntype, nodes in stats["node_samples"].items():
                print(f"      - {ntype}: {nodes}")
        
        if "edge_samples" in stats:
            print("\n    边采样 (每种类型前10个):")
            for etype, edges in stats["edge_samples"].items():
                print(f"      - {etype}: {edges}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="分析保存的 pkl 图文件")
    parser.add_argument(
        "path",
        nargs="?",
        help="图文件所在目录或单个 pkl 文件路径"
    )
    parser.add_argument(
        "--graph",
        type=str,
        help="单个图文件路径 (可选，替代位置参数)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="打印详细信息"
    )
    
    args = parser.parse_args()
    
    # 确定要分析的路径
    if args.graph:
        path = Path(args.graph)
        if path.is_file() and path.suffix == ".pkl":
            stats = analyze_single_graph(path, include_details=True)
            if stats:
                print_stats([stats])
        else:
            logger.error(f"文件不存在或不是 pkl 格式: {args.graph}")
    else:
        path = Path(args.path)
        if path.is_dir():
            stats_list = analyze_graph_directory(path, include_details_all=args.verbose)
            print_stats(stats_list)
        elif path.is_file() and path.suffix == ".pkl":
            stats = analyze_single_graph(path, include_details=True)
            if stats:
                print_stats([stats])
        else:
            logger.error(f"路径不存在或无效: {args.path}")


if __name__ == "__main__":
    main()
