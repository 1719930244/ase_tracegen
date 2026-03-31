"""
图构建器 - 模块化设计，支持 CodeGraphBuilder 为主方案
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import networkx as nx
from loguru import logger

from ..core.interfaces import GraphEngine

from .code_graph_builder import (
    build_graph_from_repo,
    NODE_TYPE_DIRECTORY,
    NODE_TYPE_FILE,
    NODE_TYPE_CLASS,
    NODE_TYPE_FUNCTION,
    EDGE_TYPE_CONTAINS,
    EDGE_TYPE_INHERITS,
    EDGE_TYPE_INVOKES,
    EDGE_TYPE_IMPORTS,
)


class CodeGraphBuilder(GraphEngine):
    """
    基于 CodeGraphBuilder 的 AST 图构建器
    完全使用 AST 解析获取精准的类/函数定义和调用关系
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.graph = nx.DiGraph()
        self.fuzzy_search = config.get("fuzzy_search", True)
        self.global_import = config.get("global_import", False)
        logger.info("CodeGraphBuilder 初始化完成")
    
    def build_graph(self, repo_path: str, entry_files: List[str], base_graph: Optional[nx.DiGraph] = None) -> nx.DiGraph:
        """
        使用 CodeGraphBuilder 构建依赖图

        Args:
            repo_path: 仓库本地路径
            entry_files: 入口文件列表 (此参数在 CodeGraphBuilder 模式下可选)
            base_graph: 可选的基础图，用于增量构建

        Returns:
            NetworkX 多重有向图
        """
        logger.info(f"开始构建代码图: {repo_path}")
        if base_graph:
            logger.info("启用增量构建模式")

        try:
            # 使用 CodeGraphBuilder 构建图
            code_graph = build_graph_from_repo(
                repo_path,
                fuzzy_search=self.fuzzy_search,
                global_import=self.global_import,
                base_graph=base_graph
            )

            # 将 CodeGraphBuilder 的 MultiDiGraph 转换为 DiGraph (去重边)
            # 使用局部变量，确保线程安全 (多个 worker 可能并发调用 build_graph)
            graph = nx.DiGraph(code_graph)

            logger.info(
                f"代码图构建完成: {graph.number_of_nodes()} 节点, "
                f"{graph.number_of_edges()} 边"
            )

            # 验证节点属性
            self._validate_and_enhance_nodes(graph)

            return graph
            
        except Exception as e:
            import traceback
            logger.error(f"CodeGraphBuilder 构建失败: {e}")
            logger.error(f"详细堆栈:\n{traceback.format_exc()}")
            raise
    
    def _validate_and_enhance_nodes(self, graph: nx.DiGraph = None):
        """
        验证和增强节点属性，确保兼容性。
        接受显式 graph 参数以支持线程安全调用。
        """
        if graph is None:
            graph = self.graph
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            
            # 确保必要的属性存在
            if "file_path" not in node_data:
                node_data["file_path"] = ""
            if "line_range" not in node_data:
                node_data["line_range"] = (0, 0)
            if "code_snippet" not in node_data:
                node_data["code_snippet"] = ""
    

