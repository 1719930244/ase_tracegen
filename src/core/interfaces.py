"""
核心接口定义 - 定义所有抽象基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from .structures import DefectChain, SWEBenchInstance, ExtractionResult
import networkx as nx


class GraphEngine(ABC):
    """
    图引擎抽象基类
    负责从代码仓库构建依赖图并执行图分析算法
    """

    @abstractmethod
    def build_graph(self, repo_path: str, entry_files: List[str]) -> nx.DiGraph:
        """
        从代码仓库构建依赖图
        
        Args:
            repo_path: 仓库本地路径
            entry_files: 入口文件列表 (相对路径)
            
        Returns:
            NetworkX 有向图对象
        """
        pass



class Extractor(ABC):
    """
    缺陷链路提取器抽象基类
    负责从 SWE-Bench 数据中挖掘结构化的缺陷链路
    """

    @abstractmethod
    def extract_chains(
        self, instance: SWEBenchInstance, graph: nx.DiGraph
    ) -> List[DefectChain]:
        """
        从单个 SWE-Bench 实例中提取缺陷链路
        
        Args:
            instance: SWE-Bench 数据实例
            graph: 代码依赖图
            
        Returns:
            缺陷链路列表
        """
        pass



class Synthesizer(ABC):
    """
    缺陷合成器抽象基类
    负责基于提取的链路生成新的反事实缺陷数据
    """

    @abstractmethod
    def synthesize(
        self,
        extraction_result: ExtractionResult,
        graph: nx.DiGraph,
        repo_path: str,
        candidate: Dict[str, Any],
        rank: int = 1
    ) -> Optional[Any]:
        """
        基于缺陷链路合成新的缺陷实例
        """
        pass

