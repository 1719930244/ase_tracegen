"""
TraceGen 核心模块
包含所有数据结构、接口定义和通用工具
"""

from .interfaces import GraphEngine, Extractor, Synthesizer
from .structures import DefectChain, ChainNode, SWEBenchInstance
from .utils import load_json, save_json, clone_repository

__all__ = [
    "GraphEngine",
    "Extractor",
    "Synthesizer",
    "DefectChain",
    "ChainNode",
    "SWEBenchInstance",
    "load_json",
    "save_json",
    "clone_repository",
]
