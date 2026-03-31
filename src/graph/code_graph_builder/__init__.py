"""
Code Graph Builder - Standalone Module
A portable Python module for building code knowledge graphs

Usage:
    from code_graph_builder import build_graph_from_repo
    
    graph = build_graph_from_repo('/path/to/repo')
"""

from .graph_builder import (
    build_graph_from_repo,
    NODE_TYPE_DIRECTORY,
    NODE_TYPE_FILE,
    NODE_TYPE_CLASS,
    NODE_TYPE_FUNCTION,
    EDGE_TYPE_CONTAINS,
    EDGE_TYPE_INHERITS,
    EDGE_TYPE_INVOKES,
    EDGE_TYPE_IMPORTS,
    VERSION
)

__version__ = VERSION
__all__ = [
    'build_graph_from_repo',
    'NODE_TYPE_DIRECTORY',
    'NODE_TYPE_FILE',
    'NODE_TYPE_CLASS',
    'NODE_TYPE_FUNCTION',
    'EDGE_TYPE_CONTAINS',
    'EDGE_TYPE_INHERITS',
    'EDGE_TYPE_INVOKES',
    'EDGE_TYPE_IMPORTS',
]
