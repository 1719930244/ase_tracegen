"""
Graph query tools for SynthesisAgent.
图查询相关的合成 Agent 工具

Tools:
- QueryGraphTool: Query graph nodes, neighbors, and paths
- SearchSimilarNodesTool: Search for structurally similar nodes
"""

from typing import Dict, Any, List, Optional
from loguru import logger
import json
import networkx as nx


from .base import BaseTool


class QueryGraphTool(BaseTool):
    """
    Tool to query the code repository graph.
    查询代码仓库图的工具
    
    Supports:
    - Getting node attributes
    - Finding neighbors (predecessors/successors)
    - Finding paths between nodes
    """
    
    name = "query_graph"
    description = (
        "Query the code repository graph to get node information, neighbors, or paths. "
        "The graph contains classes, functions, and their relationships (invokes, inherits, etc.). "
        "Use 'node_info' to get details about a specific node, 'neighbors' to find connected nodes, "
        "or 'path' to find the shortest path between two nodes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query_type": {
                "type": "string",
                "description": "Type of query: 'node_info', 'neighbors', 'predecessors', 'successors', or 'path'",
                "enum": ["node_info", "neighbors", "predecessors", "successors", "path"]
            },
            "node_id": {
                "type": "string",
                "description": "The node ID to query (format: 'file_path:entity_name', e.g., 'django/utils/text.py:slugify')"
            },
            "target_node_id": {
                "type": "string",
                "description": "Target node ID for path queries (required if query_type='path')"
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth for neighbor search (default: 1)"
            }
        },
        "required": ["query_type", "node_id"]
    }
    
    def execute(
        self,
        query_type: str,
        node_id: str,
        target_node_id: Optional[str] = None,
        max_depth: int = 1
    ) -> str:
        """
        Execute graph query.
        执行图查询
        
        Args:
            query_type: Type of query
            node_id: Source node ID
            target_node_id: Target node ID (for path queries)
            max_depth: Maximum search depth
            
        Returns:
            JSON string with query results
        """
        # Get graph from context
        # 从上下文获取图
        graph: nx.DiGraph = self.context.get("graph")
        if graph is None:
            return json.dumps({
                "error": "Graph not available in context",
                "hint": "Ensure the graph is loaded before using this tool"
            }, ensure_ascii=False)
        
        # Check if node exists
        # 检查节点是否存在
        if node_id not in graph:
            # Try fuzzy matching
            # 尝试模糊匹配
            candidates = self._fuzzy_find_node(graph, node_id)
            if candidates:
                return json.dumps({
                    "error": f"Node not found: {node_id}",
                    "similar_nodes": candidates[:5],
                    "hint": "Try using one of the similar nodes listed above"
                }, ensure_ascii=False)
            else:
                return json.dumps({
                    "error": f"Node not found: {node_id}",
                    "total_nodes": graph.number_of_nodes()
                }, ensure_ascii=False)
        
        # Execute query based on type
        # 根据类型执行查询
        if query_type == "node_info":
            return self._get_node_info(graph, node_id)
        elif query_type == "neighbors":
            return self._get_neighbors(graph, node_id, max_depth, direction="both")
        elif query_type == "predecessors":
            return self._get_neighbors(graph, node_id, max_depth, direction="in")
        elif query_type == "successors":
            return self._get_neighbors(graph, node_id, max_depth, direction="out")
        elif query_type == "path":
            if not target_node_id:
                return json.dumps({
                    "error": "target_node_id is required for path queries"
                }, ensure_ascii=False)
            return self._find_path(graph, node_id, target_node_id)
        else:
            return json.dumps({
                "error": f"Unknown query_type: {query_type}",
                "valid_types": ["node_info", "neighbors", "predecessors", "successors", "path"]
            }, ensure_ascii=False)
    
    def _fuzzy_find_node(self, graph: nx.DiGraph, query: str, limit: int = 5) -> List[str]:
        """
        Fuzzy search for node by partial match.
        通过部分匹配模糊搜索节点
        """
        query_lower = query.lower()
        matches = []
        
        for node_id in graph.nodes():
            if query_lower in node_id.lower():
                matches.append(node_id)
                if len(matches) >= limit:
                    break
        
        return matches
    
    def _get_node_info(self, graph: nx.DiGraph, node_id: str) -> str:
        """Get detailed information about a node."""
        node_data = dict(graph.nodes[node_id])
        
        # Add degree information
        # 添加度数信息
        in_degree = graph.in_degree(node_id)
        out_degree = graph.out_degree(node_id)
        
        result = {
            "node_id": node_id,
            "attributes": {},
            "metrics": {
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": in_degree + out_degree
            }
        }
        
        # Extract relevant attributes
        # 提取相关属性
        for key, value in node_data.items():
            if key in ["type", "node_type", "file_path", "file", "name", "start_line", "end_line", "code"]:
                if key == "code" and value:
                    result["attributes"][key] = value[:500] + "..." if len(str(value)) > 500 else value
                else:
                    result["attributes"][key] = value
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def _get_neighbors(
        self, 
        graph: nx.DiGraph, 
        node_id: str, 
        max_depth: int,
        direction: str
    ) -> str:
        """Get neighbors of a node."""
        neighbors = []
        
        if direction in ["in", "both"]:
            for pred in graph.predecessors(node_id):
                edge_data = graph.get_edge_data(pred, node_id)
                neighbors.append({
                    "node_id": pred,
                    "direction": "predecessor",
                    "edge_type": edge_data.get("type", "unknown") if edge_data else "unknown"
                })
        
        if direction in ["out", "both"]:
            for succ in graph.successors(node_id):
                edge_data = graph.get_edge_data(node_id, succ)
                neighbors.append({
                    "node_id": succ,
                    "direction": "successor",
                    "edge_type": edge_data.get("type", "unknown") if edge_data else "unknown"
                })
        
        result = {
            "node_id": node_id,
            "neighbor_count": len(neighbors),
            "neighbors": neighbors[:50]  # Limit to 50
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def _find_path(self, graph: nx.DiGraph, source: str, target: str) -> str:
        """Find shortest path between two nodes."""
        if target not in graph:
            return json.dumps({
                "error": f"Target node not found: {target}"
            }, ensure_ascii=False)
        
        try:
            # Try to find shortest path
            # 尝试找最短路径
            path = nx.shortest_path(graph, source, target)
            
            # Get edge types along the path
            # 获取路径上的边类型
            edges = []
            for i in range(len(path) - 1):
                edge_data = graph.get_edge_data(path[i], path[i+1])
                edges.append({
                    "from": path[i],
                    "to": path[i+1],
                    "type": edge_data.get("type", "unknown") if edge_data else "unknown"
                })
            
            result = {
                "source": source,
                "target": target,
                "path_length": len(path) - 1,
                "path": path,
                "edges": edges
            }
            
        except nx.NetworkXNoPath:
            result = {
                "source": source,
                "target": target,
                "path": None,
                "message": "No path exists between the nodes"
            }
        except nx.NetworkXError as e:
            result = {
                "error": f"Path finding failed: {str(e)}"
            }
        
        return json.dumps(result, ensure_ascii=False, indent=2)


class SearchSimilarNodesTool(BaseTool):
    """
    Tool to search for structurally similar nodes in the graph.
    在图中搜索结构相似节点的工具
    
    Searches for nodes that match specified criteria:
    - Node type (function, class, method)
    - Depth range (distance from entry points)
    - Degree range (in/out connections)
    - Code content patterns (regex, string literals, etc.)
    """
    
    name = "search_similar_nodes"
    description = (
        "Search for nodes in the graph that are structurally similar to the seed's nodes. "
        "You can filter by node type, degree range, file pattern, and **code content**. "
        "**NEW**: Use `max_distance` to limit the search to nodes within N hops of the original seed nodes. "
        "This ensures the synthesized bug is semantically related to the original issue."
    )
    parameters = {
        "type": "object",
        "properties": {
            "node_type": {
                "type": "string",
                "description": "Filter by node type: 'function', 'class', 'method', or 'any'",
                "enum": ["function", "class", "method", "any"]
            },
            "max_distance": {
                "type": "integer",
                "description": "Maximum graph distance (hops) from the original seed nodes. Suggested: 1-5."
            },
            "min_in_degree": {
                "type": "integer",
                "description": "Minimum in-degree (callers). Default: 0"
            },
            "max_in_degree": {
                "type": "integer",
                "description": "Maximum in-degree. Default: 100"
            },
            "file_pattern": {
                "type": "string",
                "description": "Filter by file path pattern (substring match). Default: no filter"
            },
            "exclude_pattern": {
                "type": "string",
                "description": "Exclude nodes matching this pattern (e.g., 'test' to exclude test files)"
            },
            "code_pattern": {
                "type": "string",
                "description": "Search for nodes whose code contains this pattern (e.g., 'regex', 'r\\'', 'validator')"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results. Default: 20"
            }
        },
        "required": ["node_type"]
    }
    
    def execute(
        self,
        node_type: str = "any",
        max_distance: Optional[int] = None,
        min_in_degree: int = 0,
        max_in_degree: int = 100,
        min_out_degree: int = 0,
        max_out_degree: int = 100,
        file_pattern: Optional[str] = None,
        exclude_pattern: Optional[str] = None,
        code_pattern: Optional[str] = None,
        limit: int = 20,
        **kwargs  # Handle unexpected arguments like degree_range
    ) -> str:
        """
        Search for similar nodes based on criteria.
        根据条件搜索相似节点
        """
        # Handle 'degree_range' if sent by LLM
        if 'degree_range' in kwargs:
            dr = kwargs['degree_range']
            if isinstance(dr, dict):
                in_range = dr.get('in', [])
                if len(in_range) >= 2:
                    min_in_degree, max_in_degree = in_range[0], in_range[1]
                out_range = dr.get('out', [])
                if len(out_range) >= 2:
                    min_out_degree, max_out_degree = out_range[0], out_range[1]

        graph: nx.DiGraph = self.context.get("graph")
        if graph is None:
            return json.dumps({
                "error": "Graph not available in context"
            }, ensure_ascii=False)
        
        # 1. Handle distance-based scope filtering
        scoped_nodes = None
        if max_distance is not None:
            extraction_result = self.context.get("extraction_result")
            if extraction_result:
                # 兼容新的 mined_data 结构
                mined_data = getattr(extraction_result, "mined_data", {})
                chains = mined_data.get("extracted_chains", [])
                
                seed_node_ids = []
                for chain in chains:
                    nodes = chain.get("nodes", []) if isinstance(chain, dict) else chain.nodes
                    seed_node_ids.extend([n.get("node_id") if isinstance(n, dict) else n.node_id for n in nodes])
                
                nodes_in_range = set()
                for seed_id in seed_node_ids:
                    if seed_id in graph:
                        lengths = nx.single_source_shortest_path_length(graph, seed_id, cutoff=max_distance)
                        nodes_in_range.update(lengths.keys())
                scoped_nodes = nodes_in_range

        # 2. Collect matching nodes
        candidates = []
        search_space = scoped_nodes if scoped_nodes is not None else graph.nodes()
        
        for node_id in search_space:
            node_data = graph.nodes[node_id]
            
            # Filter by node type
            if node_type != "any":
                actual_type = node_data.get("type", node_data.get("node_type", ""))
                if not actual_type:
                    if ":" in node_id:
                        entity_part = node_id.split(":")[-1]
                        if "." in entity_part: actual_type = "method"
                        elif entity_part and entity_part[0].isupper(): actual_type = "class"
                        else: actual_type = "function"
                    else: actual_type = "unknown"
                
                if node_type.lower() not in actual_type.lower():
                    continue
            
            # Filter by degree
            in_deg = graph.in_degree(node_id)
            if not (min_in_degree <= in_deg <= max_in_degree):
                continue
            
            # Filter by file pattern
            if file_pattern:
                import fnmatch
                if '*' in file_pattern:
                    if not fnmatch.fnmatch(node_id.lower(), file_pattern.lower()): 
                        continue
                elif file_pattern.lower() not in node_id.lower():
                    continue
            
            # Exclude pattern
            if exclude_pattern and exclude_pattern.lower() in node_id.lower():
                continue
            
            # Filter by code content pattern
            if code_pattern:
                code = node_data.get("code", "")
                if not code or code_pattern.lower() not in code.lower():
                    continue
            
            # Add to candidates
            code_snippet = node_data.get("code", "")
            candidates.append({
                "node_id": node_id,
                "in_degree": in_deg,
                "type": node_data.get("type", node_data.get("node_type", "unknown")),
                "code_preview": code_snippet[:100] if code_snippet else ""
            })
            
            if len(candidates) >= limit:
                break
        
        return json.dumps({
            "query": {"node_type": node_type, "max_distance": max_distance},
            "total_found": len(candidates),
            "candidates": candidates
        }, ensure_ascii=False, indent=2)
