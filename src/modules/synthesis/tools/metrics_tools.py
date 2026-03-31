"""
Metrics computation tools for SynthesisAgent.
指标计算相关的合成 Agent 工具

Tools:
- ComputeChainMetricsTool: Compute complexity metrics for a defect chain
"""

from typing import Dict, Any, List, Optional
from loguru import logger
import json
import networkx as nx

from .base import BaseTool


class ComputeChainMetricsTool(BaseTool):
    """
    Tool to compute complexity metrics for a defect chain.
    计算缺陷链路复杂度指标的工具
    
    Computes:
    - Chain length (number of hops)
    - Average degree along the chain
    - Graph centrality of nodes
    - Estimated localization difficulty
    """
    
    name = "compute_chain_metrics"
    description = (
        "Compute complexity metrics for a proposed defect chain. "
        "Given a list of node IDs forming a chain (from symptom to root cause), "
        "this tool calculates metrics like chain length, node degrees, centrality, "
        "and an estimated difficulty score. Use this to ensure the synthesized bug "
        "has comparable complexity to the seed."
    )
    parameters = {
        "type": "object",
        "properties": {
            "chain_nodes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of node IDs forming the chain, ordered from symptom to root cause"
            },
            "compare_to_seed": {
                "type": "boolean",
                "description": "Whether to compare metrics with the seed chain (default: true)"
            }
        },
        "required": ["chain_nodes"]
    }
    
    def execute(
        self,
        chain_nodes: List[str],
        compare_to_seed: bool = True
    ) -> str:
        """
        Compute chain metrics.
        计算链路指标
        
        Args:
            chain_nodes: List of node IDs in the chain
            compare_to_seed: Whether to compare with seed
            
        Returns:
            JSON string with computed metrics
        """
        graph: nx.DiGraph = self.context.get("graph")
        if graph is None:
            return json.dumps({
                "error": "Graph not available in context"
            }, ensure_ascii=False)
        
        if not chain_nodes:
            return json.dumps({
                "error": "chain_nodes cannot be empty"
            }, ensure_ascii=False)
        
        # Validate nodes exist
        # 验证节点存在
        missing_nodes = [n for n in chain_nodes if n not in graph]
        if missing_nodes:
            return json.dumps({
                "error": "Some nodes not found in graph",
                "missing_nodes": missing_nodes[:5]
            }, ensure_ascii=False)
        
        # Compute metrics for the proposed chain
        # 计算提议链路的指标
        metrics = self._compute_metrics(graph, chain_nodes)
        
        result = {
            "proposed_chain": {
                "nodes": chain_nodes,
                "metrics": metrics
            }
        }
        
        # Compare with seed if requested
        # 如果请求，与 seed 比较
        if compare_to_seed:
            extraction_result = self.context.get("extraction_result")
            if extraction_result and extraction_result.chains:
                first_chain = extraction_result.chains[0]
                
                # 兼容字典和对象访问
                if isinstance(first_chain, dict):
                    nodes = first_chain.get("nodes", [])
                    seed_nodes = [n.get("node_id") if isinstance(n, dict) else n.node_id for n in nodes]
                else:
                    seed_nodes = [n.node_id for n in first_chain.nodes]
                
                # Only compute if seed nodes are in current graph
                # 仅当 seed 节点在当前图中时计算
                valid_seed_nodes = [n for n in seed_nodes if n in graph]
                if valid_seed_nodes:
                    seed_metrics = self._compute_metrics(graph, valid_seed_nodes)
                    result["seed_chain"] = {
                        "nodes": valid_seed_nodes,
                        "metrics": seed_metrics
                    }
                    
                    # Compute similarity score
                    # 计算相似度分数
                    result["comparison"] = self._compare_metrics(metrics, seed_metrics)
                else:
                    result["seed_chain"] = {
                        "warning": "Seed nodes not found in current graph (different repo)"
                    }
            else:
                result["seed_chain"] = {
                    "warning": "No seed chain available for comparison"
                }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def _compute_metrics(self, graph: nx.DiGraph, chain_nodes: List[str]) -> Dict[str, Any]:
        """
        Compute detailed metrics for a chain.
        计算链路的详细指标
        """
        metrics = {}
        
        # Basic metrics
        # 基本指标
        metrics["chain_length"] = len(chain_nodes)
        
        # Degree statistics
        # 度数统计
        in_degrees = [graph.in_degree(n) for n in chain_nodes]
        out_degrees = [graph.out_degree(n) for n in chain_nodes]
        
        metrics["degree_stats"] = {
            "avg_in_degree": round(sum(in_degrees) / len(in_degrees), 2) if in_degrees else 0,
            "avg_out_degree": round(sum(out_degrees) / len(out_degrees), 2) if out_degrees else 0,
            "max_in_degree": max(in_degrees) if in_degrees else 0,
            "max_out_degree": max(out_degrees) if out_degrees else 0,
            "total_edges": sum(in_degrees) + sum(out_degrees)
        }
        
        # Path connectivity
        # 路径连通性
        connected_pairs = 0
        for i in range(len(chain_nodes) - 1):
            if graph.has_edge(chain_nodes[i], chain_nodes[i+1]) or \
               graph.has_edge(chain_nodes[i+1], chain_nodes[i]):
                connected_pairs += 1
        
        metrics["path_connectivity"] = {
            "connected_pairs": connected_pairs,
            "total_pairs": len(chain_nodes) - 1 if len(chain_nodes) > 1 else 0,
            "connectivity_ratio": round(connected_pairs / (len(chain_nodes) - 1), 2) if len(chain_nodes) > 1 else 1.0
        }
        
        # Node type distribution
        # 节点类型分布
        type_counts = {}
        for node_id in chain_nodes:
            node_data = graph.nodes[node_id]
            node_type = node_data.get("type", node_data.get("node_type", "unknown"))
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        metrics["node_types"] = type_counts
        
        # Compute difficulty score (heuristic)
        # 计算难度分数（启发式）
        difficulty = self._compute_difficulty_score(metrics)
        metrics["difficulty_score"] = difficulty
        
        return metrics
    
    def _compute_difficulty_score(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute a heuristic difficulty score.
        计算启发式难度分数
        
        Factors:
        - Chain length: longer chains are harder to trace
        - Degree: high-degree nodes have more potential paths
        - Connectivity: well-connected chains may have multiple valid paths
        """
        score = 0.0
        factors = {}
        
        # Chain length factor (0-30 points)
        # 链路长度因子
        length = metrics["chain_length"]
        if length == 1:
            length_score = 5
        elif length == 2:
            length_score = 10
        elif length <= 4:
            length_score = 20
        else:
            length_score = min(30, 20 + (length - 4) * 2)
        
        factors["length"] = length_score
        score += length_score
        
        # Degree complexity factor (0-40 points)
        # 度数复杂度因子
        avg_in = metrics["degree_stats"]["avg_in_degree"]
        avg_out = metrics["degree_stats"]["avg_out_degree"]
        degree_score = min(40, (avg_in + avg_out) * 2)
        
        factors["degree_complexity"] = round(degree_score, 2)
        score += degree_score
        
        # Connectivity factor (0-30 points)
        # 连通性因子
        connectivity = metrics["path_connectivity"]["connectivity_ratio"]
        # Lower connectivity = harder (need to infer indirect paths)
        connectivity_score = (1 - connectivity) * 30
        
        factors["indirect_paths"] = round(connectivity_score, 2)
        score += connectivity_score
        
        return {
            "total": round(min(100, score), 2),
            "factors": factors,
            "level": self._score_to_level(score)
        }
    
    def _score_to_level(self, score: float) -> str:
        """Convert score to difficulty level."""
        if score < 20:
            return "trivial"
        elif score < 40:
            return "easy"
        elif score < 60:
            return "medium"
        elif score < 80:
            return "hard"
        else:
            return "expert"
    
    def _compare_metrics(
        self, 
        proposed: Dict[str, Any], 
        seed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare proposed chain metrics with seed.
        比较提议链路与 seed 的指标
        """
        comparison = {}
        
        # Length comparison
        # 长度比较
        len_diff = proposed["chain_length"] - seed["chain_length"]
        comparison["length_diff"] = len_diff
        
        # Degree comparison
        # 度数比较
        prop_avg = proposed["degree_stats"]["avg_in_degree"] + proposed["degree_stats"]["avg_out_degree"]
        seed_avg = seed["degree_stats"]["avg_in_degree"] + seed["degree_stats"]["avg_out_degree"]
        comparison["avg_degree_diff"] = round(prop_avg - seed_avg, 2)
        
        # Difficulty comparison
        # 难度比较
        prop_diff = proposed["difficulty_score"]["total"]
        seed_diff = seed["difficulty_score"]["total"]
        comparison["difficulty_diff"] = round(prop_diff - seed_diff, 2)
        
        # Overall similarity (0-100)
        # 总体相似度
        similarity = 100 - abs(comparison["difficulty_diff"]) - abs(len_diff * 5) - abs(comparison["avg_degree_diff"] * 2)
        comparison["similarity_score"] = round(max(0, min(100, similarity)), 2)
        
        # Recommendation
        # 建议
        if comparison["similarity_score"] >= 80:
            comparison["recommendation"] = "excellent_match"
        elif comparison["similarity_score"] >= 60:
            comparison["recommendation"] = "good_match"
        elif comparison["similarity_score"] >= 40:
            comparison["recommendation"] = "acceptable"
        else:
            comparison["recommendation"] = "consider_adjustment"
        
        return comparison
