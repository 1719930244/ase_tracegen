"""
缺陷链路提取器 - 阶段一:从 SWE-Bench 数据挖掘链路
"""
from typing import List, Dict, Any, Optional
import networkx as nx
from loguru import logger
from pathlib import Path

from ...core.interfaces import Extractor
from ...core.structures import DefectChain, SWEBenchInstance, ChainNode, RepairChain
from ...core.utils import parse_diff_hunks, robust_json_load
from ...core.exceptions import LLMResponseError
from .prompts.extraction_prompts import LOC_EXTRACTION_PROMPT, REPAIR_EXTRACTION_PROMPT
from ..llm_client import LLMClient


class ChainExtractor(Extractor):
    """
    缺陷链路提取器
    """
    
    def __init__(self, synthesis_llm: LLMClient, config: Dict[str, Any], analyzer_llm: Optional[LLMClient] = None):
        """
        初始化提取器
        
        Args:
            synthesis_llm: 合成 LLM 客户端
            config: 提取配置
            analyzer_llm: 分析 LLM 客户端 (用于链路提取)
        """
        self.analyzer_llm = analyzer_llm
        self.config = config
    
    def extract_chains(
        self, instance: SWEBenchInstance, graph: nx.DiGraph
    ) -> List[DefectChain]:
        """
        从 LocAgent 日志中提取的链路与当前仓库图实体进行绑定
        
        Args:
            instance: SWE-Bench 实例
            graph: 代码依赖图 (CodeGraphBuilder 构建)
            
        Returns:
            缺陷链路列表
        """
        # 获取图统计信息以便展示
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        logger.info(f"开始提取链路: {instance.instance_id} | 图规模: {num_nodes} 节点, {num_edges} 边")
        
        chains = []
        
        # 如果有分析 LLM，使用 LLM 提取
        if self.analyzer_llm and instance.raw_output_loc:
            logger.info(f"使用分析LLM提取链路...")
            chains = self._analyze_with_llm(instance, graph)
        
        logger.info(f"提取完成: {len(chains)} 条链路")
        return chains
    
    def _analyze_with_llm(
        self, instance: SWEBenchInstance, graph: nx.DiGraph
    ) -> List[DefectChain]:
        """
        使用分析 LLM 从定位日志中提取链路
        
        Args:
            instance: SWE-Bench 实例
            graph: 代码图
            
        Returns:
            提取的链路列表
        """
        chains = []
        
        try:
            # 整理定位日志
            loc_logs = instance.raw_output_loc if isinstance(instance.raw_output_loc, list) else [instance.raw_output_loc]
            
            # 构建 LLM 提示词
            prompt = self._build_analyzer_prompt(instance, graph, loc_logs)
            
            logger.debug(f"分析 LLM 提示词:\n{prompt}")
            
            # 调用 LLM
            response = self.analyzer_llm.complete(prompt)
            
            logger.debug(f"分析 LLM 响应:\n{response}")
            
            # 解析 LLM 响应 (定位链路)
            chains = self._parse_llm_response(response, instance, graph)
            
            # 阶段二: 提取修复链路和子图上下文
            if chains:
                for chain in chains:
                    # 1. 提取修复意图
                    repair_chain = self._extract_repair_chain(instance, chain, chain.root_cause_node.node_id, graph)
                    if repair_chain:
                        chain.extraction_metadata["repair_chain"] = repair_chain.model_dump()

                    # 2. 提取子图和上下文范围 (新增强化逻辑)
                    subgraph_data = self._extract_seed_subgraph(chain.nodes, graph)
                    chain.extraction_metadata["subgraph"] = subgraph_data
                    chain.extraction_metadata["context_files"] = list(set(n.file_path for n in chain.nodes if n.file_path))

                    # 3. 计算难度特征 (depth, in_degree, betweenness_centrality)
                    rc_nid = chain.root_cause_node.node_id
                    difficulty = {}
                    if rc_nid in graph:
                        difficulty["in_degree"] = graph.in_degree(rc_nid)
                        difficulty["out_degree"] = graph.out_degree(rc_nid)
                        # depth: shortest path from any entry node (node with in_degree=0)
                        try:
                            entry_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
                            min_depth = float("inf")
                            for en in entry_nodes[:20]:  # limit to avoid long computation
                                try:
                                    d = nx.shortest_path_length(graph, en, rc_nid)
                                    min_depth = min(min_depth, d)
                                except nx.NetworkXNoPath:
                                    pass
                            difficulty["depth"] = min_depth if min_depth != float("inf") else len(chain.nodes)
                        except Exception:
                            difficulty["depth"] = len(chain.nodes)
                        # betweenness centrality (approximate for large graphs)
                        try:
                            if graph.number_of_nodes() > 5000:
                                bc = nx.betweenness_centrality(graph, k=min(100, graph.number_of_nodes()))
                            else:
                                bc = nx.betweenness_centrality(graph)
                            difficulty["betweenness_centrality"] = round(bc.get(rc_nid, 0.0), 6)
                        except Exception:
                            difficulty["betweenness_centrality"] = 0.0
                    else:
                        difficulty = {"depth": len(chain.nodes), "in_degree": 0, "out_degree": 0, "betweenness_centrality": 0.0}

                    chain.extraction_metadata["difficulty_features"] = difficulty
                    # Also store node type pattern for Stage 2 constraint
                    chain.extraction_metadata["node_type_pattern"] = [n.node_type for n in chain.nodes]
            
        except Exception as e:
            logger.warning(f"分析 LLM 提取失败: {e}")
        
        return chains

    def _extract_seed_subgraph(self, chain_nodes: List[ChainNode], graph: nx.DiGraph, hops: int = 1) -> Dict[str, Any]:
        """
        提取 Seed 链路周围的子图
        """
        seed_node_ids = [n.node_id for n in chain_nodes]
        subgraph_nodes = set(seed_node_ids)
        
        # 扩展 N 跳
        for _ in range(hops):
            new_nodes = set()
            for node in subgraph_nodes:
                if node in graph:
                    # 包含入边和出边邻居
                    new_nodes.update(graph.predecessors(node))
                    new_nodes.update(graph.successors(node))
            subgraph_nodes.update(new_nodes)
            
        # 提取子图
        sub_g = graph.subgraph(subgraph_nodes)
        
        # 序列化子图数据
        nodes_info = {}
        for n in sub_g.nodes():
            data = graph.nodes[n]
            nodes_info[n] = {
                "type": data.get("type", "unknown"),
                "file_path": data.get("file_path", "")
            }
            
        edges_info = []
        for u, v, data in sub_g.edges(data=True):
            edges_info.append({
                "source": u,
                "target": v,
                "type": data.get("type", "unknown")
            })
            
        return {
            "nodes": nodes_info,
            "edges": edges_info,
            "context_range": list(set(d["file_path"] for d in nodes_info.values() if d["file_path"]))
        }

    def _extract_repair_chain(
        self, instance: SWEBenchInstance, loc_chain: DefectChain, root_cause_node_id: str, graph: nx.DiGraph
    ) -> Optional[RepairChain]:
        """
        提取结构化修复链路
        """
        try:
            # 1. 获取缺陷代码
            buggy_code = ""
            if root_cause_node_id in graph:
                buggy_code = graph.nodes[root_cause_node_id].get("code", "")
            
            if not buggy_code:
                logger.warning(f"未能在图中找到根因节点的代码: {root_cause_node_id}")
                return None

            # 2. 总结定位链路
            loc_summary = " -> ".join([f"{n.node_id}({n.node_type})" for n in loc_chain.nodes])

            # 3. 格式化 Prompt
            prompt = REPAIR_EXTRACTION_PROMPT.format(
                problem_statement=instance.problem_statement,
                localization_summary=loc_summary,
                root_cause_node_id=root_cause_node_id,
                buggy_code_snippet=buggy_code,
                patch_diff=instance.patch,
                instance_id=instance.instance_id
            )

            # 4. 调用 LLM
            logger.info(f"正在提取修复链路: {instance.instance_id}")
            response = self.analyzer_llm.complete(prompt)
            
            # 5. 使用 robust_json_load 替代原有的正则逻辑
            try:
                data = robust_json_load(response)
                return RepairChain(**data)
            except LLMResponseError as e:
                logger.warning(f"修复链路 JSON 解析失败: {e}")
                return None

        except Exception as e:
            logger.warning(f"提取修复链路失败: {e}")
            return None
    
    def _build_analyzer_prompt(self, instance: SWEBenchInstance, graph: nx.DiGraph, loc_logs: List[str]) -> str:
        """
        构建定位链路提取的提示词
        """
        loc_text = "\n---\n".join(loc_logs) if loc_logs else ""

        # 生成图节点样本，帮助 LLM 理解正确的节点 ID 格式
        graph_nodes_sample = self._generate_graph_nodes_sample(instance, graph)

        return LOC_EXTRACTION_PROMPT.format(
            instance_id=instance.instance_id,
            repo=instance.repo,
            problem_statement=instance.problem_statement[:1000], # 限制长度
            loc_text=loc_text,
            graph_nodes_sample=graph_nodes_sample
        )

    def _generate_graph_nodes_sample(self, instance: SWEBenchInstance, graph: nx.DiGraph) -> str:
        """
        生成图节点样本，帮助 LLM 理解正确的节点 ID 格式

        优先展示与 patch 相关文件中的节点
        """
        # 从 patch 中提取涉及的文件
        patch_files = set()
        try:
            hunks = parse_diff_hunks(instance.patch)
            patch_files = set(h["file"] for h in hunks if h.get("file"))
        except (ValueError, KeyError) as e:
            logger.debug(f"Failed to parse diff hunks: {e}")
        relevant_nodes = []
        other_nodes = []

        for node_id in graph.nodes():
            if ":" not in node_id:
                continue
            file_path = node_id.split(":")[0]
            # 优先收集 patch 相关文件的节点
            if any(pf in file_path or file_path in pf for pf in patch_files):
                relevant_nodes.append(node_id)
            else:
                other_nodes.append(node_id)

        # 限制数量：优先展示相关节点，补充其他节点
        sample_nodes = relevant_nodes[:15]
        if len(sample_nodes) < 20:
            sample_nodes.extend(other_nodes[:20 - len(sample_nodes)])

        if not sample_nodes:
            return ""

        # 格式化输出
        nodes_list = "\n".join(f"  - `{n}`" for n in sample_nodes[:20])
        return f"""## Available Nodes in Graph (Sample)
The following are REAL node IDs from the code graph. Use similar format:
{nodes_list}
"""
    
    def _parse_llm_response(self, response: str, instance: SWEBenchInstance, graph: nx.DiGraph) -> List[DefectChain]:
        """
        解析 LLM 响应並构建链路
        """
        chains = []
        
        try:
            # 1. 使用 robust_json_load 解析 JSON
            try:
                data = robust_json_load(response)
            except LLMResponseError as e:
                logger.warning(f"定位链路响应解析失败: {e}")
                return []
            
            # 2. 解析新格式 (loc_chain)
            loc_chain_edges = data.get("loc_chain", [])
            if not loc_chain_edges:
                logger.warning(f"定位链路为空: {instance.instance_id}")
                return []
            
            # 从边列表中提取所有涉及的节点
            # 第一个边的source为症状，最后一个边的target为根因
            symptom_node_id = loc_chain_edges[0][0] if loc_chain_edges else None
            root_cause_node_id = loc_chain_edges[-1][1] if loc_chain_edges else None
            chain_edges = loc_chain_edges
            
            # 3. 辅助函数：从 LLM 节点格式查找图中的实际节点
            def find_node_in_graph(llm_node_str: str) -> Optional[str]:
                """
                LLM 可能返回多种格式，需要容错处理：
                - 正确格式: "path/to/file.py:func_name" 或 "path/to/file.py:Class.method"
                - 错误格式: "path/to/file.py:63-70:func_name" (带行号)
                - 错误格式: "path/to/file.py:63:func_name" (带单行号)

                CodeGraphBuilder 格式: "path/to/file.py:func_name" 或 "path/to/file.py:Class.func_name"
                """
                if not llm_node_str or not isinstance(llm_node_str, str):
                    return None

                try:
                    # 策略 0: 直接精确匹配
                    if llm_node_str in graph:
                        return llm_node_str

                    # 提取关键信息
                    parts = llm_node_str.split(":")
                    if len(parts) < 2:
                        return None

                    file_path = parts[0]  # e.g. "django/contrib/auth/validators.py"
                    # 最后一个部分通常是函数名/类名
                    func_name = parts[-1]  # e.g. "ASCIIUsernameValidator.__call__"

                    # 策略 A: 组合匹配 (去除行号，只用 file:func)
                    candidate_id = f"{file_path}:{func_name}"
                    if candidate_id in graph:
                        return candidate_id

                    # 策略 B: 处理中间有行号的情况 (file:line:func 或 file:line-line:func)
                    # 如果有 3 个或更多部分，尝试跳过中间的行号部分
                    if len(parts) >= 3:
                        # 检查中间部分是否像行号 (纯数字或 "数字-数字")
                        middle_part = parts[1]
                        if middle_part.isdigit() or (
                            "-" in middle_part and
                            all(p.isdigit() for p in middle_part.split("-"))
                        ):
                            # 跳过行号，直接用 file:func
                            candidate_id = f"{file_path}:{func_name}"
                            if candidate_id in graph:
                                return candidate_id

                    # 策略 C: 模糊路径匹配 (处理绝对路径/相对路径差异)
                    file_name = Path(file_path).name
                    for node_id in graph.nodes():
                        if ":" in node_id:
                            g_file, g_func = node_id.split(":", 1)
                            if (file_name in g_file or g_file.endswith(file_path)) and (func_name == g_func):
                                return node_id

                    # 策略 D: 仅函数名匹配 (最后的手段)
                    # 优先匹配完全相同的函数名
                    for node_id in graph.nodes():
                        if node_id.endswith(f":{func_name}"):
                            return node_id

                    # 策略 E: 部分函数名匹配 (处理 Class.method vs method)
                    if "." in func_name:
                        method_name = func_name.split(".")[-1]
                        for node_id in graph.nodes():
                            if node_id.endswith(f".{method_name}") or node_id.endswith(f":{method_name}"):
                                # 额外检查文件路径是否相关
                                if file_name in node_id.split(":")[0]:
                                    return node_id

                    return None
                except Exception as e:
                    logger.debug(f"节点匹配出错: {e}")
                    return None
            
            # 4. 构建节点对象
            nodes = []
            
            # 定义一个映射字典，防止重复匹配
            node_map = {}
            
            # 处理症状节点
            if symptom_node_id:
                matched = find_node_in_graph(symptom_node_id)
                if matched:
                    node_map["symptom"] = matched
            
            # 处理根因节点
            if root_cause_node_id:
                matched = find_node_in_graph(root_cause_node_id)
                if matched:
                    node_map["root_cause"] = matched
            
            # 处理中间节点 (从边中提取)
            intermediate_candidates = []
            for edge in chain_edges:
                if len(edge) >= 2:
                    intermediate_candidates.append(edge[0])
                    intermediate_candidates.append(edge[1])
            
            # 按顺序构建节点序列 (去重并保持拓扑顺序)
            unique_matched_ids = []
            seen = set()
            
            # 必须包含症状
            if "symptom" in node_map:
                unique_matched_ids.append(node_map["symptom"])
                seen.add(node_map["symptom"])
                
            # 添加中间节点
            for cand in intermediate_candidates:
                m = find_node_in_graph(cand)
                if m and m not in seen:
                    unique_matched_ids.append(m)
                    seen.add(m)
            
            # 必须包含根因 (如果不在 seen 中)
            if "root_cause" in node_map and node_map["root_cause"] not in seen:
                unique_matched_ids.append(node_map["root_cause"])
            
            # 转换为 ChainNode 对象
            for i, nid in enumerate(unique_matched_ids):
                node_data = graph.nodes[nid]
                ntype = "intermediate"
                if i == 0: ntype = "symptom"
                if i == len(unique_matched_ids) - 1: ntype = "root_cause"

                # 兼容 CodeGraphBuilder 的属性名 (code, start_line, end_line)
                # 注意：需要检查值是否为空，而不仅仅是键是否存在
                file_path = node_data.get("file_path") or node_data.get("file") or (nid.split(":")[0] if ":" in nid else "")

                # line_range: 优先使用 line_range，否则从 start_line/end_line 构建
                line_range = node_data.get("line_range")
                if not line_range or line_range == (0, 0):
                    start_line = node_data.get("start_line", 0)
                    end_line = node_data.get("end_line", 0)
                    line_range = (start_line, end_line) if start_line > 0 else (0, 0)

                # code_snippet: 优先使用 code_snippet，否则使用 code
                code_snippet = node_data.get("code_snippet") or node_data.get("code") or ""

                nodes.append(ChainNode(
                    node_id=nid,
                    node_type=ntype,
                    file_path=file_path,
                    line_range=line_range,
                    code_snippet=code_snippet
                ))
            
            # 5. 构成链路
            if len(nodes) >= 1: # 降低要求，至少有一个节点也算提取成功
                # 转换边为字典格式
                edges_dicts = []
                for edge in chain_edges:
                    if len(edge) >= 3:
                        edges_dicts.append({
                            "source": edge[0],
                            "target": edge[1],
                            "relation": edge[2]
                        })
                
                chain = DefectChain(
                    chain_id=f"{instance.instance_id}_chain_0",
                    source_instance_id=instance.instance_id,
                    nodes=nodes,
                    edges=edges_dicts,
                    confidence_score=0.8
                )
                chains.append(chain)
                logger.info(f"成功匹配链路节点: {[n.node_id for n in nodes]}")
            else:
                logger.warning(f"未能匹配到任何有效的图节点: {instance.instance_id}")
        
        except Exception as e:
            logger.warning(f"解析 LLM 响应失败: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return chains
