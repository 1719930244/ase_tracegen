import os
import numpy as np
import json
import networkx as nx
from collections import deque
from typing import List, Dict, Any, Optional
from loguru import logger
from pathlib import Path
from ...core.structures import ExtractionResult
from .pattern_matcher import (
    FixIntentPatternMatcher,
    check_intent_compatibility,
    format_injection_points_for_prompt,
)


def is_test_file(node_id: str) -> bool:
    """
    判断节点是否属于测试文件。

    测试文件模式：
    1. 路径包含 tests/, test/, testing/ 目录
    2. 文件名以 test_ 开头或以 _test.py, _tests.py 结尾
    3. 特殊测试文件：conftest.py, tests.py
    """
    # 提取文件路径
    if ":" in node_id:
        file_path = node_id.split(":")[0]
    else:
        file_path = node_id

    file_path_lower = file_path.lower()

    # 模式 1: 目录模式
    test_dir_patterns = ['/tests/', '/test/', '/testing/', 'tests/', 'test/']
    for pattern in test_dir_patterns:
        if pattern in file_path_lower:
            return True

    # 模式 2: 文件名模式
    filename = os.path.basename(file_path_lower)
    if filename.startswith('test_'):
        return True
    if filename.endswith('_test.py') or filename.endswith('_tests.py'):
        return True

    # 模式 3: 特殊测试文件
    if filename in ['tests.py', 'conftest.py', 'test.py']:
        return True

    return False


class SubgraphMatcher:
    """
    基于向量相似度和子图拓扑匹配的候选锚点检索器
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embedding_dir = Path(config.get("embedding_dir", ".cache/embeddings"))
        self.top_k_vector = config.get("top_k_vector", 30) # 向量初步筛选数量
        self.top_k_final = config.get("top_k_final", 5)    # 最终返回的候选数量
        self.auto_generate = config.get("auto_generate", False)  # 自动生成向量
        self.embedding_config = config.get("embedding_config", {})  # 向量生成配置
        # Match score thresholds (from method.synthesis.match_score)
        _ms = config.get("match_score", {})
        self._matcher_min_match_score = float(_ms.get("matcher_min", 0.35))
        # Ablation switches
        _abl = config.get("ablation", {})
        self._disable_graph_matching = bool(_abl.get("disable_graph_matching", False))
        self._disable_candidate_filtering = bool(_abl.get("disable_candidate_filtering", False))
        self._disable_chain_depth_filter = bool(_abl.get("disable_chain_depth_filter", False))
        
    def find_candidates(self, extraction_result: ExtractionResult, graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        为给定的提取结果寻找候选合成位置

        第四阶段改进：添加链路深度匹配约束，只选择链路深度与 Seed 匹配的候选节点。
        """
        repo = extraction_result.seed_metadata.get("repo", "").replace("/", "_")
        commit = extraction_result.seed_metadata.get("base_commit", "")

        # 1. 加载向量库
        embeddings, node_mapping = self._load_embeddings(repo, commit)
        if embeddings is None:
            logger.warning(f"未找到 {repo}@{commit} 的向量库，降级为随机或拓扑匹配")
            return []

        # 2. 获取 Seed 锚点 (Root Cause) 的向量
        seed_chains = extraction_result.mined_data.get("extracted_chains", [])
        if not seed_chains:
            return []

        # 新增 (P0): 获取 Seed 链路深度
        seed_depth = 3  # 默认值
        first_chain_for_depth = seed_chains[0]
        if isinstance(first_chain_for_depth, dict):
            depth_nodes = first_chain_for_depth.get("nodes", [])
        else:
            depth_nodes = first_chain_for_depth.nodes if hasattr(first_chain_for_depth, "nodes") else []
        if depth_nodes:
            seed_depth = len(depth_nodes)
        logger.info(f"Seed 链路深度: {seed_depth} 节点")

        # 2. 确定原始节点集（必须排除这些节点，避免注入到原始 Bug 位置）
        original_node_ids = set()
        for chain in seed_chains:
            for node in (chain.nodes if not isinstance(chain, dict) else chain.get("nodes", [])):
                nid = node.node_id if not isinstance(node, dict) else node.get("node_id")
                if nid:
                    original_node_ids.add(nid)
        
        # 我们以第一个链路的 root_cause 作为原始锚点
        first_chain = seed_chains[0]
        
        # 兼容字典和对象访问
        if isinstance(first_chain, dict):
            nodes = first_chain.get("nodes", [])
            if not nodes: return []
            seed_root_cause = nodes[-1]
            seed_node_id = seed_root_cause.get("node_id") if isinstance(seed_root_cause, dict) else seed_root_cause.node_id
            seed_subgraph = first_chain.get("extraction_metadata", {}).get("subgraph", {})
        else:
            seed_root_cause = first_chain.root_cause_node
            seed_node_id = seed_root_cause.node_id
            seed_subgraph = first_chain.extraction_metadata.get("subgraph", {})
        
        if not seed_node_id:
            return []
            
        # 构建节点→索引映射 (O(1) 查找，替代 list.index O(n))
        node_idx_map = {nid: i for i, nid in enumerate(node_mapping)}

        # 优化: 一次性预算全图链路深度 O(V+E)，替代逐候选 BFS
        depth_map = self._precompute_chain_depths(graph)

        # [Ablation] disable_graph_matching: 随机选候选替代向量匹配
        if self._disable_graph_matching:
            import random
            logger.info("[Ablation] disable_graph_matching: 使用随机选候选替代向量匹配")
            all_indices = list(range(len(node_mapping)))
            random.shuffle(all_indices)
            top_indices = []
            for idx in all_indices:
                candidate_id = node_mapping[idx]
                if candidate_id in original_node_ids:
                    continue
                if is_test_file(candidate_id):
                    continue
                top_indices.append(idx)
                if len(top_indices) >= self.top_k_vector:
                    break
            similarities = None
        else:
            # 获取 Seed 向量
            seed_idx = self._find_node_idx(seed_node_id, node_mapping, node_idx_map)

            similarities = None
            if seed_idx != -1:
                seed_vec = embeddings[seed_idx]
                # 3. 向量相似度检索 (计算原始相似度)
                similarities = np.dot(embeddings, seed_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(seed_vec) + 1e-9)

                # 按照原始相似度排序
                sorted_indices = np.argsort(similarities)[::-1]

                # 排除原始节点和测试文件，用预算深度表做链路深度匹配
                top_indices = []
                depth_mismatch_count = 0
                for idx in sorted_indices:
                    candidate_id = node_mapping[idx]
                    if candidate_id in original_node_ids:
                        continue
                    if is_test_file(candidate_id):
                        continue
                    # 优化: 使用预算深度 O(1) 查表替代 BFS
                    if not self._disable_chain_depth_filter:
                        candidate_depth = self._get_chain_depth(candidate_id, depth_map)
                        if abs(candidate_depth - seed_depth) > 1:  # 允许 ±1 的容差
                            depth_mismatch_count += 1
                            if depth_mismatch_count <= 5:
                                logger.debug(f"跳过候选 {candidate_id}: 链路深度 {candidate_depth} 与 Seed {seed_depth} 不匹配")
                            continue
                    top_indices.append(idx)
                    if len(top_indices) >= self.top_k_vector:
                        break
                if depth_mismatch_count > 0:
                    logger.info(f"链路深度匹配：跳过 {depth_mismatch_count} 个深度不匹配的候选节点")
            else:
                logger.warning(f"Seed 节点 {seed_node_id} 无法映射到向量库，将降级为纯结构/随机采样")
                all_node_ids = list(graph.nodes())
                non_original_nodes = [
                    n for n in all_node_ids
                    if n not in original_node_ids and not is_test_file(n)
                ]

                import random
                selected_ids = random.sample(non_original_nodes, min(len(non_original_nodes), self.top_k_vector))

                top_indices = []
                selected_ids_set = set(selected_ids)
                for i, nid in enumerate(node_mapping):
                    if nid in selected_ids_set:
                        top_indices.append(i)

        # 优化: 先做 intent 过滤（轻量），再只对通过的候选算拓扑（重量）
        candidates = []
        for idx in top_indices:
            candidate_node_id = node_mapping[idx]
            score = float(similarities[idx]) if similarities is not None else 0.0

            # 5. 意图兼容性过滤 (先于拓扑，提前淘汰不兼容候选)
            if self._disable_candidate_filtering:
                compatible_intents = extraction_result.fix_intents or []
            else:
                compatible_intents = self._filter_compatible_intents(candidate_node_id, extraction_result.fix_intents, graph)
                if not compatible_intents:
                    continue  # 不兼容，跳过拓扑计算

            # 4. 拓扑相似度校核 (延迟到 intent 过滤之后，只对兼容候选计算)
            candidate_subgraph = self._get_node_subgraph(candidate_node_id, graph)
            topology_score = self._calculate_subgraph_similarity(seed_subgraph, candidate_subgraph)

            candidates.append({
                "anchor_node_id": candidate_node_id,
                "vector_score": score,
                "topology_score": topology_score,
                "final_score": (0.4 * score + 0.6 * topology_score),
                "subgraph": candidate_subgraph,
                "compatible_intents": compatible_intents
            })

        # 按最终分数排序
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        logger.info(f"候选匹配: {len(top_indices)} 个向量候选 → {len(candidates)} 个意图兼容, 返回 Top {min(self.top_k_final, len(candidates))}")
        return candidates[:self.top_k_final]

    def _load_embeddings(self, repo: str, commit: str):
        """
        加载向量库，支持两种格式：
        1. V2 仓库级聚合格式: embeddings/<repo>/vector_pool.npy + commits/<commit>.json
        2. V1 旧格式: embeddings/<repo>__<commit>_embeddings.npy + _mapping.json
        """
        # === 优先尝试 V2 仓库级格式 ===
        repo_dir = self.embedding_dir / repo
        if repo_dir.is_dir():
            pool_path = repo_dir / "vector_pool.npy"
            hash_to_idx_path = repo_dir / "hash_to_idx.json"

            # 尝试精确 commit 匹配
            commit_file = repo_dir / "commits" / f"{commit[:8]}.json"
            if not commit_file.exists():
                commit_file = repo_dir / "commits" / f"{commit}.json"

            if pool_path.exists() and hash_to_idx_path.exists():
                if commit_file.exists():
                    logger.debug(f"使用 V2 仓库级向量库: {repo} @ {commit[:8]}")
                    return self._read_v2_files(pool_path, hash_to_idx_path, commit_file)
                else:
                    # V2 格式存在但没有该 commit，尝试使用最新的 commit 映射
                    commits_dir = repo_dir / "commits"
                    if commits_dir.exists():
                        commit_files = list(commits_dir.glob("*.json"))
                        if commit_files:
                            commit_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                            logger.info(f"V2 格式: 未找到 {commit[:8]}，复用最近的 commit 映射: {commit_files[0].stem}")
                            return self._read_v2_files(pool_path, hash_to_idx_path, commit_files[0])

        # === 回退到 V1 旧格式 ===
        # 1. 优先尝试精确匹配 (Exact Match)
        potential_stems = [
            f"{repo}__{commit[:8]}",
            f"{repo}__{commit}",
            f"{repo}_{commit[:8]}",
            f"{repo}_{commit}"
        ]
        for stem in potential_stems:
            npy_path = self.embedding_dir / f"{stem}_embeddings.npy"
            json_path = self.embedding_dir / f"{stem}_mapping.json"
            if npy_path.exists() and json_path.exists():
                logger.info(f"使用 V1 格式向量库: {stem}")
                return self._read_v1_files(npy_path, json_path, stem)

        # 2. 模糊匹配：如果找不到当前 Commit，尝试寻找同仓库的其他 Commit 向量
        logger.info(f"未找到精确匹配的向量库 ({repo}@{commit})，尝试寻找同仓库的其他版本...")
        existing_files = list(self.embedding_dir.glob(f"{repo}*_embeddings.npy"))
        if existing_files:
            # 取最近修改的一个
            existing_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            chosen_npy = existing_files[0]
            stem = chosen_npy.name.replace("_embeddings.npy", "")
            chosen_json = self.embedding_dir / f"{stem}_mapping.json"

            if chosen_json.exists():
                logger.info(f"复用同仓库 V1 向量库: {stem}")
                return self._read_v1_files(chosen_npy, chosen_json, stem)

        return None, None

    def _auto_generate_embeddings(self, repo: str, graph_path: Path) -> bool:
        """当向量库不存在时自动生成"""
        import os
        import sys

        # 添加 tools 目录到路径
        tools_dir = Path(__file__).parent.parent.parent.parent / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))

        try:
            from embedding_generator_dashscope import update_repo_embeddings_api
        except ImportError:
            logger.warning("无法导入 embedding_generator_dashscope，跳过自动生成")
            return False

        api_key = self.embedding_config.get("api_key") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            logger.warning("未配置 embedding.api_key 或 DASHSCOPE_API_KEY，无法自动生成向量")
            return False

        try:
            logger.info(f"自动生成向量库: {repo} (图文件: {graph_path.name})")
            update_repo_embeddings_api(
                repo_name=repo,
                graph_paths=[graph_path],
                output_dir=self.embedding_dir,
                api_key=api_key,
                dimension=self.embedding_config.get("dimension", 1024),
                batch_size=min(self.embedding_config.get("batch_size", 10), 10)
            )
            logger.info(f"向量库生成完成: {repo}")
            return True
        except Exception as e:
            logger.error(f"自动生成向量失败: {e}")
            return False

    def find_candidates_with_auto_generate(
        self, extraction_result: ExtractionResult, graph: nx.DiGraph, graph_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        带自动向量生成的候选查找
        """
        repo = extraction_result.seed_metadata.get("repo", "").replace("/", "_")
        commit = extraction_result.seed_metadata.get("base_commit", "")

        # 先尝试加载向量
        embeddings, node_mapping = self._load_embeddings(repo, commit)

        # 如果没有向量且配置了自动生成
        if embeddings is None and self.auto_generate and graph_path:
            logger.info(f"未找到向量库，尝试自动生成: {repo}@{commit[:8]}")
            if self._auto_generate_embeddings(repo, graph_path):
                # 重新加载
                embeddings, node_mapping = self._load_embeddings(repo, commit)

        if embeddings is None:
            logger.warning(f"无法加载或生成向量库 {repo}@{commit}，将使用拓扑匹配")

        # 调用原始的 find_candidates 逻辑
        return self.find_candidates(extraction_result, graph)

    def _read_v2_files(self, pool_path: Path, hash_to_idx_path: Path, commit_file: Path):
        """读取 V2 仓库级向量库格式"""
        try:
            # 加载向量池
            vector_pool = np.load(pool_path)

            # 加载哈希到索引的映射
            with open(hash_to_idx_path, 'r') as f:
                hash_to_idx = json.load(f)

            # 加载 commit 级别的节点到哈希映射
            with open(commit_file, 'r') as f:
                node_to_hash = json.load(f)

            # 构建 node_id 列表和对应的向量矩阵
            node_ids = []
            embeddings_list = []

            for node_id, content_hash in node_to_hash.items():
                if content_hash in hash_to_idx:
                    idx = hash_to_idx[content_hash]
                    node_ids.append(node_id)
                    embeddings_list.append(vector_pool[idx])

            if embeddings_list:
                embeddings = np.vstack(embeddings_list)
                logger.debug(f"V2 向量库加载成功: {len(node_ids)} 节点, 向量维度 {embeddings.shape}")
                return embeddings, node_ids
            else:
                logger.warning("V2 向量库: 无法匹配任何节点到向量")
                return None, None

        except Exception as e:
            logger.error(f"加载 V2 向量库出错: {e}")
            return None, None

    def _read_v1_files(self, npy_path: Path, json_path: Path, stem: str):
        """读取 V1 旧格式向量库"""
        try:
            embeddings = np.load(npy_path)
            with open(json_path, 'r') as f:
                node_mapping = json.load(f)
            return embeddings, node_mapping
        except Exception as e:
            logger.error(f"加载 V1 向量库出错 {stem}: {e}")
            return None, None

    def _find_node_idx(self, node_id: str, node_mapping: List[str], node_idx_map: Optional[Dict[str, int]] = None) -> int:
        """寻找节点索引，增加对文件级锚点的兼容性。使用 dict 加速精确查找。"""
        if node_idx_map is None:
            node_idx_map = {nid: i for i, nid in enumerate(node_mapping)}

        idx = node_idx_map.get(node_id)
        if idx is not None:
            return idx

        # 如果 node_id 是文件（例如 django/contrib/auth/validators.py）
        # 尝试在映射中寻找该文件下的第一个类或函数
        prefix = f"{node_id}:"
        for i, mapped_id in enumerate(node_mapping):
            if mapped_id.startswith(prefix):
                logger.debug(f"Seed 节点 {node_id} 是文件，映射到其子节点: {mapped_id}")
                return i
        return -1

    def _precompute_chain_depths(self, graph: nx.DiGraph) -> Dict[str, int]:
        """
        一次性预算所有节点到最近测试节点的链路深度。

        从所有测试节点出发做多源 BFS（沿 successors 方向），
        一次 O(V+E) 遍历得到全图深度，替代对每个候选节点独立 BFS。

        Returns:
            {node_id: depth} 字典，depth = 到最近测试节点的距离 + 1
        """
        depths = {}
        queue = deque()

        # 收集所有测试节点作为 BFS 起点 (depth=0)
        for node in graph.nodes():
            if is_test_file(node):
                depths[node] = 0
                queue.append((node, 0))

        # 多源 BFS：从测试节点沿 successors（被调用方向）扩散
        while queue:
            current, depth = queue.popleft()
            if depth >= 10:
                continue
            for succ in graph.successors(current):
                if succ not in depths:
                    depths[succ] = depth + 1
                    queue.append((succ, depth + 1))

        # 转换为链路深度（+1 包含自身），未覆盖节点返回默认值 3
        return {nid: d + 1 for nid, d in depths.items()}

    def _get_chain_depth(self, node_id: str, depth_map: Dict[str, int]) -> int:
        """从预算表查询链路深度，未覆盖节点返回默认值 3"""
        return depth_map.get(node_id, 3)

    def _calculate_chain_depth(self, node_id: str, graph: nx.DiGraph) -> int:
        """
        计算从给定节点到最近测试节点的调用链路深度（单节点版，保留兼容性）。

        使用 BFS 向后遍历 (predecessors)，找到调用该函数的测试节点。
        链路深度 = 从测试节点到目标节点的距离 + 1（包含自身）
        """
        if node_id not in graph:
            return 3

        visited = {node_id}
        queue = deque([(node_id, 0)])
        test_depth = -1

        while queue:
            current, depth = queue.popleft()
            if is_test_file(current):
                test_depth = depth
                break
            if depth >= 10:
                continue
            for pred in graph.predecessors(current):
                if pred not in visited:
                    visited.add(pred)
                    queue.append((pred, depth + 1))

        return (test_depth + 1) if test_depth != -1 else 3

    def _get_node_subgraph(self, node_id: str, graph: nx.DiGraph, hops: int = 1) -> Dict[str, Any]:
        """提取候选节点的子图 (逻辑同 extractor.py)"""
        if node_id not in graph:
            return {"nodes": {}, "edges": [], "context_range": []}
            
        subgraph_nodes = {node_id}
        for _ in range(hops):
            new_nodes = set()
            for n in subgraph_nodes:
                if n in graph:
                    new_nodes.update(graph.predecessors(n))
                    new_nodes.update(graph.successors(n))
            subgraph_nodes.update(new_nodes)
            
        sub_g = graph.subgraph(subgraph_nodes)
        nodes_info = {n: {"type": graph.nodes[n].get("type", "unknown"), "file_path": graph.nodes[n].get("file_path", "")} for n in sub_g.nodes()}
        edges_info = [{"source": u, "target": v, "type": data.get("type", "unknown")} for u, v, data in sub_g.edges(data=True)]
        
        return {
            "nodes": nodes_info,
            "edges": edges_info,
            "context_range": list(set(d["file_path"] for d in nodes_info.values() if d["file_path"]))
        }

    def _calculate_subgraph_similarity(self, seed_sub: Dict[str, Any], cand_sub: Dict[str, Any]) -> float:
        """
        计算两个子图的相似度 (基于节点类型分布和边密度)
        """
        if not seed_sub or not cand_sub:
            return 0.0
            
        def get_dist(sub):
            dist = {}
            for n in sub.get("nodes", {}).values():
                t = n.get("type", "unknown")
                dist[t] = dist.get(t, 0) + 1
            return dist
            
        dist_s = get_dist(seed_sub)
        dist_c = get_dist(cand_sub)
        
        # 计算类型分布的余弦相似度
        all_types = set(list(dist_s.keys()) + list(dist_c.keys()))
        vec_s = np.array([dist_s.get(t, 0) for t in all_types])
        vec_c = np.array([dist_c.get(t, 0) for t in all_types])
        
        norm_s = np.linalg.norm(vec_s)
        norm_c = np.linalg.norm(vec_c)
        if norm_s == 0 or norm_c == 0:
            return 0.0
            
        type_sim = np.dot(vec_s, vec_c) / (norm_s * norm_c)
        
        # 边密度对比
        edge_ratio_s = len(seed_sub.get("edges", [])) / (len(seed_sub.get("nodes", {})) + 1e-9)
        edge_ratio_c = len(cand_sub.get("edges", [])) / (len(cand_sub.get("nodes", {})) + 1e-9)
        edge_sim = 1.0 - min(abs(edge_ratio_s - edge_ratio_c), 1.0)
        
        return 0.7 * type_sim + 0.3 * edge_sim

    def _filter_compatible_intents(self, node_id: str, intents: List[Any], graph: nx.DiGraph) -> List[Any]:
        """
        根据节点特征过滤兼容的修复意图 - 三级 Fallback 版本
        使用 FixIntentPatternMatcher 进行语义级别的匹配

        Args:
            node_id: 候选节点 ID
            intents: Fix Intent 列表
            graph: 代码依赖图

        Returns:
            兼容的 intent 列表，每个 intent 会附加 _injection_points 和 _best_match_score
        """
        compatible = []
        node_data = graph.nodes.get(node_id, {})
        code = node_data.get("code_snippet", "") or node_data.get("code", "")

        if not code:
            logger.debug(f"节点 {node_id} 无代码内容，跳过")
            return []

        # 三级 Fallback 最低匹配分 (从配置读取)
        min_match_score = self._matcher_min_match_score

        for intent in intents:
            # 兼容字典和对象访问
            if isinstance(intent, dict):
                intent_dict = intent.copy()
            else:
                intent_dict = {
                    "type": getattr(intent, "type", "Unknown"),
                    "code_transformation": getattr(intent, "code_transformation", {}),
                    "summary": getattr(intent, "summary", ""),
                }

            intent_type = intent_dict.get("type", "Unknown")

            # 使用新的模式匹配器检查兼容性
            is_compatible, injection_points, reason = check_intent_compatibility(
                code, intent_dict, min_match_score
            )

            if is_compatible and injection_points:
                # 计算最高匹配分数
                best_score = max(p.match_score for p in injection_points)

                # 在 intent 中注入匹配信息
                if isinstance(intent, dict):
                    intent["_injection_points"] = injection_points
                    intent["_best_match_score"] = best_score
                    intent["_compatibility_reason"] = reason
                    compatible.append(intent)
                else:
                    # 创建一个增强的字典版本
                    enhanced_intent = intent_dict.copy()
                    enhanced_intent["_injection_points"] = injection_points
                    enhanced_intent["_best_match_score"] = best_score
                    enhanced_intent["_compatibility_reason"] = reason
                    compatible.append(enhanced_intent)

                logger.debug(
                    f"Intent '{intent_type}' 与节点 {node_id} 兼容: "
                    f"找到 {len(injection_points)} 个注入点, 最高匹配度 {best_score:.0%}"
                )
            else:
                logger.debug(f"Intent '{intent_type}' 与节点 {node_id} 不兼容: {reason}")

        return compatible
