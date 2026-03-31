"""
链路对齐评分模块

P3: 计算合成链路与 Seed 链路的结构相似度，用于评估合成 Bug 的质量。

评分维度：
1. 深度匹配 (depth_match): 链路节点数量是否一致
2. 结构相似度 (structure_similarity): 节点类型分布是否匹配
3. 语义对齐 (semantic_alignment): 节点在代码库中的位置相似性
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ChainAlignmentScore:
    """链路对齐评分结果"""
    depth_match: float           # 深度匹配分数 (0-1)
    structure_similarity: float  # 结构相似度 (0-1)
    semantic_alignment: float    # 语义对齐分数 (0-1)
    overall_score: float         # 综合分数 (加权平均)
    details: Dict[str, Any]      # 详细评分信息

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "depth_match": self.depth_match,
            "structure_similarity": self.structure_similarity,
            "semantic_alignment": self.semantic_alignment,
            "overall_score": self.overall_score,
            "details": self.details,
        }

    def summary(self) -> str:
        """生成可读的评分摘要"""
        return (
            f"对齐评分: {self.overall_score:.0%} "
            f"(深度: {self.depth_match:.0%}, "
            f"结构: {self.structure_similarity:.0%}, "
            f"语义: {self.semantic_alignment:.0%})"
        )


def calculate_chain_alignment(
    seed_chain: List[Dict[str, Any]],
    synthetic_chain: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> ChainAlignmentScore:
    """
    计算合成链路与 Seed 链路的对齐分数。

    Args:
        seed_chain: Seed 链路节点列表，每个节点应包含:
            - node_id: 节点标识符
            - node_type: 节点类型 (symptom/intermediate/root_cause)
            - file_path: 可选的文件路径
        synthetic_chain: 合成链路节点列表（格式同上）
        weights: 各维度权重，默认为 {depth: 0.4, structure: 0.4, semantic: 0.2}

    Returns:
        ChainAlignmentScore 对齐分数对象
    """
    if weights is None:
        weights = {"depth": 0.4, "structure": 0.4, "semantic": 0.2}

    details = {}

    # 1. 深度匹配分数
    seed_depth = len(seed_chain) if seed_chain else 0
    synthetic_depth = len(synthetic_chain) if synthetic_chain else 0

    if max(seed_depth, synthetic_depth) > 0:
        depth_diff = abs(seed_depth - synthetic_depth)
        depth_match = 1.0 - depth_diff / max(seed_depth, synthetic_depth)
    else:
        depth_match = 0.0

    details["seed_depth"] = seed_depth
    details["synthetic_depth"] = synthetic_depth
    details["depth_difference"] = abs(seed_depth - synthetic_depth)

    # 2. 结构相似度 (节点类型匹配)
    type_matches = 0
    type_details = []
    min_length = min(len(seed_chain), len(synthetic_chain))

    for i in range(min_length):
        seed_type = _get_node_type(seed_chain[i])
        synth_type = _get_node_type(synthetic_chain[i])

        match_status = "match" if seed_type == synth_type else "mismatch"
        type_details.append({
            "index": i,
            "seed_type": seed_type,
            "synthetic_type": synth_type,
            "status": match_status,
        })

        if seed_type == synth_type:
            type_matches += 1

    max_length = max(len(seed_chain), len(synthetic_chain))
    structure_similarity = type_matches / max_length if max_length > 0 else 0.0

    details["type_matches"] = type_matches
    details["type_comparison"] = type_details

    # 3. 语义对齐 (检查 root_cause 是否在相似类型的代码模块中)
    semantic_alignment = _calculate_semantic_alignment(seed_chain, synthetic_chain)
    details["semantic_details"] = semantic_alignment.get("details", {})

    # 计算综合分数
    overall_score = (
        weights["depth"] * depth_match +
        weights["structure"] * structure_similarity +
        weights["semantic"] * semantic_alignment["score"]
    )

    return ChainAlignmentScore(
        depth_match=depth_match,
        structure_similarity=structure_similarity,
        semantic_alignment=semantic_alignment["score"],
        overall_score=overall_score,
        details=details,
    )


def _get_node_type(node: Dict[str, Any]) -> str:
    """从节点中提取类型信息"""
    if isinstance(node, dict):
        return node.get("node_type", node.get("type", "unknown"))
    return "unknown"


def _get_node_id(node: Dict[str, Any]) -> str:
    """从节点中提取 ID"""
    if isinstance(node, dict):
        return node.get("node_id", node.get("id", ""))
    return str(node)


def _get_file_path(node: Dict[str, Any]) -> str:
    """从节点中提取文件路径"""
    if isinstance(node, dict):
        # 优先使用显式的 file_path
        if node.get("file_path"):
            return node["file_path"]
        # 从 node_id 中提取
        node_id = _get_node_id(node)
        if ":" in node_id:
            return node_id.split(":")[0]
        if node_id.endswith(".py"):
            return node_id
    return ""


def _calculate_semantic_alignment(
    seed_chain: List[Dict[str, Any]],
    synthetic_chain: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    计算语义对齐分数。

    检查维度：
    1. root_cause 节点是否在相同类型的模块中
    2. 文件路径的目录结构相似性
    3. 函数/类名称的模式相似性
    """
    score = 0.5  # 默认中等分数
    details = {}

    if not seed_chain or not synthetic_chain:
        return {"score": 0.0, "details": {"error": "Empty chain"}}

    # 获取 root_cause 节点 (最后一个节点)
    seed_root = seed_chain[-1]
    synth_root = synthetic_chain[-1]

    seed_file = _get_file_path(seed_root)
    synth_file = _get_file_path(synth_root)

    details["seed_root_file"] = seed_file
    details["synthetic_root_file"] = synth_file

    # 1. 检查是否在相同目录结构中
    if seed_file and synth_file:
        seed_parts = seed_file.split("/")
        synth_parts = synth_file.split("/")

        # 计算共同的目录层级数
        common_depth = 0
        for i in range(min(len(seed_parts) - 1, len(synth_parts) - 1)):
            if seed_parts[i] == synth_parts[i]:
                common_depth += 1
            else:
                break

        max_depth = max(len(seed_parts), len(synth_parts)) - 1
        if max_depth > 0:
            directory_similarity = common_depth / max_depth
        else:
            directory_similarity = 0.0

        details["common_directory_depth"] = common_depth
        details["directory_similarity"] = directory_similarity

        # 根据目录相似度调整分数
        if common_depth >= 2:
            score = 0.9  # 在同一子模块中
        elif common_depth >= 1:
            score = 0.7  # 在同一顶级模块中
        else:
            score = 0.4  # 不同模块

    # 2. 检查文件名模式
    if seed_file and synth_file:
        seed_filename = seed_file.split("/")[-1] if "/" in seed_file else seed_file
        synth_filename = synth_file.split("/")[-1] if "/" in synth_file else synth_file

        # 检查文件名是否相同或相似
        if seed_filename == synth_filename:
            score = min(score + 0.1, 1.0)
            details["same_filename"] = True
        else:
            details["same_filename"] = False

    # 3. 检查节点名称模式相似性
    seed_node_id = _get_node_id(seed_root)
    synth_node_id = _get_node_id(synth_root)

    if seed_node_id and synth_node_id:
        # 提取函数/类名（: 后面的部分）
        seed_name = seed_node_id.split(":")[-1] if ":" in seed_node_id else ""
        synth_name = synth_node_id.split(":")[-1] if ":" in synth_node_id else ""

        # 检查名称模式（如 validate_*, get_*, etc.）
        seed_prefix = seed_name.split("_")[0] if "_" in seed_name else seed_name[:4]
        synth_prefix = synth_name.split("_")[0] if "_" in synth_name else synth_name[:4]

        if seed_prefix == synth_prefix:
            score = min(score + 0.05, 1.0)
            details["similar_naming_pattern"] = True
        else:
            details["similar_naming_pattern"] = False

    return {"score": score, "details": details}


def align_chains_for_comparison(
    seed_chain: List[Dict[str, Any]],
    synthetic_chain: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    将两个链路对齐以便比较。

    返回对齐后的节点对列表，用于可视化或详细分析。
    """
    aligned = []
    max_len = max(len(seed_chain), len(synthetic_chain))

    for i in range(max_len):
        entry = {"index": i}

        if i < len(seed_chain):
            entry["seed"] = {
                "node_id": _get_node_id(seed_chain[i]),
                "node_type": _get_node_type(seed_chain[i]),
                "file_path": _get_file_path(seed_chain[i]),
            }
        else:
            entry["seed"] = None

        if i < len(synthetic_chain):
            entry["synthetic"] = {
                "node_id": _get_node_id(synthetic_chain[i]),
                "node_type": _get_node_type(synthetic_chain[i]),
                "file_path": _get_file_path(synthetic_chain[i]),
            }
        else:
            entry["synthetic"] = None

        # 判断匹配状态
        if entry["seed"] and entry["synthetic"]:
            if entry["seed"]["node_type"] == entry["synthetic"]["node_type"]:
                entry["match_status"] = "type_match"
            else:
                entry["match_status"] = "type_mismatch"
        else:
            entry["match_status"] = "missing"

        aligned.append(entry)

    return aligned


# 导出
__all__ = [
    "ChainAlignmentScore",
    "calculate_chain_alignment",
    "align_chains_for_comparison",
]
