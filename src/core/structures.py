"""
核心数据结构定义 - 使用 Pydantic 进行数据验证
"""
import json
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from enum import Enum

class RepairType(str, Enum):
    """修复类型分类 (基于 IEEE 1044 和 APR 模式)"""
    # Logic
    Condition_Refinement = "Condition_Refinement"
    Guard_Clause_Addition = "Guard_Clause_Addition"
    Exception_Fix = "Exception_Fix"
    # Interface
    Argument_Update = "Argument_Update"
    API_Replacement = "API_Replacement"
    # Data
    Variable_Replacement = "Variable_Replacement"
    Constant_Update = "Constant_Update"
    Type_Cast_Fix = "Type_Cast_Fix"
    Data_Initialization = "Data_Initialization"
    # Structure
    Statement_Insertion = "Statement_Insertion"
    Complex_Logic_Rewrite = "Complex_Logic_Rewrite"

class RepairOperation(BaseModel):
    """修复操作的完整描述（包含AST操作和度量）"""
    type: RepairType
    target_node: str  # 图中的节点 ID
    target_obj: str   # 函数或变量名
    code_transformation: Dict[str, str] # {"before": "...", "after": "..."}
    summary: str
    metrics: Dict[str, Any]  # 包含 operator_category, complexity_level, python_specific

class RepairChain(BaseModel):
    instance_id: str
    chain_type: Literal["repair_chain"] = "repair_chain"
    reasoning_trace: List[Dict[str, Any]] # 分析步骤
    repair_chain: RepairOperation

class ChainNode(BaseModel):
    """
    链路节点 (CodeGraphBuilder 构建)
    """
    node_id: str = Field(..., description="节点处一标识符")
    node_type: Literal["symptom", "intermediate", "root_cause"] = Field(
        ..., description="节点类型"
    )
    file_path: str = Field(..., description="所属文件路径")
    line_range: tuple[int, int] = Field(..., description="代码行范围 (start, end)")
    code_snippet: str = Field(..., description="代码片段")
    semantic_label: Optional[str] = Field(None, description="语义标签")
    
    class Config:
        frozen = True  # 不可变对象


class DefectChain(BaseModel):
    """
    缺陷链路 - 从症状节点到根因节点的路径
    """
    chain_id: str = Field(..., description="链路値一标识符")
    source_instance_id: str = Field(..., description="来源 SWE-Bench 实例 ID")
    nodes: List[ChainNode] = Field(..., description="节点序列 (症状 -> 根因)")
    edges: List[Dict[str, str]] = Field(..., description="边信息列表")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="置信度分数")
    extraction_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="提取过程的元数据"
    )
    
    @property
    def length(self) -> int:
        """链路长度"""
        return len(self.nodes)
    
    @property
    def symptom_node(self) -> ChainNode:
        """症状节点 (起点)"""
        return self.nodes[0]
    
    @property
    def root_cause_node(self) -> ChainNode:
        """根因节点 (终点)"""
        return self.nodes[-1]


class SWEBenchInstance(BaseModel):
    """
    SWE-Bench 数据格式 (严格遵循官方 Schema)
    """
    instance_id: str = Field(..., description="实例唯一标识符")
    repo: str = Field(..., description="仓库名称 (如 'django/django')")
    base_commit: str = Field(..., description="基础提交 SHA")
    problem_statement: str = Field(..., description="问题描述")
    hints_text: str = Field("", description="提示文本")
    created_at: str = Field(..., description="创建时间戳")
    version: str = Field(..., description="版本号")
    FAIL_TO_PASS: str = Field(..., description="从失败到通过的测试用例 (JSON 字符串)")
    PASS_TO_PASS: str = Field(..., description="保持通过的测试用例 (JSON 字符串)")
    environment_setup_commit: str = Field(..., description="环境设置提交 SHA")
    patch: str = Field(..., description="修复补丁 (unified diff 格式)")
    test_patch: str = Field(..., description="测试补丁")
    
    # 扩展字段 (用于 TraceGen)
    synthetic: bool = Field(False, description="是否为合成数据")
    source_chains: List[str] = Field(default_factory=list, description="来源链路 ID 列表")
    generation_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="生成元数据"
    )
    
    # 定位日志字段 (来自 LocAgent)
    raw_output_loc: Optional[List[str]] = Field(None, description="定位日志输出 (LocAgent 生成)")
    found_files: Optional[List[str]] = Field(None, description="定位到的文件列表")
    found_modules: Optional[List[str]] = Field(None, description="定位到的模块列表")
    found_entities: Optional[List[str]] = Field(None, description="定位到的实体列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "instance_id": "django__django-12345",
                "repo": "django/django",
                "base_commit": "abc123def456",
                "problem_statement": "Fix bug in QuerySet.filter()",
                "created_at": "2024-01-01T00:00:00Z",
                "version": "3.2",
                "FAIL_TO_PASS": '["tests.queries.test_filter"]',
                "PASS_TO_PASS": '["tests.queries.test_basic"]',
                "environment_setup_commit": "abc123",
                "patch": "--- a/file.py\n+++ b/file.py\n...",
                "test_patch": "...",
                "hints_text": "",
            }
        }


class ExtractionResult(BaseModel):
    """
    提取阶段的完整结果 - 清晰区分原始信息与挖掘信息
    """
    instance_id: str
    
    # 1. 原始 Seed 数据 (Original data from SWE-bench)
    seed_metadata: Dict[str, Any] = Field(
        ..., 
        description="原始 Seed 信息：包含 repo, base_commit, problem_statement, patch 等"
    )
    
    # 2. 挖掘出的数据 (Mined/Extracted data)
    mined_data: Dict[str, Any] = Field(
        ...,
        description="挖掘信息：包含 extracted_chains (Loc Chain) 和 fix_intents (Repair Chain)"
    )
    
    # 3. 运行元数据
    runtime_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="运行元数据：包含执行时间、图统计信息、时间戳等"
    )

    @property
    def chains(self) -> List[DefectChain]:
        return self.mined_data.get("extracted_chains", [])

    @property
    def fix_intents(self) -> List[RepairOperation]:
        return self.mined_data.get("fix_intents", [])


class SynthesisResult(BaseModel):
    """
    合成阶段的完整结果
    """
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    test_patch: str = Field("", description="测试补丁 (用于添加合成的测试用例)")
    FAIL_TO_PASS: List[str]
    PASS_TO_PASS: List[str] = Field(default_factory=list)

    # 元数据
    seed_id: str
    fix_intent: str
    injection_strategy: str
    seed_metadata: Dict[str, Any] = Field(default_factory=dict)
    seed_extraction_chains: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_swe_bench(self) -> Dict[str, Any]:
        """转换为标准的 SWE-bench 格式"""
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "patch": self.patch,
            "test_patch": self.test_patch,
            "FAIL_TO_PASS": json.dumps(self.FAIL_TO_PASS),
            "PASS_TO_PASS": json.dumps(self.PASS_TO_PASS),
            "version": "synthetic_v1",
            "created_at": datetime.now().isoformat(),
            "metadata": {
                "is_synthetic": True,
                "seed_id": self.seed_id,
                "seed_fix_intent": self.fix_intent,
                "seed_metadata": self.seed_metadata,
                "seed_extraction_chains": self.seed_extraction_chains,
                "injection_strategy": self.injection_strategy,
                "synthesized_chain": self.metadata.get("proposed_chain", []),
                "loc_chain": self.metadata.get("proposed_chain", []),  # 保持一致性
                # Paper-aligned naming: keep both seed and synthesized loc-chains.
                "seed_loc_chain": self.seed_extraction_chains,
                "synth_loc_chain": self.metadata.get("proposed_chain", []),
                "chain_depth": len(self.metadata.get("proposed_chain", [])),
                "chain_length": max(0, len(self.metadata.get("proposed_chain", [])) - 1),
                "injection_patch": self.metadata.get("injection_patch", ""),
                "test_case_code": self.metadata.get("test_case_code", ""),
                "candidate_info": self.metadata.get("candidate_info", {})
            }
        }
