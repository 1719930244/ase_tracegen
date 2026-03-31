"""
TraceGen 主流水线 - 编排提取和合成两个阶段
"""
import os
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
import json
import networkx as nx
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.structures import (
    SWEBenchInstance,
    DefectChain,
    ExtractionResult,
    SynthesisResult,
)
from ..core.utils import load_json, save_json, clone_repository, ensure_dir, save_pickle, load_pickle
from ..graph.builder import CodeGraphBuilder
from ..modules.llm_client import create_llm_client
from ..modules.extraction.extractor import ChainExtractor
from ..modules.synthesis.agent import SynthesisAgent
from ..modules.synthesis.matcher import SubgraphMatcher
from ..modules.validation.adapter import ValidationAdapter
from ..modules.localization.localizer import (
    FaultLocalizer,
    LocalizationResult,
    save_localization_cache,
    load_localization_cache,
)
from ..core.repo_profiles import get_repo_profile, detect_repo_from_instance_id
from ..core.repo_profile import RepoProfile

# Import validator components from local module
try:
    from ..modules.validation.validator import Validator
    from ..modules.validation.constants import ValidationConfig, ValidationStatus
    from ..modules.validation.profiles.python import PythonProfile, UnittestProfile
    _VALIDATOR_AVAILABLE = True
except ImportError:
    Validator = None
    ValidationConfig = None
    ValidationStatus = None
    PythonProfile = None
    UnittestProfile = None
    _VALIDATOR_AVAILABLE = False

GRAPH_CACHE_VERSION = "v1"

# 预筛选配置：排除的 Fix Intent 类型
EXCLUDED_INTENT_TYPES = {"Complex_Logic_Rewrite"}
MIN_CHAIN_LENGTH = 2


def should_synthesize(extraction_result: ExtractionResult) -> Tuple[bool, str]:
    """
    判断提取结果是否值得进行合成

    筛选规则:
    1. 链路长度 >= MIN_CHAIN_LENGTH
    2. 排除 Complex_Logic_Rewrite 类型
    3. 必须有有效的 code_transformation

    Args:
        extraction_result: 提取结果

    Returns:
        (should_synthesize, reason)
    """
    mined_data = extraction_result.mined_data
    chains = mined_data.get("extracted_chains", [])
    fix_intents = mined_data.get("fix_intents", [])

    # 检查基本数据
    if not chains or not fix_intents:
        return False, "缺少链路或 Fix Intent"

    # 规则 1: 链路长度检查
    first_chain = chains[0]
    nodes = first_chain.get("nodes", []) if isinstance(first_chain, dict) else getattr(first_chain, "nodes", [])
    chain_len = len(nodes)
    if chain_len < MIN_CHAIN_LENGTH:
        return False, f"链路太短 ({chain_len} < {MIN_CHAIN_LENGTH})"

    # 规则 2: 排除特定 Intent 类型
    first_intent = fix_intents[0]
    intent_type = first_intent.get("type", "") if isinstance(first_intent, dict) else getattr(first_intent, "type", "")
    if intent_type in EXCLUDED_INTENT_TYPES:
        return False, f"{intent_type} 难以泛化，已排除"

    # 规则 3: 必须有有效的 code_transformation
    ct = first_intent.get("code_transformation", {}) if isinstance(first_intent, dict) else {}
    if not ct.get("before") or not ct.get("after"):
        return False, "缺少有效的 code_transformation"

    return True, "通过筛选"


class TraceGenPipeline:
    """
    TraceGen 完整流水线
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化流水线
        
        Args:
            config: Hydra 配置对象
        """
        self.config = config
        
        # 先创建输出目录
        self.output_dir = Path(config.data.output_dir)
        ensure_dir(self.output_dir)
        
        # 然后初始化组件（需要输出目录）
        self._init_components()
        
        logger.info(f"TraceGen 流水线初始化完成")
        logger.info(f"输出目录: {self.output_dir}")
    
    def _init_components(self):
        """
        初始化所有组件
        """
        # 创建日志目录
        self.logs_dir = self.output_dir / "logs"
        ensure_dir(self.logs_dir)

        # 1. 初始化合成 LLM 客户端
        logger.info("初始化合成 LLM 客户端...")
        self.synthesis_llm_client = create_llm_client(self.config.synthesis_llm, output_dir=self.logs_dir)
        
        # 1.5. 初始化分析 LLM 客户端 (用于链路提取)
        logger.info("初始化分析 LLM 客户端...")
        self.analyzer_llm_client = create_llm_client(self.config.analyzer_llm, output_dir=self.logs_dir)

        # 1.6. 初始化 Stage 0 定位器 (为缺少 LocAgent 数据的实例生成 raw_output_loc)
        loc_config = dict(getattr(self.config, "localization", {}) or {})
        self.fault_localizer = FaultLocalizer(
            llm_client=self.analyzer_llm_client,
            config=loc_config,
        )
        self.localization_cache_dir = Path(self.config.data.cache_dir) / "localizations"
        
        # 2. 初始化图构建器
        logger.info("初始化图构建器...")
        self.graph_builder = CodeGraphBuilder(self.config.graph)
        
        # 3. 初始化提取器
        logger.info("初始化链路提取器...")
        self.extractor = ChainExtractor(
            self.synthesis_llm_client, self.config.method.extraction, analyzer_llm=self.analyzer_llm_client
        )
        
        # 3.5. 检测仓库类型，初始化 RepoProfile
        self.repo_profile = self._detect_repo_profile()
        logger.info(f"仓库 Profile: {type(self.repo_profile).__name__} (framework={self.repo_profile.test_framework})")

        # 4. 初始化合成 Agent
        logger.info("初始化合成 Agent...")
        agent_config = dict(self.config.method.synthesis.get("agent", {}))
        # 注入 match_score 配置到 agent_config
        match_score_config = dict(self.config.method.synthesis.get("match_score", {}))
        agent_config["match_score"] = match_score_config

        # 注入 ablation 开关到 agent_config 和 matcher_config
        ablation_config = dict(self.config.method.get("ablation", {}) or {})
        agent_config["ablation"] = ablation_config
        if any(ablation_config.values()):
            logger.info(f"Ablation 开关: {', '.join(k for k, v in ablation_config.items() if v)}")

        self.synthesis_agent = SynthesisAgent(
            llm_client=self.synthesis_llm_client,
            config=agent_config,
            output_dir=self.output_dir,
            repo_profile=self.repo_profile,
        )
        # 构建 matcher 配置，包含向量自动生成选项
        matcher_config = dict(self.config.method.synthesis.get("matcher", {}))
        # 注入 match_score 配置到 matcher_config
        matcher_config["match_score"] = match_score_config
        # 注入 ablation 开关到 matcher_config
        matcher_config["ablation"] = ablation_config
        # Backward compatibility: allow `method.num_candidates` to control matcher top_k_final.
        try:
            num_cand = int(getattr(self.config.method, "num_candidates", 0) or 0)
            if num_cand > 0 and not matcher_config.get("top_k_final"):
                matcher_config["top_k_final"] = num_cand
        except (TypeError, ValueError, AttributeError):
            pass
        if hasattr(self.config, "embedding"):
            matcher_config["auto_generate"] = self.config.embedding.get("auto_generate", False)
            matcher_config["embedding_config"] = dict(self.config.embedding)
        self.subgraph_matcher = SubgraphMatcher(config=matcher_config)

        # 5. 初始化验证器
        if self.config.validation.get("enabled", False) and _VALIDATOR_AVAILABLE:
            logger.info("初始化验证器...")
            try:
                self.validation_adapter = ValidationAdapter(dict(self.config.validation), repo_profile=self.repo_profile)
                # Create ValidationConfig from config
                val_config = ValidationConfig(
                    mode="injection",
                    timeout=self.config.validation.get("timeout", 300),
                    memory_limit=self.config.validation.get("memory_limit", "4g"),
                    clean_containers=self.config.validation.get("clean_containers", True),
                    verbose=self.config.validation.get("verbose", False),
                    enforce_chain_coverage=self.config.validation.get("enforce_chain_coverage", ValidationConfig.enforce_chain_coverage),
                    min_chain_coverage=float(self.config.validation.get("min_chain_coverage", ValidationConfig.min_chain_coverage) or 0.30),
                    require_target_node_in_traceback=self.config.validation.get("require_target_node_in_traceback", ValidationConfig.require_target_node_in_traceback),
                    log_dir=self.output_dir / "3_validation" / "logs"
                )
                # 使用已检测的 repo_profile 获取验证 Profile
                default_image = self.config.validation.get("default_image", "python:3.10-slim")
                self.validation_profile = self.repo_profile.get_validation_profile(default_image)

                self.validator = Validator(profile=self.validation_profile, config=val_config, repo_profile=self.repo_profile)
            except Exception as e:
                logger.warning(f"验证器初始化失败: {e}")
                self.validator = None
                self.validation_adapter = None
        else:
            self.validator = None
            self.validation_adapter = None

    def _detect_repo_profile(self) -> RepoProfile:
        """从配置中自动检测仓库类型，返回对应的 RepoProfile。"""
        swebench_path = self.config.data.get("swebench_path", "")
        target_instances_path = self.config.data.get("target_instances_path", "")

        # 策略 1: 读取 target_instances / swebench 数据内容推断（更可靠）
        # - 避免从路径字符串做 substring 匹配导致误判（例如 "target_instances_non_django.json" 含 "django" 子串）
        # - 支持 multi-repo：若检测到多个 repo，则回退到 GenericPytestProfile
        for path_str in [target_instances_path, swebench_path]:
            if not path_str:
                continue
            try:
                data = load_json(path_str)
                if data and isinstance(data, list):
                    repos: set[str] = set()
                    for item in data[:50]:
                        if isinstance(item, str):
                            repos.add(detect_repo_from_instance_id(item))
                        elif isinstance(item, dict):
                            repo = str(item.get("repo", "") or "").strip()
                            if repo:
                                repos.add(repo)
                                continue
                            iid = str(item.get("instance_id", "") or "").strip()
                            if iid:
                                repos.add(detect_repo_from_instance_id(iid))

                    repos = {r for r in repos if r}
                    if len(repos) == 1:
                        return get_repo_profile(next(iter(repos)))
                    if len(repos) > 1:
                        from ..core.repo_profiles.generic import GenericPytestProfile

                        logger.warning(
                            f"检测到 multi-repo 输入 ({len(repos)} repos)，将使用 GenericPytestProfile: {sorted(repos)[:6]}"
                        )
                        return GenericPytestProfile()
            except (json.JSONDecodeError, ValueError, OSError, KeyError) as e:
                logger.debug(f"Auto-detect repo profile failed: {e}")
        from ..core.repo_profiles.generic import GenericPytestProfile
        return GenericPytestProfile()

    def run(self, input_data_path: str = None, instances: List[SWEBenchInstance] = None) -> Dict[str, Any]:
        """
        运行完整流水线

        Args:
            input_data_path: 输入数据路径 (可选,如果不提供则使用配置中的路径)
            instances: 直接传入实例列表 (可选,如果提供则忽略 input_data_path)

        Returns:
            运行结果统计
        """
        logger.info("=" * 60)
        logger.info("TraceGen 流水线开始运行")
        logger.info("=" * 60)

        # 检查是否是验证专用模式
        validation_only_dir = self.config.runtime.get("validation_only_dir", "")
        if validation_only_dir and Path(validation_only_dir).exists():
            logger.info(f"验证专用模式: 从 {validation_only_dir} 加载已合成结果")
            return self._run_validation_only(Path(validation_only_dir))

        # 加载输入数据
        if instances is None:
            data_path = input_data_path or self.config.data.swebench_path
            instances = self._load_instances(data_path)

        # 按 target_instances_path 过滤实例 (用于 split 并行)
        target_path = self.config.data.get("target_instances_path", "")
        if target_path and Path(target_path).exists():
            try:
                target_data = load_json(target_path)
                target_ids = set()
                for item in target_data:
                    if isinstance(item, str):
                        target_ids.add(item)
                    elif isinstance(item, dict):
                        iid = item.get("instance_id", "")
                        if iid:
                            target_ids.add(iid)
                if target_ids:
                    before = len(instances)
                    instances = [i for i in instances if i.instance_id in target_ids]
                    logger.info(f"按 target_instances_path 过滤: {before} → {len(instances)} 个实例")
            except Exception as e:
                logger.warning(f"加载 target_instances_path 失败, 使用全量实例: {e}")

        # 1. 提取阶段 (可选)
        all_extraction_results = []
        if self.config.runtime.get("enable_extraction", True):
            all_extraction_results = self._run_extraction_phase(instances)
        else:
            logger.info("跳过提取阶段 (未开启)")
            # 尝试从缓存加载提取结果
            all_extraction_results = self._load_extraction_cache(instances)

        # 2. 合成阶段 (可选)
        all_synthesis_results = []
        if self.config.runtime.get("enable_synthesis", False) and all_extraction_results:
            logger.info("开启合成阶段")
            all_synthesis_results = self._run_synthesis_phase(all_extraction_results)

            # 3. 验证阶段 (可选)
            if self.config.runtime.get("enable_validation", False) and self.validator and all_synthesis_results:
                logger.info("开启验证阶段")
                all_synthesis_results = self._run_validation_phase(all_synthesis_results)
            elif not self.config.runtime.get("enable_validation", False):
                logger.info("跳过验证阶段 (未开启)")

            # Stage 4 (solving) 已移除，改用 scripts/sweagent/ 外部评测
        else:
            if not self.config.runtime.get("enable_synthesis", False):
                logger.info("跳过合成阶段 (未开启)")
        
        # 3. 统一保存结果
        self._save_unified_results(all_extraction_results, all_synthesis_results)
        
        # 统计
        stats = {
            "input_instances": len(instances),
            "extracted_chains": sum(len(r.chains) for r in all_extraction_results),
            "synthetic_instances": len(all_synthesis_results),
            "output_dir": str(self.output_dir),
        }
        
        logger.info("=" * 60)
        logger.info("流水线运行完成")
        logger.info(f"输入实例: {stats['input_instances']}")
        logger.info(f"提取链路: {stats['extracted_chains']}")
        logger.info(f"合成实例: {stats['synthetic_instances']}")
        logger.info(f"结果目录: {self.output_dir}")
        logger.info("=" * 60)
        
        return stats
    
    def _run_extraction_phase(
        self, instances: List[SWEBenchInstance]
    ) -> List[ExtractionResult]:
        """
        运行阶段一: 缺陷链路提取
        支持按仓库分组的跨仓库并行提取 (同仓库内串行，避免 git checkout 冲突)
        """
        logger.info("\n" + "=" * 60)
        logger.info("阶段一: 缺陷链路提取")
        logger.info("=" * 60)

        cache_dir = Path(self.config.data.cache_dir) / "extractions"
        ensure_dir(cache_dir)

        # 详细结果子目录 (仅在 verbose 模式下使用)
        verbose = self.config.runtime.get("verbose", False)
        details_dir = self.output_dir / "1_extraction" / "details"
        if verbose:
            ensure_dir(details_dir)

        num_workers = self.config.runtime.get("num_workers", 4)

        # 按仓库分组
        from collections import defaultdict
        repo_groups = defaultdict(list)
        for inst in instances:
            repo_groups[inst.repo].append(inst)

        logger.info(f"共 {len(instances)} 实例, {len(repo_groups)} 仓库, 并行数: {num_workers}")
        for repo, group in sorted(repo_groups.items(), key=lambda x: -len(x[1])):
            logger.info(f"  {repo}: {len(group)} 实例")

        # 并行度 <= 1 时走串行路径
        if num_workers <= 1:
            return self._run_extraction_phase_serial(instances, cache_dir, details_dir, verbose)

        # 构建并行任务列表: 大仓库拆分成多个子组，每个子组用独立 repo clone
        INTRA_REPO_SPLIT_THRESHOLD = 50  # 超过此数量的仓库内部也拆分并行
        work_units = []  # (repo, sub_group, worker_id_or_None)

        for repo, group in repo_groups.items():
            # 按 created_at 时间排序，使相邻实例 commit 接近，提高增量图复用率
            group.sort(key=lambda inst: getattr(inst, 'created_at', '') or inst.base_commit)

            if len(group) > INTRA_REPO_SPLIT_THRESHOLD:
                # 大仓库: 拆分成多个子组
                n_splits = min(num_workers, max(2, len(group) // 30))
                chunk_size = (len(group) + n_splits - 1) // n_splits
                for i in range(n_splits):
                    sub_group = group[i * chunk_size : (i + 1) * chunk_size]
                    if sub_group:
                        work_units.append((repo, sub_group, i))
                logger.info(f"  {repo}: 拆分为 {n_splits} 个并行子组 (每组 ~{chunk_size} 实例)")
            else:
                work_units.append((repo, group, None))

        # 跨仓库 + 仓库内并行提取
        from concurrent.futures import ThreadPoolExecutor, as_completed

        completed = {"count": 0}  # mutable counter for thread-safe progress
        total = len(instances)
        lock = threading.Lock()

        def process_work_unit(repo: str, group_instances: List[SWEBenchInstance], worker_id) -> List[ExtractionResult]:
            """处理一个工作单元 (可能是完整仓库或仓库的子组)"""
            results = []
            for instance in group_instances:
                try:
                    # 用 instance_id 作缓存 key (同一 commit 可能对应多个 instance)
                    cache_file = cache_dir / f"{instance.instance_id}.json"
                    legacy_cache = cache_dir / f"{instance.repo.replace('/', '_')}__{instance.base_commit[:8]}.json"

                    result = None
                    for cf in [cache_file, legacy_cache]:
                        if cf.exists():
                            try:
                                cached_data = load_json(cf)
                                result = ExtractionResult(**cached_data)
                                # 校验 instance_id 防止同 commit 不同 instance 误匹配
                                if not result.chains or result.instance_id != instance.instance_id:
                                    result = None
                                else:
                                    break
                            except Exception:
                                result = None

                    if result is None:
                        result = self._extract_single_instance(instance, worker_id=worker_id)
                        save_json(result.model_dump(), cache_file)

                    results.append(result)

                    if verbose:
                        save_json(result.model_dump(), details_dir / f"{instance.instance_id}.json")

                    with lock:
                        completed["count"] += 1
                        cnt = completed["count"]
                    logger.info(f"[{cnt}/{total}] 完成: {instance.instance_id}")

                except Exception as e:
                    logger.error(f"提取失败 {instance.instance_id}: {e}")
            return results

        extraction_results = []
        logger.info(f"启动 {len(work_units)} 个并行工作单元, max_workers={num_workers}")
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for repo, group, worker_id in work_units:
                label = f"{repo}" if worker_id is None else f"{repo}[w{worker_id}]"
                future = executor.submit(process_work_unit, repo, group, worker_id)
                futures[future] = label

            for future in as_completed(futures):
                label = futures[future]
                try:
                    results = future.result()
                    extraction_results.extend(results)
                    logger.info(f"工作单元 {label} 完成: {len(results)} 实例")
                except Exception as e:
                    logger.error(f"工作单元 {label} 异常: {e}")

        logger.info(f"提取阶段完成: {len(extraction_results)}/{total} 实例")
        return extraction_results

    def _run_extraction_phase_serial(
        self, instances: List[SWEBenchInstance],
        cache_dir: Path, details_dir: Path, verbose: bool
    ) -> List[ExtractionResult]:
        """串行提取 (fallback, num_workers<=1 或单仓库时使用)"""
        extraction_results = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task(
                f"提取链路 (0/{len(instances)})", total=len(instances)
            )

            for i, instance in enumerate(instances):
                try:
                    # 优先用 instance_id 缓存，兼容旧格式 {repo}__{commit[:8]}
                    cache_file = cache_dir / f"{instance.instance_id}.json"
                    legacy_cache = cache_dir / f"{instance.repo.replace('/', '_')}__{instance.base_commit[:8]}.json"

                    result = None
                    for cf in [cache_file, legacy_cache]:
                        if cf.exists():
                            try:
                                cached_data = load_json(cf)
                                result = ExtractionResult(**cached_data)
                                if not result.chains or result.instance_id != instance.instance_id:
                                    result = None
                                else:
                                    break
                            except Exception:
                                result = None

                    if result is None:
                        result = self._extract_single_instance(instance)
                        save_json(result.model_dump(), cache_file)

                    extraction_results.append(result)

                    if verbose:
                        save_json(result.model_dump(), details_dir / f"{instance.instance_id}.json")

                    progress.update(task, advance=1, description=f"提取链路 ({i+1}/{len(instances)})")

                except Exception as e:
                    logger.error(f"提取失败 {instance.instance_id}: {e}")

        return extraction_results

    def _load_extraction_cache(self, instances: List[SWEBenchInstance]) -> List[ExtractionResult]:
        """
        从缓存加载提取结果 (当 enable_extraction=False 时使用)
        """
        logger.info("\n" + "=" * 60)
        logger.info("从缓存加载提取结果")
        logger.info("=" * 60)

        extraction_results = []
        cache_dir = Path(self.config.data.cache_dir) / "extractions"

        for instance in instances:
            # 优先用 instance_id 缓存，兼容旧格式
            cache_file = cache_dir / f"{instance.instance_id}.json"
            legacy_cache = cache_dir / f"{instance.repo.replace('/', '_')}__{instance.base_commit[:8]}.json"

            loaded = False
            for cf in [cache_file, legacy_cache]:
                if cf.exists():
                    try:
                        cached_data = load_json(cf)
                        result = ExtractionResult(**cached_data)
                        if result.chains and result.instance_id == instance.instance_id:
                            extraction_results.append(result)
                            logger.info(f"已加载缓存: {instance.instance_id}")
                            loaded = True
                            break
                    except Exception as e:
                        logger.warning(f"缓存加载失败 {instance.instance_id}: {e}")
            if not loaded:
                logger.warning(f"未找到缓存: {instance.instance_id}")

        logger.info(f"从缓存加载了 {len(extraction_results)} 个提取结果")
        return extraction_results

    def _load_synthesis_cache(self, synthesis_dir: Path) -> Dict[str, List[SynthesisResult]]:
        """
        从历史输出目录加载已完成的合成结果。

        以 seed_id 为 key，返回该 seed 下所有已合成的 SynthesisResult 列表。
        判定标准: details/ 目录下存在 {seed_id}_rank*.json 文件。
        """
        cache: Dict[str, List[SynthesisResult]] = {}
        details_dir = synthesis_dir / "details"
        if not details_dir.exists():
            return cache

        for json_file in sorted(details_dir.glob("*_rank*.json")):
            try:
                data = load_json(json_file)
                result = SynthesisResult(**data)
                cache.setdefault(result.seed_id, []).append(result)
            except Exception as e:
                logger.debug(f"跳过无法加载的合成缓存 {json_file.name}: {e}")
        return cache

    def _run_synthesis_phase(self, extraction_results: List[ExtractionResult]) -> List[SynthesisResult]:
        """
        运行阶段二: 缺陷合成 (Agent 模式，支持断点续传)
        """
        logger.info("\n" + "=" * 60)
        logger.info("阶段二: 缺陷合成 (Agent 模式)")
        logger.info("=" * 60)

        synthesis_results = []
        synthesis_dir = self.output_dir / "2_synthesis"
        details_dir = synthesis_dir / "details"
        candidates_dir = synthesis_dir / "candidates"

        if self.config.runtime.get("verbose", False):
            ensure_dir(details_dir)
        ensure_dir(candidates_dir)

        # --- 断点续传: 加载已完成的合成结果 ---
        completed_seeds: Dict[str, List[SynthesisResult]] = {}

        # 1) 从当前输出目录加载
        completed_seeds.update(self._load_synthesis_cache(synthesis_dir))

        # 2) 自动扫描同级历史运行目录 (同一父目录下的兄弟目录)
        parent_dir = self.output_dir.parent
        if parent_dir.exists():
            for sibling in sorted(parent_dir.iterdir()):
                if sibling.is_dir() and sibling != self.output_dir:
                    sibling_synthesis = sibling / "2_synthesis"
                    if sibling_synthesis.exists():
                        for seed_id, results in self._load_synthesis_cache(sibling_synthesis).items():
                            if seed_id not in completed_seeds:
                                completed_seeds[seed_id] = results

        # 3) 从配置的额外历史目录加载 (runtime.synthesis_cache_dirs)
        cache_dirs = self.config.runtime.get("synthesis_cache_dirs", [])
        if isinstance(cache_dirs, str):
            cache_dirs = [cache_dirs] if cache_dirs else []
        for cache_dir_str in cache_dirs:
            cache_path = Path(cache_dir_str) / "2_synthesis"
            if cache_path.exists():
                for seed_id, results in self._load_synthesis_cache(cache_path).items():
                    if seed_id not in completed_seeds:
                        completed_seeds[seed_id] = results

        if completed_seeds:
            logger.info(f"断点续传: 发现 {len(completed_seeds)} 个已完成合成的 seed 实例，将跳过")

        # 预筛选：过滤低质量实例（可通过 runtime.apply_synthesis_prefilter=false 禁用）
        apply_prefilter = bool(self.config.runtime.get("apply_synthesis_prefilter", True))
        filtered_results: List[ExtractionResult] = []
        skipped_results: List[tuple[str, str]] = []
        if apply_prefilter:
            for ext_result in extraction_results:
                should_do, reason = should_synthesize(ext_result)
                if should_do:
                    filtered_results.append(ext_result)
                else:
                    skipped_results.append((ext_result.instance_id, reason))
                    logger.info(f"跳过实例 {ext_result.instance_id}: {reason}")

            if skipped_results:
                logger.info(f"预筛选: 跳过 {len(skipped_results)} 个实例, 保留 {len(filtered_results)} 个")
        else:
            filtered_results = list(extraction_results)
            logger.info(f"预筛选已禁用: 将对 {len(filtered_results)} 个实例进行合成")

        # 限制合成的实例数量
        max_synthesis = self.config.runtime.get("max_synthesis_instances", 0)
        if max_synthesis > 0 and len(filtered_results) > max_synthesis:
            logger.info(f"限制合成实例数量: {max_synthesis}/{len(filtered_results)}")
            extraction_results_to_process = filtered_results[:max_synthesis]
        else:
            extraction_results_to_process = filtered_results

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ) as progress:
            task = progress.add_task(
                f"合成缺陷 (0/{len(extraction_results_to_process)})", total=len(extraction_results_to_process)
            )

            for i, ext_result in enumerate(extraction_results_to_process):
                try:
                    if not ext_result.chains:
                        progress.update(task, advance=1)
                        continue

                    # 断点续传: 跳过已完成合成的 seed 实例
                    seed_id = ext_result.instance_id
                    if seed_id in completed_seeds:
                        cached_results = completed_seeds[seed_id]
                        synthesis_results.extend(cached_results)
                        logger.info(f"[cache] 跳过已完成的 seed {seed_id} ({len(cached_results)} 个合成结果)")
                        progress.update(task, advance=1, description=f"合成缺陷 ({i+1}/{len(extraction_results_to_process)})")
                        continue

                    repo = ext_result.seed_metadata.get("repo", "")
                    base_commit = ext_result.seed_metadata.get("base_commit", "")

                    # Multi-repo: 按当前 seed 的 repo 动态切换 synthesis agent 的 profile
                    if repo and self.synthesis_agent is not None:
                        try:
                            instance_profile = get_repo_profile(repo)
                            # 与 agent 当前 profile 比较（非全局），确保每次都切到正确 profile
                            if type(instance_profile) != type(self.synthesis_agent.repo_profile):
                                self.synthesis_agent.repo_profile = instance_profile
                                logger.debug(f"Synthesis profile switched to {type(instance_profile).__name__} for {repo}")
                        except Exception as e:
                            logger.debug(f"Synthesis profile switch failed for {repo}: {e}")

                    # 准备仓库 (合成阶段应用修复补丁，确保从安全状态开始)
                    repo_path = self._prepare_repository(instance=ext_result, apply_fix=True)

                    # 同步 repo_path 到 agent，确保候选重排阶段也能访问正确路径
                    if self.synthesis_agent is not None:
                        self.synthesis_agent.repo_path = str(repo_path)

                    # 加载图
                    graph_cache_path = (
                        Path(self.config.data.cache_dir)
                        / "graphs"
                        / f"{repo.replace('/', '_')}__{base_commit[:8]}_{GRAPH_CACHE_VERSION}.pkl"
                    )
                    graph = load_pickle(graph_cache_path)

                    # 运行 Matcher 寻找候选子图（带自动向量生成）
                    logger.info(f"正在寻找候选合成位置: {ext_result.instance_id}")
                    if self.subgraph_matcher.auto_generate:
                        candidates = self.subgraph_matcher.find_candidates_with_auto_generate(
                            ext_result, graph, graph_cache_path
                        )
                    else:
                        candidates = self.subgraph_matcher.find_candidates(ext_result, graph)
                    
                    # 保存候选结果到当前输出目录
                    if candidates:
                        # 处理 candidates 中可能包含的 dataclass 对象，使其可序列化
                        from dataclasses import asdict, is_dataclass

                        def make_serializable(obj):
                            """递归转换对象为可序列化格式"""
                            if obj is None:
                                return None
                            elif isinstance(obj, (str, int, float, bool)):
                                return obj
                            elif hasattr(obj, "model_dump"):
                                return obj.model_dump()
                            elif is_dataclass(obj) and not isinstance(obj, type):
                                return asdict(obj)
                            elif isinstance(obj, dict):
                                return {k: make_serializable(v) for k, v in obj.items()}
                            elif isinstance(obj, (list, tuple)):
                                return [make_serializable(item) for item in obj]
                            else:
                                return str(obj)  # 回退到字符串表示

                        serializable_candidates = [make_serializable(cand) for cand in candidates]

                        save_json(
                            serializable_candidates,
                            candidates_dir / f"{ext_result.instance_id}_candidates.json"
                        )
                    
                    if not candidates:
                        logger.warning(f"未找到合适的候选合成位置: {ext_result.instance_id}")
                        progress.update(task, advance=1)
                        continue

                    # P0: Prefer candidates that have identifiable, existing regression tests.
                    #
                    # Controlled by `method.synthesis.candidate_test_filtering_mode`:
                    # - "filter"    (default): filter out candidates without test-signal (existing behavior)
                    # - "rank_only": only re-rank by test signal, keep all candidates
                    # - "off":      do nothing (keep matcher order)
                    filtering_mode = (
                        str(self.config.method.synthesis.get("candidate_test_filtering_mode", "filter") or "filter")
                        .strip()
                        .lower()
                    )
                    if filtering_mode not in {"filter", "rank_only", "off"}:
                        filtering_mode = "filter"

                    if filtering_mode != "off":
                        scored_candidates: list[dict] = []
                        for cand in candidates:
                            try:
                                anchor_node_id = cand.get("anchor_node_id", "") or ""
                                file_path = anchor_node_id.split(":")[0] if ":" in anchor_node_id else anchor_node_id

                                planned = self.synthesis_agent._plan_validation_test_suite(file_path)
                                related = self.synthesis_agent._collect_related_test_details(
                                    file_path=file_path,
                                    target_node_id=anchor_node_id,
                                    planned_test_files=planned.get("test_files", []),
                                )
                                target_hit = sum(1 for t in (related or []) if t.get("target_hit"))

                                cand["related_tests_count"] = len(related or [])
                                cand["related_target_hit_count"] = int(target_hit)
                                scored_candidates.append(cand)
                            except Exception:
                                cand["related_tests_count"] = 0
                                cand["related_target_hit_count"] = 0
                                scored_candidates.append(cand)

                        if filtering_mode == "filter":
                            # Filter:
                            # - Strong signal: keep only candidates with target-hit tests (tests that directly mention the target API).
                            # - Fallback: if none have target-hit tests, keep candidates that at least have some related tests.
                            max_target_hit = max(
                                int(c.get("related_target_hit_count", 0) or 0) for c in scored_candidates
                            )
                            if max_target_hit > 0:
                                scored_candidates = [
                                    c
                                    for c in scored_candidates
                                    if int(c.get("related_target_hit_count", 0) or 0) > 0
                                ]
                            elif any((c.get("related_tests_count", 0) or 0) > 0 for c in scored_candidates):
                                scored_candidates = [
                                    c for c in scored_candidates if (c.get("related_tests_count", 0) or 0) > 0
                                ]
                            else:
                                logger.warning(
                                    f"{ext_result.instance_id}: 候选点均未找到可识别的相关测试，将继续尝试（可能出现 NO_FAIL）"
                                )

                        scored_candidates.sort(
                            key=lambda c: (
                                int(c.get("related_target_hit_count", 0) or 0),
                                int(c.get("related_tests_count", 0) or 0),
                                float(c.get("final_score", 0.0) or 0.0),
                            ),
                            reverse=True,
                        )
                        candidates = scored_candidates
                        
                    # 运行 Agent 合成 - 针对每个 Top 候选点生成独立 Bug
                    logger.info(f"正在针对 Top {len(candidates)} 个候选位置并行/顺序合成...")

                    agent_cfg = self.config.method.synthesis.get("agent", {})
                    max_attempts = int(agent_cfg.get("max_attempts", 1) or 1)
                    validate_during_synthesis = bool(agent_cfg.get("validate_during_synthesis", False))
                    if validate_during_synthesis and not (self.validator and self.validation_adapter):
                        logger.warning("已启用 validate_during_synthesis，但验证器未初始化；将跳过合成阶段验证")
                        validate_during_synthesis = False
                    if validate_during_synthesis:
                        ensure_dir(self.output_dir / "3_validation")

                    for rank, candidate in enumerate(candidates):
                        logger.info(f"--- 尝试合成候选点 {rank+1}/{len(candidates)}: {candidate['anchor_node_id']} ---")
                        try:
                            validation_feedback = ""
                            best_result = None
                            best_validation = None

                            for attempt in range(1, max_attempts + 1):
                                if max_attempts > 1:
                                    logger.info(f"候选点 {rank+1}: 合成尝试 {attempt}/{max_attempts}")

                                result = self.synthesis_agent.synthesize(
                                    extraction_result=ext_result,
                                    graph=graph,
                                    repo_path=str(repo_path),
                                    candidate=candidate,
                                    rank=rank + 1,
                                    validation_feedback=validation_feedback,
                                )

                                if not result:
                                    continue

                                # 注入候选点元数据
                                result.metadata["candidate_info"] = {
                                    "anchor_node": candidate.get("anchor_node_id"),
                                    "topology_score": candidate.get("topology_score"),
                                    "vector_score": candidate.get("vector_score"),
                                    "final_score": candidate.get("final_score"),
                                    "attempt": attempt,
                                }

                                if not validate_during_synthesis:
                                    best_result = result
                                    break

                                # 合成阶段快速验证（用 adapter 生成的最小相关测试）
                                try:
                                    instance_dict = self.validation_adapter.adapt(result)
                                    val_result = self.validator.validate(instance_dict)

                                    # 注入验证结果，后续阶段可复用
                                    result.metadata["validation"] = val_result.to_dict()
                                    result.metadata["is_valid"] = val_result.is_valid_bug()

                                    if val_result.status == ValidationStatus.MISSING_IMAGE:
                                        logger.warning(
                                            f"候选点 {rank+1}: 镜像缺失，无法合成阶段验证，直接保留该结果: {val_result.error_message}"
                                        )
                                        best_result = result
                                        best_validation = val_result
                                        break

                                    if val_result.is_valid_bug():
                                        logger.info(f"候选点 {rank+1}: 合成阶段验证通过 (VALID)")
                                        best_result = result
                                        best_validation = val_result
                                        break

                                    # 保留最后一次结果用于调试/回退
                                    best_result = result
                                    best_validation = val_result
                                    validation_feedback = (
                                        "Previous attempt did NOT trigger any failing tests.\n"
                                        f"- PASS_TO_FAIL: {len(val_result.PASS_TO_FAIL)}\n"
                                        f"- PASS_TO_PASS: {len(val_result.PASS_TO_PASS)}\n"
                                        f"- FAIL_TO_FAIL: {len(val_result.FAIL_TO_FAIL)}\n"
                                        f"- FAIL_TO_PASS: {len(val_result.FAIL_TO_PASS)}\n"
                                        "Inject a real logic bug (not a refactor) that will break at least one existing test "
                                        "from RELATED EXISTING TESTS while keeping other tests passing."
                                    )
                                except Exception as e:
                                    logger.warning(f"候选点 {rank+1}: 合成阶段验证异常: {e}")
                                    best_result = result
                                    best_validation = None
                                    validation_feedback = f"Validation crashed: {e}"

                            if best_result:
                                # 只为最终保留的结果写入 validation json，避免为每次尝试都产生文件
                                if validate_during_synthesis and best_validation is not None:
                                    save_json(
                                        best_validation.to_dict(),
                                        (self.output_dir / "3_validation") / f"{best_result.instance_id}_validation.json",
                                    )
                                synthesis_results.append(best_result)
                                save_json(
                                    best_result.model_dump(),
                                    details_dir / f"{best_result.instance_id}_rank{rank+1}.json",
                                )
                                # If we already have a VALID synthetic bug for this seed, stop trying more candidates
                                # to reduce iteration time.
                                if validate_during_synthesis and best_validation is not None and best_validation.is_valid_bug():
                                    logger.info(
                                        f"{ext_result.instance_id}: 已获得 VALID 合成 bug（候选点 {rank+1}），跳过剩余候选点。"
                                    )
                                    break
                            else:
                                logger.warning(
                                    f"候选点 {rank+1} 合成未产生有效结果 (可能因为补丁为空、代码匹配失败或 Agent 格式错误)"
                                )
                        except Exception as e:
                            logger.error(f"候选点 {rank+1} 合成失败: {e}")
                            continue
                    
                    progress.update(task, advance=1, description=f"合成缺陷 ({i+1}/{len(extraction_results)})")
                    
                except Exception as e:
                    logger.error(f"合成失败 {ext_result.instance_id}: {e}")
                    progress.update(task, advance=1)
        
        return synthesis_results

    def _run_validation_phase(self, synthesis_results: List[SynthesisResult]) -> List[SynthesisResult]:
        """
        运行阶段三: 缺陷验证 (支持多线程并行，跳过已完成的验证)
        """
        logger.info("\n" + "=" * 60)
        logger.info("阶段三: 缺陷验证")
        logger.info("=" * 60)

        validation_dir = self.output_dir / "3_validation"
        ensure_dir(validation_dir)

        # 获取并发数配置
        # Prefer `validation.num_workers` if set; otherwise fallback to `runtime.num_workers`.
        num_workers = self.config.validation.get("num_workers", self.config.runtime.get("num_workers", 8))
        try:
            num_workers = int(num_workers or 8)
        except Exception:
            num_workers = 8
        logger.info(f"验证并发数: {num_workers}")

        # 检查已完成的验证结果 (支持从多个目录加载)
        completed_validations = {}
        validation_dirs_to_check = [validation_dir]

        # 如果是验证专用模式，也检查源目录
        validation_only_dir = self.config.runtime.get("validation_only_dir", "")
        if validation_only_dir:
            source_validation_dir = Path(validation_only_dir) / "3_validation"
            if source_validation_dir.exists():
                validation_dirs_to_check.append(source_validation_dir)

        for check_dir in validation_dirs_to_check:
            if check_dir.exists():
                for json_file in check_dir.glob("*_validation.json"):
                    try:
                        data = load_json(json_file)
                        instance_id = data.get("instance_id", "")
                        status = data.get("status", "")
                        # 只跳过已完成且可复用的验证 (valid/invalid/missing_image)，不跳过 error/timeout
                        if instance_id and status in ["valid", "invalid", "missing_image"]:
                            completed_validations[instance_id] = data
                    except (json.JSONDecodeError, ValueError, OSError) as e:
                        logger.debug(f"Failed to read validation file {json_file}: {e}")
            logger.info(f"发现 {len(completed_validations)} 个已完成的验证结果，将跳过这些实例")

        # 分离已完成和待验证的实例
        to_validate = []
        already_validated = []
        for result in synthesis_results:
            if result.instance_id in completed_validations:
                # 加载已有的验证结果
                val_data = completed_validations[result.instance_id]
                result.metadata["validation"] = val_data
                result.metadata["is_valid"] = val_data.get("status") == "valid"
                already_validated.append(result)
            else:
                to_validate.append(result)

        logger.info(f"待验证: {len(to_validate)} | 已跳过: {len(already_validated)}")

        validated_results = list(already_validated)  # 先添加已完成的
        results_lock = threading.Lock()
        completed_count = [0]  # 使用列表以便在闭包中修改
        total_to_validate = len(to_validate)

        def validate_single(result: SynthesisResult) -> SynthesisResult:
            """验证单个实例"""
            try:
                # Adapt result
                instance_dict = self.validation_adapter.adapt(result)

                # Validate
                logger.info(f"验证实例: {result.instance_id}")
                val_result = self.validator.validate(instance_dict)

                # P3: 计算链路对齐评分
                # [Ablation] disable_chain_alignment: 跳过链路对齐评分
                if not self._ablation_config.get("disable_chain_alignment", False):
                    try:
                        from src.modules.synthesis.chain_alignment import calculate_chain_alignment
                        seed_chains = result.seed_extraction_chains or []
                        proposed_chain = result.metadata.get("proposed_chain", [])

                        if seed_chains and proposed_chain:
                            # 提取 seed chain 的节点结构
                            first_seed_chain = seed_chains[0]
                            if isinstance(first_seed_chain, dict):
                                seed_nodes = first_seed_chain.get("nodes", [])
                            else:
                                seed_nodes = first_seed_chain.nodes if hasattr(first_seed_chain, "nodes") else []

                            # 标准化 seed_nodes 格式
                            normalized_seed = []
                            for idx, node in enumerate(seed_nodes):
                                if isinstance(node, dict):
                                    node_id = node.get("node_id", "")
                                    file_path = node.get("file_path", "")
                                else:
                                    node_id = node.node_id if hasattr(node, "node_id") else str(node)
                                    file_path = node.file_path if hasattr(node, "file_path") else ""

                                if idx == 0:
                                    node_type = "symptom"
                                elif idx == len(seed_nodes) - 1:
                                    node_type = "root_cause"
                                else:
                                    node_type = "intermediate"

                                normalized_seed.append({
                                    "node_id": node_id,
                                    "node_type": node_type,
                                    "file_path": file_path,
                                })

                            # 计算结构对齐评分 (Eq.3: seed chain vs synthetic chain)
                            alignment_score = calculate_chain_alignment(normalized_seed, proposed_chain)
                            # 保存为独立字段，不覆盖 validator 的 traceback 覆盖度评分
                            # chain_alignment_score: traceback-based (validator, 用于准入门槛)
                            # structural_alignment:  Eq.3-based (seed vs synthetic, 用于质量评估)
                            if not val_result.chain_alignment_score:
                                val_result.chain_alignment_score = {}
                            val_result.chain_alignment_score["structural_alignment"] = alignment_score.to_dict()
                            logger.info(f"结构对齐评分: {alignment_score.summary()}")
                    except Exception as e:
                        logger.warning(f"链路对齐评分计算失败: {e}")
                else:
                    logger.debug("[Ablation] disable_chain_alignment: 跳过链路对齐评分")

                # Update metadata
                result.metadata["validation"] = val_result.to_dict()
                result.metadata["is_valid"] = val_result.is_valid_bug()

                # Post-validation: generate chain-guided PS at all levels
                if val_result.is_valid_bug():
                    repo_name = result.repo or ""
                    test_output_path = (
                        validation_dir / "logs" / repo_name / result.instance_id / "test_output.txt"
                    )
                    try:
                        from src.modules.synthesis.ps_chain_guided import ChainGuidedPSGenerator
                        meta = result.metadata or {}
                        chain_nodes = meta.get("proposed_chain", meta.get("synthesized_chain", []))
                        if not isinstance(chain_nodes, list):
                            chain_nodes = []
                        injection_patch = meta.get("injection_patch", result.patch or "")

                        # Read test output
                        test_output = ""
                        if test_output_path.exists():
                            test_output = test_output_path.read_text(errors="replace")

                        # Generate L1/L2/L3 PS variants
                        ps_gen = ChainGuidedPSGenerator()
                        ps_levels = ps_gen.generate_all_levels(
                            chain_nodes=chain_nodes,
                            injection_patch=injection_patch,
                            test_output=test_output,
                            seed_ps=result.problem_statement,
                            instance_id=result.instance_id,
                        )

                        # Store all PS variants in metadata
                        result.metadata["original_problem_statement"] = result.problem_statement
                        result.metadata["ps_levels"] = {
                            lvl: {
                                "problem_statement": r.problem_statement,
                                "metrics": r.metrics,
                            }
                            for lvl, r in ps_levels.items()
                        }

                        # Select hardest PS (L1) as default for evaluation
                        ps_select = self.config.method.synthesis.get("agent", {}).get("ps_select", "L1")
                        if ps_select in ps_levels:
                            result.problem_statement = ps_levels[ps_select].problem_statement
                            result.metadata["ps_selected_level"] = ps_select
                            logger.info(
                                f"PS level selected for {result.instance_id}: {ps_select} "
                                f"({ps_levels[ps_select].metrics.get('token_count', 0)} tokens)"
                            )
                        else:
                            result.metadata["ps_selected_level"] = "Original"
                    except Exception as e:
                        logger.debug(f"Chain-guided PS generation skipped for {result.instance_id}: {e}")

                    # Legacy: PS hybrid enhancement (only if ps_hybrid_mode is set AND no chain-guided)
                    if "ps_levels" not in result.metadata:
                        try:
                            from src.modules.synthesis.ps_hybrid import enhance_ps_post_validation
                            ps_mode = self.config.method.synthesis.get("agent", {}).get("ps_hybrid_mode", "")
                            if ps_mode:
                                ps_level = self.config.method.synthesis.get("agent", {}).get("ps_level", "standard")
                                enhanced = enhance_ps_post_validation(
                                    original_ps=result.problem_statement,
                                    test_output_path=test_output_path,
                                    target_level=ps_level,
                                    mode=ps_mode,
                                    repo=repo_name,
                                )
                                if enhanced:
                                    result.metadata["original_problem_statement"] = result.problem_statement
                                    result.problem_statement = enhanced["problem_statement"]
                                    result.metadata["ps_hybrid"] = {
                                        "mode": enhanced["mode"],
                                        "level": enhanced["level"],
                                        "token_count": enhanced["token_count"],
                                        "components": enhanced["components"],
                                    }
                        except Exception as e:
                            logger.debug(f"PS hybrid enhancement skipped: {e}")

                # Save individual validation result
                save_json(
                    val_result.to_dict(),
                    validation_dir / f"{result.instance_id}_validation.json"
                )

                status_str = "VALID" if val_result.is_valid_bug() else val_result.status.value
                with results_lock:
                    completed_count[0] += 1
                    logger.info(f"验证完成 ({completed_count[0]}/{total_to_validate}): {result.instance_id} - {status_str}")

                return result

            except Exception as e:
                logger.error(f"验证失败 {result.instance_id}: {e}")
                result.metadata["validation_error"] = str(e)
                with results_lock:
                    completed_count[0] += 1
                return result

        # 使用线程池并行验证
        if to_validate:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(validate_single, result): result for result in to_validate}

                for future in as_completed(futures):
                    try:
                        validated_result = future.result()
                        validated_results.append(validated_result)
                    except Exception as e:
                        original_result = futures[future]
                        logger.error(f"验证异常 {original_result.instance_id}: {e}")
                        original_result.metadata["validation_error"] = str(e)
                        validated_results.append(original_result)

        return validated_results


    def _run_validation_only(self, synthesis_dir: Path) -> Dict[str, Any]:
        """
        验证专用模式: 从已有合成结果目录加载数据并运行验证

        Args:
            synthesis_dir: 合成结果目录路径

        Returns:
            运行结果统计
        """
        logger.info("=" * 60)
        logger.info("验证专用模式")
        logger.info("=" * 60)

        # 加载合成结果
        final_dataset_path = synthesis_dir / "2_synthesis" / "final_dataset.json"
        if not final_dataset_path.exists():
            logger.error(f"未找到合成结果: {final_dataset_path}")
            return {"error": "未找到合成结果"}

        synthesis_data = load_json(final_dataset_path)
        logger.info(f"加载了 {len(synthesis_data)} 个合成实例")

        # 转换为 SynthesisResult 对象
        synthesis_results = []
        for item in synthesis_data:
            try:
                # 字段优先从顶层读取，回退到 metadata（兼容两种格式）
                metadata = item.get("metadata", {})
                result = SynthesisResult(
                    instance_id=item.get("instance_id", "unknown"),
                    repo=item.get("repo", "django/django"),
                    base_commit=item.get("base_commit", ""),
                    problem_statement=item.get("problem_statement", ""),
                    patch=item.get("patch", ""),
                    FAIL_TO_PASS=self._parse_test_list(item.get("FAIL_TO_PASS", [])),
                    PASS_TO_PASS=self._parse_test_list(item.get("PASS_TO_PASS", [])),
                    seed_id=item.get("seed_id", "") or metadata.get("seed_id", ""),
                    fix_intent=item.get("fix_intent", "") or metadata.get("seed_fix_intent", ""),
                    injection_strategy=item.get("injection_strategy", "") or metadata.get("injection_strategy", ""),
                    seed_metadata=item.get("seed_metadata", {}) or metadata.get("seed_metadata", {}),
                    seed_extraction_chains=item.get("seed_extraction_chains", []) or metadata.get("seed_extraction_chains", []),
                    metadata=metadata,
                )
                synthesis_results.append(result)
            except Exception as e:
                logger.warning(f"转换失败 {item.get('instance_id', 'unknown')}: {e}")

        if not synthesis_results:
            logger.error("没有有效的合成结果")
            return {"error": "没有有效的合成结果"}

        # 运行验证
        if self.validator and self.validation_adapter:
            validated_results = self._run_validation_phase(synthesis_results)
        else:
            logger.error("验证器未初始化")
            return {"error": "验证器未初始化"}

        # 保存验证结果到原目录
        validation_dir = synthesis_dir / "3_validation"
        ensure_dir(validation_dir)

        # 统计
        stats = {
            "total": len(validated_results),
            "valid": sum(1 for r in validated_results if r.metadata.get("is_valid", False)),
            "invalid": sum(1 for r in validated_results
                          if r.metadata.get("validation", {}).get("status") == "invalid"),
            "error": sum(1 for r in validated_results
                        if r.metadata.get("validation", {}).get("status") == "error"),
            "timeout": sum(1 for r in validated_results
                          if r.metadata.get("validation", {}).get("status") == "timeout"),
        }

        # 保存验证汇总
        summary = {
            "timestamp": datetime.now().isoformat(),
            "source_dir": str(synthesis_dir),
            "stats": stats,
            "success_rate": f"{stats['valid'] / stats['total'] * 100:.1f}%" if stats["total"] > 0 else "0%",
            "results": [
                {
                    "instance_id": r.instance_id,
                    "is_valid": r.metadata.get("is_valid", False),
                    "status": r.metadata.get("validation", {}).get("status", "unknown"),
                    "FAIL_TO_PASS": r.metadata.get("validation", {}).get("FAIL_TO_PASS", []),
                }
                for r in validated_results
            ]
        }
        save_json(summary, validation_dir / "validation_summary.json")

        logger.info("=" * 60)
        logger.info("验证专用模式完成")
        logger.info(f"总数: {stats['total']}")
        logger.info(f"有效: {stats['valid']}")
        logger.info(f"无效: {stats['invalid']}")
        logger.info(f"错误: {stats['error']}")
        logger.info(f"超时: {stats['timeout']}")
        logger.info(f"验证通过率: {summary['success_rate']}")
        logger.info("=" * 60)

        return stats

    def _parse_test_list(self, value: Any) -> List[str]:
        """解析测试列表 (可能是字符串或列表)"""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                import json as json_module
                return json_module.loads(value)
            except (json.JSONDecodeError, ValueError):
                return [value] if value else []
        return []

    def _save_unified_results(self, extractions: List[ExtractionResult], synthesis: List[SynthesisResult]):
        """
        统一保存最终结果，结构清晰标注合成信息
        """
        logger.info("正在统一保存结果...")
        
        # 1. 建立 Extraction 查找表
        extraction_map = {ext.instance_id: ext for ext in extractions}
        
        # 2. 链路汇总 (1_extraction/summary.json)
        summary_chains = []
        for ext in extractions:
            for chain in ext.chains:
                extraction_metadata = chain.get("extraction_metadata", {}) if isinstance(chain, dict) else getattr(chain, "extraction_metadata", {})
                fix_intents = []
                if "repair_chain" in extraction_metadata:
                    rc_meta = extraction_metadata["repair_chain"]
                    if isinstance(rc_meta, dict) and "repair_chain" in rc_meta:
                        fix_intents.append(rc_meta["repair_chain"].get("type", "Unknown"))
                
                chain_id = chain.get("chain_id") if isinstance(chain, dict) else chain.chain_id
                nodes = chain.get("nodes", []) if isinstance(chain, dict) else chain.nodes
                
                summary_chains.append({
                    "chain_id": chain_id,
                    "seed_instance_id": ext.instance_id,
                    "repo": ext.seed_metadata.get("repo", "Unknown"),
                    "fix_intents": fix_intents,
                    "path_length": len(nodes),
                    "symptom_node": nodes[0].get("node_id") if isinstance(nodes[0], dict) else nodes[0].node_id if nodes else "N/A",
                    "root_cause_node": nodes[-1].get("node_id") if isinstance(nodes[-1], dict) else nodes[-1].node_id if nodes else "N/A"
                })
        
        ensure_dir(self.output_dir / "1_extraction")
        save_json(summary_chains, self.output_dir / "1_extraction" / "summary.json")
        
        # 3. 保存合成汇总 (2_synthesis/summary.json)
        if synthesis:
            summary_synthesis = []
            for s in synthesis:
                # 获取原始 Seed 信息
                seed_ext = extraction_map.get(s.seed_id)
                seed_metadata = seed_ext.seed_metadata if seed_ext else {}
                seed_chains = seed_ext.mined_data.get("extracted_chains", []) if seed_ext else []
                
                meta = s.metadata if hasattr(s, "metadata") else {}
                chain = meta.get("proposed_chain", [])
                
                summary_synthesis.append({
                    # --- 来源信息 (Seed Info) ---
                    "seed_instance_id": s.seed_id,
                    "seed_metadata": seed_metadata,
                    "seed_extraction_chains": seed_chains,
                    "seed_fix_intent": s.fix_intent,
                    
                    # --- 合成信息 (Synthetic Info - Clearly Marked) ---
                    "synthetic_instance_id": s.instance_id,
                    "synthetic_repo": s.repo,
                    "synthetic_candidate_info": meta.get("candidate_info"),
                    "synthetic_problem_statement": s.problem_statement,
                    "synthetic_patch": s.patch,
                    "synthetic_chain": chain,
                    "synthetic_chain_depth": len(chain),
                    "synthetic_injection_patch": meta.get("injection_patch", ""),
                    "synthetic_test_case_code": meta.get("test_case_code", ""),
                    "synthetic_injection_strategy": s.injection_strategy,
                    "synthetic_timestamp": meta.get("timestamp"),

                    # --- 验证信息 ---
                    "validation_status": meta.get("validation", {}).get("status", "skipped"),
                    "is_valid_bug": meta.get("is_valid", False)
                })

            ensure_dir(self.output_dir / "2_synthesis")
            save_json(summary_synthesis, self.output_dir / "2_synthesis" / "summary.json")
            
            # 4. 保存最终的标准数据集 (2_synthesis/final_dataset.json)
            final_dataset = []
            for s in synthesis:
                swe_data = s.to_swe_bench()
                # 增强 metadata 标注
                seed_ext = extraction_map.get(s.seed_id)
                swe_data["metadata"].update({
                    "is_synthetic": True,
                    "seed_instance_id": s.seed_id,
                    "seed_fix_intent": s.fix_intent,
                    "synthetic_chain": s.metadata.get("proposed_chain", []),
                    "synthetic_injection_patch": s.metadata.get("injection_patch", ""),
                    "synthetic_test_case_code": s.metadata.get("test_case_code", "")
                })
                final_dataset.append(swe_data)
            
            save_json(final_dataset, self.output_dir / "2_synthesis" / "final_dataset.json")
        
        # 4. 运行报告 (report.json)
        seeds_total = len(extractions)
        seeds_with_synthetic = len({s.seed_id for s in synthesis}) if synthesis else 0
        coverage_rate = (seeds_with_synthetic / seeds_total * 100.0) if seeds_total else 0.0
        avg_synthetic_per_seed = (len(synthesis) / seeds_total) if seeds_total else 0.0

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_input_instances": seeds_total,
                "total_extracted_chains": len(summary_chains),
                "total_synthetic_instances": len(synthesis),
                # Backward compatible: used to be (synthetic/seed)*100 which could exceed 100%.
                # Now it represents "seed coverage": % of seeds that produced >=1 synthetic instance.
                "success_rate": f"{coverage_rate:.1f}%",
                "seeds_with_synthetic": seeds_with_synthetic,
                "seed_coverage_rate": f"{coverage_rate:.1f}%",
                "avg_synthetic_per_seed": round(avg_synthetic_per_seed, 3),
            },
            "models": {
                "analyzer": self.config.analyzer_llm.model,
                "synthesis": self.config.synthesis_llm.model
            },
            "output_structure": {
                "extractions": "1_extraction/summary.json",
                "synthesis_summary": "2_synthesis/summary.json" if synthesis else "N/A",
                "synthetic_dataset": "2_synthesis/final_dataset.json" if synthesis else "N/A",
                "agent_trace": "logs/agent_trace_<instance_id>.json"
            }
        }
        save_json(report, self.output_dir / "report.json")

    def _load_instances(self, data_path: str) -> List[SWEBenchInstance]:
        """
        加载 SWE-Bench 数据
        """
        logger.info(f"加载数据: {data_path}")
        try:
            raw_data = load_json(data_path)
            instances = []
            if isinstance(raw_data, list):
                for item in raw_data:
                    try:
                        instances.append(SWEBenchInstance(**item))
                    except Exception as e:
                        logger.warning(f"跳过无效实例: {e}")
            else:
                logger.error(f"数据格式错误,期望列表,实际为: {type(raw_data)}")
                return []
            logger.info(f"成功加载 {len(instances)} 个实例")
            return instances
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return []

    def _run_localization_for_instance(
        self, instance: SWEBenchInstance, graph: nx.DiGraph
    ) -> Optional[LocalizationResult]:
        """
        Stage 0: 为缺少 raw_output_loc 的实例运行定位，支持缓存。
        """
        # Check cache first
        cached = load_localization_cache(instance.instance_id, self.localization_cache_dir)
        if cached is not None:
            logger.debug(f"[Stage 0] 从缓存加载: {instance.instance_id}")
            return cached

        try:
            result = self.fault_localizer.localize(
                instance_id=instance.instance_id,
                repo=instance.repo,
                problem_statement=instance.problem_statement,
                patch=instance.patch,
                test_patch=instance.test_patch,
                graph=graph,
            )
            # Persist to cache
            save_localization_cache(result, self.localization_cache_dir)
            return result
        except Exception as e:
            logger.warning(f"[Stage 0] 定位失败 {instance.instance_id}: {e}")
            return None

    def _extract_single_instance(self, instance: SWEBenchInstance, worker_id=None) -> ExtractionResult:
        """提取单个实例的链路。worker_id 用于仓库内并行时隔离 repo checkout 目录。"""
        start_time = time.time()
        repo_dir = self._prepare_repository(instance, worker_id=worker_id)

        # 图缓存始终共享 (不按 worker_id 分隔)，避免重复构建
        graph_cache_dir = Path(self.config.data.cache_dir) / "graphs"
        graph_cache_dir.mkdir(parents=True, exist_ok=True)
        graph_cache_path = (
            graph_cache_dir
            / f"{instance.repo.replace('/', '_')}__{instance.base_commit[:8]}_{GRAPH_CACHE_VERSION}.pkl"
        )

        # 图缓存加文件锁，防止多 worker 并行构建同一 commit 的图时竞态写入
        # save_pickle 已改为原子写入 (tmpfile + os.replace)，快路径安全
        graph_lock_path = Path(str(graph_cache_path) + ".lock")
        import filelock

        graph = None
        if graph_cache_path.exists():
            try:
                logger.info(f"从缓存加载图: {graph_cache_path}")
                graph = load_pickle(graph_cache_path)
            except Exception as e:
                logger.warning(f"缓存图加载失败，将重建: {e}")

        if graph is None:
            with filelock.FileLock(graph_lock_path, timeout=600):
                # Double-check: 另一个 worker 可能在等锁期间已构建完成
                if graph_cache_path.exists():
                    try:
                        logger.info(f"图已由其他 worker 构建，直接加载: {graph_cache_path}")
                        graph = load_pickle(graph_cache_path)
                    except Exception:
                        graph = None
                if graph is None:
                    # 尝试寻找同一仓库的旧图作为增量构建的基础
                    base_graph = None
                    try:
                        graph_cache_dir = Path(self.config.data.cache_dir) / "graphs"
                        repo_clean = instance.repo.replace('/', '_')
                        candidate_graphs = list(graph_cache_dir.glob(f"{repo_clean}__*_{GRAPH_CACHE_VERSION}.pkl"))

                        if candidate_graphs:
                            # 选最近修改的图作为增量基础 (比 hash 前缀相似度更可靠)
                            best_graph_path = max(candidate_graphs, key=lambda p: p.stat().st_mtime)
                            logger.info(f"基于最近的图进行增量构建: {best_graph_path.name}")
                            base_graph = load_pickle(best_graph_path)
                    except Exception as e:
                        logger.warning(f"尝试加载基准图失败: {e}")

                    logger.info(f"构建新的依赖图 (增量模式: {base_graph is not None})")
                    graph = self.graph_builder.build_graph(
                        str(repo_dir),
                        entry_files=self._get_entry_files(instance),
                        base_graph=base_graph
                    )
                    save_pickle(graph, graph_cache_path)

        # --- Stage 0: 定位补全 (为缺少 raw_output_loc 的实例生成定位数据) ---
        if not instance.raw_output_loc:
            loc_result = self._run_localization_for_instance(instance, graph)
            if loc_result:
                # 筛选: 只接受 patch_coverage == 100% 的定位结果
                patch_cov = loc_result.quality.get("patch_coverage", 0.0)
                if patch_cov < 1.0:
                    logger.warning(
                        f"[Stage 0] 定位质量不足，跳过: {instance.instance_id} "
                        f"(patch_coverage={patch_cov:.0%})"
                    )
                    return ExtractionResult(
                        instance_id=instance.instance_id,
                        seed_metadata={
                            "repo": instance.repo,
                            "base_commit": instance.base_commit,
                            "problem_statement": instance.problem_statement,
                            "patch": instance.patch,
                            "test_patch": instance.test_patch,
                            "source_file": "data/swebench_converted.json",
                        },
                        mined_data={"extracted_chains": [], "fix_intents": []},
                        runtime_metadata={"skipped_reason": f"low_patch_coverage_{patch_cov:.2f}"},
                    )
                instance.raw_output_loc = loc_result.raw_output_loc
                instance.found_files = loc_result.found_files or instance.found_files
                instance.found_modules = loc_result.found_modules or instance.found_modules
                instance.found_entities = loc_result.found_entities or instance.found_entities
                logger.info(f"[Stage 0] 已补全定位数据: {instance.instance_id}")
        else:
            logger.debug(f"[Stage 0] 跳过已有定位数据: {instance.instance_id}")

        chains = self.extractor.extract_chains(instance, graph)
        graph_stats = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
        }
        
        # 提取 Fix Intents (支持多个)
        fix_intents = []
        for chain in chains:
            if "repair_chain" in chain.extraction_metadata:
                rc_meta = chain.extraction_metadata["repair_chain"]
                if isinstance(rc_meta, dict) and "repair_chain" in rc_meta:
                    fix_intents.append(rc_meta["repair_chain"])
        
        return ExtractionResult(
            instance_id=instance.instance_id,
            seed_metadata={
                "repo": instance.repo,
                "base_commit": instance.base_commit,
                "problem_statement": instance.problem_statement,
                "patch": instance.patch,
                "test_patch": instance.test_patch,
                "source_file": "data/swebench_converted.json" # 说明原始数据来源
            },
            mined_data={
                "extracted_chains": [c.model_dump() for c in chains],
                "fix_intents": fix_intents
            },
            runtime_metadata={
                "graph_stats": graph_stats,
                "execution_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _prepare_repository(self, instance: Any, apply_fix: bool = False, worker_id=None) -> Path:
        """准备仓库。worker_id 非 None 时使用独立的 repo 目录以支持仓库内并行。"""
        if hasattr(instance, "repo"):
            repo_name = instance.repo
            base_commit = instance.base_commit
            patch = getattr(instance, "patch", None)
        else:
            # 兼容 ExtractionResult 的字典结构
            metadata = getattr(instance, "seed_metadata", {})
            repo_name = metadata.get("repo", "")
            base_commit = metadata.get("base_commit", "")
            patch = metadata.get("patch", None)

        repo_url = f"https://github.com/{repo_name}.git"
        repo_dir_name = repo_name.replace("/", "_")
        if worker_id is not None:
            repo_dir_name = f"{repo_dir_name}_w{worker_id}"
        cache_dir = Path(self.config.data.cache_dir) / "repos" / repo_dir_name
        
        try:
            repo = clone_repository(repo_url, cache_dir, commit=base_commit)
            
            if apply_fix and patch:
                logger.info(f"正在对仓库应用修复补丁以进入安全状态...")
                try:
                    # 将补丁写入临时文件并应用，避免 GitPython 的 stdin 处理问题
                    import tempfile
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as tmp:
                        tmp.write(patch)
                        tmp_path = tmp.name
                    
                    try:
                        repo.git.apply(tmp_path)
                        logger.info("修复补丁应用成功")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                except Exception as e:
                    logger.warning(f"修复补丁应用失败 (可能已应用): {e}")
                    
            return cache_dir
        except Exception as e:
            logger.error(f"仓库准备失败 {repo_name}: {e}")
            return ensure_dir(cache_dir)
    
    def _get_entry_files(self, instance: SWEBenchInstance) -> List[str]:
        """获取入口文件列表"""
        from ..core.utils import parse_diff_hunks
        hunks = parse_diff_hunks(instance.patch)
        return list(set(hunk["file"] for hunk in hunks if hunk["file"]))
