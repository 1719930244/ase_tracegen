#!/usr/bin/env python
"""
测试验证阶段的脚本。

使用已有的合成结果进行验证测试，避免重复合成开销。

用法:
    python scripts/test_validation.py [synthesis_result_path]

如果不提供路径，会使用最近的合成结果。
"""
import sys
import json
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.core.structures import SynthesisResult
from src.modules.validation.adapter import ValidationAdapter
from src.modules.validation.validator import Validator, ValidationConfig


def find_latest_synthesis_result() -> Path:
    """查找最近的合成结果文件"""
    project_root = Path(__file__).resolve().parent.parent
    outputs_dir = project_root.parent / "tracegen-outputs"

    # 按日期和时间排序，找到最新的
    date_dirs = sorted(outputs_dir.glob("*"), reverse=True)
    for date_dir in date_dirs:
        if not date_dir.is_dir():
            continue
        time_dirs = sorted(date_dir.glob("*"), reverse=True)
        for time_dir in time_dirs:
            if not time_dir.is_dir():
                continue
            # 查找合成结果
            synthesis_dir = time_dir / "2_synthesis" / "details"
            if synthesis_dir.exists():
                results = list(synthesis_dir.glob("synthetic_*.json"))
                if results:
                    return results[0]

    raise FileNotFoundError("未找到合成结果文件")


def load_synthesis_result(path: Path) -> SynthesisResult:
    """加载合成结果"""
    with open(path) as f:
        data = json.load(f)

    # 构建 SynthesisResult（填充所有必需字段）
    return SynthesisResult(
        instance_id=data.get("instance_id", ""),
        repo=data.get("repo", ""),
        base_commit=data.get("base_commit", ""),
        problem_statement=data.get("problem_statement", ""),
        seed_id=data.get("seed_id", ""),
        patch=data.get("patch", ""),
        fix_intent=data.get("fix_intent", ""),
        injection_strategy=data.get("injection_strategy", ""),
        FAIL_TO_PASS=data.get("FAIL_TO_PASS", []),
        PASS_TO_PASS=data.get("PASS_TO_PASS", []),
        metadata=data.get("metadata", {}),
    )


def main():
    # 确定使用哪个合成结果
    if len(sys.argv) > 1:
        result_path = Path(sys.argv[1])
    else:
        result_path = find_latest_synthesis_result()

    logger.info(f"使用合成结果: {result_path}")

    # 加载合成结果
    result = load_synthesis_result(result_path)
    logger.info(f"实例 ID: {result.instance_id}")
    logger.info(f"仓库: {result.repo}")

    # 初始化验证组件
    config = {
        "enabled": True,
        "default_image": "docker.io/library/python:3.10-slim",
        "timeout": 300,
        "memory_limit": "4g",
        "auto_pull": True,
        "dockerhub_org": "swebench",
    }

    adapter = ValidationAdapter(config)
    instance = adapter.adapt(result)

    logger.info(f"镜像: {instance.get('image_name')}")
    logger.info(f"测试命令: {instance.get('test_cmd')}")
    logger.info(f"Patch 大小: {len(instance.get('patch', ''))} bytes")

    # 显示 patch 内容
    patch = instance.get("patch", "")
    if patch:
        logger.info("Patch 内容预览:")
        for line in patch.split("\n")[:20]:
            print(f"  {line}")

    # 创建验证器并运行
    validator_config = ValidationConfig(
        timeout=300,
        memory_limit="4g",
        platform="linux/amd64",
        clean_containers=True,
    )

    validator = Validator(config=validator_config, profile=None)

    # 创建输出目录
    output_dir = Path("/tmp/tracegen_validation_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"验证日志目录: {output_dir}")
    logger.info("开始验证...")

    # 运行验证（validate 只接收 instance 参数）
    validation_result = validator.validate(instance)

    # 显示结果（ValidationResult 是 dataclass）
    logger.info("=" * 60)
    logger.info("验证结果:")
    logger.info(f"  状态: {validation_result.status}")
    logger.info(f"  FAIL_TO_PASS: {validation_result.FAIL_TO_PASS}")
    logger.info(f"  PASS_TO_PASS: {validation_result.PASS_TO_PASS}")
    logger.info(f"  FAIL_TO_FAIL: {validation_result.FAIL_TO_FAIL}")
    logger.info(f"  PASS_TO_FAIL: {validation_result.PASS_TO_FAIL}")
    logger.info("=" * 60)

    # 保存结果（转换为 dict）
    result_dict = {
        "instance_id": validation_result.instance_id,
        "status": validation_result.status.value if hasattr(validation_result.status, 'value') else str(validation_result.status),
        "FAIL_TO_PASS": validation_result.FAIL_TO_PASS,
        "PASS_TO_PASS": validation_result.PASS_TO_PASS,
        "FAIL_TO_FAIL": validation_result.FAIL_TO_FAIL,
        "PASS_TO_FAIL": validation_result.PASS_TO_FAIL,
    }
    result_file = output_dir / "validation_result.json"
    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=2)
    logger.info(f"结果已保存: {result_file}")


if __name__ == "__main__":
    main()
