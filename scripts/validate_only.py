"""
单独验证已合成的 Bug 实例

用法:
    python scripts/validate_only.py --synthesis_dir <合成结果目录>

示例:
    python scripts/validate_only.py --synthesis_dir ../tracegen-outputs/2026-01-28/16-31-45
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.utils import load_json, save_json, ensure_dir
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class SimpleSynthesisResult:
    """简化的合成结果，用于验证"""
    instance_id: str
    repo: str
    patch: str
    seed_id: str
    base_commit: str = ""
    FAIL_TO_PASS: List[str] = field(default_factory=list)
    PASS_TO_PASS: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
from src.modules.validation.adapter import ValidationAdapter
from src.modules.validation.validator import Validator
from src.modules.validation.constants import ValidationConfig, ValidationStatus
from src.modules.validation.profiles.python import PythonProfile, UnittestProfile
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console


console = Console()


def load_synthesis_results(synthesis_dir: Path) -> list:
    """
    从合成目录加载已合成的实例

    支持两种格式:
    1. final_dataset.json (完整数据集)
    2. details/*.json (单个实例文件)
    """
    results = []

    # 尝试加载 final_dataset.json
    final_dataset = synthesis_dir / "2_synthesis" / "final_dataset.json"
    if final_dataset.exists():
        logger.info(f"从 final_dataset.json 加载合成结果")
        data = load_json(final_dataset)
        for item in data:
            # 转换为 SynthesisResult 兼容格式
            result = _convert_to_synthesis_result(item)
            if result:
                results.append(result)
        return results

    # 尝试从 details 目录加载
    details_dir = synthesis_dir / "2_synthesis" / "details"
    if details_dir.exists():
        logger.info(f"从 details 目录加载合成结果")
        for json_file in sorted(details_dir.glob("*.json")):
            try:
                data = load_json(json_file)
                result = _convert_to_synthesis_result(data)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning(f"加载 {json_file} 失败: {e}")

    return results


def _convert_to_synthesis_result(data: dict) -> SimpleSynthesisResult:
    """将 JSON 数据转换为 SimpleSynthesisResult 对象"""
    try:
        # 提取必要字段
        instance_id = data.get("instance_id") or data.get("synthetic_instance_id", "unknown")
        repo = data.get("repo", "django/django")

        # 获取 patch (优先使用 injection_patch)
        patch = data.get("patch") or data.get("synthetic_injection_patch", "")

        # 获取 seed_id (优先从 metadata 中获取)
        seed_id = data.get("seed_id") or data.get("seed_instance_id", "")
        metadata = data.get("metadata", {})
        if not seed_id and metadata:
            seed_id = metadata.get("seed_id") or metadata.get("seed_instance_id", "")

        # 获取 base_commit
        base_commit = data.get("base_commit", "")
        if not base_commit and metadata:
            seed_metadata = metadata.get("seed_metadata", {})
            base_commit = seed_metadata.get("base_commit", "")

        # 处理 FAIL_TO_PASS 和 PASS_TO_PASS (可能是 JSON 字符串)
        fail_to_pass = data.get("FAIL_TO_PASS", [])
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except (json.JSONDecodeError, ValueError):
                fail_to_pass = [fail_to_pass]

        pass_to_pass = data.get("PASS_TO_PASS", [])
        if isinstance(pass_to_pass, str):
            try:
                pass_to_pass = json.loads(pass_to_pass)
            except (json.JSONDecodeError, ValueError):
                pass_to_pass = [pass_to_pass]

        # 构建 metadata
        metadata = data.get("metadata", {})
        if not metadata:
            metadata = {
                "injection_patch": data.get("synthetic_injection_patch", patch),
                "target_node": data.get("metadata", {}).get("target_node", ""),
                "base_commit": base_commit,
            }

        # 确保 injection_patch 在 metadata 中
        if "injection_patch" not in metadata:
            metadata["injection_patch"] = data.get("synthetic_injection_patch", patch)

        # 确保 metadata 包含必要字段用于镜像选择
        if "seed_id" not in metadata and seed_id:
            metadata["seed_id"] = seed_id

        # 创建 SimpleSynthesisResult
        result = SimpleSynthesisResult(
            instance_id=instance_id,
            repo=repo,
            patch=patch,
            seed_id=seed_id,
            base_commit=base_commit,
            FAIL_TO_PASS=fail_to_pass if isinstance(fail_to_pass, list) else [],
            PASS_TO_PASS=pass_to_pass if isinstance(pass_to_pass, list) else [],
            metadata=metadata,
        )

        return result
    except Exception as e:
        logger.error(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_validation(
    synthesis_results: list,
    output_dir: Path,
    timeout: int = 300,
    verbose: bool = True,
) -> dict:
    """
    运行验证阶段

    Args:
        synthesis_results: SynthesisResult 列表
        output_dir: 输出目录
        timeout: 超时时间 (秒)
        verbose: 详细输出

    Returns:
        验证统计结果
    """
    validation_dir = output_dir / "3_validation"
    ensure_dir(validation_dir)
    logs_dir = validation_dir / "logs"
    ensure_dir(logs_dir)

    # 初始化验证组件
    validation_config = {
        "enabled": True,
        "timeout": timeout,
        "memory_limit": "4g",
        "clean_containers": True,
        "verbose": verbose,
        "auto_pull": True,
        "dockerhub_org": "swebench",
    }

    adapter = ValidationAdapter(validation_config)

    val_config = ValidationConfig(
        mode="injection",
        timeout=timeout,
        memory_limit="4g",
        clean_containers=True,
        verbose=verbose,
        enforce_chain_coverage=bool(validation_config.get("enforce_chain_coverage", False)),
        min_chain_coverage=float(validation_config.get("min_chain_coverage", 0.34) or 0.34),
        require_target_node_in_traceback=bool(validation_config.get("require_target_node_in_traceback", False)),
        log_dir=logs_dir,
    )

    # 选择合适的 Profile 解析测试输出
    # Django 的 ./tests/runtests.py 输出更接近 unittest 格式
    is_django = False
    for r in synthesis_results[:5]:
        repo = getattr(r, "repo", "") or ""
        seed_id = getattr(r, "seed_id", "") or ""
        if "django" in repo.lower() or "django" in seed_id.lower():
            is_django = True
            break

    profile = UnittestProfile(image_name="python:3.10-slim") if is_django else PythonProfile(image_name="python:3.10-slim")
    validator = Validator(profile=profile, config=val_config)

    # 统计
    stats = {
        "total": len(synthesis_results),
        "valid": 0,
        "invalid": 0,
        "missing_image": 0,
        "error": 0,
        "timeout": 0,
    }

    validated_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    ) as progress:
        task = progress.add_task(
            f"验证缺陷 (0/{len(synthesis_results)})",
            total=len(synthesis_results)
        )

        for i, result in enumerate(synthesis_results):
            try:
                # 适配实例
                instance_dict = adapter.adapt(result)

                logger.info(f"验证实例: {result.instance_id}")
                logger.info(f"  镜像: {instance_dict.get('image_name', 'N/A')}")
                logger.info(f"  测试命令: {instance_dict.get('test_cmd', 'N/A')[:80]}...")

                # 执行验证
                val_result = validator.validate(instance_dict)

                # 更新统计
                status_str = val_result.status.value
                if val_result.status == ValidationStatus.VALID:
                    stats["valid"] += 1
                elif val_result.status == ValidationStatus.INVALID:
                    stats["invalid"] += 1
                elif val_result.status == ValidationStatus.MISSING_IMAGE:
                    stats["missing_image"] += 1
                elif val_result.status == ValidationStatus.TIMEOUT:
                    stats["timeout"] += 1
                else:
                    stats["error"] += 1

                # 保存单个验证结果
                save_json(
                    val_result.to_dict(),
                    validation_dir / f"{result.instance_id}_validation.json"
                )

                # 更新 result metadata
                result.metadata["validation"] = val_result.to_dict()
                result.metadata["is_valid"] = val_result.is_valid_bug()
                validated_results.append(result)

                progress.update(
                    task,
                    advance=1,
                    description=f"验证缺陷 ({i+1}/{len(synthesis_results)}) - {status_str}"
                )

                logger.info(f"  结果: {status_str}")
                if val_result.FAIL_TO_PASS:
                    logger.info(f"  FAIL_TO_PASS: {val_result.FAIL_TO_PASS}")

            except Exception as e:
                logger.error(f"验证失败 {result.instance_id}: {e}")
                stats["error"] += 1
                result.metadata["validation_error"] = str(e)
                validated_results.append(result)
                progress.update(task, advance=1)

    # 保存验证汇总
    summary = {
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "success_rate": f"{stats['valid'] / stats['total'] * 100:.1f}%" if stats["total"] > 0 else "0%",
        "results": [
            {
                "instance_id": r.instance_id,
                "is_valid": r.metadata.get("is_valid", False),
                "status": r.metadata.get("validation", {}).get("status", "unknown"),
            }
            for r in validated_results
        ]
    }
    save_json(summary, validation_dir / "validation_summary.json")

    return stats, validated_results


def main():
    parser = argparse.ArgumentParser(description="单独验证已合成的 Bug 实例")
    parser.add_argument(
        "--synthesis_dir",
        type=str,
        required=True,
        help="合成结果目录 (包含 2_synthesis/final_dataset.json)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录 (默认使用 synthesis_dir)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="单个实例验证超时时间 (秒)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--instance_ids",
        type=str,
        nargs="+",
        default=None,
        help="只验证指定的实例 ID"
    )

    args = parser.parse_args()

    synthesis_dir = Path(args.synthesis_dir)
    output_dir = Path(args.output_dir) if args.output_dir else synthesis_dir

    if not synthesis_dir.exists():
        console.print(f"[red]错误: 目录不存在: {synthesis_dir}[/red]")
        sys.exit(1)

    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO" if args.verbose else "WARNING"
    )
    logger.add(
        output_dir / "3_validation" / "validation.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG"
    )

    console.print(f"\n[bold cyan]TraceGen 单独验证模式[/bold cyan]")
    console.print(f"合成目录: {synthesis_dir}")
    console.print(f"输出目录: {output_dir}")
    console.print()

    # 加载合成结果
    console.print("[yellow]正在加载合成结果...[/yellow]")
    synthesis_results = load_synthesis_results(synthesis_dir)

    if not synthesis_results:
        console.print("[red]未找到合成结果[/red]")
        sys.exit(1)

    console.print(f"[green]找到 {len(synthesis_results)} 个合成实例[/green]")

    # 过滤指定实例
    if args.instance_ids:
        synthesis_results = [
            r for r in synthesis_results
            if r.instance_id in args.instance_ids
        ]
        console.print(f"[yellow]过滤后: {len(synthesis_results)} 个实例[/yellow]")

    # 运行验证
    console.print("\n[bold]开始验证...[/bold]\n")
    stats, validated_results = run_validation(
        synthesis_results,
        output_dir,
        timeout=args.timeout,
        verbose=args.verbose,
    )

    # 输出结果
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]验证结果汇总[/bold cyan]")
    console.print("=" * 60)
    console.print(f"总数: {stats['total']}")
    console.print(f"[green]有效 (VALID): {stats['valid']}[/green]")
    console.print(f"[yellow]无效 (INVALID): {stats['invalid']}[/yellow]")
    console.print(f"[cyan]缺少镜像 (MISSING_IMAGE): {stats['missing_image']}[/cyan]")
    console.print(f"[red]错误 (ERROR): {stats['error']}[/red]")
    console.print(f"[orange1]超时 (TIMEOUT): {stats['timeout']}[/orange1]")

    if stats["total"] > 0:
        rate = stats["valid"] / stats["total"] * 100
        console.print(f"\n[bold]验证通过率: {rate:.1f}%[/bold]")

    console.print(f"\n结果保存至: {output_dir / '3_validation'}")


if __name__ == "__main__":
    main()
