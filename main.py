"""
TraceGen 主入口
"""
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import sys
import os
import logging
import re
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


from src.pipeline.runner import TraceGenPipeline
from src.core.logging_utils import configure_logging


def load_env_file():
    """
    加载 .env 文件中的环境变量
    """
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()
        logger.info(f"已加载环境变量: {env_file}")


def _redact_config_yaml(yaml_text: str) -> str:
    """
    Best-effort redaction for secrets before logging configs.

    This prevents accidentally leaking API keys in logs and artifacts while keeping the
    rest of the config readable.
    """
    if not yaml_text:
        return yaml_text

    # 1) Mask common YAML "api_key: <value>" lines.
    redacted = re.sub(r"(?im)^(\s*api_key\s*:\s*)(.+)$", r"\1<REDACTED>", yaml_text)

    # 2) Mask any inline OpenAI-style key patterns that may appear elsewhere.
    redacted = re.sub(r"sk-[A-Za-z0-9_-]{10,}", "sk-<REDACTED>", redacted)

    return redacted


def _write_run_records(*, output_dir: Path, cfg: DictConfig) -> None:
    """
    Persist run records for reproducibility.

    Writes:
    - run_manifest.json: command line, runtime flags, and Hydra overrides (best-effort redacted)
    - config_resolved.yaml: fully-resolved Hydra config (redacted)
    - command.txt: human-friendly repro snippet
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    # Hydra metadata (best-effort)
    hydra_info: dict[str, object] = {}
    try:
        from hydra.core.hydra_config import HydraConfig  # type: ignore

        hc = HydraConfig.get()
        hydra_info = {
            "config_name": getattr(getattr(hc, "job", None), "config_name", None),
            "overrides": {
                "task": list(getattr(getattr(hc, "overrides", None), "task", []) or []),
                "hydra": list(getattr(getattr(hc, "overrides", None), "hydra", []) or []),
            },
            "runtime": {
                "cwd": str(getattr(getattr(hc, "runtime", None), "cwd", "")),
                "output_dir": str(getattr(getattr(hc, "runtime", None), "output_dir", "")),
            },
        }
    except Exception:
        hydra_info = {}

    # Build manifest
    argv = list(sys.argv)
    command_line = " ".join(argv)
    redacted_command_line = _redact_config_yaml(command_line)
    redacted_overrides = hydra_info.get("overrides", {}) if isinstance(hydra_info, dict) else {}
    if isinstance(redacted_overrides, dict):
        for k, v in list(redacted_overrides.items()):
            if isinstance(v, list):
                redacted_overrides[k] = [_redact_config_yaml(str(x)) for x in v]

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " ").strip(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "command_line": redacted_command_line,
        "argv": argv,
        "hydra": {
            "config_name": (hydra_info.get("config_name") if isinstance(hydra_info, dict) else None),
            "overrides": redacted_overrides,
        },
        "tracegen": {
            "output_dir": str(output_dir),
            "data": {
                "swebench_path": str(getattr(getattr(cfg, "data", None), "swebench_path", "")),
                "target_instances_path": str(getattr(getattr(cfg, "data", None), "target_instances_path", "")),
                "cache_dir": str(getattr(getattr(cfg, "data", None), "cache_dir", "")),
            },
            "runtime": dict(getattr(cfg, "runtime", {}) or {}),
            "validation": {
                "enabled": bool(getattr(getattr(cfg, "validation", None), "enabled", False)),
                "test_selection_mode": str(getattr(getattr(cfg, "validation", None), "test_selection_mode", "")),
                "timeout": int(getattr(getattr(cfg, "validation", None), "timeout", 0) or 0),
            },
            "evaluation": {
                "note": "External SWE-agent evaluation; see scripts/sweagent/",
            },
        },
    }

    # Persist manifest
    try:
        (output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except (OSError, IOError) as e:
        logger.debug(f"Failed to write run_manifest.json: {e}")

    # Persist fully-resolved config snapshot (redacted)
    try:
        resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
        (output_dir / "config_resolved.yaml").write_text(
            _redact_config_yaml(resolved_yaml),
            encoding="utf-8",
        )
    except (OSError, IOError) as e:
        logger.debug(f"Failed to write config_resolved.yaml: {e}")

    # Human-friendly repro snippet
    try:
        task_overrides: list[str] = []
        if isinstance(hydra_info, dict):
            ov = hydra_info.get("overrides", {})
            if isinstance(ov, dict):
                raw_task = ov.get("task", [])
                if isinstance(raw_task, list):
                    task_overrides = [str(x) for x in raw_task if x is not None]
        lines = [
            f"# TraceGen run record (UTC): {manifest['timestamp_utc']}",
            f"# CWD: {manifest['cwd']}",
            f"# Python: {manifest['python_executable']}",
            "",
            "# Original command (redacted):",
            redacted_command_line,
            "",
        ]
        if task_overrides:
            lines += [
                "# Hydra task overrides (redacted):",
                *[f"- {_redact_config_yaml(x)}" for x in task_overrides],
                "",
            ]
        lines += [
            "# Output directory:",
            str(output_dir),
            "",
        ]
        (output_dir / "command.txt").write_text("\n".join(lines), encoding="utf-8")
    except (OSError, IOError, TypeError) as e:
        logger.debug(f"Failed to write command.txt: {e}")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    主函数
    
    Args:
        cfg: Hydra 配置对象
    """
    # 加载环境变量
    load_env_file()
    
    # 配置日志（统一 Loguru + stdlib logging 输出格式）
    configure_logging(verbose=bool(cfg.runtime.verbose), stream=sys.stderr)

    # Reproducibility (best-effort)
    try:
        import random

        seed_value = cfg.runtime.get("seed", None)
        if seed_value is not None and str(seed_value).strip() != "":
            random.seed(int(seed_value))
    except (ValueError, TypeError, AttributeError) as e:
        logger.debug(f"Failed to set random seed: {e}")
    logger.info("TraceGen 配置:")
    logger.info("\n" + _redact_config_yaml(OmegaConf.to_yaml(cfg)))
    
    # 创建并运行流水线
    pipeline = TraceGenPipeline(cfg)

    # Persist reproducibility artifacts as early as possible.
    try:
        out_dir = getattr(pipeline, "output_dir", None)
        out_dir_path = Path(str(out_dir)) if out_dir else Path(str(cfg.data.output_dir))
        _write_run_records(output_dir=out_dir_path, cfg=cfg)
    except (OSError, IOError, TypeError, AttributeError) as e:
        logger.debug(f"Failed to write run records: {e}")
    all_instances = pipeline._load_instances(cfg.data.swebench_path)
    
    # 处理针对性运行逻辑
    target_instances_path = Path(cfg.data.get("target_instances_path", ""))
    filtered_instances = all_instances
    
    if target_instances_path and target_instances_path.exists():
        import json
        try:
            with open(target_instances_path, "r", encoding="utf-8") as f:
                target_ids = json.load(f)
            
            if isinstance(target_ids, list) and target_ids:
                filtered_instances = [inst for inst in all_instances if inst.instance_id in target_ids]
                logger.info(f"从 {target_instances_path} 加载了 {len(filtered_instances)} 个目标实例")
            else:
                logger.info(f"{target_instances_path} 为空或格式不正确，将运行所有实例")
        except Exception as e:
            logger.warning(f"加载目标实例列表失败 ({e})，将运行所有实例")
    else:
        logger.info("未指定目标实例列表或文件不存在，将运行所有实例")

    # Optional cap on number of input instances (kept for backward compatibility with existing configs).
    try:
        batch_size = int(cfg.runtime.get("batch_size", 0) or 0)
        if batch_size > 0 and len(filtered_instances) > batch_size:
            logger.info(f"限制输入实例数量(batch_size): {batch_size}/{len(filtered_instances)}")
            filtered_instances = filtered_instances[:batch_size]
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"batch_size 配置解析失败: {e}，将处理所有实例")

    # Check if we have any instances to process
    if not filtered_instances:
        logger.warning("没有可运行的实例，程序退出")
        return

    results = pipeline.run(instances=filtered_instances)
    
    # 输出结果摘要
    logger.info("\n" + "=" * 60)
    logger.info("运行结果摘要:")
    if isinstance(results, dict) and "input_instances" in results:
        # Full pipeline mode
        logger.info(f"  输入实例: {results.get('input_instances')}")
        logger.info(f"  提取链路: {results.get('extracted_chains')}")
        logger.info(f"  合成实例: {results.get('synthetic_instances')}")
        logger.info(f"  输出目录: {results.get('output_dir')}")
    else:
        # validation_only_dir mode returns validation stats (no extraction/synthesis counts)
        logger.info("  模式: validation_only_dir")
        if isinstance(results, dict):
            logger.info(f"  合成实例总数: {results.get('total')}")
            logger.info(f"  valid: {results.get('valid')} | invalid: {results.get('invalid')} | timeout: {results.get('timeout')} | error: {results.get('error')}")
        logger.info(f"  输出目录: {str(pipeline.output_dir)}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
