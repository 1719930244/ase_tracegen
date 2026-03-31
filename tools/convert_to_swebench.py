"""
SWE-Bench 格式转换工具
将 TraceGen 生成的合成数据转换为标准 SWE-Bench 格式
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import sys

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.structures import SWEBenchInstance
from src.core.utils import load_json, save_json


def convert_to_swebench_format(
    input_file: str, output_file: str, filter_synthetic: bool = True
) -> None:
    """
    将 TraceGen 输出转换为 SWE-Bench 标准格式
    
    Args:
        input_file: TraceGen 输出文件 (synthetic_instances.json)
        output_file: 输出文件路径
        filter_synthetic: 是否只保留合成数据
    """
    print(f"加载数据: {input_file}")
    data = load_json(input_file)
    
    # 解析为 Pydantic 对象
    instances = []
    for item in data:
        try:
            inst = SWEBenchInstance(**item)
            if filter_synthetic and not inst.synthetic:
                continue
            instances.append(inst)
        except Exception as e:
            print(f"跳过无效实例: {e}")
    
    print(f"加载了 {len(instances)} 个实例")
    
    # 转换为标准格式
    swebench_data = []
    for inst in instances:
        # 移除扩展字段,只保留标准字段
        standard_fields = {
            "instance_id": inst.instance_id,
            "repo": inst.repo,
            "base_commit": inst.base_commit,
            "problem_statement": inst.problem_statement,
            "hints_text": inst.hints_text,
            "created_at": inst.created_at,
            "version": inst.version,
            "FAIL_TO_PASS": inst.FAIL_TO_PASS,
            "PASS_TO_PASS": inst.PASS_TO_PASS,
            "environment_setup_commit": inst.environment_setup_commit,
            "patch": inst.patch,
            "test_patch": inst.test_patch,
        }
        swebench_data.append(standard_fields)
    
    # 保存
    print(f"保存到: {output_file}")
    save_json(swebench_data, output_file)
    print(f"转换完成: {len(swebench_data)} 个实例")


def validate_swebench_format(file_path: str) -> bool:
    """
    验证文件是否符合 SWE-Bench 格式
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否有效
    """
    print(f"验证格式: {file_path}")
    
    required_fields = [
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "created_at",
        "version",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "patch",
    ]
    
    try:
        data = load_json(file_path)
        
        if not isinstance(data, list):
            print("错误: 数据应为列表")
            return False
        
        for i, item in enumerate(data):
            missing_fields = [f for f in required_fields if f not in item]
            if missing_fields:
                print(f"实例 {i} 缺少字段: {missing_fields}")
                return False
        
        print(f"验证通过: {len(data)} 个实例")
        return True
        
    except Exception as e:
        print(f"验证失败: {e}")
        return False


def main():
    """
    命令行入口
    """
    parser = argparse.ArgumentParser(
        description="TraceGen 到 SWE-Bench 格式转换工具"
    )
    parser.add_argument(
        "input", type=str, help="输入文件路径 (TraceGen synthetic_instances.json)"
    )
    parser.add_argument("output", type=str, help="输出文件路径")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅验证格式,不进行转换",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="包含所有实例 (默认只保留合成数据)",
    )
    
    args = parser.parse_args()
    
    if args.validate_only:
        validate_swebench_format(args.input)
    else:
        convert_to_swebench_format(
            args.input, args.output, filter_synthetic=not args.include_all
        )
        
        # 验证输出
        print("\n验证输出文件...")
        validate_swebench_format(args.output)


if __name__ == "__main__":
    main()
