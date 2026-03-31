#!/usr/bin/env python3
"""
转换脚本：将 loc_claude_outputs.jsonl 转换为 swebench_lite.json 格式

输入：loc_claude_outputs.jsonl (JSONL格式，每行是一个dict)
输出：swebench_lite.json (JSON格式，数组)
"""
import json
from pathlib import Path

def convert_loc_to_swebench(input_file: str, output_file: str):
    """
    将 loc_claude_outputs.jsonl 转换为 swebench_lite.json
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSON文件路径
    """
    data_list = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                loc_data = json.loads(line)
                
                # 构建swebench格式
                swebench_item = {
                    "repo": loc_data["meta_data"]["repo"],
                    "instance_id": loc_data["instance_id"],
                    "base_commit": loc_data["meta_data"]["base_commit"],
                    "patch": loc_data["meta_data"]["patch"],
                    "test_patch": loc_data["meta_data"].get("test_patch", ""),
                    "problem_statement": loc_data["meta_data"]["problem_statement"],
                    "hints_text": loc_data["meta_data"].get("hints_text", ""),
                    "created_at": loc_data["meta_data"].get("created_at", ""),
                    "version": loc_data["meta_data"].get("version", ""),
                    "FAIL_TO_PASS": loc_data["meta_data"].get("FAIL_TO_PASS", "[]"),
                    "PASS_TO_PASS": loc_data["meta_data"].get("PASS_TO_PASS", "[]"),
                    "environment_setup_commit": loc_data["meta_data"].get("environment_setup_commit", ""),
                    "found_files": loc_data.get("found_files", []),
                    "found_modules": loc_data.get("found_modules", []),
                    "found_entities": loc_data.get("found_entities", []),
                    "raw_output_loc": loc_data.get("raw_output_loc", []),
                }
                
                # 添加gt_file_changes（如果存在）
                if "gt_file_changes" in loc_data["meta_data"]:
                    swebench_item["gt_file_changes"] = loc_data["meta_data"]["gt_file_changes"]
                
                data_list.append(swebench_item)
                print(f"✓ 处理第 {line_num} 行: {loc_data['instance_id']}")
                
            except json.JSONDecodeError as e:
                print(f"✗ 第 {line_num} 行JSON解析失败: {e}")
                continue
            except KeyError as e:
                print(f"✗ 第 {line_num} 行缺少字段 {e}: {line[:100]}")
                continue
    
    # 保存为JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data_list[:10], f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 转换完成！")
    print(f"  输入文件: {input_file}")
    print(f"  输出文件: {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert loc_claude_outputs.jsonl to SWE-bench format")
    parser.add_argument("input_file", help="Input JSONL file path")
    parser.add_argument("output_file", help="Output JSON file path")
    args = parser.parse_args()

    convert_loc_to_swebench(args.input_file, args.output_file)
