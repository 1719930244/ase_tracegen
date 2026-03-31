#!/usr/bin/env python3
"""
Phase 5: 准备 114 Django 实例的完整运行数据

功能:
1. 从 loc_claude_outputs copy.json 提取 114 个 Django 实例
2. 生成 configs/target_instances_full.json (实例 ID 列表)
3. 生成 data/swebench_converted_full.json (完整 Pipeline 格式)
4. 识别需要构建的代码图 commit
"""

import json
import os
from pathlib import Path


def load_swebench_test_info():
    """从 SWE-bench 原始数据集加载测试信息"""
    from datasets import load_dataset

    print("正在加载 SWE-bench 数据集...")
    ds = load_dataset('princeton-nlp/SWE-bench', split='test')

    test_info = {}
    for item in ds:
        instance_id = item['instance_id']
        test_info[instance_id] = {
            'FAIL_TO_PASS': item['FAIL_TO_PASS'],
            'PASS_TO_PASS': item['PASS_TO_PASS'],
        }

    print(f"已加载 {len(test_info)} 个实例的测试信息")
    return test_info


def main():
    project_root = Path(__file__).resolve().parent.parent

    # 1. 读取源数据
    print("=" * 60)
    print("Step 1: 读取 loc_claude_outputs copy.json")
    print("=" * 60)

    with open(project_root / "data/loc_claude_outputs copy.json", "r") as f:
        all_instances = json.load(f)

    print(f"总实例数: {len(all_instances)}")

    # 2. 提取 Django 实例
    django_instances = [d for d in all_instances if "django" in d.get("instance_id", "").lower()]
    print(f"Django 实例数: {len(django_instances)}")

    # 2.5 加载 SWE-bench 测试信息
    swebench_test_info = load_swebench_test_info()

    # 3. 生成 target_instances_full.json
    print("\n" + "=" * 60)
    print("Step 2: 生成 target_instances_full.json")
    print("=" * 60)

    instance_ids = [d["instance_id"] for d in django_instances]
    target_file = project_root / "configs/target_instances_full.json"
    with open(target_file, "w") as f:
        json.dump(instance_ids, f, indent=2)
    print(f"已保存: {target_file}")
    print(f"实例数: {len(instance_ids)}")

    # 4. 转换为 swebench_converted 格式
    print("\n" + "=" * 60)
    print("Step 3: 生成 swebench_converted_full.json")
    print("=" * 60)

    converted_instances = []
    missing_test_info = []
    for d in django_instances:
        meta = d.get("meta_data", {})
        instance_id = d["instance_id"]

        # 从 SWE-bench 获取测试信息
        test_info = swebench_test_info.get(instance_id, {})
        fail_to_pass = test_info.get('FAIL_TO_PASS', [])
        pass_to_pass = test_info.get('PASS_TO_PASS', [])

        if not test_info:
            missing_test_info.append(instance_id)

        converted = {
            "repo": meta.get("repo", "django/django"),
            "instance_id": instance_id,
            "base_commit": meta.get("base_commit", ""),
            "patch": meta.get("patch", ""),
            "test_patch": "",
            "problem_statement": meta.get("problem_statement", ""),
            "hints_text": "",
            "created_at": "",
            "version": "",
            "FAIL_TO_PASS": fail_to_pass,
            "PASS_TO_PASS": pass_to_pass,
            "environment_setup_commit": "",
            "found_files": d.get("found_files", []),
            "found_modules": d.get("found_modules", []),
            "found_entities": d.get("found_entities", []),
            "raw_output_loc": d.get("raw_output_loc", []),
            "gt_file_changes": meta.get("gt_file_changes", [])
        }
        converted_instances.append(converted)

    if missing_test_info:
        print(f"警告: {len(missing_test_info)} 个实例缺少测试信息")
        for iid in missing_test_info[:5]:
            print(f"  - {iid}")
        if len(missing_test_info) > 5:
            print(f"  ... 还有 {len(missing_test_info) - 5} 个")

    output_file = project_root / "data/swebench_converted_full.json"
    with open(output_file, "w") as f:
        json.dump(converted_instances, f, indent=2)
    print(f"已保存: {output_file}")
    print(f"实例数: {len(converted_instances)}")

    # 5. 统计 commit 和图文件
    print("\n" + "=" * 60)
    print("Step 4: 分析代码图状态")
    print("=" * 60)

    # 收集所有 commit
    all_commits = set()
    for d in converted_instances:
        commit = d.get("base_commit", "")[:8]
        if commit:
            all_commits.add(commit)

    print(f"唯一 commit 数: {len(all_commits)}")

    # 检查已有图
    graph_dir = project_root / "data/assets/graphs"
    existing_graphs = set()
    if graph_dir.exists():
        for f in graph_dir.glob("django_django__*_v1.pkl"):
            # 从文件名提取 commit: django_django__004b4620_v1.pkl
            commit = f.stem.replace("django_django__", "").replace("_v1", "")
            existing_graphs.add(commit)

    print(f"已有图数: {len(existing_graphs)}")

    # 计算缺失
    missing_commits = all_commits - existing_graphs
    print(f"缺失图数: {len(missing_commits)}")

    # 保存缺失 commit 列表
    missing_file = project_root / "configs/missing_commits.txt"
    with open(missing_file, "w") as f:
        for commit in sorted(missing_commits):
            f.write(commit + "\n")
    print(f"缺失 commit 列表: {missing_file}")

    # 保存所有 commit 列表 (用于向量生成)
    all_commits_file = project_root / "configs/all_django_commits.txt"
    with open(all_commits_file, "w") as f:
        for commit in sorted(all_commits):
            f.write(commit + "\n")
    print(f"所有 commit 列表: {all_commits_file}")

    # 6. 打印摘要
    print("\n" + "=" * 60)
    print("摘要")
    print("=" * 60)
    print(f"Django 实例数: {len(django_instances)}")
    print(f"唯一 commit 数: {len(all_commits)}")
    print(f"已有图: {len(existing_graphs)}")
    print(f"需构建图: {len(missing_commits)}")

    print("\n生成的文件:")
    print(f"  1. configs/target_instances_full.json ({len(instance_ids)} IDs)")
    print(f"  2. data/swebench_converted_full.json ({len(converted_instances)} instances)")
    print(f"  3. configs/missing_commits.txt ({len(missing_commits)} commits)")
    print(f"  4. configs/all_django_commits.txt ({len(all_commits)} commits)")

    # 检查已有向量
    embedding_dir = project_root / "data/assets/embeddings/django_django/commits"
    existing_embeddings = set()
    if embedding_dir.exists():
        for f in embedding_dir.glob("*.json"):
            existing_embeddings.add(f.stem)

    missing_embeddings = all_commits - existing_embeddings
    print(f"\n向量状态:")
    print(f"  已有向量: {len(existing_embeddings)}")
    print(f"  需生成向量: {len(missing_embeddings)}")

    # 保存需要生成向量的 commit 列表
    if missing_embeddings:
        missing_emb_file = project_root / "configs/commits_need_embeddings.txt"
        with open(missing_emb_file, "w") as f:
            for commit in sorted(missing_embeddings):
                f.write(commit + "\n")
        print(f"  列表文件: {missing_emb_file}")


if __name__ == "__main__":
    main()
