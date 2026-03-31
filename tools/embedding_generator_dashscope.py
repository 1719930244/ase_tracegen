"""
TraceGen 仓库级聚合向量库生成器 (DashScope API 版本)

使用阿里云 DashScope API 生成代码向量，无需本地 GPU。

支持的模型:
- text-embedding-v1: 1536 维, 最大 2048 tokens
- text-embedding-v2: 1536 维, 最大 2048 tokens
- text-embedding-v3: 1024/768/512 维, 最大 8192 tokens (推荐)

输出结构与 V2 格式兼容:
    data/assets/embeddings/
    └── <repo>/
        ├── vector_pool.npy          # 聚合向量池 [N, dim]
        ├── hash_to_idx.json         # {content_hash: vector_idx}
        ├── stats.json               # 统计信息
        └── commits/
            ├── <commit1>.json       # {node_id: content_hash}
            └── <commit2>.json
"""

import os
import pickle
import json
import hashlib
import argparse
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


# 默认配置
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "text-embedding-v3"
DEFAULT_BATCH_SIZE = 10  # API 批量请求大小 (DashScope 单次最多 10 条)
DEFAULT_DIMENSION = 1024  # text-embedding-v3 默认维度


def compute_code_hash(code: str) -> str:
    """计算代码内容的 MD5 哈希"""
    normalized = code.strip()
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def extract_nodes_with_hash(graph_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    从图中提取叶子节点 (function/class)，返回:
    - node_to_hash: {node_id: content_hash}
    - hash_to_code: {content_hash: code} (去重后的代码)

    优化策略：避免类和方法的代码重叠
    - 如果一个类有方法节点，只保留方法节点，跳过类节点
    - 如果一个类没有方法节点（如只有属性定义），保留类节点
    """
    print(f"Reading graph: {graph_path}")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    total_nodes = len(graph.nodes())
    print(f"Graph loaded. Total nodes: {total_nodes}")

    # 第一遍：收集所有节点信息
    all_nodes = {}
    for node_id, data in graph.nodes(data=True):
        ntype = data.get("type", "unknown")
        code = data.get("code_snippet") or data.get("code", "")
        if ntype in ["function", "class"] and code.strip():
            all_nodes[node_id] = {
                "type": ntype,
                "code": code
            }

    # 第二遍：识别有方法的类（应该跳过）
    classes_with_methods = set()
    for node_id in all_nodes:
        if all_nodes[node_id]["type"] == "function":
            # 检查是否是类的方法 (格式: path/file.py:ClassName.method_name)
            if "." in node_id.split(":")[-1]:
                # 提取类节点 ID
                parts = node_id.rsplit(".", 1)
                if len(parts) == 2:
                    class_node_id = parts[0]
                    if class_node_id in all_nodes and all_nodes[class_node_id]["type"] == "class":
                        classes_with_methods.add(class_node_id)

    # 第三遍：构建最终的节点映射（跳过有方法的类）
    node_to_hash = {}
    hash_to_code = {}
    skipped_classes = 0

    for node_id, info in tqdm(all_nodes.items(), desc="Extracting nodes"):
        # 跳过有方法的类（避免重叠）
        if node_id in classes_with_methods:
            skipped_classes += 1
            continue

        code = info["code"]
        content_hash = compute_code_hash(code)
        node_to_hash[node_id] = content_hash
        if content_hash not in hash_to_code:
            hash_to_code[content_hash] = code

    print(f"Extracted {len(node_to_hash)} leaf nodes (function/class)")
    print(f"Unique code snippets: {len(hash_to_code)}")
    print(f"Skipped {skipped_classes} classes with methods (to avoid overlap)")

    return node_to_hash, hash_to_code


def load_existing_pool(repo_dir: Path) -> Tuple[Optional[np.ndarray], Dict[str, int]]:
    """加载已有的向量池和哈希索引"""
    pool_path = repo_dir / "vector_pool.npy"
    index_path = repo_dir / "hash_to_idx.json"

    if pool_path.exists() and index_path.exists():
        print(f"Loading existing vector pool from {repo_dir}")
        pool = np.load(pool_path)
        with open(index_path, 'r') as f:
            hash_to_idx = json.load(f)
        print(f"Loaded {len(hash_to_idx)} existing vectors")
        return pool, hash_to_idx

    return None, {}


def generate_embeddings_api(
    texts: List[str],
    client: OpenAI,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dimension: int = DEFAULT_DIMENSION,
    retry_times: int = 10,
    retry_delay: float = 5.0
) -> np.ndarray:
    """
    使用阿里云 DashScope API 批量生成向量

    Args:
        texts: 代码文本列表
        client: OpenAI 兼容客户端
        model: 模型名称
        batch_size: 批量大小
        dimension: 向量维度 (仅 text-embedding-v3 支持)
        retry_times: 重试次数
        retry_delay: 重试延迟(秒)
    """
    all_embeddings = []

    print(f"Generating embeddings using {model} (dim={dimension})")

    for i in tqdm(range(0, len(texts), batch_size), desc="API calls"):
        batch_texts = texts[i:i + batch_size]

        # 截断过长的文本 (代码 token 化率约 1:2~1:3，保守取 2)
        max_chars = 8192 * 2 if "v3" in model else 2000 * 2
        batch_texts = [t[:max_chars] if len(t) > max_chars else t for t in batch_texts]

        current_delay = retry_delay
        for attempt in range(retry_times):
            try:
                # 调用 API
                if "v3" in model or "Qwen3" in model:
                    response = client.embeddings.create(
                        model=model,
                        input=batch_texts,
                        dimensions=dimension
                    )
                else:
                    response = client.embeddings.create(
                        model=model,
                        input=batch_texts
                    )

                # 提取向量
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                break

            except Exception as e:
                err_str = str(e)
                # 400 参数错误(如超长)不可重试，强制截断后再试一次
                if "InvalidParameter" in err_str or "input length" in err_str:
                    print(f"\nInput too long, truncating to 6000 chars and retrying...")
                    batch_texts = [t[:6000] for t in batch_texts]
                    try:
                        if "v3" in model or "Qwen3" in model:
                            response = client.embeddings.create(
                                model=model, input=batch_texts, dimensions=dimension)
                        else:
                            response = client.embeddings.create(
                                model=model, input=batch_texts)
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                    except Exception as e2:
                        print(f"\nStill failed after truncation: {e2}, using zero vectors")
                        all_embeddings.extend([[0.0] * dimension] * len(batch_texts))
                    break
                elif attempt < retry_times - 1:
                    print(f"\nAPI error (attempt {attempt + 1}): {e}")
                    print(f"Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= 2  # 指数退避
                else:
                    print(f"\nFailed after {retry_times} attempts: {e}")
                    # 填充零向量作为占位
                    all_embeddings.extend([[0.0] * dimension] * len(batch_texts))

        # 避免 API 限流
        time.sleep(0.1)

    return np.array(all_embeddings)


def _save_checkpoint(repo_dir: Path, vector_pool: np.ndarray, hash_to_idx: Dict[str, int]):
    """保存 checkpoint: vector_pool + hash_to_idx"""
    if vector_pool is not None and len(vector_pool) > 0:
        np.save(repo_dir / "vector_pool.npy", vector_pool)
    with open(repo_dir / "hash_to_idx.json", 'w') as f:
        json.dump(hash_to_idx, f)
    print(f"\n[Checkpoint] Saved {len(hash_to_idx)} vectors to {repo_dir}")


CHECKPOINT_CHUNK_SIZE = 500  # 每 500 个 hash 保存一次 (= 50 API batches)


def update_repo_embeddings_api(
    repo_name: str,
    graph_paths: List[Path],
    output_dir: Path,
    api_key: str,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dimension: int = DEFAULT_DIMENSION
):
    """
    使用 API 更新仓库级向量库 (V2 格式)
    支持 checkpoint: 每处理 CHECKPOINT_CHUNK_SIZE 个新 hash 后自动保存，
    中断后重启可从断点继续。
    """
    repo_dir = output_dir / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)
    commits_dir = repo_dir / "commits"
    commits_dir.mkdir(exist_ok=True)

    # 1. 加载已有向量池
    existing_pool, hash_to_idx = load_existing_pool(repo_dir)

    # 2. 收集所有 commit 的节点信息，commit mapping 立即落盘
    all_new_hashes = {}

    for graph_path in graph_paths:
        stem = graph_path.stem
        parts = stem.split("__")
        if len(parts) >= 2:
            commit = parts[1].split("_")[0]
        else:
            commit = stem

        print(f"\n--- Processing commit: {commit} ---")

        commit_file = commits_dir / f"{commit}.json"
        if commit_file.exists():
            print(f"Commit {commit} already processed, skipping.")
            continue

        node_to_hash, hash_to_code = extract_nodes_with_hash(graph_path)

        # 立即保存 commit mapping，避免重启后重复扫描
        with open(commit_file, 'w') as f:
            json.dump(node_to_hash, f, indent=2)
        print(f"Saved commits/{commit}.json: {len(node_to_hash)} nodes")

        for h, code in hash_to_code.items():
            if h not in hash_to_idx and h not in all_new_hashes:
                all_new_hashes[h] = code

        new_count = len([h for h in hash_to_code if h not in hash_to_idx])
        print(f"Commit {commit}: {len(node_to_hash)} leaf nodes, {new_count} new unique codes")

    # 3. 分 chunk 生成新向量，每 chunk 保存 checkpoint
    if all_new_hashes:
        print(f"\n--- Generating {len(all_new_hashes)} new embeddings via API ---")
        client = OpenAI(api_key=api_key, base_url=api_base)

        new_hashes = sorted(all_new_hashes.keys())
        new_codes = [all_new_hashes[h] for h in new_hashes]
        total = len(new_hashes)
        vector_pool = existing_pool

        for chunk_start in range(0, total, CHECKPOINT_CHUNK_SIZE):
            chunk_end = min(chunk_start + CHECKPOINT_CHUNK_SIZE, total)
            chunk_hashes = new_hashes[chunk_start:chunk_end]
            chunk_codes = new_codes[chunk_start:chunk_end]

            print(f"\n--- Chunk [{chunk_start+1}-{chunk_end}]/{total} ---")

            chunk_embeddings = generate_embeddings_api(
                chunk_codes, client, model, batch_size, dimension
            )

            # 更新索引
            start_idx = len(hash_to_idx)
            for i, h in enumerate(chunk_hashes):
                hash_to_idx[h] = start_idx + i

            # 合并向量池
            if vector_pool is not None and len(vector_pool) > 0:
                vector_pool = np.vstack([vector_pool, chunk_embeddings])
            else:
                vector_pool = chunk_embeddings

            # 保存 checkpoint
            _save_checkpoint(repo_dir, vector_pool, hash_to_idx)

        print(f"Vector pool updated: {vector_pool.shape}")
    else:
        print("\nNo new embeddings needed")
        vector_pool = existing_pool if existing_pool is not None else np.array([])

    # 4. 最终保存 (含 indent 格式化)
    print(f"\n--- Saving final results to {repo_dir} ---")

    if vector_pool is not None and len(vector_pool) > 0:
        np.save(repo_dir / "vector_pool.npy", vector_pool)
        print(f"Saved vector_pool.npy: {vector_pool.shape}")

    with open(repo_dir / "hash_to_idx.json", 'w') as f:
        json.dump(hash_to_idx, f, indent=2)
    print(f"Saved hash_to_idx.json: {len(hash_to_idx)} entries")

    # 5. 统计报告
    commit_files = list(commits_dir.glob("*.json"))
    stats = {
        "repo": repo_name,
        "total_vectors": len(hash_to_idx),
        "vector_dim": int(vector_pool.shape[1]) if len(vector_pool) > 0 else dimension,
        "commits_processed": [f.stem for f in commit_files],
        "model": model,
        "api_base": api_base
    }

    with open(repo_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Repository: {repo_name}")
    print(f"Total unique vectors: {len(hash_to_idx)}")
    print(f"Commits processed: {len(commit_files)}")
    print(f"Model: {model}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="TraceGen Embedding Generator (DashScope API) - V2 Format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用环境变量中的 API Key
  export DASHSCOPE_API_KEY=sk-xxx
  python embedding_generator_dashscope.py --graph data/assets/graphs/django_django__d26b2424_v1.pkl

  # 直接指定 API Key
  python embedding_generator_dashscope.py --repo django_django --api_key sk-xxx

  # 使用不同的模型和维度
  python embedding_generator_dashscope.py --graph xxx.pkl --model text-embedding-v2
  python embedding_generator_dashscope.py --graph xxx.pkl --dimension 512
        """
    )

    parser.add_argument("--graph", type=str, help="Path to a single graph .pkl file")
    parser.add_argument("--repo", type=str, help="Repository name (process all commits)")
    parser.add_argument("--graphs_dir", type=str, default="data/assets/graphs",
                        help="Directory containing graph files")
    parser.add_argument("--output_dir", type=str, default="data/assets/embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--api_key", type=str, default=os.environ.get("DASHSCOPE_API_KEY"),
                        help="DashScope API key")
    parser.add_argument("--api_base", type=str, default=DEFAULT_API_BASE,
                        help="API base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        choices=["text-embedding-v1", "text-embedding-v2", "text-embedding-v3", "Qwen3-Embedding-0.6B"],
                        help="Embedding model")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for API calls")
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION,
                        help="Vector dimension (only for text-embedding-v3)")

    args = parser.parse_args()

    if not args.api_key:
        print("Error: API key required. Set DASHSCOPE_API_KEY environment variable or use --api_key")
        return 1

    output_dir = Path(args.output_dir)
    graphs_dir = Path(args.graphs_dir)

    if args.graph:
        graph_path = Path(args.graph)
        if not graph_path.exists():
            print(f"Error: Graph file not found: {graph_path}")
            return 1

        stem = graph_path.stem
        repo_name = stem.split("__")[0] if "__" in stem else stem.rsplit("_", 1)[0]

        update_repo_embeddings_api(
            repo_name=repo_name,
            graph_paths=[graph_path],
            output_dir=output_dir,
            api_key=args.api_key,
            api_base=args.api_base,
            model=args.model,
            batch_size=args.batch_size,
            dimension=args.dimension
        )

    elif args.repo:
        # 支持多种文件名格式
        patterns = [f"{args.repo}__*_v1.pkl", f"{args.repo}__*.pkl"]
        graph_paths = []
        for pattern in patterns:
            graph_paths.extend(graphs_dir.glob(pattern))

        # 去重
        graph_paths = list(set(graph_paths))

        if not graph_paths:
            print(f"Error: No graph files found for repo {args.repo} in {graphs_dir}")
            return 1

        print(f"Found {len(graph_paths)} graph files for {args.repo}")

        update_repo_embeddings_api(
            repo_name=args.repo,
            graph_paths=graph_paths,
            output_dir=output_dir,
            api_key=args.api_key,
            api_base=args.api_base,
            model=args.model,
            batch_size=args.batch_size,
            dimension=args.dimension
        )

    else:
        print("Error: Please specify either --graph or --repo")
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
