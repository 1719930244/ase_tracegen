"""
TraceGen GPU 向量生成器 (本地推理版本)

支持模型:
- Qwen/Qwen3-Embedding-0.6B: 1024 维, 8192 tokens
- jinaai/jina-embeddings-v2-base-code: 768 维, 8192 tokens (代码专用)

针对 4x V100 32G 优化:
- Qwen3-Embedding-0.6B: batch_size=128 per GPU (模型小，显存充足)
- jina-embeddings-v2-base-code: batch_size=64 per GPU (模型稍大)

多 GPU 策略: 每个 GPU 独立加载模型，数据均匀分配，避免 DataParallel 的 GPU 0 瓶颈

输出格式与 DashScope API 版本兼容 (V2 格式)
"""

import os
import pickle
import json
import hashlib
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


# ============ 模型配置 ============

@dataclass
class ModelConfig:
    name: str
    hf_name: str
    dimension: int
    max_length: int
    batch_size_per_gpu: int  # 针对 V100 32G 优化
    pooling: str  # "mean", "cls", "last"
    normalize: bool


MODEL_CONFIGS = {
    "qwen3-0.6b": ModelConfig(
        name="qwen3-0.6b",
        hf_name="Qwen/Qwen3-Embedding-0.6B",
        dimension=1024,
        max_length=8192,
        batch_size_per_gpu=128,  # 0.6B 模型很小，可以用大 batch
        pooling="last",  # Qwen3 使用 last token pooling
        normalize=True,
    ),
    "jina-code": ModelConfig(
        name="jina-code",
        hf_name="jinaai/jina-embeddings-v2-base-code",
        dimension=768,
        max_length=8192,
        batch_size_per_gpu=64,  # 稍大的模型，batch 稍小
        pooling="mean",
        normalize=True,
    ),
}


# ============ 单 GPU 工作函数 ============

def worker_init(gpu_id: int, model_name: str, use_fp16: bool):
    """在子进程中初始化模型"""
    import torch
    from transformers import AutoModel, AutoTokenizer

    global _worker_model, _worker_tokenizer, _worker_config, _worker_device

    _worker_config = MODEL_CONFIGS[model_name]
    _worker_device = f"cuda:{gpu_id}"

    # 设置当前进程只使用指定 GPU
    torch.cuda.set_device(gpu_id)

    _worker_tokenizer = AutoTokenizer.from_pretrained(
        _worker_config.hf_name,
        trust_remote_code=True,
    )

    _worker_model = AutoModel.from_pretrained(
        _worker_config.hf_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if use_fp16 else torch.float32,
    ).to(_worker_device)

    _worker_model.eval()


def worker_encode(texts: List[str]) -> np.ndarray:
    """在子进程中编码文本"""
    import torch
    import torch.nn.functional as F

    global _worker_model, _worker_tokenizer, _worker_config, _worker_device

    with torch.no_grad():
        inputs = _worker_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=_worker_config.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(_worker_device) for k, v in inputs.items()}

        outputs = _worker_model(**inputs)

        if hasattr(outputs, 'last_hidden_state'):
            hidden_state = outputs.last_hidden_state
        else:
            hidden_state = outputs[0]

        # 池化
        if _worker_config.pooling == "mean":
            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_state.size()).float()
            embeddings = (hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        elif _worker_config.pooling == "cls":
            embeddings = hidden_state[:, 0]
        elif _worker_config.pooling == "last":
            seq_lens = inputs['attention_mask'].sum(dim=1) - 1
            embeddings = hidden_state[torch.arange(hidden_state.size(0), device=_worker_device), seq_lens]

        if _worker_config.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()


def process_batch_on_gpu(args):
    """处理单个批次（在子进程中运行）"""
    batch_texts, batch_idx = args
    embeddings = worker_encode(batch_texts)
    return batch_idx, embeddings


# ============ 多 GPU 生成器 ============

class MultiGPUEmbeddingGenerator:
    """多 GPU 向量生成器 - 每个 GPU 独立进程"""

    def __init__(
        self,
        model_name: str = "qwen3-0.6b",
        device_ids: Optional[List[int]] = None,
        use_fp16: bool = True,
    ):
        import torch

        self.config = MODEL_CONFIGS[model_name]
        self.model_name = model_name
        self.use_fp16 = use_fp16

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.num_gpus = len(device_ids)

        print(f"Model: {self.config.hf_name}")
        print(f"Using {self.num_gpus} GPU(s): {device_ids}")
        print(f"Batch size per GPU: {self.config.batch_size_per_gpu}")
        print(f"Total batch size: {self.config.batch_size_per_gpu * self.num_gpus}")

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """使用多进程并行生成向量"""
        import torch

        batch_size = self.config.batch_size_per_gpu
        total_batches = (len(texts) + batch_size - 1) // batch_size

        # 准备批次
        batches = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batches.append((batch_texts, i // batch_size))

        # 分配批次到各 GPU
        gpu_batches = [[] for _ in range(self.num_gpus)]
        for idx, batch in enumerate(batches):
            gpu_batches[idx % self.num_gpus].append(batch)

        # 多进程处理
        all_results = [None] * len(batches)

        # 使用 spawn 方式创建进程
        ctx = mp.get_context('spawn')

        with ProcessPoolExecutor(max_workers=self.num_gpus, mp_context=ctx) as executor:
            futures = []

            for gpu_idx, gpu_id in enumerate(self.device_ids):
                if not gpu_batches[gpu_idx]:
                    continue

                # 每个 GPU 一个进程
                for batch in gpu_batches[gpu_idx]:
                    # 创建一个包含初始化信息的任务
                    future = executor.submit(
                        self._process_single_batch,
                        gpu_id,
                        self.model_name,
                        self.use_fp16,
                        batch[0],  # texts
                        batch[1],  # batch_idx
                    )
                    futures.append(future)

            # 收集结果
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Generating embeddings")

            for future in iterator:
                batch_idx, embeddings = future.result()
                all_results[batch_idx] = embeddings

        # 合并结果
        return np.vstack([r for r in all_results if r is not None])

    @staticmethod
    def _process_single_batch(gpu_id, model_name, use_fp16, texts, batch_idx):
        """静态方法：在独立进程中处理批次"""
        import torch
        from transformers import AutoModel, AutoTokenizer
        import torch.nn.functional as F

        config = MODEL_CONFIGS[model_name]
        device = f"cuda:{gpu_id}"

        torch.cuda.set_device(gpu_id)

        tokenizer = AutoTokenizer.from_pretrained(
            config.hf_name,
            trust_remote_code=True,
        )

        model = AutoModel.from_pretrained(
            config.hf_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
        ).to(device)
        model.eval()

        with torch.no_grad():
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            if hasattr(outputs, 'last_hidden_state'):
                hidden_state = outputs.last_hidden_state
            else:
                hidden_state = outputs[0]

            # 池化
            if config.pooling == "mean":
                mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_state.size()).float()
                embeddings = (hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            elif config.pooling == "cls":
                embeddings = hidden_state[:, 0]
            elif config.pooling == "last":
                seq_lens = inputs['attention_mask'].sum(dim=1) - 1
                embeddings = hidden_state[torch.arange(hidden_state.size(0), device=device), seq_lens]

            if config.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            result = embeddings.cpu().numpy()

        # 清理显存
        del model, tokenizer, inputs, outputs
        torch.cuda.empty_cache()

        return batch_idx, result


# ============ 单 GPU 简化版（推荐，模型只加载一次）============

class SingleLoadMultiGPUGenerator:
    """优化版：模型只加载一次到各 GPU，批次轮流分发"""

    def __init__(
        self,
        model_name: str = "qwen3-0.6b",
        device_ids: Optional[List[int]] = None,
        use_fp16: bool = True,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.config = MODEL_CONFIGS[model_name]
        self.use_fp16 = use_fp16

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        self.device_ids = device_ids
        self.num_gpus = len(device_ids)

        print(f"Model: {self.config.hf_name}")
        print(f"Using {self.num_gpus} GPU(s): {device_ids}")

        # 加载 tokenizer（共享）
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_name,
            trust_remote_code=True,
        )

        # 在每个 GPU 上加载模型
        self.models = []
        for gpu_id in device_ids:
            print(f"  Loading model on GPU {gpu_id}...")
            model = AutoModel.from_pretrained(
                self.config.hf_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if use_fp16 else torch.float32,
            ).to(f"cuda:{gpu_id}")
            model.eval()
            self.models.append(model)

        self.batch_size = self.config.batch_size_per_gpu
        print(f"Batch size per GPU: {self.batch_size}")
        print(f"Total batch size: {self.batch_size * self.num_gpus}")

    def _encode_on_gpu(self, texts: List[str], gpu_idx: int) -> np.ndarray:
        """在指定 GPU 上编码"""
        import torch
        import torch.nn.functional as F

        device = f"cuda:{self.device_ids[gpu_idx]}"
        model = self.models[gpu_idx]

        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)

            if hasattr(outputs, 'last_hidden_state'):
                hidden_state = outputs.last_hidden_state
            else:
                hidden_state = outputs[0]

            # 池化
            if self.config.pooling == "mean":
                mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden_state.size()).float()
                embeddings = (hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            elif self.config.pooling == "cls":
                embeddings = hidden_state[:, 0]
            elif self.config.pooling == "last":
                seq_lens = inputs['attention_mask'].sum(dim=1) - 1
                embeddings = hidden_state[torch.arange(hidden_state.size(0), device=device), seq_lens]

            if self.config.normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            result = embeddings.cpu().numpy()

        # 清理显存
        del inputs, outputs, hidden_state, embeddings
        torch.cuda.empty_cache()

        return result

    def generate_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """轮流在各 GPU 上生成向量"""
        from concurrent.futures import ThreadPoolExecutor

        all_embeddings = []

        # 按 GPU 数量分组批次
        total_batch_size = self.batch_size * self.num_gpus

        iterator = range(0, len(texts), total_batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings",
                          total=(len(texts) + total_batch_size - 1) // total_batch_size)

        for i in iterator:
            # 取出 num_gpus 个批次
            mega_batch = texts[i:i + total_batch_size]

            # 分配到各 GPU
            gpu_batches = []
            for gpu_idx in range(self.num_gpus):
                start = gpu_idx * self.batch_size
                end = start + self.batch_size
                if start < len(mega_batch):
                    gpu_batches.append((mega_batch[start:end], gpu_idx))

            # 并行处理
            with ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
                futures = [
                    executor.submit(self._encode_on_gpu, batch, gpu_idx)
                    for batch, gpu_idx in gpu_batches
                ]

                for future in futures:
                    all_embeddings.append(future.result())

        return np.vstack(all_embeddings)


# ============ 图处理函数 ============

def compute_code_hash(code: str) -> str:
    """计算代码内容的 MD5 哈希"""
    normalized = code.strip()
    return hashlib.md5(normalized.encode('utf-8')).hexdigest()


def extract_nodes_with_hash(graph_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """从图中提取叶子节点"""
    print(f"Reading graph: {graph_path}")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    print(f"Graph loaded. Total nodes: {len(graph.nodes())}")

    all_nodes = {}
    for node_id, data in graph.nodes(data=True):
        ntype = data.get("type", "unknown")
        code = data.get("code_snippet") or data.get("code", "")
        if ntype in ["function", "class"] and code.strip():
            all_nodes[node_id] = {"type": ntype, "code": code}

    classes_with_methods = set()
    for node_id in all_nodes:
        if all_nodes[node_id]["type"] == "function":
            if "." in node_id.split(":")[-1]:
                parts = node_id.rsplit(".", 1)
                if len(parts) == 2:
                    class_node_id = parts[0]
                    if class_node_id in all_nodes and all_nodes[class_node_id]["type"] == "class":
                        classes_with_methods.add(class_node_id)

    node_to_hash = {}
    hash_to_code = {}
    skipped = 0

    for node_id, info in tqdm(all_nodes.items(), desc="Extracting nodes"):
        if node_id in classes_with_methods:
            skipped += 1
            continue

        code = info["code"]
        content_hash = compute_code_hash(code)
        node_to_hash[node_id] = content_hash
        if content_hash not in hash_to_code:
            hash_to_code[content_hash] = code

    print(f"Extracted {len(node_to_hash)} leaf nodes")
    print(f"Unique code snippets: {len(hash_to_code)}")
    print(f"Skipped {skipped} classes with methods")

    return node_to_hash, hash_to_code


def load_existing_pool(repo_dir: Path) -> Tuple[Optional[np.ndarray], Dict[str, int]]:
    """加载已有的向量池"""
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


# ============ 主流程 ============

def process_repository(
    repo_name: str,
    graph_paths: List[Path],
    output_dir: Path,
    generator,
    model_config: ModelConfig,
):
    """处理整个仓库的所有 commit"""
    repo_dir = output_dir / repo_name
    repo_dir.mkdir(parents=True, exist_ok=True)
    commits_dir = repo_dir / "commits"
    commits_dir.mkdir(exist_ok=True)

    existing_pool, hash_to_idx = load_existing_pool(repo_dir)

    all_new_hashes = {}
    commit_mappings = {}

    for graph_path in graph_paths:
        stem = graph_path.stem
        parts = stem.split("__")
        commit = parts[1].split("_")[0] if len(parts) >= 2 else stem

        print(f"\n--- Processing commit: {commit} ---")

        commit_file = commits_dir / f"{commit}.json"
        if commit_file.exists():
            print(f"Commit {commit} already processed, loading mapping...")
            with open(commit_file, 'r') as f:
                commit_mappings[commit] = json.load(f)
            continue

        node_to_hash, hash_to_code = extract_nodes_with_hash(graph_path)
        commit_mappings[commit] = node_to_hash

        for h, code in hash_to_code.items():
            if h not in hash_to_idx and h not in all_new_hashes:
                all_new_hashes[h] = code

        new_count = len([h for h in hash_to_code if h not in hash_to_idx])
        print(f"Commit {commit}: {len(node_to_hash)} nodes, {new_count} new unique codes")

    if all_new_hashes:
        print(f"\n--- Generating {len(all_new_hashes)} new embeddings ---")

        new_hashes = sorted(all_new_hashes.keys())
        new_codes = [all_new_hashes[h] for h in new_hashes]

        new_embeddings = generator.generate_embeddings(new_codes)

        start_idx = len(hash_to_idx)
        for i, h in enumerate(new_hashes):
            hash_to_idx[h] = start_idx + i

        if existing_pool is not None and len(existing_pool) > 0:
            vector_pool = np.vstack([existing_pool, new_embeddings])
        else:
            vector_pool = new_embeddings

        print(f"Vector pool updated: {vector_pool.shape}")
    else:
        print("\nNo new embeddings needed")
        vector_pool = existing_pool if existing_pool is not None else np.array([])

    print(f"\n--- Saving results to {repo_dir} ---")

    if vector_pool is not None and len(vector_pool) > 0:
        np.save(repo_dir / "vector_pool.npy", vector_pool)
        print(f"Saved vector_pool.npy: {vector_pool.shape}")

    with open(repo_dir / "hash_to_idx.json", 'w') as f:
        json.dump(hash_to_idx, f, indent=2)
    print(f"Saved hash_to_idx.json: {len(hash_to_idx)} entries")

    for commit, mapping in commit_mappings.items():
        commit_file = commits_dir / f"{commit}.json"
        with open(commit_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Saved commits/{commit}.json: {len(mapping)} nodes")

    stats = {
        "repo": repo_name,
        "total_vectors": len(hash_to_idx),
        "vector_dim": int(vector_pool.shape[1]) if len(vector_pool) > 0 else model_config.dimension,
        "commits_processed": list(commit_mappings.keys()),
        "model": model_config.hf_name,
        "device": "gpu",
        "num_gpus": generator.num_gpus,
    }

    with open(repo_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"Repository: {repo_name}")
    print(f"Total vectors: {len(hash_to_idx)}")
    print(f"Vector dimension: {stats['vector_dim']}")
    print(f"Commits: {len(commit_mappings)}")
    print(f"Model: {model_config.hf_name}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="TraceGen GPU Embedding Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用 Qwen3 模型处理 django 仓库（推荐方式）
  python embedding_generator_gpu.py --repo django_django --model qwen3-0.6b

  # 使用 Jina Code 模型
  python embedding_generator_gpu.py --repo django_django --model jina-code

  # 指定 GPU
  python embedding_generator_gpu.py --repo django_django --gpus 0,1,2,3

  # 处理单个图文件
  python embedding_generator_gpu.py --graph path/to/graph.pkl --model qwen3-0.6b
        """
    )

    parser.add_argument("--graph", type=str, help="Path to a single graph .pkl file")
    parser.add_argument("--repo", type=str, help="Repository name (process all commits)")
    parser.add_argument("--graphs_dir", type=str, default="data/assets/graphs",
                        help="Directory containing graph files")
    parser.add_argument("--output_dir", type=str, default="data/assets/embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--model", type=str, default="qwen3-0.6b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Embedding model to use")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size per GPU")
    parser.add_argument("--fp32", action="store_true",
                        help="Use FP32 instead of FP16")

    args = parser.parse_args()

    device_ids = None
    if args.gpus:
        device_ids = [int(x) for x in args.gpus.split(",")]

    model_config = MODEL_CONFIGS[args.model]

    if args.batch_size:
        model_config.batch_size_per_gpu = args.batch_size

    # 使用优化版生成器（模型只加载一次）
    generator = SingleLoadMultiGPUGenerator(
        model_name=args.model,
        device_ids=device_ids,
        use_fp16=not args.fp32,
    )

    output_dir = Path(args.output_dir)
    graphs_dir = Path(args.graphs_dir)

    if args.graph:
        graph_path = Path(args.graph)
        if not graph_path.exists():
            print(f"Error: Graph file not found: {graph_path}")
            return 1

        stem = graph_path.stem
        repo_name = stem.split("__")[0] if "__" in stem else stem.rsplit("_", 1)[0]

        process_repository(
            repo_name=repo_name,
            graph_paths=[graph_path],
            output_dir=output_dir,
            generator=generator,
            model_config=model_config,
        )

    elif args.repo:
        patterns = [f"{args.repo}__*_v1.pkl", f"{args.repo}__*.pkl"]
        graph_paths = []
        for pattern in patterns:
            graph_paths.extend(graphs_dir.glob(pattern))
        graph_paths = list(set(graph_paths))

        if not graph_paths:
            print(f"Error: No graph files found for {args.repo} in {graphs_dir}")
            return 1

        print(f"Found {len(graph_paths)} graph files for {args.repo}")

        process_repository(
            repo_name=args.repo,
            graph_paths=graph_paths,
            output_dir=output_dir,
            generator=generator,
            model_config=model_config,
        )

    else:
        print("Error: Please specify --graph or --repo")
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
