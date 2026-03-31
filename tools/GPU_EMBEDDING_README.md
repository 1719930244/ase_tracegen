# TraceGen GPU 向量生成器使用说明

## 概述

本工具用于在 GPU 服务器上生成代码向量，支持多 GPU 并行推理。

**支持的模型：**

| 模型 | HuggingFace ID | 维度 | Batch/GPU | 特点 |
|------|----------------|------|-----------|------|
| qwen3-0.6b | Qwen/Qwen3-Embedding-0.6B | 1024 | 128 | 通用，模型小速度快 |
| jina-code | jinaai/jina-embeddings-v2-base-code | 768 | 64 | 代码专用 |

**硬件要求：**
- 推荐：4x V100 32G（总 batch_size=512）
- 最低：单卡 16G

## 快速开始

### 1. 解压文件

```bash
unzip gpu_embedding_package.zip
cd <解压目录>
```

### 2. 配置环境

```bash
chmod +x tools/gpu_env_setup.sh
./tools/gpu_env_setup.sh
```

这会：
- 创建 conda 环境 `tracegen_gpu`
- 安装 PyTorch (CUDA 11.8)
- 安装 transformers 等依赖

### 3. 激活环境

```bash
conda activate tracegen_gpu
```

### 4. 运行向量生成

**使用 Qwen3 模型（推荐）：**
```bash
python tools/embedding_generator_gpu.py \
    --repo django_django \
    --model qwen3-0.6b \
    --gpus 0,1,2,3
```

**使用 Jina Code 模型：**
```bash
python tools/embedding_generator_gpu.py \
    --repo django_django \
    --model jina-code \
    --gpus 0,1,2,3
```

**单卡运行：**
```bash
python tools/embedding_generator_gpu.py \
    --repo django_django \
    --model qwen3-0.6b \
    --gpus 0
```

### 5. 打包结果传回

```bash
zip -r embeddings_result.zip data/assets/embeddings/django_django/
```

## 命令行参数

```
--repo          仓库名称（处理该仓库所有 commit）
--graph         单个图文件路径（二选一）
--model         模型选择：qwen3-0.6b | jina-code
--gpus          GPU ID，逗号分隔（如 0,1,2,3）
--graphs_dir    图文件目录（默认 data/assets/graphs）
--output_dir    输出目录（默认 data/assets/embeddings）
--batch_size    覆盖每 GPU 的 batch size
--fp32          使用 FP32 精度（默认 FP16）
```

## 输出结构

```
data/assets/embeddings/django_django/
├── vector_pool.npy      # 向量池 [N, dim]
├── hash_to_idx.json     # {content_hash: vector_idx}
├── stats.json           # 统计信息
└── commits/
    ├── 004b4620.json    # {node_id: content_hash}
    ├── 0456d3e4.json
    └── ...
```

## 预估时间

| 配置 | 代码片段数 | 预估时间 |
|------|-----------|----------|
| 4x V100 32G | ~57,000 | 5-10 分钟 |
| 1x V100 32G | ~57,000 | 15-20 分钟 |

## 增量生成

脚本支持增量生成：
- 已处理的 commit 会跳过
- 相同代码内容（相同 hash）不会重复生成向量
- 新增 commit 时只计算新代码的向量

## 常见问题

**Q: 首次运行很慢？**
A: 首次运行需要从 HuggingFace 下载模型（~1-2GB），后续运行会使用缓存。

**Q: 显存不足？**
A: 减小 batch_size：`--batch_size 32`

**Q: 网络问题无法下载模型？**
A: 设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q: 如何验证结果？**
A: 检查 `stats.json` 中的 `total_vectors` 和 `model` 字段。
