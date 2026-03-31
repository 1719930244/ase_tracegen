#!/bin/bash
# TraceGen GPU 向量生成环境配置脚本
#
# 使用方法:
#   chmod +x gpu_env_setup.sh
#   ./gpu_env_setup.sh
#
# 支持的模型:
#   - Qwen/Qwen3-Embedding-0.6B (推荐，1024维)
#   - jinaai/jina-embeddings-v2-base-code (代码专用，768维)
#
# 硬件要求:
#   - 4x V100 32G 可以跑 batch_size=512 (128 per GPU)
#   - 单卡也可以运行，自动调整 batch_size

set -e

echo "=========================================="
echo "TraceGen GPU Embedding Generator Setup"
echo "=========================================="

# 配置
ENV_NAME="tracegen_gpu"
PYTHON_VERSION="3.10"

# 检查 conda
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

# 创建 conda 环境
echo ""
echo "[1/4] Creating conda environment: $ENV_NAME"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment $ENV_NAME already exists, updating..."
    conda activate $ENV_NAME || source activate $ENV_NAME
else
    conda create -n $ENV_NAME python=$PYTHON_VERSION -y
    conda activate $ENV_NAME || source activate $ENV_NAME
fi

# 安装 PyTorch (CUDA 11.8)
echo ""
echo "[2/4] Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
echo ""
echo "[3/4] Installing dependencies..."
pip install \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    sentence-transformers>=2.5.0 \
    numpy>=1.24.0 \
    tqdm \
    networkx

# 验证安装
echo ""
echo "[4/4] Verifying installation..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)')
else:
    print('WARNING: CUDA not available!')

from transformers import AutoModel, AutoTokenizer
print('Transformers: OK')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  conda activate $ENV_NAME"
echo ""
echo "  # 使用 Qwen3 模型 (推荐)"
echo "  python tools/embedding_generator_gpu.py \\"
echo "      --repo django_django \\"
echo "      --model qwen3-0.6b \\"
echo "      --gpus 0,1,2,3"
echo ""
echo "  # 使用 Jina Code 模型"
echo "  python tools/embedding_generator_gpu.py \\"
echo "      --repo django_django \\"
echo "      --model jina-code \\"
echo "      --gpus 0,1,2,3"
echo ""
echo "Model info:"
echo "  qwen3-0.6b:  Qwen/Qwen3-Embedding-0.6B  (1024 dim, batch=128/GPU)"
echo "  jina-code:   jinaai/jina-embeddings-v2-base-code (768 dim, batch=64/GPU)"
echo ""
echo "First run will download models from HuggingFace (~1-2GB each)"
echo ""
