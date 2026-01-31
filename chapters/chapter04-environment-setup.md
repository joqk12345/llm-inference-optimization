# 第4章: 环境搭建

> **💰 商业动机**: 正确的环境配置可以避免80%的部署问题。根据行业数据,环境不当导致的故障平均排查时间为4-8小时,而正确配置可以在30分钟内完成部署。

## 简介

在深入优化技术之前,我们需要先搭建一个可靠的开发环境。很多工程师在这一步花费了太多时间——CUDA版本冲突、Docker权限问题、依赖包版本不匹配……这些问题都会阻碍你的进度。

本章将帮你:
- 理解为什么使用 Docker 进行环境隔离
- 从零搭建完整的 LLM 推理环境
- 快速启动你的第一个 vLLM 推理服务
- 掌握容器化部署的最佳实践
- 学会排查常见的环境问题

**学完本章,你将拥有一个可以立即用于生产的推理环境。**

---

## 4.1 开发环境概览

### 4.1.1 为什么使用 Docker

你可能听过这样的话:"在我机器上能运行,为什么在你那就不行?"

**传统方式的问题**:
```
工程师 A 的机器:
- Ubuntu 20.04
- CUDA 11.8
- Python 3.9
- PyTorch 2.0.1

工程师 B 的机器:
- Ubuntu 22.04
- CUDA 12.1
- Python 3.10
- PyTorch 2.1.0

结果:
→ 同样的代码,不同的结果
→ 难以复现 bug
→ 生产环境部署噩梦
```

**Docker 的解决方案**:
```
Docker 容器:
- 固定的基础镜像
- �装的 CUDA 版本
- 锁定的依赖版本
- 标准的运行环境

结果:
→ 任何机器,同样的行为
→ 易于复现和调试
→ 一键部署到生产
```

**商业价值**:
- 减少 80% 的环境相关 bug
- 新人上手时间从 2 天降到 30 分钟
- 部署时间从数小时降到数分钟

---

### 4.1.2 环境一致性: 本地 vs 生产

**三层环境一致性**:

```
┌─────────────────────────────────────────┐
│  开发环境 (Development)                  │
│  - 你的笔记本电脑                        │
│  - 快速迭代,频繁重启                    │
│  - 使用较小的模型进行测试               │
└──────────────┬──────────────────────────┘
               │ Docker 镜像复用
┌──────────────▼──────────────────────────┐
│  测试环境 (Staging)                     │
│  - 与生产相同的配置                     │
│  - 真实负载测试                         │
│  - 验证性能和稳定性                     │
└──────────────┬──────────────────────────┘
               │ 同一个 Docker 镜像
┌──────────────▼──────────────────────────┐
│  生产环境 (Production)                  │
│  - 云端 GPU 实例                        │
│  - 高可用部署                           │
│  - 监控和告警                           │
└─────────────────────────────────────────┘
```

**关键原则**:
1. **开发容器化**: 从第一天开始就用 Docker
2. **版本锁定**: 使用 `requirements.txt` 或 `pyproject.toml` 锁定依赖
3. **配置外部化**: 环境变量、配置文件不要硬编码
4. **最小权限**: 生产容器不要包含开发工具

---

### 4.1.3 完整技术栈

```
┌─────────────────────────────────────────────────────┐
│  应用层 (Application Layer)                        │
│  - FastAPI / Flask (API 服务)                     │
│  - vLLM / SGLang (推理引擎)                       │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  框架层 (Framework Layer)                          │
│  - PyTorch / TensorFlow (深度学习框架)            │
│  - Transformers (模型库)                           │
│  - Hugging Face Hub (模型下载)                    │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  运行时层 (Runtime Layer)                          │
│  - Python 3.8+                                     │
│  - CUDA 12.x                                       │
│  - cuDNN / cuBLAS (CUDA 加速库)                   │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  驱动层 (Driver Layer)                             │
│  - NVIDIA Driver (525+)                            │
│  - GPU 硬件 (A100 / H100 / RTX 4090)              │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  容器层 (Container Layer)                          │
│  - Docker                                          │
│  - NVIDIA Container Toolkit                       │
│  - Docker Compose                                  │
└─────────────────────────────────────────────────────┘
```

**每一层都很重要**:
- 应用层: 你的业务逻辑
- 框架层: 推理引擎的基础
- 运行时层: Python 和 CUDA 的版本兼容性
- 驱动层: 必须与 GPU 硬件匹配
- 容器层: 隔离和可移植性

---

## 4.2 基础环境安装

### 4.2.1 NVIDIA 驱动安装

**检查当前驱动版本**:

```bash
nvidia-smi
```

你应该看到类似这样的输出:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   32C    P0    54W / 400W |  18939MiB / 81920MiB |     28%      Default |
+-------------------------------+----------------------+----------------------+
```

**关键信息**:
- **Driver Version**: 至少 525+ (推荐 535+)
- **CUDA Version**: 这是最高的 CUDA 版本支持,不一定是已安装的版本

**如果驱动版本过低或未安装**:

**Ubuntu/Debian**:
```bash
# 添加 NVIDIA 仓库
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装驱动
sudo apt-get update
sudo apt-get install -y nvidia-driver-535

# 重启
sudo reboot
```

**CentOS/RHEL**:
```bash
# 添加 NVIDIA 仓库
sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
sudo yum install -y https://nvidia.github.io/libnvidia-container/rhel8/nvidia-container-toolkit.repo

# 安装驱动
sudo yum install -y nvidia-driver

# 重启
sudo reboot
```

**云平台 (AWS/GCP/Azure)**:
- 通常已经预装 NVIDIA 驱动
- 使用官方的 GPU 优化 AMI/Image

---

### 4.2.2 CUDA Toolkit 配置

**重要说明**: Docker 容器中的 CUDA 不需要宿主机安装 CUDA Toolkit!

**为什么?**
```
宿主机:
- 只需要 NVIDIA 驱动
- 驱动提供 GPU 访问能力

Docker 容器:
- 包含 CUDA Toolkit
- 包含 CUDA 运行时库
- 隔离的 CUDA 版本
```

**最佳实践**:
- ✅ 宿主机: 只安装 NVIDIA 驱动
- ✅ Docker 容器: 使用带 CUDA 的基础镜像
- ❌ 避免在宿主机安装多个 CUDA 版本

**如果你确实需要在宿主机安装 CUDA** (例如本地开发):

```bash
# 从 NVIDIA 官网下载 CUDA Toolkit
# https://developer.nvidia.com/cuda-downloads

# 安装 (Ubuntu 示例)
wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
sudo sh cuda_12.2.0_535.54.03_linux.run --toolkit --silent

# 配置环境变量
echo 'export PATH=/usr/local/cuda-12.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 验证
nvcc --version
```

---

### 4.2.3 Docker 与 NVIDIA Container Toolkit

**安装 Docker**:

```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 将当前用户添加到 docker 组
sudo usermod -aG docker $USER

# 重新登录或运行
newgrp docker

# 验证
docker --version
docker run hello-world
```

**安装 NVIDIA Container Toolkit**:

```bash
# 添加 NVIDIA 仓库
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

如果成功,你应该看到 `nvidia-smi` 的输出。

---

### 4.2.4 Python 环境管理

**推荐方式**: 使用 pyenv 或 conda 管理多个 Python 版本

**使用 pyenv** (推荐):

```bash
# 安装 pyenv
curl https://pyenv.run | bash

# 添加到 shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# 安装 Python 3.10
pyenv install 3.10.12
pyenv global 3.10.12

# 验证
python --version
```

**使用 conda** (可选):

```bash
# 下载 Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建虚拟环境
conda create -n llm-inference python=3.10
conda activate llm-inference
```

**使用 venv** (Docker 内推荐):

```bash
# 在 Dockerfile 中
python3 -m venv /opt/venv
source /opt/venv/bin/activate
pip install --upgrade pip
```

---

## 4.3 vLLM 快速入门

### 4.3.1 什么是 vLLM

**vLLM** 是目前最流行的开源 LLM 推理引擎之一,由 UC Berkeley 的团队开发。

**核心特性**:
- ⚡ **高性能**: PagedAttention 算法,吞吐量比 HuggingFace Transformers 高 24 倍
- 🚀 **连续批处理**: Continuous Batching,最大化 GPU 利用率
- 🎯 **易用性**: 兼容 OpenAI API,一行代码启动服务
- 🔧 **灵活性**: 支持多种量化格式、投机解码、前缀缓存

**适用场景**:
- ✅ 高吞吐量推理服务
- ✅ 多模型并发部署
- ✅ 需要低延迟的实时应用
- ✅ 生产环境部署

**不适用场景**:
- ❌ 研究和实验 (建议使用 Transformers)
- ❌ 需要最大化的模型灵活性
- ❌ 超大模型的模型并行 (vLLM 支持有限)

---

### 4.3.2 vLLM vs 其他推理框架

| 特性 | vLLM | SGLang | TensorRT-LLM | Transformers |
|------|------|--------|--------------|--------------|
| **性能** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **生态** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **OpenAI API** | ✅ | ✅ | ❌ | ❌ |
| **生产就绪** | ✅ | ⚠️ | ✅ | ❌ |
| **学习曲线** | 低 | 中 | 高 | 低 |

**选择建议**:
- **vLLM**: 大多数场景的首选,性能与易用性的最佳平衡
- **SGLang**: 需要结构化生成或高级调度功能
- **TensorRT-LLM**: 极致性能要求,愿意投入时间优化
- **Transformers**: 快速原型,学习研究

---

### 4.3.3 安装 vLLM

**方式 1: pip 安装** (推荐用于开发):

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装 vLLM
pip install vllm

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

**方式 2: 从源码安装** (用于开发或最新功能):

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 安装依赖
pip install -r requirements.txt

# 安装 vLLM (可编辑模式)
pip install -e .
```

**方式 3: Docker 镜像** (推荐用于生产):

```bash
# 拉取官方镜像
docker pull vllm/vllm-openai:latest

# 或者使用本书提供的镜像
docker pull your-registry/llm-inference-book:latest
```

---

### 4.3.4 启动第一个推理服务

**最简单的启动方式**:

```bash
# OpenAI API 兼容服务器
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --host 0.0.0.0 \
    --port 8000
```

**使用 Docker**:

```bash
docker run --gpus all \
    --shm-size 10g \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Llama-2-7b-chat-hf
```

**测试推理服务**:

```bash
# 使用 curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'

# 使用 Python
import openai

# 配置本地端点
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "dummy"  # vLLM 不验证 key

response = openai.ChatCompletion.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```

**重要启动参数**:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \  # 模型名称或路径
    --tensor-parallel-size 2 \                # 张量并行度 (多GPU)
    --gpu-memory-utilization 0.9 \            # GPU 内存利用率 (0-1)
    --max-model-len 4096 \                    # 最大序列长度
    --dtype half \                            # 数据类型 (half, bfloat16)
    --quantization awq \                      # 量化格式 (awq, gptq, squeezellm)
    --host 0.0.0.0 \                          # 监听地址
    --port 8000                               # 监听端口
```

---

## 4.4 Docker 容器化部署

### 4.4.1 Dockerfile 编写

**基础版 Dockerfile**:

```dockerfile
# 基础镜像: 包含 CUDA 12.1
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 安装 Python 和基础工具
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# 安装 vLLM
RUN pip3 install --no-cache-dir vllm

# 设置工作目录
WORKDIR /app

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
```

**生产级 Dockerfile** (多阶段构建):

```dockerfile
# ==========================================
# 阶段 1: 构建阶段
# ==========================================
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级 pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 复制依赖文件
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# 阶段 2: 运行阶段
# ==========================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 只安装运行时依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制虚拟环境
COPY --from=builder /opt/venv /opt/venv

# 设置环境变量
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3

# 创建非 root 用户
RUN useradd -m -u 1000 appuser

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY --chown=appuser:appuser . .

# 切换到非 root 用户
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--model", "${MODEL_PATH}"]
```

**requirements.txt**:

```txt
vllm==0.6.0
torch==2.3.0
transformers==4.41.0
accelerate==0.30.0
fastapi==0.111.0
uvicorn[standard]==0.29.0
pydantic==2.7.0
```

---

### 4.4.2 Docker Compose 配置

**docker-compose.yml**:

```yaml
version: '3.8'

services:
  vllm-server:
    build:
      context: .
      dockerfile: Dockerfile
    image: llm-inference:latest
    container_name: vllm-server

    # GPU 配置
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

    # 环境变量
    environment:
      - MODEL_PATH=meta-llama/Llama-2-7b-chat-hf
      - GPU_MEMORY_UTILIZATION=0.9
      - MAX_MODEL_LEN=4096
      - NUM_GPU=1

    # 端口映射
    ports:
      - "8000:8000"

    # 共享内存
    shm_size: '10g'

    # 数据卷
    volumes:
      - model-cache:/root/.cache/huggingface
      - logs:/app/logs

    # 网络
    networks:
      - llm-network

    # 重启策略
    restart: unless-stopped

    # 健康检查
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

    # 日志
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"

  # 可选: Nginx 反向代理
  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    networks:
      - llm-network
    depends_on:
      - vllm-server
    restart: unless-stopped

  # 可选: Prometheus 监控
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - llm-network
    restart: unless-stopped

  # 可选: Grafana 可视化
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    networks:
      - llm-network
    restart: unless-stopped

# 数据卷
volumes:
  model-cache:
  logs:
  prometheus-data:
  grafana-data:

# 网络
networks:
  llm-network:
    driver: bridge
```

**启动服务**:

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f vllm-server

# 停止服务
docker-compose down

# 停止并删除数据卷
docker-compose down -v
```

---

### 4.4.3 多阶段构建优化

**为什么要多阶段构建?**

```
单阶段构建:
├── 基础镜像: 5GB
├── 构建工具: 2GB
├── 源代码: 500MB
├── 编译产物: 3GB
└── 最终镜像: 10.5GB ❌

多阶段构建:
┌─ 构建阶段 ─────────────────┐
│ 基础镜像: 5GB              │
│ 构建工具: 2GB              │
│ 源代码: 500MB              │
└──────────────┬─────────────┘
               │ 只复制编译产物
┌─ 运行阶段 ───▼─────────────┐
│ 基础镜像: 5GB              │
│ 编译产物: 3GB              │
└────────────────────────────┘
│ 最终镜像: 8GB ✅           │
```

**优势**:
- ✅ 更小的镜像体积 (节省存储和传输)
- ✅ 更高的安全性 (不包含源代码和构建工具)
- ✅ 更快的部署速度

---

### 4.4.4 数据卷管理

**三种挂载方式**:

```yaml
volumes:
  # 1. 命名卷 (Docker 管理)
  - model-cache:/root/.cache/huggingface

  # 2. 绑定挂载 (宿主机路径)
  - /path/on/host:/path/in/container

  # 3. 临时卷 (tmpfs)
  - tmpfs-data:/tmp:rw,size=1g
```

**最佳实践**:

```yaml
volumes:
  # 模型缓存 (持久化)
  - model-cache:/root/.cache/huggingface

  # 日志 (持久化)
  - ./logs:/app/logs

  # 配置文件 (只读)
  - ./config:/app/config:ro

  # 临时文件 (内存)
  - /tmp:rw,size=1g
```

---

## 4.5 基础推理示例

### 4.5.1 单次推理

**Python API**:

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    gpu_memory_utilization=0.9,
    max_model_len=4096,
)

# 采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)

# 输入文本
prompts = [
    "Hello, my name is",
    "The future of AI is",
]

# 推理
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for i, output in enumerate(outputs):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt {i}: {prompt}")
    print(f"Generated: {generated_text}\n")
```

**OpenAI API**:

```python
import openai

# 配置本地端点
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "dummy"

response = openai.ChatCompletion.create(
    model="meta-llama/Llama-2-7b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=100,
)

print(response.choices[0].message.content)
```

---

### 4.5.2 批量推理

**Python API**:

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=2,  # 使用 2 个 GPU
)

# 批量输入
prompts = [
    "Write a short story about a robot.",
    "Explain quantum computing.",
    "What is the meaning of life?",
    "Describe the perfect day.",
    "How does the internet work?",
]

# 采样参数
sampling_params = SamplingParams(
    n=1,  # 每个 prompt 生成 1 个结果
    temperature=0.8,
    top_p=0.95,
    max_tokens=256,
)

# 批量推理
outputs = llm.generate(prompts, sampling_params)

# 保存结果
results = []
for output in outputs:
    results.append({
        "prompt": output.prompt,
        "generated": output.outputs[0].text,
        "tokens": len(output.outputs[0].token_ids),
    })

# 打印统计
import json
print(json.dumps(results, indent=2, ensure_ascii=False))
```

**性能优化建议**:
- ✅ 使用更大的 batch size 提高吞吐量
- ✅ 预处理 prompt,减少运行时开销
- ✅ 使用异步 API 处理大量请求

---

### 4.5.3 流式输出

**服务器端配置**:

```python
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams

# 启用流式输出
engine_args = AsyncEngineArgs(
    model="meta-llama/Llama-2-7b-chat-hf",
    enable_prefix_caching=True,
)

engine = AsyncLLMEngine.from_engine_args(engine_args)
```

**客户端使用**:

```python
import asyncio
from openai import AsyncOpenAI

async def stream_chat():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy",
    )

    stream = await client.chat.completions.create(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=[
            {"role": "user", "content": "Tell me a long story."}
        ],
        stream=True,
        max_tokens=500,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)

# 运行
asyncio.run(stream_chat())
```

**curl 示例**:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

### 4.5.4 性能基准测试

**简单的基准测试脚本**:

```python
import time
import numpy as np
from vllm import LLM, SamplingParams

def benchmark(llm, prompts, sampling_params, num_iterations=10):
    latencies = []

    for _ in range(num_iterations):
        start_time = time.time()

        outputs = llm.generate(prompts, sampling_params)

        end_time = time.time()
        latencies.append(end_time - start_time)

    # 统计
    latencies = np.array(latencies)
    print(f"平均延迟: {np.mean(latencies):.3f} 秒")
    print(f"P50 延迟: {np.percentile(latencies, 50):.3f} 秒")
    print(f"P99 延迟: {np.percentile(latencies, 99):.3f} 秒")
    print(f"吞吐量: {len(prompts) / np.mean(latencies):.2f} 请求/秒")

# 运行基准测试
llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=128,
)

prompts = ["Hello, world!"] * 32  # 批量 32 个请求

benchmark(llm, prompts, sampling_params)
```

**使用 Apache Bench**:

```bash
# 安装 ab
sudo apt-get install apache2-utils

# 运行基准测试
ab -n 1000 -c 10 -T 'application/json' \
  -p request.json \
  http://localhost:8000/v1/chat/completions
```

**request.json**:
```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

---

## 4.6 开发工具推荐

### 4.6.1 代码编辑器配置

**VS Code** (推荐):

**推荐插件**:
- Python
- Pylance
- Docker
- Jupyter
- GitLens
- Thunder Client (API 测试)

**VS Code 配置** (`.vscode/settings.json`):

```json
{
  "python.defaultInterpreterPath": "/opt/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true
  }
}
```

**PyCharm**:
- 内置强大的 Python 支持
- Docker 集成
- 性能分析工具

---

### 4.6.2 调试工具

**Python 调试器**:

```python
# 使用 pdb
import pdb; pdb.set_trace()

# 使用 ipdb (更友好)
import ipdb; ipdb.set_trace()

# VS Code 调试配置
# .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Current File",
      "type": "debugpy",
      "request": "launch",
      "program": "${file}",
      "console": "integratedTerminal",
      "env": {
        "CUDA_VISIBLE_DEVICES": "0"
      }
    }
  ]
}
```

**NVIDIA Nsight** (GPU 性能分析):
```bash
# 安装 Nsight Systems
sudo apt-get install nsight-systems

# 分析 GPU 性能
nsys profile python your_script.py

# 查看结果
nsys stats report.nsys-rep
```

---

### 4.6.3 性能分析工具

**nvtop** (GPU 监控):

```bash
# 安装
sudo apt-get install nvtop

# 运行
nvtop
```

**GPUtil** (Python):

```python
import GPUtil
GPUtil.showUtilization()
```

**自定义监控脚本**:

```python
import time
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

while True:
    # GPU 利用率
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"GPU 利用率: {util.gpu}%")

    # 内存使用
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"内存: {info.used / 1024**3:.2f}GB / {info.total / 1024**3:.2f}GB")

    # 温度
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    print(f"温度: {temp}°C")

    # 功耗
    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    print(f"功耗: {power}W")

    print("-" * 40)
    time.sleep(1)
```

---

### 4.6.4 可视化工具

**TensorBoard**:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

# 记录指标
writer.add_scalar('Latency', latency, step)
writer.add_scalar('Throughput', throughput, step)
writer.add_scalar('GPU_Memory', gpu_memory, step)

# 启动 TensorBoard
# tensorboard --logdir runs
```

**Grafana + Prometheus**:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
```

---

## 4.7 常见问题排查

### 4.7.1 CUDA 版本不兼容

**问题**: `CUDA_ERROR_INVALID_DEVICE`

**原因**: 驱动版本与 CUDA 版本不匹配

**解决方案**:

```bash
# 1. 检查驱动版本
nvidia-smi

# 2. 检查容器内 CUDA 版本
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 3. 使用兼容的 Docker 镜像
# CUDA 12.1 需要 Driver >= 525
# CUDA 11.8 需要 Driver >= 450
```

**版本兼容表**:
| CUDA 版本 | 最低驱动版本 |
|-----------|-------------|
| 12.x      | 525+        |
| 11.x      | 450+        |
| 10.x      | 410+        |

---

### 4.7.2 Docker GPU 访问问题

**问题**: `could not select device driver`

**原因**: NVIDIA Container Toolkit 配置不正确

**解决方案**:

```bash
# 1. 检查 Docker 运行时
docker info | grep nvidia

# 2. 重新配置
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. 测试
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 4. 如果还不行,检查默认运行时
# 编辑 /etc/docker/daemon.json
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

sudo systemctl restart docker
```

---

### 4.7.3 端口冲突处理

**问题**: `port is already allocated`

**解决方案**:

```bash
# 1. 查看占用端口的进程
sudo lsof -i :8000

# 2. 杀掉占用端口的进程
sudo kill -9 <PID>

# 3. 或者使用其他端口
docker run --gpus all -p 8001:8000 vllm/vllm-openai:latest
```

---

### 4.7.4 依赖安装失败

**问题**: pip 安装失败

**解决方案**:

```bash
# 1. 升级 pip
pip install --upgrade pip

# 2. 清理缓存
pip cache purge

# 3. 使用预编译包
pip install --only-binary :all: vllm

# 4. 如果还是失败,使用 conda
conda install -c conda-forge vllm

# 5. 检查 Python 版本 (需要 3.8+)
python --version
```

---

## ✅ 章节检查清单

完成本章后,你应该能够:

- [ ] 理解为什么使用 Docker 进行环境隔离
- [ ] 在本地搭建完整的 LLM 推理环境
- [ ] 使用 vLLM 启动推理服务
- [ ] 编写生产级的 Dockerfile 和 docker-compose.yml
- [ ] 使用 OpenAI API 兼容的接口进行推理
- [ ] 排查常见的环境问题

---

## 📚 动手练习

**练习 4.1**: 从零搭建 vLLM 开发环境

1. 安装 Docker 和 NVIDIA Container Toolkit
2. 拉取 vLLM Docker 镜像
3. 启动 Llama-2-7b 推理服务
4. 使用 curl 发送测试请求
5. 验证服务正常工作

**练习 4.2**: Docker 化一个推理服务

1. 编写 Dockerfile,构建自定义 vLLM 镜像
2. 配置 docker-compose.yml,包含:
   - vLLM 服务
   - Nginx 反向代理
   - 基本的监控
3. 启动完整的服务栈
4. 测试服务的可用性
5. 清理所有资源

---

## 🎯 总结

**关键要点**:
- Docker 是确保环境一致性的最佳方式
- 从宿主机到生产环境使用同一个 Docker 镜像
- vLLM 提供高性能、易用的推理服务
- Docker Compose 简化多服务编排
- 掌握基本的调试和监控工具

**下一章**: 第5章 vLLM 深入——理解 PagedAttention 和 Continuous Batching 的原理。

---

**有问题?加入 [第4章 Discord 频道](https://discord.gg/TODO) 讨论!**
