---
id: "chapters-chapter04-environment-setup"
title: "第4章：环境搭建"
slug: "chapters-chapter04-environment-setup"
date: "2026-03-11"
type: "article"
topics:
  - "environment-setup"
concepts: []
tools:
  - "docker"
  - "cuda"
  - "vllm"
architecture_layer:
  - "hardware-and-runtime"
learning_stage: "foundations"
optimization_axes:
  - "operability"
  - "latency"
related:
  - "chapters-chapter03-gpu-basics"
  - "chapters-chapter05-llm-inference-basics"
  - "appendix-b-troubleshooting"
references:
  - "https://docs.vllm.ai/en/latest/getting_started/installation/gpu/"
  - "https://docs.vllm.ai/en/stable/deployment/docker/"
  - "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
status: "published"
display_order: 5
---
# 第4章 环境搭建

> **💰 商业动机**：环境问题是最“无聊”但也最昂贵的推理成本之一。可复现的制品、兼容矩阵和清晰的排障路径，能减少版本漂移造成的发布失败，并让恢复时间成为可度量、可改进的指标。

## 简介

在深入优化技术之前，我们需要先搭建一个**可复现、可观测、可回滚**的开发环境。很多工程师在这一步花费了太多时间：CUDA 版本冲突、Docker 权限问题、驱动与容器运行时不匹配、依赖版本漂移……这些问题会持续拖慢你后续所有章节的推进速度。

为了更像“书”而不是“安装教程”，本章用同一套叙事框架组织内容：

- **背景**：为什么推理环境问题会反复出现，并且一旦出现就很难排查？
- **决策**：什么时候该用 Docker，什么时候可以用 venv/conda？宿主机需要装什么、不需要装什么？
- **落地**：按顺序把驱动、容器、Python、推理框架跑通，并跑一个最小闭环的验证。
- **踩坑**：列出高频故障模式（版本不兼容、GPU 访问、端口/权限/依赖）以及最快的定位路线。
- **指标**：环境是否“可用”，不是凭感觉，而是能否通过一组最小验证用例（GPU 可见性、容器可跑、服务可请求）。

本章将帮你：
- 理解为什么使用 Docker 进行环境隔离
- 从零搭建完整的 LLM 推理环境
- 快速启动你的第一个 vLLM 推理服务
- 掌握容器化部署的最佳实践
- 学会排查常见的环境问题

**学完本章，你将拥有一个可复现的推理开发/验证环境，并且具备向生产演进所需的最小骨架。**

---

## 4.1 开发环境概览

### 4.1.1 为什么使用 Docker

你可能听过这样的话:"在我机器上能运行,为什么在你那就不行?"

**传统方式的问题**：
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

**Docker 的解决方案**：
```
Docker 容器:
- 固定的基础镜像
- 封装的 CUDA 版本
- 锁定的依赖版本
- 标准的运行环境

结果:
→ 用户态依赖与启动方式更容易复现
→ 镜像 tag/digest 可以进入发布和回滚记录
→ 宿主驱动、GPU、内核与容器运行时仍需单独核验
```

**如何验证价值**：

- 环境相关发布失败率与回滚率；
- 从拉取制品到通过 smoke test 的时间；
- 新环境首次成功部署时间；
- 相同 digest 在开发、测试和生产的差异项数量。

---

### 4.1.2 环境一致性: 本地 vs 生产

**三层环境一致性**：

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

**关键原则**：
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
│  - 目标版本支持的 Python                            │
│  - 与后端匹配的 CUDA / ROCm / XPU / Metal          │
│  - cuDNN / cuBLAS (CUDA 加速库)                   │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  驱动层 (Driver Layer)                             │
│  - 目标后端支持的宿主驱动                           │
│  - 已验证的 GPU / 加速器                            │
└──────────────┬──────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────┐
│  容器层 (Container Layer)                          │
│  - Docker                                          │
│  - NVIDIA Container Toolkit                       │
│  - Docker Compose                                  │
└─────────────────────────────────────────────────────┘
```

**每一层都很重要**：
- 应用层: 你的业务逻辑
- 框架层: 推理引擎的基础
- 运行时层: Python 和 CUDA 的版本兼容性
- 驱动层: 必须与 GPU 硬件匹配
- 容器层: 隔离和可移植性

---

## 4.2 基础环境安装

### 4.2.1 NVIDIA 驱动安装

**检查当前驱动版本**：

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

**关键信息**：
- **Driver Version**: 至少 525+ (推荐 535+)
- **CUDA Version**: 这是最高的 CUDA 版本支持,不一定是已安装的版本

**如果驱动版本过低或未安装**：

**Ubuntu/Debian**：
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

**CentOS/RHEL**：
```bash
# 添加 NVIDIA 仓库
sudo yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
sudo yum install -y https://nvidia.github.io/libnvidia-container/rhel8/nvidia-container-toolkit.repo

# 安装驱动
sudo yum install -y nvidia-driver

# 重启
sudo reboot
```

**云平台 (AWS/GCP/Azure)**：
- 通常已经预装 NVIDIA 驱动
- 使用官方的 GPU 优化 AMI/Image

---

### 4.2.2 CUDA Toolkit 配置

**重要说明**：Docker 容器中的 CUDA 不需要宿主机安装 CUDA Toolkit!

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

**最佳实践**：
- 宿主机: 只安装 NVIDIA 驱动
- Docker 容器: 使用带 CUDA 的基础镜像
- 避免在宿主机安装多个 CUDA 版本

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

**安装 Docker**：

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

**安装 NVIDIA Container Toolkit**：

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

**最小验证用例（推荐）**：

- 你可以直接运行本仓库的环境检查脚本，它会按顺序验证：宿主机 GPU 可见性、Docker 是否可用、容器内 GPU 是否可见。

```bash
bash code/chapter04/check_env.sh
```

---

### 4.2.4 Python 环境管理

**推荐方式**：使用 pyenv 或 conda 管理多个 Python 版本

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

**核心特性**：
- **高性能**: PagedAttention 等优化在多场景中可显著提升吞吐
- **连续批处理**: Continuous Batching,最大化 GPU 利用率
- **易用性**: 兼容 OpenAI API,一行代码启动服务
- **灵活性**: 支持多种量化格式、投机解码、前缀缓存

**适用场景**：
- 高吞吐量推理服务
- 多模型并发部署
- 需要低延迟的实时应用
- 生产环境部署

**不适用场景**：
- 需要极高灵活性的研究实验 (Transformers 更灵活)
- 需要最大化的模型灵活性
- 超大模型的模型并行 (vLLM 支持有限)

---

### 4.3.2 vLLM vs 其他推理框架

框架能力和支持矩阵变化很快，不使用“高/中/低”给出永久排名。先按以下问题筛选候选项，再在同一模型制品、流量和 SLO 下验证：

| 决策维度 | 需要核对的证据 |
|----------|----------------|
| 模型与硬件支持 | 目标版本支持矩阵、能否加载指定制品 |
| 服务接口 | 流式输出、鉴权、取消、结构化输出和错误语义 |
| 性能 | 同负载下的 TTFT、TPOT、goodput、显存和成本 |
| 运维 | metrics、trace、滚动升级、故障回退和多租户隔离 |
| 扩展能力 | 并行、量化、KV 传输、插件与自定义 kernel 边界 |

本书以 vLLM 作为贯穿示例，不代表它对所有模型、硬件和组织都是默认最优解。

---

### 4.3.3 安装 vLLM

安装命令必须与 GPU 后端、Python 和目标 vLLM 版本配套。官方文档当前推荐使用隔离环境，并针对后端选择预编译 wheel；出版后这些条件仍可能变化。[CITE: vllm-installation-gpu-stable]

**方式 1：预编译 wheel（开发验证）**：

```bash
# Python 版本按目标版本支持矩阵选择
uv venv --python <supported-python> --seed
source .venv/bin/activate

# NVIDIA CUDA 后端示例；其他后端不能照抄
uv pip install "vllm==<validated-version>" --torch-backend=auto

# 验证安装
python -c "import vllm; print(vllm.__version__)"
```

安装后应保存 lockfile、Python 版本、wheel 来源、GPU 和驱动信息；只有一个安装命令不足以复现环境。

**方式 2：从源码安装（仅在需要修改源码时）**：

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout <validated-commit>

# 具体构建命令按该 commit 的官方文档执行
uv pip install -e . --torch-backend=auto
```

**方式 3：官方 Docker 镜像**：

```bash
# 选定经过验证的 release tag，并记录解析后的 digest
docker pull vllm/vllm-openai:<validated-tag>
docker image inspect vllm/vllm-openai:<validated-tag> --format '{{index .RepoDigests 0}}'

# 生产部署使用 registry/repository@sha256:<validated-digest>
```

`latest` 可以用于一次性试跑，但不能作为可回滚的生产制品标识。

---

### 4.3.4 启动第一个推理服务

**最简单的启动方式**：

```bash
# OpenAI API 兼容服务器
vllm serve MODEL --host 0.0.0.0 --port 8000
```

**使用 Docker**：

```bash
docker run --gpus all \
    --ipc=host \
    -p 8000:8000 \
    registry/repository@sha256:<validated-digest> \
    --model MODEL
```

官方镜像为 `vllm/vllm-openai`，并要求按运行方式配置 GPU 与共享内存；生产环境应在此基础上锁定 tag/digest。[CITE: vllm-installation-gpu-stable]

**测试推理服务**：

```bash
# 使用 curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'

# 使用 Python (OpenAI SDK v1+ 接口)
from openai import OpenAI

# 配置本地端点
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy",  # vLLM 不验证 key
)

response = client.chat.completions.create(
    model="MODEL",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

print(response.choices[0].message.content)
```

**启动参数实验骨架**：

```bash
vllm serve MODEL \
    --tensor-parallel-size <validated> \
    --gpu-memory-utilization <candidate> \
    --max-model-len <product-limit> \
    --dtype <artifact-compatible-dtype> \
    --host 0.0.0.0 \
    --port 8000
```

量化格式不是任意模型都能启用的服务开关；它必须与模型制品、kernel 和质量回归配套。

---

## 4.4 Docker 容器化部署

### 4.4.1 Dockerfile 编写

不要从任意 CUDA 基础镜像手工拼接一组 PyTorch、vLLM、Transformers 和 CUDA 版本，并把它称为“生产级”。这些包存在编译和 ABI 约束，表面上的版本锁定不等于兼容。

更稳妥的路线是从经过验证的官方 release 镜像派生，只增加业务必需层：

```dockerfile
# 构建系统将该 ARG 替换为已验证且带 digest 的官方制品
ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}

# 只复制业务需要的、已经锁定的附加依赖
COPY requirements-extra.lock /tmp/requirements-extra.lock
RUN uv pip install --system --require-hashes \
    -r /tmp/requirements-extra.lock

# 不在镜像里写死模型密钥；模型制品也应有独立 revision/digest
```

发布物必须同时记录：基础镜像 digest、附加依赖 lockfile、模型 revision、启动参数、GPU/驱动/运行时矩阵、smoke test 结果和回滚 digest。官方文档特别提醒，基于官方镜像增加 optional dependencies 时，附加安装的 vLLM 版本必须与基础镜像匹配。[CITE: vllm-installation-gpu-stable]

---

### 4.4.2 Docker Compose 配置

**docker-compose.yml 骨架**（所有 digest、模型和容量值由验证流水线填入）：

```yaml
services:
  vllm-server:
    image: registry/llm-inference@sha256:<validated-digest>
    container_name: vllm-server

    # GPU 配置 (Compose 模式)
    gpus: all

    # 环境变量
    environment:
      - MODEL_PATH=${MODEL_PATH}

    # 端口映射
    ports:
      - "8000:8000"

    # 共享内存
    shm_size: '10g'

    # 数据卷
    volumes:
      - model-cache:/home/vllm/.cache/huggingface
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
    image: nginx@sha256:<validated-digest>
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
    image: prom/prometheus@sha256:<validated-digest>
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
    image: grafana/grafana@sha256:<validated-digest>
    container_name: grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD_FILE=/run/secrets/grafana_admin_password
    secrets:
      - grafana_admin_password
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

secrets:
  grafana_admin_password:
    file: ./secrets/grafana_admin_password
```

**启动服务**：

```bash
# 构建并启动
docker compose up -d

# 查看日志
docker compose logs -f vllm-server

# 停止服务
docker compose down

# 停止并删除数据卷
docker compose down -v
```

---

### 4.4.3 多阶段构建优化

多阶段构建适用于必须在本地编译扩展或业务组件的情况：构建阶段包含编译器和源码，运行阶段只复制运行时产物。是否真的减小镜像、漏洞面或发布时间，应由镜像 layer、SBOM、漏洞扫描和拉取时间验证，不能套用固定 GB 数字。

若完全使用官方预编译 vLLM 镜像且只添加少量 Python 依赖，多阶段构建未必带来价值；优先保持派生层最少并锁定全部输入制品。

---

### 4.4.4 数据卷管理

**三种挂载方式**：

```yaml
volumes:
  # 1. 命名卷 (Docker 管理)
  - model-cache:/root/.cache/huggingface

  # 2. 绑定挂载 (宿主机路径)
  - /path/on/host:/path/in/container

  # 3. 临时卷 (tmpfs)
  - tmpfs-data:/tmp:rw,size=1g
```

**最佳实践**：

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

**Python API**：

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="MODEL",
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

**OpenAI API**：

```python
from openai import OpenAI

# 配置本地端点
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="MODEL",
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

**Python API**：

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="MODEL",
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

**性能优化建议**：
- 扫描并发和 token budget；更大的 batch 不保证更好的尾延迟或 goodput
- 预处理 prompt,减少运行时开销
- 使用异步 API 处理大量请求

---

### 4.5.3 流式输出

OpenAI-compatible server 已提供流式接口。不要为了流式输出依赖 `AsyncLLMEngine` 等内部模块路径；它们可能随版本调整。

**客户端使用**：

```python
import asyncio
from openai import AsyncOpenAI

async def stream_chat():
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy",
    )

    stream = await client.chat.completions.create(
        model="MODEL",
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

**curl 示例**：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

### 4.5.4 性能基准测试

不要用十次同步调用计算“P99”。优先使用目标版本随附的 benchmark 工具，并先查看其子命令与参数：

```bash
vllm bench --help
```

基准记录至少包括：版本和 digest、模型 revision、硬件拓扑、dtype/量化、prompt/output 分布、到达过程、并发、warm-up、样本量、TTFT/TPOT/E2E 分位数、输入/输出 token 吞吐、错误率与原始结果文件。只有这些条件一致，结果才可比较。

---

## 4.6 开发工具推荐

### 4.6.1 代码编辑器配置

**VS Code** (推荐):

**推荐插件**：
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

**PyCharm**：
- 内置强大的 Python 支持
- Docker 集成
- 性能分析工具

---

### 4.6.2 调试工具

**Python 调试器**：

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

**自定义监控脚本**：

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

**TensorBoard**：

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

**Grafana + Prometheus**：

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

**问题**：`CUDA_ERROR_INVALID_DEVICE`

**原因**：可能涉及宿主驱动、容器所带 CUDA compatibility libraries、GPU 架构或 PyTorch/vLLM wheel 不匹配，不能只比较两个版本号。

**解决方案**：

```bash
# 1. 检查驱动版本
nvidia-smi

# 2. 记录目标镜像 digest，并运行该镜像的 smoke test
docker run --rm --gpus all <validated-cuda-image@sha256:digest> nvidia-smi

# 3. 对照目标镜像、GPU 和后端版本的官方支持矩阵
```

不要维护“CUDA 12.x → 某个最低驱动”这种过度简化表。CUDA minor compatibility、forward compatibility、GPU 类型和容器内兼容库会改变边界，应记录实际驱动与镜像 digest，并链接该制品对应的官方兼容说明。

---

### 4.7.2 Docker GPU 访问问题

**问题**：`could not select device driver`

**原因**：NVIDIA Container Toolkit 配置不正确

**解决方案**：

```bash
# 1. 检查 Docker 运行时
docker info | grep nvidia

# 2. 重新配置
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. 测试
docker run --rm --gpus all <validated-cuda-image@sha256:digest> nvidia-smi
```

上述 `nvidia-ctk` 流程来自 NVIDIA Container Toolkit 官方安装指南；rootless Docker、containerd、CRI-O 与 Podman 使用不同配置路径，不应手写一份通用 `daemon.json`。[CITE: nvidia-container-toolkit-install]

---

### 4.7.3 端口冲突处理

**问题**：`port is already allocated`

**解决方案**：

```bash
# 1. 查看占用端口的进程
sudo lsof -i :8000

# 2. 识别进程归属后再决定停止方式；不要默认使用 SIGKILL
sudo kill <PID>

# 3. 或者使用其他端口
docker run --gpus all --ipc=host -p 8001:8000 \
  registry/repository@sha256:<validated-digest> --model MODEL
```

---

### 4.7.4 依赖安装失败

**问题**：pip 安装失败

**解决方案**：

```bash
# 1. 记录 Python、OS/glibc、GPU 后端和驱动
python --version
nvidia-smi

# 2. 对照目标版本官方安装矩阵，使用新的隔离环境复现
uv venv --python <supported-python> --seed .venv-repro

# 3. 安装明确版本并保存完整日志；不要在失败环境里反复无界升级
uv pip install "vllm==<validated-version>" --torch-backend=auto -v
```

---

## 章节检查清单

完成本章后,你应该能够:

- [ ] 理解为什么使用 Docker 进行环境隔离
- [ ] 在本地搭建完整的 LLM 推理环境
- [ ] 使用 vLLM 启动推理服务
- [ ] 编写可锁定 digest、依赖和模型 revision 的容器配置
- [ ] 使用 OpenAI API 兼容的接口进行推理
- [ ] 排查常见的环境问题

---

## 动手练习

**练习 4.1**：从零搭建 vLLM 开发环境

1. 安装 Docker 和 NVIDIA Container Toolkit
2. 拉取 vLLM Docker 镜像
3. 选择一个目标版本支持且有权限访问的小模型，启动推理服务
4. 使用 curl 发送测试请求
5. 验证服务正常工作

**练习 4.2**：Docker 化一个推理服务

1. 编写 Dockerfile,构建自定义 vLLM 镜像
2. 配置 docker-compose.yml,包含:
   - vLLM 服务
   - Nginx 反向代理
   - 基本的监控
3. 启动完整的服务栈
4. 测试服务的可用性
5. 清理所有资源

---

## 本章小结

关键要点：
- Docker 是确保环境一致性的最佳方式
- 从宿主机到生产环境使用同一个 Docker 镜像
- vLLM 提供高性能、易用的推理服务
- Docker Compose 简化多服务编排
- 掌握基本的调试和监控工具

## 章节衔接

到这一章为止，实验与部署底座已经搭好。下一章开始，书的主线会正式进入推理过程本身：先用第5章把 prefill、decode、attention 和核心指标讲清楚，再顺着这些代价结构进入后面的 KV 管理、请求调度和量化优化。

---
