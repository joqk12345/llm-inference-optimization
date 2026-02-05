# 第10章: 生产环境部署

> "在开发环境能运行是运气,在生产环境稳定运行才是本事。" - 佚名

## 简介

从开发环境到生产环境,这不是一个简单的"复制粘贴"过程,而是一次质的飞跃。经验上,环境问题的排查常常需要数小时,而规范化配置可显著缩短部署与排障时间。

**💰 成本影响**(经验区间,需基准测试验证)
- **可用性提升**: 设定更高SLA目标通常会降低故障成本,但也增加架构与运维投入
- **监控ROI**: 及时发现问题,避免资源浪费
- **成本优化**: 通过Spot实例等策略可能显著降低云GPU成本

在本章中,你将学习:
- 生产环境与开发环境的关键差异
- 如何设计高可用的部署架构
- Kubernetes部署最佳实践
- 监控与可观测性体系建设
- 性能调优的完整流程
- 成本优化策略与ROI监控
- 安全性与灾备方案

本章结束后,你将能够:
- ✅ 设计并部署高可用的LLM服务
- ✅ 搭建完整的监控体系
- ✅ 实施有效的成本优化策略
- ✅ 处理生产环境的常见问题

> **数值说明**: 本章出现的阈值、价格、成本与性能数字均为示例或经验值,需结合你的硬件、负载与SLA目标进行校准。

---

## 10.1 生产环境 vs 开发环境

### 10.1.1 关键差异

| 维度 | 开发环境 | 生产环境 |
|------|---------|---------|
| **可用性要求** | 可以接受停机 | 有SLA目标(示例: 99.9%) |
| **负载特征** | 低并发,测试流量 | 高并发,真实用户 |
| **监控** | 基本日志即可 | 完整可观测性体系 |
| **安全** | 宽松 | 严格的认证授权 |
| **成本优化** | 不关心 | 必须优化 |
| **故障恢复** | 手动重启 | 自动恢复 |
| **容量规划** | 猜估 | 基于数据的预测 |

### 10.1.2 生产环境的特殊要求

**1. 高可用性(High Availability)**

```yaml
# 单点故障是生产环境的大忌
架构设计:
  - 多副本部署(至少2个实例)
  - 跨可用区分布(AZ分布)
  - 自动故障转移
  - 健康检查机制
```

**2. 可观测性(Observability)**

生产环境需要三大支柱:

```python
# Metrics(指标)
- 请求延迟(P50, P95, P99)
- 吞吐量(tokens/s)
- GPU利用率
- 错误率

# Logs(日志)
- 结构化日志(JSON格式)
- 日志分级(ERROR, WARN, INFO)
- 请求追踪ID

# Traces(追踪)
- 端到端请求追踪
- 依赖关系可视化
- 性能瓶颈定位
```

**3. 弹性伸缩(Elasticity)**

```yaml
自动伸缩策略:
  - 基于CPU/GPU利用率
  - 基于请求队列长度
  - 基于时间窗口(业务高峰)
  - Spot实例自动替换
```

### 10.1.3 SLA定义

**SLA(Service Level Agreement)是生产环境的承诺**

| 指标 | 定义 | 目标值 | 监控方式 |
|------|------|--------|---------|
| **可用性** | 服务正常运行时间比例 | 按SLA目标 | 健康检查 + 告警 |
| **延迟** | 请求响应时间 | 按SLA目标 | Prometheus |
| **吞吐量** | 每秒处理的token数 | 依模型与硬件 | 指标监控 |
| **错误率** | 失败请求比例 | 尽量低 | 日志分析 |

**可用性与成本的关系**:

```
更高的可用性目标通常意味着更高的架构与运维成本。
具体SLA应结合业务风险、流量规模与预算综合确定。
```

---

## 10.2 部署架构设计

### 10.2.1 单机部署

**适用场景**:
- 开发测试环境
- 小规模内部工具
- 低并发场景

```bash
# 单机部署示例
vllm serve meta-llama/Llama-3.1-8B \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 8192 \
  --port 8000
```

**架构图**:

```
┌─────────────────────────────────────┐
│         单服务器 (RTX 4090)          │
│                                     │
│  ┌─────────┐  ┌─────────┐          │
│  │ vLLM    │  │ vLLM    │  多副本  │
│  │ :8000   │  │ :8001   │          │
│  └────┬────┘  └────┬────┘          │
│       │            │               │
│       └─────┬──────┘               │
│             ▼                      │
│       ┌──────────┐                 │
│       │ Nginx    │  负载均衡       │
│       │ :80      │                 │
│       └──────────┘                 │
└─────────────────────────────────────┘
```

### 10.2.2 多机部署(模型并行)

**适用场景**:
- 大模型(70B+)无法放入单卡
- 需要更高吞吐量
- 生产环境高可用

```bash
# 使用Ray启动多节点集群
# 在头节点(head node)
ray start --head --port=6379

# 在工作节点(worker nodes)
ray start --address=<head-node-ip>:6379

# 启动vLLM服务(自动分布式)
vllm serve meta-llama/Llama-3.1-70B \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

**架构图**:

```
          ┌──────────────────┐
          │   Load Balancer  │
          │   (Nginx/ALB)    │
          └────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼────┐    ┌───▼────┐    ┌───▼────┐
│ Node 1 │    │ Node 2 │    │ Node 3 │
│ GPU 0-3│    │ GPU 0-3│    │ GPU 0-3│
│ TP=4   │    │ TP=4   │    │ TP=4   │
└────────┘    └────────┘    └────────┘
```

### 10.2.3 负载均衡策略

**1. 轮询(Round Robin)**

```nginx
upstream vllm_backend {
    server 10.0.1.10:8000;
    server 10.0.1.11:8000;
    server 10.0.1.12:8000;
}

server {
    listen 80;
    location /v1/chat/completions {
        proxy_pass http://vllm_backend;
    }
}
```

**2. 最少连接(Least Connections)**

```nginx
upstream vllm_backend {
    least_conn;
    server 10.0.1.10:8000;
    server 10.0.1.11:8000;
    server 10.0.1.12:8000;
}
```

**3. Session-Aware路由**(重要!)

```python
# 对于有状态服务(LLM对话),需要session stickiness
# 使用一致性哈希确保同一请求路由到同一节点

import hashlib

def get_worker_id(session_id: str, num_workers: int) -> int:
    """一致性哈希路由"""
    hash_val = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
    return hash_val % num_workers
```

**为什么需要Session-Aware路由?**

```
问题场景:
- 用户A的第一轮对话 → Node 1
- 用户A的第二轮对话 → Node 2 (不同的节点!)
- Node 2没有KV Cache → 需要重新prefill整个历史

解决方案:
- 使用session_id进行一致性哈希
- 确保同一session的请求路由到同一节点
- KV Cache可以复用,TTFT通常降低
```

### 10.2.4 高可用架构

**完整的高可用架构**:

```
┌──────────────────────────────────────────────────┐
│                   CDN/WAF                        │
│              (CloudFlare/AWS WAF)                │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│              Load Balancer (ALB/SLB)              │
│              Health Checks + Auto Failover        │
└────┬──────────────────────────────┬──────────────┘
     │                              │
┌────▼───────┐              ┌───────▼────┐
│  AZ 1      │              │   AZ 2     │
│  ┌──────┐  │              │   ┌──────┐ │
│  │Node1 │  │              │   │Node3 │ │
│  │Node2 │  │              │   │Node4 │ │
│  └──────┘  │              │   └──────┘ │
│  共享存储   │              │   共享存储  │
│  (EFS/S3)  │              │  (EFS/S3)  │
└────────────┘              └────────────┘
     │                              │
     └──────────┬───────────────────┘
                │
┌───────────────▼──────────────────────┐
│         监控告警系统                  │
│  (Prometheus + Grafana + AlertMgr)   │
└──────────────────────────────────────┘
```

---

## 10.3 Kubernetes部署

### 10.3.1 K8s基础概念

**核心概念映射**:

| K8s概念 | LLM服务对应 |
|---------|-----------|
| **Pod** | 一个vLLM实例 |
| **Deployment** | vLLM副本管理 |
| **Service** | 服务发现与负载均衡 |
| **ConfigMap** | 配置管理(模型参数) |
| **Secret** | 敏感信息(API密钥) |
| **HPA** | 水平自动伸缩 |

### 10.3.2 部署vLLM到K8s

**Deployment配置**:

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-8b
  labels:
    app: vllm
    model: llama3-8b
spec:
  replicas: 3  # 3个副本
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 1  # 每个Pod 1个GPU
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-3.1-8B"
        - name: GPU_MEMORY_UTILIZATION
          value: "0.9"
        - name: MAX_MODEL_LEN
          value: "8192"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
```

**Service配置**:

```yaml
# vllm-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer  # 或 ClusterIP
```

**部署命令**:

```bash
# 应用配置
kubectl apply -f vllm-deployment.yaml
kubectl apply -f vllm-service.yaml

# 查看状态
kubectl get pods -w
kubectl logs -f deployment/vllm-llama3-8b

# 扩缩容
kubectl scale deployment vllm-llama3-8b --replicas=5
```

### 10.3.3 配置管理

**使用ConfigMap管理配置**:

```yaml
# vllm-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-config
data:
  MODEL_NAME: "meta-llama/Llama-3.1-8B"
  GPU_MEMORY_UTILIZATION: "0.9"
  MAX_MODEL_LEN: "8192"
  DTYPE: "half"
  KV_CACHE_DTYPE: "fp8"
```

**引用ConfigMap**:

```yaml
envFrom:
- configMapRef:
    name: vllm-config
```

### 10.3.4 资源调度与GPU共享

**GPU共享(NVIDIA GPU Operator)**:

```yaml
# 使用时间切片共享GPU
apiVersion: v1
kind: Pod
metadata:
  name: vllm-shared-gpu
spec:
  containers:
  - name: vllm
    image: vllm/vllm-openai:latest
    resources:
      limits:
        nvidia.com/gpu: 1  # 请求1个GPU
        nvidia.com/mig-1g.5gb: 2  # 或使用MIG分区
```

**节点选择与亲和性**:

```yaml
# 确保Pod调度到GPU节点
spec:
  containers:
  - name: vllm
    resources:
      limits:
        nvidia.com/gpu: 1
  nodeSelector:
    gpu-type: nvidia-a100  # 选择特定GPU类型
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

**优先级调度**(保证关键任务):

```yaml
# priority-class.yaml
apiVersion: scheduling.k8s.io/v1
kind: PriorityClass
metadata:
  name: high-priority-vllm
value: 1000
globalDefault: false
description: "高优先级vLLM服务"
```

---

## 10.4 监控与可观测性

### 10.4.1 关键监控指标

**业务指标**:

```python
# 1. 延迟指标
指标:
  - TTFT (Time To First Token)
  - TPOT (Time Per Output Token)
  - 端到端延迟

目标:
  - TTFT / TPOT 需与SLA一致
```

```python
# 2. 吞吐量指标
指标:
  - tokens/second
  - requests/second
  - GPU利用率

目标:
  - tokens/s 与GPU利用率需结合模型与硬件设定
```

```python
# 3. 质量指标
指标:
  - 错误率
  - 超时率
  - OOM频率

目标:
  - 错误率与超时率尽量低,按SLA设定
```

**系统指标**:

```bash
# GPU指标
nvidia-smi dmon -s u -c 1  # GPU利用率
nvidia-smi dmon -s m -c 1  # 显存使用

# 系统指标
top -bn1 | grep "Cpu(s)"  # CPU使用率
free -h                    # 内存使用
df -h                      # 磁盘使用
```

### 10.4.2 Prometheus + Grafana

**vLLM内置Prometheus支持**:

```bash
# 启动vLLM时启用metrics
vllm serve meta-llama/Llama-3.1-8B \
  --metrics-port 8000 \
  --enable-prometheus
```

**Prometheus配置**:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['vllm-service:8000']
    metrics_path: /metrics
```

**Grafana仪表盘JSON片段**:

```json
{
  "dashboard": {
    "title": "vLLM Performance Dashboard",
    "panels": [
      {
        "title": "TTFT (P95)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(vllm:ttft_seconds_bucket[5m]))"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "nvidia_gpu_utilization"
          }
        ]
      },
      {
        "title": "Tokens per Second",
        "targets": [
          {
            "expr": "rate(vllm:tokens_total[5m])"
          }
        ]
      }
    ]
  }
}
```

**关键PromQL查询**:

```promql
# TTFT P95
histogram_quantile(0.95, rate(vllm_ttft_seconds_bucket[5m]))

# 吞吐量
rate(vllm_tokens_total[5m])

# GPU利用率
nvidia_gpu_utilization

# 请求错误率
rate(vllm_requests_failed_total[5m]) / rate(vllm_requests_total[5m])

# KV Cache命中率
vllm_kv_cache_hit_rate
```

### 10.4.3 日志收集与分析

**结构化日志配置**:

```python
# vllm_logging_config.json
{
  "version": 1,
  "formatters": {
    "json": {
      "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
      "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
    }
  },
  "handlers": {
    "console": {
      "class": "logging.StreamHandler",
      "formatter": "json",
      "stream": "ext://sys.stdout"
    }
  },
  "root": {
    "level": "INFO",
    "handlers": ["console"]
  }
}
```

**使用ELK Stack收集日志**:

```yaml
# filebeat.yml
filebeat.inputs:
- type: container
  paths:
    - /var/log/containers/vllm*.log
  processors:
  - add_kubernetes_metadata:

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

**关键日志字段**:

```json
{
  "timestamp": "2025-01-31T10:00:00Z",
  "level": "INFO",
  "request_id": "req-123456",
  "session_id": "sess-789",
  "model": "Llama-3-8B",
  "input_tokens": 150,
  "output_tokens": 50,
  "ttft_ms": 1200,
  "tpot_ms": 80,
  "gpu_memory_mb": 16384,
  "cache_hit": true
}
```

### 10.4.4 分布式追踪

**使用OpenTelemetry**:

```python
# 安装
pip install opentelemetry-api opentelemetry-sdk

# 追踪代码
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

def generate_request(prompt: str):
    with tracer.start_as_current_span("generate_request") as span:
        span.set_attribute("prompt_length", len(prompt))

        with tracer.start_as_current_span("prefill"):
            # Prefill阶段
            pass

        with tracer.start_as_current_span("decode"):
            # Decode阶段
            pass
```

**Jaeger UI查看追踪**:

```bash
# 启动Jaeger
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 14250:14250 \
  jaegertracing/all-in-one:latest

# 访问UI
open http://localhost:16686
```

---

## 10.5 性能调优实战

### 10.5.1 调优流程

**完整的调优流程**:

```
┌─────────────────────────────────────────────────────┐
│  Step 1: 建立基线                                   │
│  - 使用benchmark_serving.py测试                     │
│  - 记录TTFT、TPOT、吞吐量                           │
│  - 记录GPU利用率                                    │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  Step 2: 识别瓶颈                                   │
│  - GPU利用率低 → 内存/CPU瓶颈?                      │
│  - GPU利用率高但慢 → 计算瓶颈?                      │
│  - 使用Nsight Systems分析                           │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  Step 3: 实施优化                                   │
│  - 调整batch size                                  │
│  - 启用Prefix Caching                              │
│  - 量化(FP16→INT8)                                 │
│  - 调整max_model_len                               │
└──────────────────┬──────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────┐
│  Step 4: 验证效果                                   │
│  - 重新运行benchmark                               │
│  - 对比优化前后指标                                │
│  - 确认没有regression                              │
└─────────────────────────────────────────────────────┘
```

### 10.5.2 瓶颈定位方法

**使用vLLM内置benchmark**:

```bash
# 运行benchmark
python benchmark_serving.py \
  --model meta-llama/Llama-3.1-8B \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --request-rate 10

# 输出关键指标
# 示例输出(依模型与硬件而定):
# TTFT: ...
# TPOT: ...
# Throughput: ...
# GPU Util: ...
```

**使用Nsight Systems**:

```bash
# 1. 安装Nsight Systems
# https://developer.nvidia.com/nsight-systems

# 2. 采集trace
nsys profile -o report.qdrep \
  python your_vllm_app.py

# 3. 分析结果
nsys-ui report.qdrep

# 查看指标:
# - GPU利用率是否达到预期?
# - Memory bandwidth是否饱和?
# - CPU overhead是否过高?
```

**诊断决策树**:

```
GPU利用率偏低?
  ├─ 是 → 内存使用率高?
  │   ├─ 是 → 内存受限
  │   │   解决: 减少batch size, 量化
  │   └─ 否 → CPU/IO受限
  │       解决: 检查CPU、磁盘、网络
  └─ 否 → 计算受限
      解决: 更好的GPU, tensor parallelism
```

### 10.5.3 常见性能问题

**问题1: TTFT过长**

```yaml
症状: 首个token返回时间偏高

原因:
  - KV Cache未命中
  - Prompt太长
  - 内存带宽不足

解决方案:
  # 1. 启用Prefix Caching
  vllm serve ... --enable-prefix-caching

  # 2. 使用Chunked Prefill
  vllm serve ... --max-model-len 32768

  # 3. 优化prompt
  - 移除冗余内容
  - 压缩系统提示词
```

**问题2: 吞吐量低**

```yaml
症状: tokens/s 明显低于预期

原因:
  - Batch size太小
  - GPU利用率低
  - 频繁的OOM

解决方案:
  # 1. 增加batch size
  vllm serve ... --max-num-seqs 256

  # 2. 调整GPU内存利用率
  vllm serve ... --gpu-memory-utilization 0.95

  # 3. 启用continuous batching
  vllm serve ... --enable-chunked-context
```

**问题3: OOM频繁**

```yaml
症状: CUDA out of memory错误

原因:
  - max_model_len太大
  - KV Cache占用过多
  - 批次大小过大

解决方案:
  # 1. 减少max_model_len
  vllm serve ... --max-model-len 4096

  # 2. KV Cache量化
  vllm serve ... --kv-cache-dtype fp8

  # 3. 减少并发请求数
  vllm serve ... --max-num-batched-tokens 8192
```

### 10.5.4 调优参数参考(示例)

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|--------|---------|------|
| **gpu-memory-utilization** | 0.9 | 0.85-0.95 | 过高可能导致OOM |
| **max-num-seqs** | 256 | 64-512 | 并发请求数 |
| **max-model-len** | 模型max | 2048-8192 | 根据实际需求 |
| **dtype** | auto | half/bf16 | FP16/BF16 |
| **kv-cache-dtype** | auto | fp8/int8 | KV缓存量化 |

---

## 10.5.5 性能分析工具

> **工具分类**: Profiling工具(定位内核级瓶颈) vs Benchmark工具(端到端性能评估)

### 10.5.5.1 PyTorch Profiler

**快速诊断Python/CUDA瓶颈**:

```python
import torch
from vllm import LLM, SamplingParams

# 启用profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    llm = LLM(model="meta-llama/Llama-3.1-8B")
    params = SamplingParams(max_tokens=100)

    for i in range(5):
        llm.generate(["Hello world"] * 10, params)
        prof.step()

# 查看结果
# tensorboard --logdir=./logs
```

**查看Chrome Trace**:

```bash
# 打开Chrome://tracing
# 加载trace.json
# 查看CUDA kernel时间线
```

### 10.5.5.2 Nsight Systems

**系统级性能分析**:

```bash
# 采集trace
nsys profile -y 30 -o vllm_report \
  --force-overwrite=true \
  python your_vllm_app.py

# 查看GUI
nsys-ui vllm_report.qdrep

# 或导出报告
nsys stats vllm_report.qdrep --report csv > stats.csv
```

**关键指标解读**:

```yaml
GPU Utilization:
  - 理想值: 偏高
  - 偏低 → 可能存在内存或CPU瓶颈

Memory Bandwidth:
  - 不同硬件峰值带宽不同
  - 达到峰值的一定比例通常意味着带宽成为瓶颈

Compute Throughput:
  - Tensor Core利用率
  - 理想值: 偏高

CUDA Kernel Duration:
  - Top kernels占用较大时间比例
  - 优化slow kernels

CPU Overhead:
  - 理想值: 较低
  - 过高 → 优化Python代码
```

### 10.5.5.3 Nsight Compute

**Kernel级深度分析**:

```bash
# 分析特定kernel
ncu --set full \
  --target-processes all \
  -o output_report \
  python your_vllm_app.py

# 查看报告
ncu-ui output_report.ncu-rep

# 关键指标:
# - Memory Workload: 读写吞吐
# - Compute Throughput: FLOPs利用率
# - Occupancy: Warp并行度
# - Warp Efficiency: 分支分歧程度
```

### 10.5.5.4 vLLM内置性能分析

```bash
# vLLM 0.6.0+内置profiling
VLLM_USE_TRACING=1 vllm serve meta-llama/Llama-3.1-8B

# 查看trace
# 生成的chrome trace文件: /tmp/vllm_trace.json
```

### 10.5.5.5 性能优化checklist

**Step 1: 基线测试**
- 使用`benchmark_serving.py`建立性能基线
- 记录关键指标: throughput (tokens/s), TTFT, TPOT, GPU利用率

**Step 2: PyTorch Profiler快速诊断**
- 找出top CUDA time operators
- 检查是否有unexpected的CPU overhead

**Step 3: Nsight Systems系统级分析**
- 验证GPU利用率是否合理
- 定位内存带宽瓶颈
- 分析CPU-GPU overlap

**Step 4: Nsight Compute kernel优化**(如需要)
- 针对slow kernel进行深度分析
- 优化memory access pattern
- 调整block/grid配置

**Step 5: 验证优化效果**
- 重新运行benchmark
- 对比优化前后的指标
- 确认没有regression

### 10.5.5.6 LLM性能测试工具

> **工具定位**: 除了profiling工具,还需要端到端的benchmark工具来评估LLM推理性能。

**GuideLLM** (Intel)

- **项目地址**: https://github.com/intel/guidellm
- **核心功能**:
  - 端到端LLM推理性能测试
  - 支持多种硬件: Intel Gaudi2、Habana、Xeon、NVIDIA GPU
  - 标准化benchmark: MMLU、GSM8K、HumanEval等
- **关键特性**:
  - 硬件对比测试
  - 推理框架选型评估
  - 模型性能验证

```bash
# 安装
pip install guidellm

# 运行benchmark
guidellm benchmark \
  --model meta-llama/Llama-3.1-8B \
  --framework vllm \
  --dataset mmlu \
  --output results.json
```

**EvalScope** (ModelScope)

- **项目地址**: https://github.com/modelscope/evalscope
- **核心功能**:
  - 综合LLM评估框架
  - 性能+准确率测试
  - 支持离线评估和在线推理
- **关键特性**:
  - 多维度评估(性能、准确率、鲁棒性)
  - 生产环境性能验证
  - A/B测试支持

**llm-bench** (vLLM内置)

```bash
# vLLM官方benchmark工具
python benchmark_serving.py \
  --model meta-llama/Llama-3.1-8B \
  --dataset-name sharegpt \
  --num-prompts 1000 \
  --request-rate 10

# 输出:
# - TTFT (Time To First Token)
# - TPOT (Time Per Output Token)
# - Throughput (tokens/s)
# - GPU利用率
```

**完整性能测试工作流**:

```bash
# Step 1: 快速评估(EvalScope)
evalscope eval \
  --model meta-llama/Llama-3.1-8B \
  --datasets mmlu,gsm8k

# Step 2: 推理性能(llm-bench)
python benchmark_serving.py \
  --model meta-llama/Llama-3.1-8B \
  --num-prompts 1000

# Step 3: 硬件对比(GuideLLM)
guidellm benchmark \
  --model meta-llama/Llama-3.1-8B \
  --framework vllm

# Step 4: vLLM专用优化
# 使用vLLM内置benchmark_serving.py
# 验证特定优化效果
```

---

## 10.6 成本优化

### 10.6.1 云GPU选择策略

> **说明**: 以下价格与规格仅为示例,实际以云厂商与地区报价为准。

**成本vs性能权衡**:

| GPU | 成本/小时 | 性能 | 适用场景 |
|-----|----------|------|---------|
| RTX 4090 | 示例区间 | 中 | 开发、小模型 |
| A100 (40GB) | 示例区间 | 高 | 生产环境 |
| A100 (80GB) | 示例区间 | 很高 | 大模型 |
| H100 | 示例区间 | 顶级 | 高性能需求 |

**选择决策树**:

```
模型大小 < 30B?
  ├─ 是 → 预算有限?
  │   ├─ 是 → RTX 4090 (自建或低成本云GPU)
  │   └─ 否 → A100 40GB (云GPU)
  └─ 否 → 预算充足?
      ├─ 否 → A100 80GB
      └─ 是 → H100
```

### 10.6.2 Spot实例使用

**💰 成本节省**: 可能显著

**什么是Spot实例?**
- 云厂商的闲置GPU资源
- 价格通常低于按需实例 (依地区与供给而定)
- 可能被随时回收

**使用策略**:

```python
# 使用Ray Autoscaler自动管理Spot实例
# cluster.yaml

cluster_name: vllm-spot-cluster

provider:
  type: aws
  region: us-west-2

available_node_types:
  spot_head:
    resources:
      CPU: 8
      GPU: 0
    node_config:
      InstanceType: m5.xlarge
      InstanceMarketOptions:
        MarketType: spot

  spot_worker:
    resources:
      CPU: 32
      GPU: 4
    node_config:
      InstanceType: p4d.24xlarge
      InstanceMarketOptions:
        MarketType: spot
        SpotOptions:
          InstanceInterruptionBehavior: terminate

autoscaling_mode: default
autoscaler_options:
  idle_timeout_minutes: 10
```

**处理中断**:

```python
# 检测Spot中断
import requests

def check_spot_interruption():
    """AWS Spot中断检测"""
    try:
        resp = requests.get(
            "http://169.254.169.254/latest/meta-data/spot/instance-action",
            timeout=1
        )
        return resp.json().get("action") == "terminate"
    except requests.exceptions.RequestException:
        return False

# 优雅降级
def graceful_shutdown():
    # 1. 停止接受新请求
    # 2. 等待现有请求完成
    # 3. 保存checkpoint
    # 4. 自动重启到新节点
    pass
```

### 10.6.3 自动伸缩

**基于负载的自动伸缩**:

```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-llama3-8b
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 120
```

**基于时间的伸缩**:

```python
# 业务高峰期提前扩容
from apscheduler.schedulers.background import BackgroundScheduler

def scale_up_before_peak():
    """在业务高峰前扩容"""
    os.system("kubectl scale deployment vllm --replicas=10")

def scale_down_after_peak():
    """业务高峰后缩容"""
    os.system("kubectl scale deployment vllm --replicas=2")

scheduler = BackgroundScheduler()
scheduler.add_job(scale_up_before_peak, 'cron', hour=8)  # 早上8点
scheduler.add_job(scale_down_after_peak, 'cron', hour=20)  # 晚上8点
scheduler.start()
```

### 10.6.4 成本监控工具

**AWS Cost Explorer**:

```bash
# 设置成本告警
aws budgets create-budget \
  --account-id <account-id> \
  --budget '{
    "BudgetName": "vLLM-GPU-Budget",
    "BudgetLimit": {"Amount": "1000", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }'
```

**自定义成本追踪**:

```python
import psutil
import pynvml
from datetime import datetime

def calculate_cost_per_token():
    """计算每1000 tokens的成本"""

    # GPU小时成本(示例值,以实际报价为准)
    gpu_cost_per_hour = 3.0

    # 获取GPU数量和利用率
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()
    total_util = 0
    for i in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        total_util += util.gpu

    avg_util = total_util / gpu_count / 100

    # 实际GPU使用量
    effective_gpus = gpu_count * avg_util

    # 每小时成本
    cost_per_hour = effective_gpus * gpu_cost_per_hour

    # 每秒成本
    cost_per_second = cost_per_hour / 3600

    return cost_per_second

# 记录每个请求的成本
def log_request_cost(tokens: int, time_seconds: float):
    cost = calculate_cost_per_token() * time_seconds
    cost_per_1k_tokens = (cost / tokens) * 1000

    print(f"Cost: ${cost_per_1k_tokens:.6f} per 1K tokens")
```

### 10.6.5 Agent系统的成本优化策略

> **来源**: [Manus - Context Engineering for AI Agents](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
>
> **核心观点**: 围绕KV-Cache设计Agent系统——这是成本优化的"银弹"

#### 10.6.5.1 成本对比: Cached vs Uncached

**价格说明**:
以厂商最新价格表为准,多数平台对 cached / uncached tokens 的定价差异较大。

**Agent系统的成本放大效应**:
```
典型Agent任务往往包含大量重复前缀与多步调用。
在高复用场景下,缓存可显著降低成本与延迟。
```

#### 10.6.5.2 四大优化手段

**优化1: 移除动态内容**

```python
# ❌ Before: 每次请求都不同
system_prompt = f"""
You are Manus AI assistant.
Current time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Today's date: {datetime.now().date()}
User ID: {user_id}
Session ID: {session_id}
...
"""

# ✅ After: 固定前缀
system_prompt = """
You are Manus AI assistant.
Current time: {{current_time}}
Today's date: {{today_date}}
"""

# 动态内容放在最后
request = system_prompt + fixed_tools + dynamic_content
```

**优化2: 使用稳定的JSON序列化**

```python
# ❌ Before: 无序序列化
import json
prompt_json = json.dumps(tools_definition)

# ✅ After: 有序序列化
prompt_json = json.dumps(tools_definition, sort_keys=True)
```

**优化3: 使用append而非modify**

```python
# ❌ Before: 修改整个prompt
for tool_call in tool_calls:
    prompt += f"\nTool result: {tool_call.result}"

# ✅ After: append新内容
for tool_call in tool_calls:
    cache_manager.append(tool_call.result)
```

**优化4: Session-Aful路由**

```python
class SessionAwareRouter:
    """确保同一session路由到同一节点"""

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.worker_cache = {}

    def get_worker(self, session_id: str) -> int:
        if session_id in self.worker_cache:
            return self.worker_cache[session_id]

        # 一致性哈希
        worker_id = hash(session_id) % self.num_workers
        self.worker_cache[session_id] = worker_id
        return worker_id
```

**效果**:
- Prefix cache复用率提升
- TTFT 通常降低
- 吞吐量通常提升

#### 10.6.5.3 成本优化Checklist

**基线测量**:
- [ ] 测量当前KV-cache hit rate
- [ ] 计算平均每个任务的token数
- [ ] 统计prefill vs decode比例
- [ ] 记录每1000个任务的cost

**快速优化**(1天内):
- [ ] 移除prompt中的timestamp等动态内容
- [ ] 检查JSON序列化是否使用`sort_keys=True`
- [ ] 确保prompt结构是"固定prefix + 动态suffix"
- [ ] 启用Prefix Caching

**中期优化**(1周内):
- [ ] 实现Session-Aful路由
- [ ] 添加file system fallback机制
- [ ] 监控cache hit rate指标

**长期优化**(持续):
- [ ] 建立成本监控dashboard
- [ ] A/B测试不同context策略
- [ ] 优化工具调用频率
- [ ] 实施context压缩策略

#### 10.6.5.4 实战案例对比

| 场景 | 优化前 | 优化后 | 节省 |
|------|--------|--------|------|
| 简单任务 | 相对较高 | 相对较低 | 可能显著 |
| 中等任务 | 相对较高 | 相对较低 | 可能显著 |
| 复杂任务 | 相对较高 | 相对较低 | 通常更明显 |
| 超长任务 | 相对较高 | 相对较低 | 通常更明显 |

**关键洞察**: 任务越复杂,优化收益往往越明显——因为上下文累积更多。

### 10.6.6 轻量级参考实现:Mini-SGLang

> **💡 深度来源**: [Mini-SGLang Blog](https://lmsys.org/blog/2025-12-17-minisgl/)
>
> **核心价值**: 以相对较小代码规模实现完整推理引擎,适合学习和研究原型
>
> **适用场景**: 教育学习、快速研究验证、内核开发调试

#### 10.6.6.1 为什么需要轻量级实现?

**问题**:
- **vLLM代码规模**: 数十万行量级
  - 新手学习曲线陡峭
  - 修改风险高(破坏隐式不变量)
  - 研究原型难以快速验证

- **SGLang代码规模**: 数十万行量级
  - 功能完整,但复杂度高
  - 不适合教学场景

**Mini-SGLang的答案**:
- **更少的代码规模**(便于理解与改动)
- **保留核心优化**:
  - Radix Attention (KV Cache复用)
  - Overlap Scheduling (CPU-GPU并行)
  - Chunked Prefill (内存控制)
  - Tensor Parallelism (分布式服务)
  - JIT CUDA kernels (FlashAttention-3, FlashInfer)
- **性能表现**: 需基准测试验证

#### 10.6.6.2 轻量级实现的核心功能

**代码结构**:
```
mini-sglang/
├── server.py          # OpenAI兼容API server
├── tokenizer.py       # Tokenizer服务
├── scheduler.py       # 调度器(含Overlap Scheduling)
├── radix_cache.py     # Radix Cache实现
├── model_runner.py    # 模型执行(Tensor Parallelism)
└── kernels/
    ├── flashattention.py    # FlashAttention-3集成
    └── flashinfer.py        # FlashInfer集成
```

**启动示例**:

```bash
# 安装
pip install mini-sglang

# 启动server
python -m mini_sglang.server \
  --model meta-llama/Llama-3.1-8B \
  --tp 1 \
  --port 8000

# OpenAI兼容API
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**关键设计决策**:
- **简洁性优先**: 移除边缘case处理,专注核心逻辑
- **教育导向**: 代码注释丰富,易于理解
- **研究友好**: 易于修改和实验新想法

#### 10.6.6.3 核心组件解析

**1. Radix Cache**(radix_cache.py)

```python
class RadixCache:
    """轻量实现的Radix Tree"""

    def __init__(self):
        self.root = RadixNode()

    def lookup(self, tokens: List[int]) -> CacheHit:
        """查找最长共享前缀"""
        node = self.root
        matched = 0

        for token in tokens:
            if token in node.children:
                node = node.children[token]
                matched += 1
            else:
                break

        return CacheHit(node, matched)

    def insert(self, tokens: List[int]) -> None:
        """插入新prompt"""
        node = self.root
        for token in tokens:
            if token not in node.children:
                node.children[token] = RadixNode()
            node = node.children[token]
```

**2. Overlap Scheduling**(scheduler.py)

```python
class OverlapScheduler:
    """CPU-GPU并行调度器"""

    def schedule(self):
        """Overlap CPU准备和GPU执行"""
        while True:
            # CPU: 准备下一个batch
            next_batch = self.prepare_batch_async()

            # GPU: 执行当前batch
            self.execute_batch(current_batch)

            # 交换
            current_batch = next_batch
```

**3. Tensor Parallelism**(model_runner.py)

```python
class TensorParallelRunner:
    """简化的TP实现"""

    def __init__(self, model, tp_size):
        self.tp_size = tp_size
        # NCCL初始化
        # GPU kernel启动
```

#### 10.6.6.4 学习价值

**与vLLM对比**:

| 维度 | vLLM | Mini-SGLang |
|------|------|-------------|
| 代码行数 | 数十万行 | 数千行 |
| 学习曲线 | 陡峭 | 平缓 |
| 核心功能 | ✅ | ✅ |
| 生产就绪 | ✅ | ❌ (教育/研究) |
| 修改难度 | 高 | 低 |
| 阅读时间 | 数周 | 数小时 |

**适用场景**:

✅ **Mini-SGLang适合**:
- 学习LLM推理原理
- 快速验证研究想法
- 开发新的CUDA内核
- 理解Radix Cache实现

❌ **vLLM/SGLang适合**:
- 生产环境部署
- 需要完整功能
- 需要长期维护

---

## 10.7 ROI监控与成本追踪

### 10.7.1 如何追踪推理成本

**完整的成本追踪系统**:

```python
class CostTracker:
    """LLM推理成本追踪器"""

    def __init__(self):
        self.requests = []
        self.gpu_cost_per_hour = 3.0  # 示例值,以实际报价为准

    def track_request(self,
                     request_id: str,
                     input_tokens: int,
                     output_tokens: int,
                     ttft_ms: float,
                     gpu_utilization: float):
        """追踪单个请求的成本"""

        # 计算实际GPU时间
        gpu_time_hours = (ttft_ms / 1000) / 3600

        # 计算成本(考虑GPU利用率)
        effective_gpus = gpu_utilization / 100
        cost = gpu_time_hours * effective_gpus * self.gpu_cost_per_hour

        # 记录
        self.requests.append({
            "request_id": request_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": cost,
            "cost_per_1k_tokens": (cost / (input_tokens + output_tokens)) * 1000
        })

    def get_summary(self):
        """获取成本汇总"""
        total_cost = sum(r["cost"] for r in self.requests)
        total_tokens = sum(r["total_tokens"] for r in self.requests)

        return {
            "total_requests": len(self.requests),
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "avg_cost_per_1k_tokens": (total_cost / total_tokens) * 1000
        }

# 使用示例
tracker = CostTracker()
tracker.track_request(
    request_id="req-001",
    input_tokens=150,
    output_tokens=50,
    ttft_ms=1200,
    gpu_utilization=75
)

print(tracker.get_summary())
# {'total_cost': 0.00075, 'avg_cost_per_1k_tokens': 0.00375}
```

### 10.7.2 优化措施的ROI计算

**ROI计算公式**:

```python
def calculate_roi(
    optimization_cost: float,  # 优化投入的成本($)
    before_cost_per_hour: float,  # 优化前成本($/小时)
    after_cost_per_hour: float,   # 优化后成本($/小时)
    hours_per_month: float = 730   # 每月小时数
):
    """计算ROI"""

    # 每月节省
    monthly_savings = (before_cost_per_hour - after_cost_per_hour) * hours_per_month

    # 回本周期(月)
    payback_period = optimization_cost / monthly_savings

    # 年化ROI
    annual_savings = monthly_savings * 12
    annual_roi = (annual_savings - optimization_cost) / optimization_cost * 100

    return {
        "monthly_savings": monthly_savings,
        "payback_period_months": payback_period,
        "annual_roi_percent": annual_roi
    }

# 示例: 启用Prefix Caching的ROI(示意)
roi = calculate_roi(
    optimization_cost=2000,  # 示例值
    before_cost_per_hour=10,  # 示例值
    after_cost_per_hour=3,    # 示例值
    hours_per_month=730
)

print(roi)
# 输出为示例
```

**ROI仪表盘**:

```python
import matplotlib.pyplot as plt

def plot_roi_dashboard():
    """绘制ROI仪表盘"""

    # 数据
    optimizations = [
        "Prefix Caching",
        "INT8量化",
        "Spot实例",
        "自动伸缩"
    ]

    investment = [2000, 1000, 500, 1500]
    monthly_savings = [5110, 3640, 4380, 2190]
    payback_months = [i / s * 30 for i, s in zip(investment, monthly_savings)]

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 投资vs节省
    ax1.bar(optimizations, investment, label='投资($)')
    ax1.bar(optimizations, monthly_savings, label='月节省($)')
    ax1.set_title('投资 vs 月节省')
    ax1.legend()

    # 回本周期
    ax2.bar(optimizations, payback_months)
    ax2.set_title('回本周期(天)')
    ax2.set_ylabel('天数')

    plt.tight_layout()
    plt.savefig('roi_dashboard.png')
```

### 10.7.3 持续优化流程

**PDCA循环**:

```
┌─────────────────────────────────────────────────┐
│  Plan: 制定优化计划                              │
│  - 分析成本数据                                  │
│  - 识别优化机会                                  │
│  - 估算ROI                                      │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  Do: 实施优化                                    │
│  - 代码实现                                      │
│  - 灰度发布                                      │
│  - 监控指标                                      │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  Check: 验证效果                                 │
│  - 对比优化前后成本                              │
│  - 计算实际ROI                                   │
│  - 检查是否有副作用                              │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│  Act: 标准化或调整                               │
│  - ROI达标 → 全面推广                            │
│  - ROI不达标 → 分析原因,调整策略                 │
│  - 文档化最佳实践                                │
└─────────────────────────────────────────────────┘
```

**优化优先级矩阵**:

```
高ROI, 低难度 → 优先实施
  - 启用Prefix Caching
  - 移除prompt动态内容

高ROI, 高难度 → 中期规划
  - INT4量化
  - 自定义kernel优化

低ROI, 低难度 → 填充实施
  - 日志优化
  - 监控完善

低ROI, 高难度 → 暂缓实施
  - 自研推理框架
  - 硬件定制
```

---

## 10.8 安全性考虑

### 10.8.1 API认证与授权

**API Key认证**:

```python
from fastapi import FastAPI, Header, HTTPException

app = FastAPI()

VALID_API_KEYS = {
    "sk-1234567890": "user1",
    "sk-0987654321": "user2"
}

@app.post("/v1/chat/completions")
async def chat_completions(
    authorization: str = Header(None)
):
    # 验证API Key
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing API key")

    api_key = authorization.replace("Bearer ", "")
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")

    # 处理请求
    user = VALID_API_KEYS[api_key]
    # ...
```

**速率限制**:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/v1/chat/completions")
@limiter.limit("10/minute")  # 每分钟10次请求
async def chat_completions(request: Request):
    # ...
```

**基于Token的限流**:

```python
class TokenBucketRateLimiter:
    """基于token的限流器"""

    def __init__(self, rate: int, capacity: int):
        self.rate = rate  # tokens/秒
        self.capacity = capacity  # 桶容量
        self.tokens = capacity
        self.last_time = time.time()

    def consume(self, tokens: int) -> bool:
        """消费tokens"""
        now = time.time()
        elapsed = now - self.last_time

        # 补充tokens
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_time = now

        # 检查是否有足够tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

# 使用示例
rate_limiter = TokenBucketRateLimiter(rate=100, capacity=1000)

@app.post("/v1/generate")
async def generate(prompt: str, max_tokens: int):
    if not rate_limiter.consume(max_tokens):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    # ...
```

### 10.8.2 内容安全过滤

**输入过滤**:

```python
import re

def validate_input(prompt: str):
    """验证输入安全性"""

    # 检查恶意prompt
    malicious_patterns = [
        r"忽略.*指令",
        r"ignore.*instruction",
        r"<\|.*\|>",  # 特殊token注入
    ]

    for pattern in malicious_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            raise ValueError("Malicious input detected")

    # 检查prompt长度
    if len(prompt) > 100000:
        raise ValueError("Prompt too long")

    return True
```

**输出过滤**:

```python
from transformers import pipeline

# 加载安全分类器
safety_classifier = pipeline("text-classification",
                             model="distilbert-base-uncased")

def filter_output(text: str) -> str:
    """过滤不安全输出"""

    result = safety_classifier(text)[0]

    if result["label"] == "UNSAFE" and result["score"] > 0.8:
        return "[内容已被过滤]"

    return text
```

### 10.8.3 数据隐私

**敏感数据脱敏**:

```python
import re

def mask_pii(text: str) -> str:
    """脱敏敏感信息"""

    # Email
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  '[EMAIL]', text)

    # Phone
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b',
                  '[PHONE]', text)

    # Credit Card
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                  '[CARD]', text)

    # SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b',
                  '[SSN]', text)

    return text
```

**数据加密存储**:

```python
from cryptography.fernet import Fernet

class EncryptedStorage:
    """加密存储"""

    def __init__(self, key: bytes):
        self.cipher = Fernet(key)

    def encrypt(self, data: str) -> bytes:
        return self.cipher.encrypt(data.encode())

    def decrypt(self, encrypted: bytes) -> str:
        return self.cipher.decrypt(encrypted).decode()

# 使用示例
storage = EncryptedStorage(Fernet.generate_key())
encrypted = storage.encrypt("sensitive data")
decrypted = storage.decrypt(encrypted)
```

### 10.8.4 审计日志

**完整的审计日志**:

```python
import logging
from datetime import datetime

class AuditLogger:
    """审计日志记录器"""

    def __init__(self):
        self.logger = logging.getLogger("audit")
        handler = logging.FileHandler("audit.log")
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log_request(self,
                   user_id: str,
                   request_id: str,
                   prompt: str,
                   response: str,
                   tokens_used: int,
                   cost: float):
        """记录请求审计信息"""

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "request_id": request_id,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "tokens_used": tokens_used,
            "cost": cost
        }

        self.logger.info(json.dumps(log_entry))

# 使用示例
audit = AuditLogger()
audit.log_request(
    user_id="user123",
    request_id="req-001",
    prompt="Hello",
    response="Hi there!",
    tokens_used=5,
    cost=0.0001
)
```

---

## 10.9 灾备与容错

### 10.9.1 失败场景分析

**常见失败场景**:

| 失败类型 | 概率 | 影响 | 检测方式 |
|---------|------|------|---------|
| GPU硬件故障 | 中 | 高 | NVIDIA健康检查 |
| OOM | 高 | 中 | 监控显存使用 |
| 网络分区 | 低 | 高 | 心跳检测 |
| Spot回收 | 高 | 低 | AWS元数据服务 |
| 进程崩溃 | 中 | 高 | 健康检查 |

### 10.9.2 健康检查

**Liveness Probe**(存活检查):

```yaml
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 60
  periodSeconds: 10
  failureThreshold: 3
```

**Readiness Probe**(就绪检查):

```yaml
# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 5
  failureThreshold: 3
```

**自定义健康检查端点**:

```python
from fastapi import FastAPI
import pynvml

app = FastAPI()

@app.get("/health")
def health_check():
    """健康检查端点"""

    try:
        # 检查GPU状态
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # 检查显存使用
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if mem_info.used / mem_info.total > 0.98:
            return {"status": "unhealthy", "reason": "OOM"}

        # 检查温度
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        if temp > 90:
            return {"status": "unhealthy", "reason": "Overheating"}

        return {"status": "healthy"}

    except Exception as e:
        return {"status": "unhealthy", "reason": str(e)}

@app.get("/ready")
def readiness_check():
    """就绪检查端点"""

    # 检查模型是否加载完成
    # 检查是否有足够资源接受新请求

    return {"status": "ready"}
```

### 10.9.3 自动重启策略

**Kubernetes重启策略**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    spec:
      restartPolicy: Always
```

**自动恢复脚本**:

```python
import time
import subprocess
import requests

def monitor_and_restart():
    """监控并自动重启"""

    while True:
        try:
            # 检查健康状态
            resp = requests.get("http://localhost:8000/health", timeout=5)

            if resp.json()["status"] != "healthy":
                print("Unhealthy detected, restarting...")
                subprocess.run(["docker-compose", "restart"])

        except Exception as e:
            print(f"Health check failed: {e}")
            subprocess.run(["docker-compose", "restart"])

        time.sleep(10)

if __name__ == "__main__":
    monitor_and_restart()
```

### 10.9.4 降级方案

**优雅降级策略**:

```python
class DegradationManager:
    """降级管理器"""

    def __init__(self):
        self.current_level = 0  # 0=正常, 1=轻度降级, 2=重度降级

    def check_and_degrade(self):
        """检查系统状态并降级"""

        # 检查GPU可用数量
        gpu_count = get_available_gpu_count()

        if gpu_count < 2:
            self.current_level = 2
            # 重度降级: 拒绝新请求
            return {"action": "reject_new", "reason": "Insufficient GPUs"}

        elif gpu_count < 4:
            self.current_level = 1
            # 轻度降级: 减少max_model_len
            return {"action": "reduce_context", "new_max_len": 4096}

        else:
            self.current_level = 0
            return {"action": "normal"}

    def should_reject_request(self) -> bool:
        """是否应该拒绝请求"""
        return self.current_level == 2

# 使用示例
degradation = DegradationManager()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # 检查是否需要降级
    action = degradation.check_and_degrade()

    if action["action"] == "reject_new":
        raise HTTPException(status_code=503, detail="Service overloaded")

    if action["action"] == "reduce_context":
        # 调整请求参数
        request.max_tokens = min(request.max_tokens, 1000)

    # 处理请求
    # ...
```

---

## 10.10 RL系统部署 ⚠️ 开源生态缺失

> **💡 2025年技术趋势**(来源:2025"青稞"AI嘉年华 - 朱子林@质朴、朱立耕@NVIDIA)
>
> RL(强化学习)系统的部署面临独特挑战:
> - Training和Rollout的分离
> - 异构GPU的协同
> - 弹性资源分配
> - 低延迟的inference serving

### 10.10.1 RL系统的特殊需求

**与普通推理的区别**:

| 维度 | 普通推理 | RL系统 |
|------|---------|--------|
| **工作负载** | 仅推理 | Training + Rollout |
| **延迟要求** | 秒级 | 毫秒级(Rollout) |
| **吞吐量** | 重要 | 极其重要 |
| **GPU类型** | 同构 | 常异构(训练+推理) |
| **调度** | 简单 | 复杂(PD分离) |

### 10.10.2 开源项目现状

**当前状况**:
- ✅ **Ray/RLlib**: 训练框架成熟
- ❌ **Rollout服务**: 开源生态缺失
- ❌ **统一框架**: 生产级方案少

**主要项目**:

- **slime** (质朴科技)
  - GitHub: https://github.com/zizai/slime
  - **定位**: RL训练和推理的统一框架
  - **特点**:
    - Training和Rollout共享GPU
    - 支持异构GPU(H100+H200)
    - 弹性资源分配

### 10.10.3 关键挑战

**挑战1: Training vs Rollout的资源竞争**

```python
# 问题: Training和Rollout竞争GPU资源
# 解决方案: 动态资源分配

class DynamicResourceManager:
    """动态资源管理器"""

    def __init__(self, total_gpus: int):
        self.total_gpus = total_gpus
        self.training_gpus = 0
        self.rollout_gpus = 0

    def allocate(self, rollout_queue_length: int):
        """根据队列长度动态分配"""

        if rollout_queue_length > 100:
            # Rollout压力大,增加资源
            self.rollout_gpus = min(self.total_gpus * 0.8, self.rollout_gpus + 1)
            self.training_gpus = self.total_gpus - self.rollout_gpus

        elif rollout_queue_length < 10:
            # Rollout压力小,减少资源
            self.rollout_gpus = max(self.total_gpus * 0.2, self.rollout_gpus - 1)
            self.training_gpus = self.total_gpus - self.rollout_gpus

        return {
            "training": self.training_gpus,
            "rollout": self.rollout_gpus
        }
```

**挑战2: 异构GPU协同**

```python
# H100用于training,H200用于rollout

class HeterogeneousCluster:
    """异构集群管理"""

    def __init__(self):
        self.h100_count = 8
        self.h200_count = 4

    def schedule_task(self, task_type: str):
        """调度任务到合适的GPU"""

        if task_type == "training":
            # Training → H100(性价比高)
            return "h100"

        elif task_type == "rollout":
            # Rollout → H200(低延迟)
            return "h200"
```

### 10.10.4 部署架构

**单机部署**:

```
┌─────────────────────────────────────┐
│         单服务器 (H100)              │
│                                     │
│  ┌─────────────┐  ┌──────────────┐ │
│  │  Training   │  │   Rollout    │ │
│  │   (70%)     │  │    (30%)     │ │
│  └─────────────┘  └──────────────┘ │
└─────────────────────────────────────┘
```

**适合**: 小规模实验

**多机部署**:

```
┌──────────────────┐      ┌──────────────────┐
│   Training Node  │      │  Rollout Nodes   │
│   (H100 x 8)     │      │  (H200 x 4)      │
│                  │      │                  │
│  ┌────────────┐  │      │  ┌────────────┐  │
│  │   PPO     │  │      │  │  vLLM/SGL  │  │
│  │  Training  │  │      │  │  Serving   │  │
│  └────────────┘  │      │  └────────────┘  │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         └────────────┬────────────┘
                      │
              ┌───────▼────────┐
              │  Parameter     │
              │   Server       │
              └────────────────┘
```

**适合**: 生产环境

### 10.10.5 实战案例

**案例1: 使用slime部署简单RL任务**

```bash
# 安装slime
pip install slime-rl

# 启动RL训练+rollout服务
slime launch \
  --model meta-llama/Llama-3.1-8B \
  --task rlhf \
  --training-gpus 4 \
  --rollout-gpus 2 \
  --rollout-framework vllm
```

**案例2: 异构GPU的RL部署(H100+H200)**

```python
# slime配置文件
# slime_config.yaml

cluster:
  training_nodes:
    - type: H100
      count: 8
      use: training

  rollout_nodes:
    - type: H200
      count: 4
      use: rollout

training:
  framework: ppo
  batch_size: 512
  learning_rate: 1e-5

rollout:
  framework: vllm
  tensor_parallel_size: 4
  gpu_memory_utilization: 0.9
```

**案例3: 大规模RL的弹性资源分配**

```python
class ElasticRLScheduler:
    """弹性RL调度器"""

    def __init__(self):
        self.cloud_provider = AWS()
        self.spot_instances = []

    def scale_rollout_workers(self, demand: int):
        """根据需求弹性扩缩容"""

        current_workers = len(self.spot_instances)

        if demand > current_workers * 100:
            # 需要扩容
            new_instances = self.cloud_provider.launch_spot_instances(
                instance_type="p4d.24xlarge",
                count=(demand // 100) - current_workers
            )
            self.spot_instances.extend(new_instances)

        elif demand < current_workers * 50:
            # 需要缩容
            instances_to_terminate = self.spot_instances[(demand // 50):]
            for inst in instances_to_terminate:
                self.cloud_provider.terminate_instance(inst)
            self.spot_instances = self.spot_instances[:(demand // 50)]
```

---

## 🚫 常见误区

### ❌ "生产环境只需要更多GPU"

**实际情况**: 架构和优化比硬件更重要。

```
场景1: 多卡中端 vs 少量高端
- 结论: 取决于模型大小、通信开销与部署复杂度

场景2: 优化前 vs 优化后
- 优化后通常可在更少资源下达到相似或更好效果
- 结论: 优化往往比单纯增加GPU更有效
```

### ❌ "K8s能自动处理所有故障"

**实际情况**: K8s只是工具,需要合理配置。

```yaml
# ❌ 错误配置
livenessProbe:
  initialDelaySeconds: 0  # 太短,模型还未加载
  periodSeconds: 1        # 太频繁,浪费资源

# ✅ 正确配置
livenessProbe:
  initialDelaySeconds: 60  # 给模型加载时间
  periodSeconds: 10        # 合理间隔
  failureThreshold: 3      # 允许偶尔失败
```

### ❌ "监控越详细越好"

**实际情况**: 关注关键指标,避免信息过载。

```python
# ❌ 监控所有指标
metrics = [
    "cpu_usage",
    "memory_usage",
    "disk_io",
    "network_io",
    "gpu_temperature",
    "gpu_fan_speed",
    "gpu_power_usage",
    # ... 100+ 指标
]

# ✅ 监控关键指标
metrics = [
    "ttft_p95",         # 首token延迟
    "tokens_per_second", # 吞吐量
    "gpu_utilization",   # GPU利用率
    "error_rate",        # 错误率
    "kv_cache_hit_rate"  # 缓存命中率
]
```

### ❌ "Spot实例不可靠,不适合生产"

**实际情况**: 合理的设计可以可靠使用Spot实例。

```python
# ✅ 最佳实践
1. 使用混合实例(按需 + Spot)
   - 按需: 最小容量
   - Spot: 弹性扩容

2. 检查中断信号
   - AWS: http://169.254.169.254/latest/meta-data/spot/instance-action
   - 优雅关闭: 保存checkpoint, 完成当前请求

3. 自动替换
   - Spot被回收 → 自动启动新Spot实例
   - 使用Autoscaling Group

4. 分布式训练
   - 使用checkpoint定期保存
   - 从checkpoint恢复训练
```

---

## ✅ 章节检查清单

阅读本章后,你应该能够:

- [ ] 解释生产环境与开发环境的关键差异
- [ ] 设计高可用的部署架构
- [ ] 编写Kubernetes部署配置
- [ ] 搭建Prometheus + Grafana监控系统
- [ ] 使用profiling工具定位性能瓶颈
- [ ] 实施成本优化策略(Spot实例、自动伸缩)
- [ ] 计算优化措施的ROI
- [ ] 配置API认证和速率限制
- [ ] 实现健康检查和自动恢复
- [ ] 避免常见的生产环境误区

---

## 📚 动手练习

**练习10.1**: 部署vLLM到Kubernetes

目标: 将vLLM服务部署到K8s集群

任务:
1. 编写Deployment配置文件
2. 配置Service和LoadBalancer
3. 设置健康检查探针
4. 验证部署成功

验收:
```bash
kubectl get pods  # 3个Pod运行中
kubectl port-forward service/vllm-service 8000:80
curl http://localhost:8000/v1/models
```

---

**练习10.2**: 搭建完整的监控系统

目标: 使用Prometheus + Grafana监控vLLM

任务:
1. 配置Prometheus采集vLLM metrics
2. 创建Grafana仪表盘
3. 添加关键指标(TTFT、吞吐量、GPU利用率)
4. 配置告警规则

验收:
- Grafana显示实时指标
- GPU利用率超过阈值时告警
- TTFT P95超过阈值时告警

---

**练习10.3**: 建立ROI监控仪表盘

目标: 追踪推理成本和优化ROI

任务:
1. 实现CostTracker类
2. 记录每个请求的成本
3. 计算优化措施的ROI
4. 可视化成本趋势

验收:
- 显示每1000 tokens的成本
- 计算Prefix Caching的ROI
- 生成月度成本报告

---

**练习10.4**: 使用slime部署简单RL任务 ⭐

目标: 部署一个简单的RL训练+rollout系统

任务:
1. 安装slime框架
2. 配置训练和rollout节点
3. 启动RLHF任务
4. 监控训练进度

验收:
- Training节点正常运行
- Rollout服务响应满足SLA
- 模型reward收敛

---

**练习10.5**: 开发并部署vLLM自定义插件 ⭐⭐

目标: 实现一个vLLM插件来定制行为

任务:
1. 使用vLLM Plugin System
2. 实现自定义调度器(如优先级调度)
3. 添加自定义日志和监控
4. 部署到生产环境

验收:
- 插件正确加载
- 优先级调度生效
- 高优先级请求TTFT明显降低

---

## ✅ 练习参考答案

**练习10.1: 部署vLLM到Kubernetes**

```yaml
# vllm-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-8b
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm
  template:
    metadata:
      labels:
        app: vllm
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: MODEL_NAME
          value: "meta-llama/Llama-3.1-8B"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

**练习10.2: 搭建监控系统**

```yaml
# prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'vllm'
      static_configs:
        - targets: ['vllm-service:8000']
      metrics_path: /metrics
---
apiVersion: v1
kind: Pod
metadata:
  name: prometheus
spec:
  containers:
  - name: prometheus
    image: prom/prometheus:latest
    args:
    - '--config.file=/etc/prometheus/prometheus.yml'
    volumeMounts:
    - name: config
      mountPath: /etc/prometheus
  volumes:
  - name: config
    configMap:
      name: prometheus-config
```

**练习10.3: ROI监控**

```python
# cost_tracker.py
import time
import json
from typing import List, Dict

class CostTracker:
    def __init__(self, gpu_cost_per_hour: float = 3.0):
        self.requests: List[Dict] = []
        self.gpu_cost_per_hour = gpu_cost_per_hour

    def track_request(self,
                     request_id: str,
                     input_tokens: int,
                     output_tokens: int,
                     ttft_ms: float,
                     gpu_utilization: float):
        total_tokens = input_tokens + output_tokens
        gpu_time_hours = (ttft_ms / 1000) / 3600
        effective_gpus = gpu_utilization / 100
        cost = gpu_time_hours * effective_gpus * self.gpu_cost_per_hour

        self.requests.append({
            "request_id": request_id,
            "total_tokens": total_tokens,
            "cost": cost,
            "cost_per_1k_tokens": (cost / total_tokens) * 1000,
            "timestamp": time.time()
        })

    def get_summary(self):
        total_cost = sum(r["cost"] for r in self.requests)
        total_tokens = sum(r["total_tokens"] for r in self.requests)
        return {
            "total_requests": len(self.requests),
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "avg_cost_per_1k_tokens": (total_cost / total_tokens) * 1000 if total_tokens > 0 else 0
        }

    def save_to_file(self, filename: str):
        with open(filename, "w") as f:
            json.dump(self.requests, f, indent=2)
```

---

## 🎯 总结

关键要点:
- **生产环境≠开发环境**: 需要高可用、监控、安全、灾备
- **监控是基础**: Metrics、Logs、Traces三大支柱
- **优化先于扩容**: Prefix Caching、量化、自动伸缩
- **成本可控**: Spot实例、ROI监控、持续优化
- **安全第一**: 认证、授权、审计日志
- **未雨绸缪**: 健康检查、自动恢复、降级方案

**下一步**: 第11章高级话题(异构硬件、MoE、未来趋势)

---

**有问题?加入 [第10章 Discord频道](https://discord.gg/TODO) 讨论!**
