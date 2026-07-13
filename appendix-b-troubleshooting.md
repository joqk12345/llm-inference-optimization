---
id: "appendix-b-troubleshooting"
title: "附录B: 故障排查指南"
slug: "appendix-b-troubleshooting"
date: "2026-03-11"
type: "reference"
topics:
  - "troubleshooting"
concepts:
  - "observability"
  - "latency-budget"
tools:
  - "vllm"
  - "nsight-systems"
architecture_layer:
  - "production-systems"
learning_stage: "production"
optimization_axes:
  - "operability"
  - "latency"
  - "memory"
  - "throughput"
related:
  - "chapters-chapter04-environment-setup"
  - "chapters-chapter07-request-scheduling"
  - "chapters-chapter10-production-deployment"
references:
  - "https://docs.vllm.ai/en/stable/usage/metrics/"
status: "published"
display_order: 14
---
# 附录B: 故障排查指南

> "当所有方法都失效时,请阅读手册。" - 佚名

本附录提供了LLM推理部署过程中常见问题的诊断和解决方法,帮助你快速定位并修复问题。

---

## B.1 常见错误及解决

### B.1.1 CUDA相关错误

**错误1: CUDA out of memory**

```
CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 23.70 GiB total capacity; 21.80 GiB already allocated)
```

**原因分析**:
- 模型权重 + KV Cache + 激活值超过GPU显存
- batch size太大
- max_model_len设置过大

**解决方案**:

```bash
# 方案1: 减少max_model_len
vllm serve meta-llama/Llama-3-8B \
  --max-model-len 4096  # 从8192降到4096

# 方案2: 降低GPU内存利用率
vllm serve meta-llama/Llama-3-8B \
  --gpu-memory-utilization 0.8  # 从0.9降到0.8

# 方案3: 减少并发请求
vllm serve meta-llama/Llama-3-8B \
  --max-num-seqs 64  # 从256降到64

# 方案4: 启用KV Cache量化
vllm serve meta-llama/Llama-3-8B \
  --kv-cache-dtype fp8

# 方案5: 使用量化模型
vllm serve TheBloke/Llama-3-8B-Instruct-AWQ
```

---

**错误2: CUDA error: invalid configuration argument**

```
RuntimeError: CUDA error: invalid configuration argument
```

**原因分析**:
- tensor-parallel-size与GPU数量不匹配
- GPU不在同一节点

**解决方案**:

```bash
# 检查可用GPU数量
nvidia-smi --list-gpus

# 确保tensor-parallel-size <= GPU数量
# 错误: 只有1个GPU,但设置TP=2
vllm serve meta-llama/Llama-3-70B \
  --tensor-parallel-size 2  # ❌ 错误

# 正确
vllm serve meta-llama/Llama-3-70B \
  --tensor-parallel-size 1  # ✅ 正确
```

---

**错误3: CUDA error: no kernel image is available for execution**

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**原因分析**:
- CUDA版本与GPU架构不匹配
- 编译时使用的CUDA版本过低

**解决方案**:

```bash
# 检查CUDA版本
nvcc --version

# 检查GPU架构
nvidia-smi --query-gpu=compute_cap --format=csv

# 升级CUDA
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda

# 或使用conda
conda install cuda -c nvidia
```

---

### B.1.2 显存不足 (OOM)

**症状**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
Kill process or sacrifice child
```

**诊断步骤**:

```python
# 1. 检查显存使用
import torch
print(torch.cuda.memory_allocated() / 1024**3, "GB")
print(torch.cuda.memory_reserved() / 1024**3, "GB")

# 2. 检查vLLM显存使用
import pynvml
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
info = pynvml.nvmlDeviceGetMemoryInfo(handle)
print(f"已用: {info.used / 1024**3:.2f} GB")
print(f"总计: {info.total / 1024**3:.2f} GB")
```

**候选动作矩阵**（参数存在性和允许值按目标版本核验）：

| 场景 | 解决方案 | 命令示例 |
|------|---------|----------|
| **模型太大** | 评估兼容的量化制品或并行方案 | 质量、吞吐、加载与显存 |
| **上下文预算过高** | 按真实长度分布调整上限 | 拒绝率、并发、TTFT |
| **并发过高** | 扫描最大并发与 admission | 排队、抢占、尾延迟 |
| **KV Cache大** | 评估 KV 精度或上下文策略 | 质量、TPOT、容量 |
| **临时 buffer 峰值大** | 分解 kernel 与 batch 峰值 | OOM、吞吐、尾延迟 |

---

### B.1.3 性能问题

**问题1: TTFT过长**

```
症状: TTFT 分位数超过该服务定义的 SLO 或健康基线
```

**诊断**:

先比较等待时间、prefill 执行时间、prefix cache 查询与网络时间。Tracing 的启用方式按目标版本官方文档核验；不要预设环境变量或输出路径。

**解决方案**:

```bash
# 1. 启用Prefix Caching
vllm serve meta-llama/Llama-3-8B \
  --enable-prefix-caching

# 2. 若目标版本支持，评估 chunked prefill
# 先运行 vllm serve --help 核对参数；max-model-len 不是该功能的开关。

# 3. 减少prompt长度
# - 压缩系统提示词
# - 移除冗余内容

# 4. 若 decode 占主导，再用独立实验评估投机采样
# draft/target 模型组合、采纳率与质量必须一起验证。
```

---

**问题2: 吞吐量低**

```
症状: 在相同模型、硬件、精度和流量回放下，tokens/s 明显低于已验证基线
```

**诊断**:

```bash
# 使用目标版本随附的 benchmark 工具；路径和参数先查该版本文档
vllm bench --help

# 查看GPU利用率
nvidia-smi dmon -s u

# nvidia-smi 只能作为线索；瓶颈结论需要 profiler、队列和服务指标共同支持。
```

**解决方案**:

```bash
# 1. 对 max-num-seqs、max-num-batched-tokens 做单变量扫描
# 2. 调整显存预算时保留 OOM 和临时 buffer 余量
# 3. 对 chunked prefill / prefix caching 分别做 A/B，不使用未核验参数

# 4. 检查CPU瓶颈
# - 升级CPU
# - 优化数据加载
# - 使用多进程
```

---

**问题3: GPU利用率低**

```
症状: GPU 活动率低于该工作负载的健康基线，同时吞吐或 SLO 恶化
```

**诊断**:

```bash
# 持续监控GPU
watch -n 1 nvidia-smi

# 使用Nsight Systems
nsys profile -o report \
  python your_app.py

# 查看GPU空闲时间
```

**解决方案**:

```bash
# 1. 先确认是否有足够请求以及等待队列；再逐步增加并发候选值

# 2. 检查是否CPU瓶颈
# - 查看CPU使用率
# - 优化数据预处理
# - 使用多worker

# 3. 检查是否网络瓶颈
# - 查看网络带宽
# - 优化数据传输

# 4. 使用Overlap Scheduling (SGLang)
# - Mini-SGLang: CPU-GPU并行
# - 隐藏CPU开销
```

---

### B.1.4 模型加载失败

**错误1: Model not found**

```
OSError: meta-llama/Llama-3-8B is not a local folder
```

**解决方案**:

```bash
# 方案1: 使用Hugging Face Hub
vllm serve meta-llama/Llama-3-8B

# 方案2: 指定本地路径
vllm serve /path/to/local/model

# 方案3: 下载模型到本地
# 使用huggingface-cli
huggingface-cli download \
  meta-llama/Llama-3-8B \
  --local-dir ./models/Llama-3-8B

# 然后使用本地路径
vllm serve ./models/Llama-3-8B
```

---

**错误2: Permission denied**

```
OSError: You are trying to access a gated repo.
```

**解决方案**:

```bash
# 1. 访问模型页面,同意许可
# https://huggingface.co/meta-llama/Llama-3-8B

# 2. 登录Hugging Face
huggingface-cli login

# 3. 输入token
# 从 https://huggingface.co/settings/tokens 获取

# 4. 重新启动服务
vllm serve meta-llama/Llama-3-8B
```

---

**错误3: Checkpoint mismatch**

```
ValueError: Checkpoint shards have conflicting tensor names
```

**解决方案**:

```bash
# 1. 验证模型文件
python -c "
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3-8B')
print(model.config)
"

# 2. 重新下载模型
rm -rf ./models/Llama-3-8B
huggingface-cli download \
  meta-llama/Llama-3-8B \
  --local-dir ./models/Llama-3-8B

# 3. 使用vLLM的conversion
python -m vllm.model_conversion.convert \
  --model meta-llama/Llama-3-8B \
  --output ./models/Llama-3-8B-vllm
```

---

### B.1.5 推理速度慢

**症状**：在固定模型、输入/输出长度、并发和采样参数后，生成速度低于同环境基线。

**诊断流程**:

```
1. 固定并记录模型制品、dtype/量化、上下文、输出长度、并发与采样参数
2. 查看等待请求、TTFT、TPOT、KV 使用率、抢占和错误
3. 用 Nsight Systems 定位 CPU、GPU、通信和同步空洞
4. 再用 Nsight Compute 判断主要 kernel 的计算、内存与 stall 指标
```

**解决方案**:

```bash
# 一次只改变一个候选变量，例如并发、token budget、prefix caching 或量化。
# 每次变更都回归 TTFT、TPOT、吞吐、错误率、显存、质量与单位成本。
# 不按 GPU 型号名称预设固定加速比。
```

---

## B.2 调试技巧

### B.2.1 日志分析

**启用详细日志**:

```bash
# vLLM日志级别
export VLLM_LOGGING_LEVEL=DEBUG

# tracing 的启用方式按目标版本官方文档核验，不预设环境变量。

# 启动服务
vllm serve meta-llama/Llama-3-8B
```

**关键日志位置**:

```bash
# 查看vLLM日志
tail -f /var/log/vllm/vllm.log

# 查看CUDA日志
export CUDA_LAUNCH_BLOCKING=1

# 查看PyTorch日志
export TORCH_LOGS="+dynamo"
```

**日志分析工具**:

```python
import json
import re

def analyze_vllm_logs(log_file):
    """分析vLLM日志"""

    with open(log_file) as f:
        logs = f.readlines()

    # 统计错误
    errors = [l for l in logs if "ERROR" in l]
    print(f"总错误数: {len(errors)}")

    # 统计警告
    warnings = [l for l in logs if "WARNING" in l]
    print(f"总警告数: {len(warnings)}")

    # 查找OOM
    oom_logs = [l for l in logs if "out of memory" in l.lower()]
    if oom_logs:
        print("发现OOM错误:")
        for log in oom_logs[-5:]:  # 显示最后5个
            print(log)

    # 查找CUDA错误
    cuda_errors = [l for l in logs if "CUDA error" in l]
    if cuda_errors:
        print("发现CUDA错误:")
        for log in cuda_errors[-5:]:
            print(log)

# 使用
analyze_vllm_logs("/var/log/vllm/vllm.log")
```

---

### B.2.2 性能profiling

**使用PyTorch Profiler**:

```python
import torch
from vllm import LLM, SamplingParams

# 启用profiler
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=6,
        repeat=1
    ),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    llm = LLM(model="meta-llama/Llama-3-8B")
    params = SamplingParams(max_tokens=100)

    for i in range(10):
        llm.generate(["Hello world"] * 10, params)
        prof.step()

# 查看结果
print(prof.key_averages().table(sort_by="cuda_time_total"))

# 启动TensorBoard
# tensorboard --logdir=./logs
```

**使用Nsight Systems**:

```bash
# 1. 安装Nsight Systems
# https://developer.nvidia.com/nsight-systems

# 2. 采集trace
nsys profile \
  -y 30 \
  -o vllm_report \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  python your_vllm_app.py

# 3. 查看GUI
nsys-ui vllm_report.qdrep

# 4. 查看统计
nsys stats vllm_report.qdrep --report csv > stats.csv
```

**关键指标解读**：这些指标没有跨模型通用的“理想值”，应与同工作负载的健康基线和业务 SLO 对比。

| 指标 | 能说明什么 | 不能单独说明什么 |
|------|------------|------------------|
| **GPU activity** | 采样窗口内 GPU 是否有工作 | 高就是高效、低就是 CPU 瓶颈 |
| **内存吞吐** | kernel 的数据移动压力 | 未达峰值就一定是计算瓶颈 |
| **Occupancy** | 可驻留 warp 比例 | occupancy 越高性能一定越好 |
| **L2 hit rate** | 该访问模式的缓存复用 | 命中率越高端到端延迟一定越低 |

vLLM 服务指标名称与弃用边界以稳定版生产指标文档为准。[CITE: vllm-production-metrics-stable]

---

### B.2.3 逐步排查法

**问题排查决策树**:

```
问题: 推理慢
│
├─ Step 1: 检查服务指标与GPU时间线
│  ├─ 有持续等待队列 → 做容量与调度分析
│  └─ 无等待但GPU有空洞 → 检查流量、CPU、同步、通信
│      │
│      ├─ Step 2: 检查显存使用
│      │  ├─ 接近上限 → 内存受限
│      │  │   ├─ 减少batch size
│      │  │   └─ KV Cache量化
│      │  └─ 远低于上限 → CPU/网络瓶颈
│      │      ├─ 检查CPU使用率
│      │      └─ 检查网络带宽
│
└─ Step 3: 使用profiler定位瓶颈
   ├─ PyTorch Profiler
   ├─ Nsight Systems
   └─ Nsight Compute
```

**排障记录模板**：记录问题时间窗、版本和制品、负载分布、服务指标、GPU/CPU 时间线、唯一变量、变更前后结果及回滚条件。没有这些信息的“500 → 1200 tok/s”不能视为可复用案例。

---

### B.2.4 社区求助技巧

**提问前检查清单**:

```markdown
- [ ] 我是否搜索过现有issues?
- [ ] 我是否阅读了文档?
- [ ] 我是否尝试了常见解决方案?
- [ ] 我是否提供了最小可复现示例?
- [ ] 我是否提供了环境信息?
```

**好的Issue模板**:

```markdown
## 问题描述
vLLM在推理Llama-3-8B时出现OOM错误

## 环境信息
- GPU: RTX 4090 (24GB)
- CUDA: 12.2
- vLLM: 0.6.0
- Python: 3.10

## 命令
```bash
vllm serve meta-llama/Llama-3-8B \
  --max-model-len 8192 \
  --max-num-seqs 256
```

## 错误信息
```
CUDA out of memory. Tried to allocate 2.00 GiB
```

## 已尝试的方案
- [x] 降低max-model-len到4096 → 仍然OOM
- [x] 降低max-num-seqs到64 → 可以运行
- [x] 启用KV Cache量化 → 可以运行,但精度下降

## 预期行为
期望在max-model-len=8192, max-num-seqs=256下正常运行

## 实际行为
在max-model-len=8192时OOM
```

**哪里求助**:

1. **GitHub Issues**:
   - vLLM: https://github.com/vllm-project/vllm/issues
   - SGLang: https://github.com/sgl-project/sglang/issues

2. **Discord**:
   - vLLM: https://discord.gg/vllm
   - LMSys: https://discord.gg/msys

3. **Stack Overflow**:
   - 标签: `vllm`, `llm`, `cuda`

4. **Reddit**:
   - r/LocalLLaMA

---

## B.3 性能问题诊断清单

### B.3.1 硬件层面

**GPU检查**:

```bash
# 1. GPU信息
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv

# 2. GPU温度
nvidia-smi --query-gpu=temperature.gpu --format=csv

# 3. GPU带宽(理论上限)
# RTX 4090: ~1 TB/s
# A100: ~2 TB/s
# H100: ~3.35 TB/s

# 4. GPU计算能力
# RTX 4090: ~83 TFLOPS (FP16)
# A100: ~312 TFLOPS (FP16)
# H100: ~1000 TFLOPS (FP16)
```

**检查清单**:
- [ ] GPU型号是否满足要求?
- [ ] GPU驱动是否最新?
- [ ] CUDA版本是否兼容?
- [ ] GPU温度是否正常(<85°C)?
- [ ] GPU带宽是否饱和?

---

### B.3.2 软件层面

**版本检查**:

```bash
# 1. CUDA版本
nvcc --version

# 2. PyTorch版本
python -c "import torch; print(torch.__version__)"

# 3. vLLM版本
python -c "import vllm; print(vllm.__version__)"

# 4. GPU驱动版本
nvidia-smi
```

**依赖检查**:

```bash
# 检查关键依赖
pip list | grep -E "torch|vllm|transformers|flash-attn"

# 检查版本兼容性
python -c "
import torch
import vllm
print(f'PyTorch: {torch.__version__}')
print(f'vLLM: {vllm.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
"
```

**检查清单**:
- [ ] CUDA版本是否>=11.8?
- [ ] PyTorch版本是否>=2.0?
- [ ] vLLM版本是否最新?
- [ ] flash-attn是否安装?
- [ ] 依赖版本是否兼容?

---

### B.3.3 配置层面

**vLLM配置检查**:

```bash
# 1. 查看默认配置
vllm serve meta-llama/Llama-3-8B --help

# 2. 关键参数检查
# - max-model-len: 是否合理?
# - gpu-memory-utilization: 是否>=0.9?
# - max-num-seqs: 是否太小?
# - tensor-parallel-size: 是否匹配GPU数量?

# 3. 推荐配置
vllm serve meta-llama/Llama-3-8B \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --dtype auto
```

**检查清单**:
- [ ] max-model-len是否合理?
- [ ] gpu-memory-utilization是否>=0.9?
- [ ] max-num-seqs是否合理?
- [ ] tensor-parallel-size是否正确?
- [ ] 是否启用了Prefix Caching?
- [ ] 是否启用了Chunked Prefill?

---

### B.3.4 应用层面

**请求模式分析**:

```python
import time
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B")
params = SamplingParams(max_tokens=100)

# 测试不同batch size
for batch_size in [1, 10, 50, 100]:
    prompts = ["Hello world"] * batch_size

    start = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - start

    tokens = sum(len(out.outputs[0].tokens) for out in outputs)
    throughput = tokens / elapsed

    print(f"Batch {batch_size}: {throughput:.2f} tok/s")
```

**瓶颈分析**:

```python
def diagnose_bottleneck():
    """诊断瓶颈"""

    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    # GPU利用率
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    print(f"GPU利用率: {util.gpu}%")

    # 显存使用
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"显存使用: {mem.used / mem.total * 100:.1f}%")

    # 诊断
    if util.gpu > 80:
        print("→ 计算瓶颈: 考虑使用更好的GPU或量化")
    elif mem.used / mem.total > 0.8:
        print("→ 内存瓶颈: 减少batch size或max_model_len")
    else:
        print("→ CPU/网络瓶颈: 检查CPU使用率和网络带宽")

diagnose_bottleneck()
```

**检查清单**:
- [ ] 请求模式是否合理?
- [ ] 是否有长尾请求?
- [ ] 是否启用了Session路由?
- [ ] 是否有超时设置?
- [ ] 是否有重试机制?

---

**💡 调试最佳实践**

1. **从简单开始**: 先用小模型、小数据集测试
2. **逐步增加**: 每次只改变一个变量
3. **记录基准**: 保留性能基准数据
4. **使用工具**: Nsight Systems、PyTorch Profiler
5. **查看日志**: DEBUG级别日志提供详细信息
6. **社区求助**: 提供最小可复现示例

---

**需要设计压测时，可参考 [附录C: 基准测试与ROI教学算例（待验证）](appendix-c-benchmarks-roi.md) 中的字段模板；不要把其中数字当作性能基线。**
