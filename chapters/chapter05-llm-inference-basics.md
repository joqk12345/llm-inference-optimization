---
id: "chapters-chapter05-llm-inference-basics"
title: "第5章 推理的本质：Prefill 和 Decode 为什么要分开讲"
slug: "chapters-chapter05-llm-inference-basics"
date: "2026-03-11"
type: "article"
topics:
  - "inference-mechanics"
concepts:
  - "kv-cache"
  - "paged-attention"
  - "continuous-batching"
tools:
  - "vllm"
architecture_layer:
  - "inference-mechanics"
learning_stage: "foundations"
optimization_axes:
  - "latency"
  - "throughput"
  - "memory"
related:
  - "chapters-chapter06-kv-cache-optimization"
  - "chapters-chapter07-request-scheduling"
references:
  - "https://arxiv.org/abs/2309.06180"
  - "https://aclanthology.org/2023.emnlp-main.298/"
  - "https://docs.vllm.ai/en/stable/usage/metrics/"
status: "published"
display_order: 6
---
# 第5章 推理机制：Prefill 与 Decode

>  教学理念 (参考: Hugging Face "Continuous batching from first principles"、Hamza Elshafie "Paged Attention from First Principles")
>
> 核心思路: 从第一性原理出发,理解LLM推理的基本流程和优化动机。
>
> 学习路径: Prefill/Decode → Attention → KV Cache → 系统瓶颈 → 为第6-7章做准备

## 简介

在深入 vLLM 的复杂优化技术之前，我们需要先理解 LLM 推理的基础原理。很多工程师直接跳到“调参与技巧”，却忽略了第一性原理：prefill 与 decode 是两种不同的工作负载；Attention 的二次复杂度决定了 KV Cache 的必要性；而“端到端延迟/成本”往往由系统瓶颈迁移决定，而不是由某个单点优化决定。

为了把这一章写成“可反复复用的认知框架”，你可以用同一套问题来读：

- **背景**：为什么推理看起来简单（生成 token）但优化极难（带宽、缓存、调度、尾延迟）？
- **决策**：当你要优化时，先动 KV、先动调度、先动量化，分别在什么条件下成立？
- **落地**：你如何用 TTFT/TPOT、P50/P95/P99、吞吐与显存曲线把结论量化出来？
- **踩坑**：为什么“跑分快”不等于“线上更便宜”？为什么长上下文会把系统从 compute-bound 推向 memory-bound？

本章将带你从零开始,逐步理解:
- 训练vs推理: 为什么推理是内存密集型而训练是计算密集型
- LLM 如何生成文本 (Prefill 和 Decode 阶段)
- Attention 机制的工作原理和计算复杂度
- KV Cache 如何避免历史 token 的重复计算，并区分单步与完整生成的计算量
- 内存碎片化问题: 为什么后续会出现更复杂的 KV 管理方案
- 操作系统类比: 如何用虚拟内存视角理解分页式 KV 管理
- Chunked Prefill、PagedAttention、Continuous Batching 分别在解决什么问题
- vLLM 架构如何把这些优化组织成完整系统

学完本章，你将能够解释 vLLM 为什么能在很多场景里显著快于朴素实现，并知道哪些问题会在第6章和第7章被展开为独立主题。本章的目标是建立共同语言，而不是替代后续章节的工程细节。

---

## 5.1 训练 vs 推理: 工作负载的本质差异

>  核心洞察：理解训练和推理的工作负载差异,是理解优化策略的第一步。

### 5.1.1 训练: 计算密集型的并行工作负载

训练流程:

```
输入数据 → Forward Pass → Loss Calculation → Backward Pass → Weight Update
    ↓           ↓              ↓                ↓               ↓
  token      预测          计算误差         计算梯度        更新参数
```

特点:
-  Compute-bound: 计算密集型
-  高度并行: 整个batch同时处理
-  GPU利用率高: 矩阵运算充分利用GPU
-  极其昂贵: 需要大规模GPU集群

示例: 训练一个7B模型
- 数据量: 数万亿tokens
- 硬件: 数百个A100 GPU
- 时间: 数周到数月
- 成本: 数百万美元

### 5.1.2 推理: 内存带宽密集型的串行工作负载

推理流程:

```
用户Prompt → Prefill (并行) → Decode (串行) → 返回结果
    ↓            ↓               ↓
  整个输入    首token生成    逐token生成
             (一次处理)      (每次生成1个)
```

特点:
-  Memory-bound: 内存带宽密集型
-  串行生成: Decode阶段必须逐个生成
-  GPU利用率低: 大量时间在移动数据而非计算
-  持续运行: 7×24小时服务

关键差异:

```
训练工作负载:
- Forward pass: O(n²·d) 计算
- Backward pass: O(n²·d) 计算
- 内存访问: 连续、可预测
- GPU: 计算单元饱和

推理工作负载 (Prefill):
- Forward pass: O(n²·d) 计算
- 内存访问: 连续、可预测
- GPU: 计算单元较饱和

推理工作负载 (Decode):
- Forward pass: O(n·d) 计算 (每步只计算1个token)
- 内存访问: 频繁加载KV Cache和模型权重
- GPU: 常见为内存带宽或访存延迟瓶颈，但需要按实际 kernel 验证
```

### 5.1.3 为什么优化推理更关键

商业现实不是一组可以跨公司复用的固定数字，而是一条持续发生的成本流。下面给出计算框架：

```
月推理成本
= 实例单价 × 实例数量 × 运行小时
+ 网络、存储、CPU 与运维成本

单位输出成本
= 月推理总成本 / 实际交付的输出 token 数
```

如果一次优化把同一 SLO 和质量约束下的单卡有效吞吐从 `Q_before` 提高到 `Q_after`，理论容量需求比例为：

```
容量需求比例 = Q_before / Q_after
```

这只是容量上界。真实成本还取决于流量峰谷、最小副本、高可用冗余、计费粒度和尾延迟；吞吐提高 3 倍并不自动等于账单下降 67%。

---

## 5.2 LLM 如何生成文本

### 5.2.1 自回归生成的基本过程

LLM 的本质: "Fancy next token predictors" (花哨的下一个词预测器)

```
用户输入: "The capital of France is"

模型思考: 前面是"法国的首都是",下一个词最可能是什么?

模型输出: "Paris"
```

生成过程:

```
步骤 1: 输入整个 prompt
输入: "Hello, my name is"
↓
模型处理整个 prompt
↓
生成第 1 个 token: " John"

步骤 2: 添加新 token,再次预测
输入: "Hello, my name is John"
↓
模型处理整个序列 (包括之前所有的 token)
↓
生成第 2 个 token: " and"

步骤 3: 重复...
输入: "Hello, my name is John and"
↓
生成第 3 个 token: " I"
...

步骤 n: 生成 <eos> (end of sequence)
停止生成
```

关键观察:
- 第一个 token 出现较慢 (需要处理整个 prompt)
- 后续 token 逐个出现 (每次只生成 1 个)
- 每次生成新 token 时,都需要重新读取之前所有内容

---

### 5.2.2 Prefill 阶段: 并行处理 prompt

定义: 处理初始 prompt,生成第一个 token 的阶段

```
输入: "Once upon a time, there was a"
       └─────────┬─────────┘
            10 个 tokens

模型: 一次 forward pass 处理全部 10 个 tokens
     并行计算所有 token 的表示
↓
输出: 第一个 token: " little"
```

特点:
-  计算密集型: 大量矩阵乘法
-  可以并行处理: 所有 token 同时计算
-  时间: TTFT (Time To First Token), 首字延迟

为什么 Prefill 可以并行?
- 所有输入 token 是已知的
- 不需要考虑因果关系
- Attention 矩阵可以一次性计算

示例:
```
Prompt: 100 tokens
单次 Prefill 的耗时通常明显低于完整生成阶段,但具体数值取决于模型规模、硬件与实现。
```

---

### 5.2.3 Decode 阶段: 逐 token 生成

定义: 逐个生成后续 token 的阶段

```
步骤 1:
输入: "Once upon a time, there was a little"
      ↓ 模型处理
输出: " girl"

步骤 2:
输入: "Once upon a time, there was a little girl"
      ↓ 模型处理
输出: " who"

步骤 3:
输入: "Once upon a time, there was a little girl who"
      ↓ 模型处理
输出: " lived"

...重复 100 次...
```

特点:
-  内存带宽密集型: 每次只生成 1 个 token
-  无法并行: 必须逐个生成 (因果关系)
-  时间: TBT (Time Between Tokens), 字间延迟

为什么 Decode 不能并行?
- 每个新 token 依赖于之前生成的 token
- 必须等待前一个 token 生成完成
- 这是自回归模型的本质限制

示例:
```
生成 100 tokens:
- Decode 阶段会逐 token 串行生成
- 总耗时通常显著高于一次 Prefill
- 具体倍数取决于模型规模、KV Cache 实现与硬件带宽
```

---

### 5.2.4 图解完整流程

```
时间线:
0ms      200ms   220ms   240ms   260ms   ...   2200ms
│        │       │       │       │             │
└──Prefill┬─Decode1┬─Decode2┬─Decode3─ ... ─Decode100─
         │       │       │       │
         │       │       │       └─ 生成 "lived"
         │       │       └─ 生成 "who"
         │       └─ 生成 "girl"
         └─ 生成 "little" (第一个 token)

Prefill 阶段:
- 输入: 10 tokens
- 时间: 200ms
- 计算: 并行处理

Decode 阶段:
- 输入: 每次加 1 个 token
- 时间: 每个 20ms
- 计算: 串行生成 100 个 tokens

总时间: 200ms + 100 × 20ms = 2200ms
```

优化方向:
- Prefill: 优化计算 (更快的 GPU, 更好的内核)
- Decode: 优化内存带宽 (KV Cache, PagedAttention)

---

### 5.2.5 延迟分解：端到端延迟从哪来？

> **工程视角**：理解延迟分解是优化工作的第一步。只有知道"慢在哪里"，才能选择正确的优化方向。

**总延迟构成**：

```
总延迟 = TTFT + Σ(TPOT_i) + 网络开销 + 调度开销

其中：
- TTFT (Time To First Token): 首字延迟，Prefill 阶段
- TPOT (Time Per Output Token): 字间延迟，Decode 阶段
- 网络开销: gRPC/HTTP 序列化、反序列化、网络传输
- 调度开销: 队列等待、调度器决策、KV Cache 管理
```

**延迟分解记录模板**：

| 阶段 | 测量值 | 证据 | 候选优化 |
|------|--------|------|----------|
| 排队 + TTFT | P50/P95/P99 | 服务 histogram + trace | admission、prefill、缓存、路由 |
| TPOT / ITL | P50/P95/P99 | 服务 histogram + token timeline | KV、kernel、batch、量化 |
| 网络与序列化 | 客户端和服务端时间戳 | 分布式追踪 | 连接复用、协议与 payload |
| 调度与 CPU | iteration 时间线 | profiler / trace | 调度流水线、采样与 launch |

**不同场景下的瓶颈迁移**：

| 场景特征 | 优先检查 | 不能直接推出 |
|----------|----------|--------------|
| prompt 相对较长 | prefill 执行、队列和 prefix 命中 | TTFT 一定由算力主导 |
| 输出相对较长 | TPOT 随 batch、上下文增长的曲线 | decode 一定只受带宽限制 |
| 高并发 | 排队、抢占、KV 使用率和尾延迟 | GPU 利用率高就代表系统健康 |
| 显存接近容量边界 | OOM、抢占、重算和临时 buffer | 问题一定来自碎片化 |

**实战诊断方法**：

```bash
# 1. 监控 TTFT vs TPOT 分布
# 当前稳定文档中的代表性指标：
# - vllm:time_to_first_token_seconds
# - vllm:request_time_per_output_token_seconds
# - vllm:num_requests_waiting
# - vllm:kv_cache_usage_perc
# 指标名会随弃用策略演进，升级前检查目标版本 /metrics。

# 2. 判断瓶颈类型
# 把主要延迟贡献与业务 SLO 对齐，再选择优化对象。

# 3. 使用 nsight systems 做深度分析
nsys profile -o inference_trace ./run_inference.py
# 查看 GPU 时间线，确认是计算还是内存等待
```

上述指标口径以 vLLM 稳定版生产指标文档为边界。[CITE: vllm-production-metrics-stable]

**优化实验矩阵**：

| 优化手段 | 机制假设 | 同时观察 | 主要风险 |
|----------|----------|----------|----------|
| 权重/KV 量化 | 减少容量或数据移动 | 质量、TTFT、TPOT、显存 | 反量化开销与质量回归 |
| PagedAttention | 降低外部碎片与分配浪费 | 并发、KV 使用率、尾延迟 | block 元数据与调度开销 |
| Chunked Prefill | 控制长 prefill 对 decode 的干扰 | TTFT、TPOT、公平性 | prefill 完成更晚 |
| 连续批处理 | 提高 iteration 中的有效工作量 | 吞吐、排队、尾延迟 | 高负载下排队扩大 |
| PD 分离 | 隔离阶段并独立扩缩容 | goodput、KV 传输、成本 | 网络和运维复杂度 |

> **关键洞察**：优化必须"对症下药"。如果瓶颈在 TTFT，却花时间优化 TPOT，往往事倍功半。

>  为什么重要: Attention 是唯一让不同 token 产生交互的地方。理解 Attention,就理解了 LLM 的核心。

### 5.3.1 Token 的表示: 向量与 hidden dimension

Tokenization: 文本 → token 序列

```
文本: "Hello, world!"
↓ Tokenizer
Tokens: [15496, 11, 2159, 0]
       │      │   │    │
     Hello    ,  world  <eos>
```

Embedding: 每个 token → d 维向量

```
Tokens: [15496, 11, 2159, 0]
       │      │    │    │
       ▼      ▼    ▼    ▼
Embeddings:
  Token 15496 → [0.12, -0.34, 0.56, ..., 0.78]  (d=4096)
  Token 11    → [-0.23, 0.45, -0.67, ..., 0.89]
  Token 2159  → [0.34, -0.56, 0.78, ..., -0.12]
  Token 0     → [-0.45, 0.67, -0.89, ..., 0.23]
```

Tensor 形状: `[batch_size, sequence_length, hidden_dim]`

```python
import torch

# 7 个 tokens
input_ids = torch.randint(0, 32000, (1, 7))  # [batch=1, seq_len=7]
# 形状: [1, 7, 4096] (假设 hidden_dim=4096)
embeddings = model.embeddings(input_ids)
print(embeddings.shape)  # torch.Size([1, 7, 4096])
```

Hidden Dimension: 模型的"表示能力"
- GPT-2: d = 768 或 1024
- Llama-2-7B: d = 4096
- Llama-2-70B: d = 8192

---

### 5.3.2 Query、Key、Value 投影

三个权重矩阵: Wq、Wk、Wv

```python
# 每个 token 的表示: x
x = embeddings[i]  # shape: [hidden_dim]

# 投影到 Q、K、V
Q = x @ Wq  # Query: 这个 token 想找什么?
K = x @ Wk  # Key: 这个 token 能提供什么?
V = x @ Wv  # Value: 这个 token 的实际内容

# Wq, Wk, Wv 的形状: [hidden_dim, head_dim]
# Q, K, V 的形状: [head_dim]
```

物理意义:

```
Token: "apple"
- Query: "我是水果,我想找与上下文相关的信息"
- Key: "我是水果,我可以被与食物相关的查询找到"
- Value: "我的具体语义内容是'苹果'"

Token: "company"
- Query: "我是组织,我想找与商业相关的信息"
- Key: "我是组织,我可以被与商业相关的查询找到"
- Value: "我的具体语义内容是'公司'"
```

多头 Attention (Multi-Head Attention):

```python
# 32 个 attention heads
num_heads = 32
head_dim = hidden_dim // num_heads  # 4096 // 32 = 128

# 每个 head 学习不同的关系模式
# Head 1: 关注语法关系
# Head 2: 关注语义关系
# Head 3: 关注指代关系
# ...
```

---

### 5.3.3 Attention 计算: QK^T 与二次复杂度

计算步骤:

```python
# 输入: Q, K, V
# 形状: [batch, num_heads, seq_len, head_dim]
Q, K, V = ...

# 步骤 1: 计算 Q @ K^T
# 相似度矩阵: 每个 token 对其他 token 的关注度
scores = Q @ K.transpose(-2, -1)  # [batch, num_heads, seq_len, seq_len]

# 步骤 2: 缩放
scores = scores / (head_dim ** 0.5)

# 步骤 3: Softmax 归一化
attn_weights = torch.softmax(scores, dim=-1)

# 步骤 4: 加权求和
output = attn_weights @ V  # [batch, num_heads, seq_len, head_dim]
```

复杂度分析:

```
Q @ K^T:
- Q: [n, d]
- K^T: [d, n]
- 结果: [n, n]
- 计算: n × d × n = O(n²·d)

Softmax:
- 输入: [n, n]
- 计算: O(n²)

attn_weights @ V:
- attn_weights: [n, n]
- V: [n, d]
- 结果: [n, d]
- 计算: n × n × d = O(n²·d)

总复杂度: O(n²·d)
```

关键洞察: Attention 的二次复杂度是性能瓶颈!

```
序列长度 n = 1000:
- Attention 计算: 1000² × 4096 = 4,096,000,000 次运算

序列长度 n = 10000:
- Attention 计算: 10000² × 4096 = 409,600,000,000 次运算 (100倍!)
```

---

### 5.3.4 Attention Mask: 控制 token 交互

什么是 Mask: 布尔矩阵,决定哪些 token 可以交互

```python
# Attention Mask
# shape: [seq_len, seq_len]
mask = torch.tensor([
    [True,  False, False, False],  # Token 0 只能 attend to 0
    [True,  True,  False, False],  # Token 1 可以 attend to 0,1
    [True,  True,  True,  False],  # Token 2 可以 attend to 0,1,2
    [True,  True,  True,  True],   # Token 3 可以 attend to 0,1,2,3
])
```

作用: Mask=False 的位置,attention 权重 = 0

```python
# 应用 mask
scores = Q @ K.T
scores = scores.masked_fill(~mask, float('-inf'))
attn_weights = torch.softmax(scores, dim=-1)
```

可视化方法:
```
绿色方块 (True)  = 可以交互
白色方块 (False) = 不能交互
```

---

### 5.3.5 Causal Mask: 因果关系的可视化

定义: 每个 token 只能与之前的 token 交互

直觉: 因必须在果之前

```
Token序列: <bos>  I     am    sure

Token 0 (<bos>):
  可以 attend to: [<bos>]
  Mask:          [  ]

Token 1 (I):
  可以 attend to: [<bos>, I]
  Mask:          [   ,  ]

Token 2 (am):
  可以 attend to: [<bos>, I, am]
  Mask:          [   ,  ,  ]

Token 3 (sure):
  可以 attend to: [<bos>, I, am, sure]
  Mask:          [   ,  ,  ,  ]
```

Mask 形状: 下三角矩阵

```python
def generate_causal_mask(seq_len):
    """生成 causal mask"""
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=bool))
    return mask

# 示例: seq_len = 4
mask = generate_causal_mask(4)
print(mask)
# tensor([[ True, False, False, False],
#         [ True,  True, False, False],
#         [ True,  True,  True, False],
#         [ True,  True,  True,  True]])
```

读 mask 方法:
- 行: 当前 token
- 列: 历史 token
- True: 当前 token 可以 attend 到该历史 token

---

### 5.3.6 为什么 Attention 是唯一让 token 交互的地方

其他操作: token-wise,每个 token 独立处理

```python
# Layer Normalization
x = LayerNorm(x)  # 每个 token 独立归一化

# 激活函数
x = GELU(x)  # 每个 token 独立激活

# 矩阵乘法 (非 Attention)
x = x @ W  # 每个 token 独立投影
```

Attention 的作用: 让 token 之间"交流"

```python
# Attention
output = Attention(Q, K, V)  # token 之间聚合信息!
```

结论: 理解了 attention mask,就理解了 LLM 的信息流

---

## 5.4 从朴素生成到 KV Cache

### 5.4.1 朴素方法: 每次重新计算 (O(n²))

问题场景: 生成第 n+1 个 token

朴素做法:
```
已生成: "Hello, my name is John" (7 tokens)
目标:   生成第 8 个 token

朴素方法:
1. 将所有 8 个 tokens 重新输入模型
   ["Hello", ",", "my", "name", "is", "John", 新token]

2. 重新计算所有 token 的 K 和 V

3. 只使用最后一个 token 的输出
```

计算复杂度:
```
生成第 1 个 token:  O(1²)
生成第 2 个 token:  O(2²)
生成第 3 个 token:  O(3²)
...
生成第 n 个 token:  O(n²)

总复杂度: O(1² + 2² + ... + n²) = O(n³)
```

可视化浪费:
```
第 2 步: 重新计算 Token 1 的 K、V  浪费
第 3 步: 重新计算 Token 1, 2 的 K、V  浪费
...
第 n 步: 重新计算 Token 1, 2, ..., n-1 的 K、V  浪费
```

---

### 5.4.2 重复计算问题的可视化

关键观察: 新 token (如"will") 不影响旧 token 的 attention 计算

```
Token 序列:
["I", "am", "sure", "I", "will"]

Token 4 ("will") 生成后:

对于 Token 0 ("I"):
- 计算 attention 时,只会看 Token 0-3
- Token 4 不影响 Token 0 的 attention
- 原因: Causal mask!

对于 Token 1 ("am"):
- 计算 attention 时,只会看 Token 0-1
- Token 4 同样不影响
```

原因: Causal mask,未来 token 不影响过去

```python
# Token 4 的 attention mask
mask = [True, True, True, True, True]
       └───────────────┬───────────────┘
          可以 attend to 之前 + 自己
```

图示:
```
最后一个 token 只关心自己的预测,
不影响其他 token 的 attention 计算!
```

---

### 5.4.3 KV Cache 的核心思想

核心洞察: 旧 token 的 K、V 已经计算过,缓存起来!

做法:
```
Prefill 阶段:
输入: "Hello, my name is"
计算:
  Token 0: K0, V0
  Token 1: K1, V1
  Token 2: K2, V2
  Token 3: K3, V3
存储: 缓存所有 K, V

Decode 阶段 - 第 1 步:
输入: 新 Token 4
计算:
  Token 4: K4, V4
复用: K0, K1, K2, K3 (从缓存读取)
组合: [K0, K1, K2, K3, K4]
      [V0, V1, V2, V3, V4]

Decode 阶段 - 第 2 步:
输入: 新 Token 5
计算:
  Token 5: K5, V5
复用: K0, K1, K2, K3, K4 (从缓存读取)
组合: [K0, K1, K2, K3, K4, K5]
      [V0, V1, V2, V3, V4, V5]
```

效果: 避免重复计算
-  每个 token 的 K、V 只计算一次
-  大幅减少计算量

代价: 显存占用 O(n)
-  需要存储所有历史 token 的 K、V
-  序列越长,显存占用越大

---

### 5.4.4 计算量如何变化：区分 prompt 与生成长度

不能只用一个 `n` 同时表示输入长度和生成长度。设 prompt 长度为 `P`，随后生成 `T` 个 token，并暂时忽略层数、head 数和隐藏维度等共同因子。

**不使用 KV Cache** 时，第 `t` 个生成步骤需要把长度约为 `P+t` 的整段序列重新送入模型，既重复计算旧 token 的投影，也重复执行旧位置之间的 Attention：

```text
总计算量 ≈ Σ(t=1..T) O((P+t)²)
```

**使用 KV Cache** 时，prompt 只做一次 prefill；每个 decode 步骤只为新 token 计算 Q/K/V，并让新 query 读取已有的 K/V：

```text
Prefill ≈ O(P²)
Decode  ≈ Σ(t=1..T) O(P+t)
总计算量 ≈ O(P² + TP + T²)
```

因此，KV Cache 消除的是跨 decode 步骤对历史 token 的重复计算；它不会把 Attention 本身变成常数复杂度，也不会自动给出某个固定加速倍数。实际收益还取决于模型结构、`P/T` 比例、batch、显存带宽、kernel 实现与调度开销，应通过端到端基准测试确认。

---

### 5.4.5 显存代价: 每个 token 需要多少显存?

单 token 的 cache 大小:
```
Size = 2 × L × H × A × bytes

其中:
- 2: K 和 V
- L: 层数 (如 32)
- H: heads 数 (如 32)
- A: head dimension (如 128)
- bytes: 每个元素的字节数 (FP16 = 2 bytes)
```

示例计算:
```
Llama-2-7B:
- L = 32 层
- H = 32 heads
- A = 128 head_dim
- bytes = 2 (FP16)

单 token cache:
= 2 × 32 × 32 × 128 × 2
= 524,288 bytes
≈ 0.5 MB/token

1000 tokens:
= 1000 × 0.5 MB
= 500 MB

10000 tokens:
= 10000 × 0.5 MB
= 5000 MB = 5 GB
```

权衡: 用显存换计算
-  计算: 大幅加速
-  显存: 线性增长

### 5.4.6 不同Attention变体的内存优化

问题: KV Cache的显存占用仍然很大

解决方案: 减少每个token的K、V大小

Multi-Query Attention (MQA):

```
传统MHA:
- 每个head有自己的K、V
- 32 heads → 32组K、V

MQA:
- 所有heads共享一组K、V
- 32 heads → 1组K、V
- 仅按 KV 元素数量计算，约为对应 MHA 的 1/32
- 代价: 模型质量可能下降
```

Grouped-Query Attention (GQA) - 现代模型中的常见折中:

```
折中方案:
- 32 heads分成8组
- 每组(4个heads)共享一组K、V
- 仅按 KV 元素数量计算，约为对应 MHA 的 1/4
- 质量: 需要在模型训练与下游任务中验证
```

Multi-Head Latent Attention (MLA) - DeepSeek V2/V3使用:

```
激进压缩:
- K、V投影到低维latent空间
- DeepSeek-V3: 671B参数,37B激活
- 内存减少: 显著
- 质量: 保持竞争力
```

对比总结:

| 变体 | K、V数量 | 内存占用 | 模型质量 | 使用场景 |
|------|---------|---------|---------|---------|
| MHA | H组 | 基准 | 最佳 | 推理不受限 |
| GQA | 可配置 KV heads | 按 KV head 比例减少 | 任务相关 | 平衡容量与质量 |
| MQA | 1组 | 减少H倍 | 可能下降 | 内存极度受限 |
| MLA | 1组latent | 最小 | 有竞争力 | 超大模型 |

如果把这几种做法和近两年的前沿模型放在一起看，你会发现它们其实对应了三类不同路线：

**第一类：表示压缩路线**

- 代表：`MQA`、`GQA`、`MLA`
- 核心问题：每个 token 的 K/V 表示是否可以更小
- 典型收益：在不改写整体 attention 机制的前提下，直接降低 KV Cache 占用

这类方法最像“给每个 token 减肥”。token 数量没有变，但每个 token 背后的 K/V 存储成本下降了。

这里的比例只描述理论 KV 元素数量，不包含元数据、block 舍入和 kernel 效率；GQA 的质量结论也必须限定在具体训练与评测设置中。[CITE: gqa-emnlp-2023]

**第二类：线性化路线**

- 代表：`Lightning Attention`、`Kimi Linear`
- 核心问题：能否让 attention 不再随序列长度保持二次增长
- 典型收益：在超长上下文下，把显存与 decode 吞吐重新拉回可接受范围

这类方法不是简单缩小 K/V，而是在重写 attention 的计算方式。`MiniMax-01` 把 `Lightning Attention` 推到百万级上下文与大规模 MoE 体系里，`Kimi Linear` 则继续把 hybrid linear attention 往“full attention 的可替代方案”方向推进。

**第三类：压缩 + 稀疏 + 局部保真路线**

- 代表：`DeepSeek-V4` 的 `CSA/HCA`
- 核心问题：上下文变长后，KV 条目数量本身能不能减少，同时尽量保住局部依赖
- 典型收益：既压缩 KV，又让模型在超长上下文下不必对所有历史位置做同样昂贵的处理

这类方法更像是在改写“哪些历史信息值得被完整保留、哪些可以被压缩摘要”。

所以这里有一个很重要的认知边界：

- `MQA/GQA/MLA` 主要在解决 **每个 token 存多大**
- `Lightning/Kimi Linear` 主要在解决 **attention 怎么算**
- `CSA/HCA` 主要在解决 **历史 token 是否都要原样保留**

这也是为什么第6章会继续把 KV 问题拆开讲：你不能把“分页管理”“低比特量化”“线性 attention”“压缩 attention”都混成同一类优化，它们作用在不同层面。

把这组路线和 DeepSeek-V4 放在一起看，会更容易理解 attention 为什么正在被重新组织：`MQA/GQA/MLA` 更像表示压缩，`MiniMax-01 / Kimi Linear` 更像线性化路线，而 `DeepSeek-V4` 的 `CSA/HCA` 更像压缩 + 稀疏 + 局部保真路线。它们共同说明的一件事是，长上下文问题已经不只是“KV 够不够大”，而是 attention 该以什么结构继续向前扩展。

---

## 5.5 KV Cache的内存管理挑战

KV 会随上下文和并发增长。按最大长度预留会产生未使用空间，要求连续大块的分配方式还可能遇到外部碎片。问题是“状态怎么放、何时释放”，不是 Attention 的数学定义。

例如总内存 128 MB，中间 24 MB 被占用，两端分别空闲 32 MB 和 72 MB。总空闲为 104 MB，但仍无法提供连续的 80 MB。固定块管理如何改变这一约束，留到 [第6章](chapter06-kv-cache-optimization.md)展开。

## 5.6 操作系统类比: 虚拟内存与分页

分页类比只需记住：逻辑连续不要求物理连续，映射表把请求的逻辑 KV 块指向物理块。GPU Attention kernel 按映射访问数据；这不意味着操作系统 MMU 自动管理 KV，也不自动提供磁盘换页。

页大小、分配、释放、共享和写时复制统一见 [第6章](chapter06-kv-cache-optimization.md)。

## 5.7 Chunked Prefill: 处理长 prompt

把长 prompt 的 prefill 拆成多轮，可以控制单轮计算和中间激活，并给已有 decode 请求留下执行机会。但已处理 token 的 KV 仍要保留；切块不能保证任意长度都不 OOM。

chunk 大小影响 TTFT、其他请求的 TPOT 和公平性。如何与每轮 token budget 配合，见 [第7章](chapter07-request-scheduling.md)。[CITE: sarathi-serve-osdi-2024]

## 5.8 PagedAttention入门: 借鉴OS的内存管理

PagedAttention 通过块表访问非连续的 KV 存储，使按需分配及共享更容易。它改变存储与访问组织，不改变 Attention 的数学目标。尾块仍可能存在内部浪费，映射和管理也有开销。[CITE: pagedattention-sosp-2023]

容量估算、前缀复用和工程权衡见 [第6章](chapter06-kv-cache-optimization.md)。

## 5.9 批处理的挑战: 从静态到动态

请求的输入长度、输出长度和到达时间不同。固定批次可能等待最慢请求，稠密 padding 实现还会产生额外计算；两种浪费要分开讨论。ragged/packed 表示能减少 padding，但不等同于迭代级调度。

## 5.10 Continuous Batching 入门

核心是每个迭代边界重新选择活跃请求：移出完成者、加入等待者，并分配本轮 token 预算。它不以“去掉 batch 维度”为定义，也不能保证 GPU 始终满载。[CITE: orca-osdi-2022]

| 方式 | 改变工作集的时机 |
| --- | --- |
| Static batching | 固定批次完成后 |
| Dynamic batching | 请求进入执行前按等待窗口凑批 |
| Continuous batching | 生成过程中的迭代边界 |

实际框架可能组合这些机制。准入、抢占、公平性与延迟权衡见 [第7章](chapter07-request-scheduling.md)。

## 5.11 vLLM 架构预览

用与框架版本无关的职责理解请求路径：

```text
接口接收与校验 → 调度分配预算 → KV 管理准备状态
             → 运行时执行 → 采样与返回 → 下一轮
```

| 职责 | 需要回答的问题 | 后续章节 |
| --- | --- | --- |
| 接口 | 请求如何校验、限流和观测 | 第10章 |
| 调度 | 本轮谁运行、分配多少 token | 第7章 |
| 状态管理 | KV 如何分配、共享和释放 | 第6章 |
| 运行时 | 算子与通信耗时在哪里 | [独立专题](../docs/topics/runtime-kernel-optimization.md) |

具体内部类名会随版本改变。学完本章，应能把问题定位到某类职责，再进入对应专题，而不是记住某个版本的对象树。

---

## 章节检查清单

完成本章后,你应该能够:

- [ ] 解释 LLM 生成文本的两个阶段 (Prefill 和 Decode)
- [ ] 理解 Attention 机制的计算过程和复杂度
- [ ] 区分 KV Cache 对 prefill、单步 decode 和完整生成计算量的影响
- [ ] 计算 KV Cache 的显存占用
- [ ] 解释 Chunked Prefill 的原理和应用场景
- [ ] 对比 Static Batching、Dynamic Batching 和 Continuous Batching
- [ ] 描述 vLLM 的三层架构
- [ ] 将性能问题初步归类到接口层、调度层、KV层或运行时层

---

## 动手练习

练习 5.1: 计算 KV Cache 显存占用

Llama-2-7B 的配置:
- 层数: 32
- Attention heads: 32
- Head dimension: 128
- 数据类型: FP16 (2 bytes)

问题:
1. 单个 token 的 KV cache 大小是多少?
2. 1000 tokens 的 KV cache 需要多少显存?
3. 如果有 10 个并发请求,每个请求平均 500 tokens,总共需要多少显存?

练习 5.2: 对比不同 Batching 方法的效率

假设有以下 3 个请求:
- Request A: "Hi" (2 tokens)
- Request B: "Hello, how are you?" (5 tokens)
- Request C: "The quick brown fox jumps over the lazy dog" (10 tokens)

任务:
1. 计算 Static Batching 的 padding 数量
2. 假设三个请求在不同迭代完成，画出移出完成者、加入新请求的调度时间线
3. 计算 padding 实现的有效 token 比例，并解释为什么它不等于 GPU 利用率

练习 5.3: 用模块地图分析一个请求流程

启动 vLLM server,发送一个请求,并观察日志:
1. 将日志粗分到接口层、调度/KV层、运行时层
2. 记录请求从进入到返回的时间
3. 标出最可能影响 TTFT 的两个模块

---

## 本章小结

关键要点：
- 训练、prefill 和 decode 的资源画像不同，瓶颈要结合 batch、形状、硬件与实际 kernel 判断
- Prefill 通常具有较高计算强度，低 batch 的 decode 常受带宽限制，但不是无条件分类
- 在标准 Transformer block 中，self-attention 负责跨 token 聚合；完整序列 Attention 的计算复杂度随序列长度二次增长
- KV Cache 缓存历史 token 的 K、V，使单个 decode 步骤只为新 token 计算投影，并读取随上下文线性增长的历史状态
- 传统连续 KV 分配会产生预留浪费、内部/外部碎片与复制开销，具体比例依工作负载而变
- PagedAttention 借鉴操作系统虚拟内存技术,将 KV Cache 分块管理
- PagedAttention 通过按需分配固定大小 blocks 限制预留浪费并缓解碎片
- 论文中的 vLLM 设计支持在序列内部及请求之间灵活共享 KV blocks [CITE: pagedattention-sosp-2023]
- Chunked Prefill 控制单轮预算，但不能消除长期 KV 容量限制
- Continuous Batching 在迭代边界调整活跃请求，收益取决于负载与调度策略
- vLLM 的三层架构 (接口层、引擎/策略层、运行时层) 提供了清晰的抽象

## 章节衔接

这一章的目标是把“问题空间”搭出来,而不是把所有优化都讲完。下一章自然会先进入第6章,因为一旦你理解了 prefill/decode 和 Attention 的代价,最先需要精细化管理的就是 KV 这类长期占用显存的资产; 再往后第7章才会在这个基础上讨论调度器如何消费这些 KV 状态来做系统级决策。

---

## 参考资料
- [Paged Attention from First Principles: A View Inside vLLM](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vLLM/) by Hamza Elshafie
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM原始论文
