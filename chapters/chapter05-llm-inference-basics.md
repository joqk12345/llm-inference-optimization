---
id: "chapters-chapter05-llm-inference-basics"
title: "第5章：LLM推理基础"
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
  - "vLLM"
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
references: []
status: "published"
display_order: 6
---
# 第5章：LLM推理基础

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
- KV Cache 如何将复杂度从 O(n²) 降到 O(n)
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
- GPU: 内存带宽瓶颈
```

### 5.1.3 为什么优化推理更关键

商业现实:

```
训练成本: 一次性投入
- GPT-3训练: 约$4.6M
- LLaMA-2训练: 约$2-3M

推理成本: 持续运营
- 每天处理1M请求
- 每请求平均1000 tokens
- 每token成本: $0.0001
- 每月成本: $3M
```

优化推理的收益:

```
优化前: 每GPU处理10 req/s
优化后: 每GPU处理30 req/s (3x提升)
→ GPU需求减少 67%
→ 成本降低 67%
```

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

**典型分布（Llama-2-7B, A100-80GB, 512 input / 128 output）**：

| 阶段 | 耗时 | 占比 | 瓶颈类型 | 优化方向 |
|------|------|------|----------|----------|
| TTFT | 120ms | 50% | Compute-bound | 更快的 kernel、FP8、Chunked Prefill |
| TPOT (×128) | 80ms | 33% | Memory-bound | KV Cache、量化、PagedAttention |
| 网络开销 | 20ms | 8% | Network | gRPC 优化、连接复用 |
| 调度开销 | 20ms | 8% | CPU | 连续批处理、减少锁竞争 |

**不同场景下的瓶颈迁移**：

| 场景特征 | TTFT 占比 | TPOT 占比 | 瓶颈判断 |
|----------|-----------|-----------|----------|
| 短 prompt (< 256 tokens) | 30% | 50% | TPOT 主导 |
| 长 prompt (> 2K tokens) | 70% | 20% | TTFT 主导 |
| 长输出 (> 512 tokens) | 15% | 75% | TPOT 主导 |
| 高并发 (> 32 concurrent) | 变化大 | 变化大 | 调度开销上升 |
| 显存接近上限 | 可能增加 | 可能增加 | 碎片化导致抖动 |

**实战诊断方法**：

```bash
# 1. 监控 TTFT vs TPOT 分布
# vLLM 默认暴露以下指标：
# - vLLM:prompt_tokens_total
# - vLLM:generation_tokens_total
# - vLLM:request_latency_seconds (包含 TTFT + generation)

# 2. 判断瓶颈类型
# 如果 TTFT 占比 > 60% → 优先优化 prefill
# 如果 TPOT 占比 > 60% → 优先优化 decode/内存

# 3. 使用 nsight systems 做深度分析
nsys profile -o inference_trace ./run_inference.py
# 查看 GPU 时间线，确认是计算还是内存等待
```

**优化收益预估表**：

| 优化手段 | 预期 TTFT 改善 | 预期 TPOT 改善 | 适用场景 |
|----------|---------------|---------------|----------|
| FP8 量化 | +20% | +30% | 显存紧张、带宽瓶颈 |
| PagedAttention | +5% | +15% | 长序列、多请求 |
| Chunked Prefill | +30% | -5% | 长 prompt 场景 |
| 连续批处理 | +10% | +20% | 高并发场景 |
| PD 分离 | +40% | +25% | 混合负载、对延迟敏感 |

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
    [True,  True,  True,  True],  # Token 0 可以 attend to 0,1,2,3
    [False, True,  True,  True],  # Token 1 可以 attend to 1,2,3
    [False, False, True,  True],  # Token 2 可以 attend to 2,3
    [False, False, False, True],  # Token 3 只能 attend to 3
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

### 5.4.4 计算复杂度降低: 从 O(n²) 到 O(n)

无 KV Cache:
```
每个 token: O(n²)
总复杂度: O(n³)
```

有 KV Cache:
```
第 1 个 token (Prefill): O(n²)
第 2 个 token (Decode): O(n)  (只计算新 token)
第 3 个 token (Decode): O(n)
...
第 n 个 token (Decode): O(n)

总复杂度: O(n²) + (n-1) × O(n) = O(n²)
平均复杂度: O(n)
```

加速效果: 序列越长,加速越明显

```
序列长度 n = 10:
- 无 KV Cache: 10³ = 1000 次运算
- 有 KV Cache: 10² + 9×10 = 190 次运算
- 加速比: 1000/190 = 5.26x

序列长度 n = 100:
- 无 KV Cache: 100³ = 1,000,000 次运算
- 有 KV Cache: 100² + 99×100 = 19,900 次运算
- 加速比: 1,000,000/19,900 = 50.25x

序列长度 n = 1000:
- 加速比: ~500x!
```

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
- 内存减少: 32x!
- 代价: 模型质量可能下降
```

Grouped-Query Attention (GQA) - LLaMA-2使用:

```
折中方案:
- 32 heads分成8组
- 每组(4个heads)共享一组K、V
- 内存减少: 4x
- 质量: 接近MHA
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
| GQA | H/G组 | 减少G倍 | 接近MHA | 平衡性能与质量 |
| MQA | 1组 | 减少H倍 | 可能下降 | 内存极度受限 |
| MLA | 1组latent | 最小 | 有竞争力 | 超大模型 |

---

## 5.5 KV Cache的内存管理挑战

>  为什么重要: 理解内存碎片化问题,才能理解PagedAttention的设计动机。

### 5.5.1 内存碎片化: 隐形的性能杀手

场景: 在A100 40GB上运行LLaMA-2-13B

```
内存分配:
- 模型权重: ~26 GB (固定)
- KV Cache可用: ~12 GB

单个请求的KV Cache (FP16):
- 每token: 0.78 MB
- 2048 token窗口: 1.56 GB
- 理论并发: 12 / 1.56 = 7个请求
```

问题: 实际只能运行2-3个请求!

原因: 内存碎片化浪费了60-80%的KV Cache内存

### 5.5.2 内部碎片化 (Internal Fragmentation)

定义: 已分配但未使用的内存

原因: 预分配策略

```
传统做法:
- 为每个请求预分配最大可能需要的内存
- 例如: 预分配2048 token的空间

问题:
- 短请求: 实际生成100 tokens就结束
- 浪费: 1948个token的空间被占用但未使用
- 其他请求无法使用这块内存
```

示例:

```
Request A: 预分配2048 slots, 实际使用100
Request B: 预分配2048 slots, 实际使用200
Request C: 预分配2048 slots, 实际使用300

已分配内存: 3 × 2048 = 6144 slots
实际使用: 100 + 200 + 300 = 600 slots
浪费: (6144 - 600) / 6144 = 90%!
```

### 5.5.3 外部碎片化 (External Fragmentation)

定义: 内存总量足够,但无法分配连续的大块

原因: Buddy Allocator等内存分配器的行为

示例:

```
初始状态: 128 MB连续内存

步骤1: 分配32 MB给Request A
  [0-31: A][32-127: 空闲]

步骤2: 分配16 MB给Request B
  [0-31: A][32-47: B][48-127: 空闲]

步骤3: 分配8 MB给Request C
  [0-31: A][32-47: B][48-55: C][56-127: 空闲]

步骤4: Request A完成,释放32 MB
  [0-31: 空闲][32-47: B][48-55: C][56-127: 空闲]

问题: 虽然有40 + 72 = 112 MB空闲,
      但无法分配64 MB的连续块!
      → 外部碎片化
```

可视化:

```
内存布局:
┌────────┬──────┬─────┬──────────┐
│  32MB  │ 16MB │ 8MB │   72MB   │
│ 空闲   │  B   │  C  │  空闲    │
└────────┴──────┴─────┴──────────┘
   ↓      ↓      ↓       ↓
  可用   可用   可用    可用

  总空闲: 112 MB
  但最大连续块: 72 MB

  需要分配: 64 MB
  结果:  可以

  需要分配: 80 MB
  结果:  失败! (虽然有112 MB空闲)
```

### 5.5.4 传统解决方案的困境

静态分配:
-  简单
-  大量内部碎片

动态分配:
-  减少内部碎片
-  严重外部碎片
-  分配器开销大

结论: 需要新的内存管理策略!

---

## 5.6 操作系统类比: 虚拟内存与分页

>  核心洞察：PagedAttention的设计思想直接借鉴了操作系统的虚拟内存机制。

### 5.6.1 操作系统面临的内存管理问题

场景: 运行总大小超过物理内存的程序

```
程序需求: 1 GB
物理内存: 512 MB

传统做法: 无法运行

虚拟内存解决方案:
- 将程序分成固定大小的页 (pages)
- 将物理内存分成帧 (frames)
- 只将需要的页保持在内存中
- 其他页存储在磁盘上
```

### 5.6.2 虚拟内存的核心概念

页 (Page): 虚拟地址空间中的固定大小块

页帧 (Frame): 物理内存中的固定大小块

页表 (Page Table): 记录页到帧的映射

MMU (Memory Management Unit): 硬件单元,负责地址翻译

示例:

```
虚拟地址空间 (64 KB):
  页0: [0-4 KB]
  页1: [4-8 KB]
  页2: [8-12 KB]
  ...
  页15: [60-64 KB]

物理内存 (32 KB):
  帧0: [0-4 KB]
  帧1: [4-8 KB]
  帧2: [8-12 KB]
  ...
  帧7: [28-32 KB]

页表映射:
  页0 → 帧2 (在内存中)
  页1 → 帧5 (在内存中)
  页2 → 缺失 (在磁盘上)
  页3 → 帧0 (在内存中)
  ...
```

### 5.6.3 地址翻译流程

```
程序访问: 虚拟地址 10000
↓
MMU计算: 页号 = 10000 / 4096 = 2, 偏移 = 10000 % 4096 = 1808
↓
查页表: 页2 → 缺失 (在磁盘)
↓
触发缺页中断 (Page Fault)
↓
操作系统:
  1. 找一个空闲帧 (或驱逐一个旧页)
  2. 从磁盘加载页2到该帧
  3. 更新页表: 页2 → 帧3
  4. 重启指令
↓
MMU重新翻译: 页2 → 帧3
↓
物理地址: 帧3起始 + 偏移 = 12288 + 1808 = 14096
↓
访问内存:
```

### 5.6.4 虚拟内存的优势

解决外部碎片化:
- 所有分配都是固定大小的页
- 任何空闲帧都可以满足任何页请求

解决内部碎片化:
- 只分配实际需要的页
- 最后一页可能有少量浪费,但最多一个页

透明性:
- 程序不需要知道实际的物理内存大小
- 程序认为自己拥有连续的大地址空间

灵活性:
- 可以运行比物理内存大的程序
- 可以动态加载/卸载代码段

### 5.6.5 从操作系统到LLM推理

类比映射:

| 操作系统概念 | LLM推理对应 |
|-------------|-----------|
| 虚拟地址空间 | 请求的逻辑KV Cache |
| 物理内存 | GPU的KV Cache存储 |
| 页 (Page) | KV Block |
| 页帧 (Frame) | Physical KV Block |
| 页表 (Page Table) | Block Table |
| MMU | PagedAttention Kernel |
| 缺页中断 | Block分配请求 |

关键洞察:
- 操作系统通过分页解决了内存碎片化问题
- LLM推理面临的KV Cache碎片化问题与OS类似
- 可以借鉴OS的分页机制!

---

## 5.7 Chunked Prefill: 处理长 prompt

> 这一节只建立问题直觉和基本流程。更完整的调度策略放到第7章展开。

### 5.7.1 为什么会需要它

当 prompt 很长时,问题往往不是模型参数放不下,而是一次性 prefill 需要的中间激活和 KV 增长超过了当前显存预算。

典型场景:

```text
一个请求携带超长上下文
→ 一次性 prefill 太大
→ TTFT 变高,甚至直接 OOM
```

### 5.7.2 它在做什么

Chunked Prefill 的核心不是新算法,而是一种更保守的执行方式:

- 不一次吞下整个 prompt
- 把 prefill 拆成若干段
- 每处理完一段,就把已经得到的 KV 保留下来
- 下一段在已有 KV 的基础上继续推进

如果只记一句话,就是:

```text
把“大 prompt 的一次性峰值显存压力”换成“多轮可控的增量处理”。
```

### 5.7.3 在系统里它意味着什么

Chunk size 不是越大越好,而是要和三件事一起看:

- 显存预算: 太大容易 OOM
- TTFT: 太小会让 prefill 被切得过碎
- 调度公平性: 超长请求不能长期霸占整轮资源

所以到这里你只需要建立直觉:

- 它解决的是"长 prompt 怎么进系统"
- 它通常和调度器一起工作,而不是单独存在
- 第7章会继续讲它如何和 Continuous Batching、token budget 结合

---

## 5.8 PagedAttention入门: 借鉴OS的内存管理

>  核心洞察：将KV Cache分成固定大小的blocks,就像OS将内存分成pages一样。
>
> 这一节聚焦为什么需要分页式 KV 管理。更具体的实现与工程权衡放到第6章继续展开。

### 5.8.1 传统连续分配为什么麻烦

如果把每个请求的 KV 都当成一整块连续内存来管理,很快就会遇到三个问题:

- 请求提前结束时,预留空间被浪费
- 请求继续变长时,需要搬迁或重新分配大块空间
- 多请求并发时,连续大块越来越难找

也就是说,麻烦不在"KV Cache 有没有",而在"KV Cache 怎么放"。

### 5.8.2 PagedAttention 的基本直觉

vLLM 的关键想法是:

- 不把 KV 当成一整段连续空间
- 改成固定大小的 blocks
- 用一张映射表记录"逻辑顺序"和"物理位置"

可以把它类比成操作系统里的分页:

```text
逻辑上的连续序列
≠
物理上的连续存储
```

这个变化本身不会改变 Attention 的数学结果,改变的是内存管理方式。

### 5.8.3 你现在只需要记住的收益

分页式 KV 管理最重要的收益有三类:

- 更少的碎片化: 显存利用率更稳定
- 更容易复用: 相同前缀更容易共享 blocks
- 更利于调度: 多请求并发时,分配和回收更灵活

对应地,复杂度也被转移到了系统实现里:

- 需要额外的 block table
- 需要更复杂的分配与回收策略
- kernel 不能再假设 KV 一定连续

这些实现细节正是第6章的重点。

---

## 5.9 批处理的挑战: 从静态到动态

### 5.9.1 静态批处理为什么会失效

批处理的初心没有错: 把多个请求一起送进 GPU,提高吞吐。

问题在于,在线流量几乎从来都不是整齐的:

- prompt 长度不同
- 输出长度不同
- 新请求会不断到达
- 有些请求已经进入 decode,有些请求还在 prefill

一旦长度不一致,传统 batch 就会用 padding 把所有请求补到同一尺寸。

### 5.9.2 真正被浪费的是什么

padding 不是"显示上补几个空位"这么简单,而是会把本来不需要的计算也一起带进去。

例如三个请求长度分别是 2、5、10:

```text
静态 batch 需要按 10 对齐
→ 实际 token: 2 + 5 + 10 = 17
→ 计算 token: 10 + 10 + 10 = 30
→ 有效占比: 17 / 30 ≈ 57%
```

这还只是 prefill 阶段的简单例子。到了在线 decode 阶段,问题通常更严重,因为:

- 老请求每轮只新增 1 个 token
- 新请求可能一下子带来很长的 prompt

这时如果还坚持传统 batch,就会出现"少量真实 token + 大量对齐填充"的局面。

### 5.9.3 这一节想把你带到哪里

所以第5章在这里要完成的认知转折只有一个:

```text
在线推理的问题,不只是“能不能 batch”,
而是“怎样 batch 才不会浪费掉绝大多数计算”。
```

接下来 Continuous Batching 就是在回答这个问题,而更完整的调度设计会在第7章展开。

---

## 5.10 Continuous Batching 入门 
>  核心洞察：去掉 batch 维度,用 attention mask 控制 token 交互,让 GPU 时刻满载。
>
> 本节只解释核心直觉。更系统的请求调度、负载取舍与线上指标放到第7章。

### 5.10.1 它到底改变了什么

Continuous Batching 的关键变化是:

- 不再把每个请求硬塞进一个固定矩形 batch
- 改成把当前活跃的工作拼接起来
- 用 attention mask 保证不同请求彼此隔离

所以它真正去掉的不是"批处理",而是"必须按同一长度对齐"这件事。

### 5.10.2 为什么它适合在线推理

在线推理的典型状态是:

- 有些请求刚进入系统,还在 prefill
- 有些请求已经在 decode,每轮只新增 1 个 token
- 有些请求刚好结束,可以被新请求替换

Continuous Batching 适合这种场景,因为它允许调度器在每一轮都重新组织工作集:

- 完成的请求移出
- 等待中的请求补进
- 长 prompt 可以按 chunk 逐步进入

这也是为什么它本质上属于"调度问题",而不是单纯的 tensor 形状技巧。

### 5.10.3 用一句话区分三种思路

| 方法 | 你该记住的核心特征 |
|------|--------------------|
| Static Batching | 简单,但长度不齐时 padding 浪费大 |
| Dynamic Batching | 能动态凑 batch,但仍容易被 padding 拖累 |
| Continuous Batching | 让调度器按轮重组活跃请求,尽量只计算真正有意义的 token |

如果只看第5章,你只需要接受这个结论:

```text
Continuous Batching 不是“又一种 batch 技巧”,
而是在线推理系统从静态批处理走向迭代级调度的入口。
```

更具体的工作流程、调度器实现、PD 分离与参数调优,统一放到第7章。

---

## 5.11 vLLM 架构预览
**参考链接（可选）**：[Berkeley EECS-2025-192 - Deconstructing vLLM](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf)

**核心价值**：用一张模块地图把前面出现的概念串起来,为后续章节铺垫架构知识。

**为什么重要**：
- 从"会用 vLLM"到"理解 vLLM"的关键转变
- 调试问题、性能优化、扩展开发的基础
- 为第6章 (KV Cache)、第7章 (调度)、第10章 (部署) 铺垫

本节只回答一个问题: 当一个请求进入 vLLM 后,哪些模块分别负责接口、内存、调度和执行? 具体实现细节不要在这里展开,否则会和后续章节重复。

### 5.11.1 vLLM 的三层架构

可以先把 vLLM 看成三层:

```
接口层 → 引擎/策略层 → 运行时层
```

| 层级 | 代表模块 | 主要职责 | 对应后续章节 |
|------|----------|----------|--------------|
| 接口层 | OpenAI-Compatible Server、API Server | 接收请求、参数校验、协议兼容 | 第10章 |
| 引擎/策略层 | LLMEngine、Scheduler、BlockManager | 请求编排、调度决策、KV 资源管理 | 第6章、第7章 |
| 运行时层 | Worker、CacheEngine、GPU kernels | 执行 forward、维护 cache、调用底层 kernel | 第6章、第11章 |

读这张表时要抓住一点: vLLM 的性能并不是某一个 kernel 单独决定的,而是接口、调度、内存管理和运行时共同作用的结果。

---

### 5.11.2 一个请求如何穿过这三层

可以把请求路径简化成下面这条线:

```text
用户请求
  → API 层接收和校验
  → LLMEngine 创建请求对象
  → Scheduler 决定本轮谁先执行
  → BlockManager / CacheEngine 准备 KV 资源
  → Worker 在 GPU 上执行 prefill 或 decode
  → 结果返回给用户
```

这条链路里最值得记住的是两个分工:

- `Scheduler` 负责"这轮该跑谁、跑多少"
- `BlockManager / CacheEngine` 负责"KV 放哪、怎么复用、什么时候释放"

前者主要决定吞吐、排队和尾延迟,后者主要决定显存效率、上下文承载能力和 OOM 风险。

---

### 5.11.3 与后续章节的衔接

从这里开始,后面几章其实是在把这张地图拆开看:

- 第6章回答: KV Cache 为什么难管,`PagedAttention / Prefix Caching` 如何提升显存效率与复用能力
- 第7章回答: 调度器如何在吞吐、延迟、公平性之间做取舍
- 第10章回答: 当这些模块进入生产环境后,如何部署、观测、扩缩容和回滚
- 第11章回答: 当系统继续演化到 Agent、异构硬件和底层优化时,运行时会遇到什么新问题

如果你能把一个性能问题先归类到"接口层 / 调度层 / KV 层 / 运行时层",后面的大部分章节就会更容易读。

---

## ✅ 章节检查清单

完成本章后,你应该能够:

- [ ] 解释 LLM 生成文本的两个阶段 (Prefill 和 Decode)
- [ ] 理解 Attention 机制的计算过程和复杂度
- [ ] 说明 KV Cache 如何将复杂度从 O(n²) 降到 O(n)
- [ ] 计算 KV Cache 的显存占用
- [ ] 解释 Chunked Prefill 的原理和应用场景
- [ ] 对比 Static Batching、Dynamic Batching 和 Continuous Batching
- [ ] 描述 vLLM 的三层架构
- [ ] 将性能问题初步归类到接口层、调度层、KV层或运行时层

---

## 📚 动手练习

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
2. 画出 Continuous Batching 的 attention mask
3. 计算 GPU 利用率 (有效 tokens / 总 tokens)

练习 5.3: 用模块地图分析一个请求流程

启动 vLLM server,发送一个请求,并观察日志:
1. 将日志粗分到接口层、调度/KV层、运行时层
2. 记录请求从进入到返回的时间
3. 标出最可能影响 TTFT 的两个模块

---

## 🎯 总结

关键要点：
- 训练vs推理: 训练是计算密集型,推理是内存带宽密集型
- LLM 推理分为 Prefill (计算密集) 和 Decode (带宽密集) 两个阶段
- Attention 是唯一让 token 交互的操作,复杂度为 O(n²)
- KV Cache 通过缓存历史 token 的 K、V,将复杂度降到 O(n)
- 内存碎片化: 传统KV Cache管理的60-80%内存浪费问题
- PagedAttention 借鉴操作系统虚拟内存技术,将 KV Cache 分块管理
- PagedAttention 通过按需分配 blocks 解决内部碎片
- PagedAttention 通过固定大小的 blocks 重用缓解外部碎片
- PagedAttention 支持相同前缀请求共享物理 blocks
- Chunked Prefill 允许处理超长 prompt,避免显存溢出
- Continuous Batching 通过去除 padding 和动态调度,大幅提升 GPU 利用率
- vLLM 的三层架构 (接口层、引擎/策略层、运行时层) 提供了清晰的抽象

## 章节衔接

这一章的目标是把“问题空间”搭出来,而不是把所有优化都讲完。下一章自然会先进入第6章,因为一旦你理解了 prefill/decode 和 Attention 的代价,最先需要精细化管理的就是 KV 这类长期占用显存的资产; 再往后第7章才会在这个基础上讨论调度器如何消费这些 KV 状态来做系统级决策。

---

## 📎 参考资料
- [Paged Attention from First Principles: A View Inside vLLM](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vLLM/) by Hamza Elshafie
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM原始论文
