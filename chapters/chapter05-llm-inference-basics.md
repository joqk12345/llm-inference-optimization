# 第5章: LLM推理基础

>  教学理念 (参考: Hugging Face "Continuous batching from first principles"、Hamza Elshafie "Paged Attention from First Principles")
>
> 核心思路: 从第一性原理出发,理解LLM推理的基本流程和优化动机。
>
> 学习路径: Attention → KV Cache → Chunked Prefill → Continuous Batching → PagedAttention

## 简介

在深入 vLLM 的复杂优化技术之前,我们需要先理解 LLM 推理的基础原理。很多工程师直接跳到高级优化技巧,却忽略了基础知识——这就像在没学会走路之前就想跑步。

本章将带你从零开始,逐步理解:
- 训练vs推理: 为什么推理是内存密集型而训练是计算密集型
- LLM 如何生成文本 (Prefill 和 Decode 阶段)
- Attention 机制的工作原理和计算复杂度
- KV Cache 如何将复杂度从 O(n²) 降到 O(n)
- 内存碎片化问题: 传统KV Cache管理的致命缺陷
- 操作系统类比: 从虚拟内存理解PagedAttention的设计思想
- Chunked Prefill 如何处理超长 prompt
- 为什么需要 Continuous Batching
- vLLM 的三层架构全景

学完本章,你将能够解释为什么 vLLM 比传统方法快 24 倍,并理解PagedAttention如何通过借鉴操作系统虚拟内存技术解决内存碎片化问题。

---

## 5.0 训练 vs 推理: 工作负载的本质差异

>  核心洞察: 理解训练和推理的工作负载差异,是理解优化策略的第一步。

### 5.0.1 训练: 计算密集型的并行工作负载

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

### 5.0.2 推理: 内存带宽密集型的串行工作负载

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

### 5.0.3 为什么优化推理更关键

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

## 5.1 LLM 如何生成文本

### 5.1.1 自回归生成的基本过程

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

### 5.1.2 Prefill 阶段: 并行处理 prompt

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
GPU RTX 4090:
- Prefill 时间: ~200ms (一次处理 100 个 tokens)
```

---

### 5.1.3 Decode 阶段: 逐 token 生成

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
- 每个 token: ~20ms
- 总时间: 100 × 20ms = 2000ms
- 是 Prefill 的 10 倍!
```

---

### 5.1.4 图解完整流程

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

## 5.2 Attention 机制详解

>  为什么重要: Attention 是唯一让不同 token 产生交互的地方。理解 Attention,就理解了 LLM 的核心。

### 5.2.1 Token 的表示: 向量与 hidden dimension

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

### 5.2.2 Query、Key、Value 投影

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

### 5.2.3 Attention 计算: QK^T 与二次复杂度

计算步骤:

```python
# 输入: Q, K, V
# 形状: [batch, num_heads, seq_len, head_dim]
Q, K, V = ...

# 步骤 1: 计算 Q @ K^T
# 相似度矩阵: 每个 token 对其他 token 的关注度
scores = Q @ K.transpose(-2, -1)  # [batch, num_heads, seq_len, seq_len]

# 步骤 2: 缩放
scores = scores / (head_dim  0.5)

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

### 5.2.4 Attention Mask: 控制 token 交互

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

### 5.2.5 Causal Mask: 因果关系的可视化

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

### 5.2.6 为什么 Attention 是唯一让 token 交互的地方

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

## 5.3 从朴素生成到 KV Cache

### 5.3.1 朴素方法: 每次重新计算 (O(n²))

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

### 5.3.2 重复计算问题的可视化

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
mask = [False, False, False, False, True]
       └────────┬────────┘          └──┬──┘
      不能 attend to 之前           只能attend to自己
```

图示:
```
最后一个 token 只关心自己的预测,
不影响其他 token 的 attention 计算!
```

---

### 5.3.3 KV Cache 的核心思想

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

### 5.3.4 计算复杂度降低: 从 O(n²) 到 O(n)

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

### 5.3.5 显存代价: 每个 token 需要多少显存?

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

### 5.3.6 不同Attention变体的内存优化

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

## 5.4 KV Cache的内存管理挑战

>  为什么重要: 理解内存碎片化问题,才能理解PagedAttention的设计动机。

### 5.4.1 内存碎片化: 隐形的性能杀手

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

### 5.4.2 内部碎片化 (Internal Fragmentation)

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

### 5.4.3 外部碎片化 (External Fragmentation)

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

### 5.4.4 传统解决方案的困境

静态分配:
-  简单
-  大量内部碎片

动态分配:
-  减少内部碎片
-  严重外部碎片
-  分配器开销大

结论: 需要新的内存管理策略!

---

## 5.5 操作系统类比: 虚拟内存与分页

>  核心洞察: PagedAttention的设计思想直接借鉴了操作系统的虚拟内存机制。

### 5.5.1 操作系统面临的内存管理问题

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

### 5.5.2 虚拟内存的核心概念

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

### 5.5.3 地址翻译流程

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
访问内存: ```

### 5.5.4 虚拟内存的优势

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

### 5.5.5 从操作系统到LLM推理

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

## 5.6 Chunked Prefill: 处理长 prompt

### 5.6. 问题: 大 prompt 超过显存

场景: Cursor 添加整个代码仓库到 prompt

```
代码仓库: 10,000 行代码
Tokens:   ~100,000 tokens

问题: 100,000 个 tokens 的激活值超过 GPU 显存!
```

约束: 每次 forward pass 最多处理 m 个 token

```
GPU: RTX 4090 24GB
显存: 24 GB
约束: 每次 ~4,096 tokens
```

---

### 5.6. 解决方案: 分块处理

思路: 将 n 个 token 的 prompt 分成 ⌈n/m⌉ 个 chunks

示例: n=7, m=4 → 分成 2 个 chunks

```
原始: [t0, t1, t2, t3, t4, t5, t6]

Chunk 1: [t0, t1, t2, t3]
Chunk 2: [t4, t5, t6]
```

关键: 如何保持信息连续性?

---

### 5.6. KV Cache 在 chunked prefill 中的作用

Chunk 1:
```
输入: [t0, t1, t2, t3]
计算:
  K0, V0
  K1, V1
  K2, V2
  K3, V3
存储: 缓存 K0-3, V0-3
```

Chunk 2:
```
输入: [t4, t5, t6]
复用: K0-3, V0-3 (从缓存读取)
计算:
  K4, V4
  K5, V5
  K6, V6
拼接: [K0, K1, K2, K3, K4, K5, K6]
      [V0, V1, V2, V3, V4, V5, V6]
```

Attention mask 调整: 确保跨 chunk 的 token 正确交互

```python
# Chunk 1 mask (4 tokens)
mask1 = [
    [True,  False, False, False],
    [True,  True,  False, False],
    [True,  True,  True,  False],
    [True,  True,  True,  True ],
]

# Chunk 2 mask (3 tokens, but can attend to chunk 1)
mask2 = [
    [True,  True,  True,  True,  True,  False, False],  # t4
    [True,  True,  True,  True,  True,  True,  False],  # t5
    [True,  True,  True,  True,  True,  True,  True ],  # t6
    └─────┬────┘ └───────────┬───────────┘
      Chunk 1         Chunk 2
```

---

### 5.6. 图解分块处理流程

无 chunked prefill:
```
输入: 100,000 tokens
问题: 显存不足! ```

有 chunked prefill:
```
Chunk 1: 处理 tokens 0-4095
  → 计算 K0-4095, V0-4095
  → 缓存到 GPU 内存

Chunk 2: 处理 tokens 4096-8191
  → 复用 K0-4095, V0-4095
  → 计算 K4096-8191, V4096-8191
  → 拼接完整 cache

Chunk 3: 处理 tokens 8192-12287
  → 复用 K0-8191, V0-8191
  → 计算 K8192-12287, V8192-12287
  → 拼接完整 cache

...重复 25 次 (100,000 / 4,096)

完成: 整个 100,000 tokens 的 KV cache
```

灵活性: 可根据内存约束动态调整 chunk 大小

```
GPU 内存充足: chunk_size = 8192
GPU 内存紧张: chunk_size = 2048
```

---

## 5.7 PagedAttention入门: 借鉴OS的内存管理

>  核心思想: 将KV Cache分成固定大小的blocks,就像OS将内存分成pages一样。

### 5.6. 传统KV Cache的问题

连续内存分配:

```
Request A的KV Cache:
  [Token 0-2047] 连续存储在GPU内存的某个区域

问题:
  1. 预分配整个2048 token的空间
  2. 如果只生成100 tokens,浪费1948个位置
  3. 如果需要超过2048 tokens,需要重新分配更大的空间
```

### 5.6. Paged KV Cache的核心设计

分块存储:

```
传统方式:
  Request A: [连续的2048个token的KV]

Paged方式:
  Request A:
    Block 0: [Token 0-15]
    Block 1: [Token 16-31]
    Block 2: [Token 32-47]
    ...
    Block 127: [Token 2032-2047]

每个Block:
  - 固定大小 (如16 tokens)
  - 可以存储在GPU内存的任意位置
  - 通过Block Table追踪
```

Block Table:

```python
# 逻辑块 → 物理块的映射
block_table = {
    "request_A": {
        "logical_block_0": "physical_block_42",
        "logical_block_1": "physical_block_17",
        "logical_block_2": "physical_block_93",
        # ...
    }
}
```

### 5.6. PagedAttention如何工作

Attention计算的挑战:

```
传统Attention:
  - 假设K、V连续存储
  - 一次性加载整个序列
  - 简单的矩阵乘法

PagedAttention:
  - K、V分散在多个blocks
  - 需要遍历block table
  - 逐block计算attention
```

PagedAttention算法:

```
输入:
  - Query: qi (当前token的query向量)
  - Block Table: [B0, B1, B2, ..., Bn]

步骤:
  1. 初始化:
     output = 0
     running_max = -∞
     running_sum = 0

  2. 遍历每个block j:
     a. 从Block Table获取物理块位置
     b. 加载该块的Kj、Vj
     c. 计算注意力分数: scores = qi ⊤ Kj / √d
     d. 更新running_max和running_sum (online softmax)
     e. 计算权重: weights = exp(scores - running_max)
     f. 累加输出: output += weights × Vj

  3. 归一化并返回output
```

关键特性:

1. 数学等价性: 结果与传统Attention完全相同
2. 内存灵活性: blocks可以分散在任意位置
3. 增量计算: 只加载需要的blocks

### 5.6. PagedAttention的优势

解决内部碎片化:

```
传统方式:
  Request A: 生成100 tokens
  → 预分配2048 slots
  → 浪费: 1948 slots (95%)

Paged方式:
  Request A: 生成100 tokens
  → 分配⌈100/16⌉ = 7个blocks
  → 最后一个block只用了4个slots
  → 浪费: 12 slots (12/112 = 11%)
```

解决外部碎片化:

```
传统方式:
  内存分散,无法分配大块

Paged方式:
  所有blocks大小相同
  任何空闲block都可以使用
  无外部碎片化
```

内存共享 (Copy-on-Write):

```
Request A: "解释量子计算"
Request B: "解释量子物理"

共享前缀: "解释量子"

Paged方式:
  - 前缀的blocks可以共享
  - 两个请求指向相同的物理blocks
  - 只在分叉点分配新blocks
  → 节省内存!
```

### 5.7.5 性能对比

vLLM论文数据:

```
传统系统 (Orca, TGI):
  - 内存浪费: 60-80%
  - 实际吞吐: 基准

vLLM (PagedAttention):
  - 内存浪费: <4%
  - 吞吐提升: 2-3x
```

为什么能提升吞吐?

```
更少的内存浪费 → 可以服务更多并发请求 → 更好的GPU利用率 → 更高吞吐
```

---

## 5.8 批处理的挑战: 从静态到动态

### 5.8.1 静态批处理

目标: 提高吞吐量 (throughput)

方法: 将多个 prompt 打包成一个 batch

```python
# 3 个 prompt
prompt1 = "Hello"
prompt2 = "Hi there, how are you doing today?"
prompt3 = "Hey"

# 问题: 长度不一致!
```

约束: 所有 prompt 必须有相同长度

解决方案: 左侧 padding,右侧对齐

```python
# Padding
prompt1 = "<pad><pad><pad><pad><pad><pad><pad><pad>Hello"
prompt2 = "Hi there, how are you doing today?"
prompt3 = "<pad><pad><pad><pad><pad><pad><pad><pad><pad>Hey"

# 统一长度: max(len(prompt1, prompt2, prompt3)) = 36
```

---

### 5.8.2 Padding 的问题: 计算浪费

Padding 位置: 左侧 (添加`<pad>` token)

Attention mask: padding 位置设为 False

```python
# Prompt 1 的 attention mask
mask1 = [
    [False, False, False, ..., True,  True],  # "H"
    [False, False, False, ..., True,  True,  True],  # "e"
    [False, False, False, ..., True,  True,  True,  True],  # "l"
    [False, False, False, ..., True,  True,  True,  True,  True],  # "l"
    [False, False, False, ..., True,  True,  True,  True,  True,  True],  # "o"
    └─────────┬────────┘ └─────────────┬─────────────┘
     padding (浪费)          实际内容
]
```

问题: padding token 占用了计算资源,但没有实际贡献

```python
# GPU 仍然计算 padding tokens 的 attention!
# 虽然结果被 mask 掉,但计算量没有减少
```

---

### 5.8.3 不同序列长度的困境

场景: batch 中有多个 prompt,长度差异大

```python
batch = [
    "Hi",          # 2 tokens
    "Hello",       # 3 tokens
    "How are you?", # 12 tokens
]

# 需要全部 padding 到 12 tokens
padded_batch = [
    "<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>Hi",
    "<pad><pad><pad><pad><pad><pad><pad><pad><pad>Hello",
    "How are you?",
]
```

问题 1: 短 prompt 完成后,长 prompt 还在生成

```
时间线:
0ms    100ms   200ms   300ms   ...   1200ms
│      │       │       │             │
Hi    Hi     Hi     ...          Hi   (完成)
       Hello  Hello  ...          Hello   (完成)
                How ...         How are you?   (完成)

问题: "Hi" 和 "Hello" 完成后,
      仍在 batch 中等待"How are you?"完成
      → GPU 计算浪费在 padding 上!
```

问题 2: 动态调度引入大量 padding

```
Decode 阶段:
- 正在 decode 的 prompt 每次只加 1 个 token
- 新加入的 prompt 需要 prefill 很多 tokens

Batch 状态:
  Request 1: 已生成 100 tokens (decode 阶段)
  Request 2-8: 已生成 50 tokens (decode 阶段)
  Request 9: 新加入,有 1000 tokens 的 prompt

问题:
  - Request 9 需要 prefill 1000 tokens
  - Request 1-8 只需要 decode 1 个 token
  - 如何组织 batch?
```

Padding 数量 = (n-1) × (B-1)

---

### 5.8.4 示例: 为什么 padding 成本随 batch 和长度二次增长

参数:
```
B = 8   (batch 中 8 个 prompt 在 decode)
n = 100 (新 prompt 有 100 个 token)
```

计算 padding:
```
新 prompt (Request 9):
- 需要 prefill 100 tokens
- 其他 7 个 request 只加 1 个 token

Padding 数量:
= (100 - 1) × (8 - 1)
= 99 × 7
= 693 个 padding tokens!

实际计算:
- Request 9: 100 tokens (实际内容)
- Request 1-8: 每个 99 个 padding tokens + 1 个实际 token
- 总计: 100 + 8×1 + 7×99 = 791 tokens
- 有效: 100 + 8 = 108 tokens
- 浪费: 791 - 108 = 683 tokens (86%!)
```

结论: 动态调度 + 传统 batching = 灾难

---

## 5.9 Continuous Batching 入门 
>  核心洞察: 去掉 batch 维度,用 attention mask 控制 token 交互,让 GPU 时刻满载。

### 5.9.1 核心思想: 去掉 batch 维度

问题根源: batch 维度引入了 padding

激进想法: 不要 batch 维度!

替代方案: 拼接所有 prompt

```python
# 传统 batching
batch = [
    [token1, token2, token3],
    [token4, token5],
]
shape: [2, 3]  # batch_size=2, seq_len=3 (padding!)

# 新方法: 拼接
sequence = [token1, token2, token3, token4, token5]
shape: [5]  # 只有 seq_len!
```

新问题: 如何防止不同 prompt 的 token 互相干扰?

---

### 5.9.2 Ragged Batching: 用 attention mask 控制交互

方法:
1. 将多个 prompt 拼接成一个序列
2. 用 attention mask 控制 token 交互
3. Prompt A 的 token 不能 attend to Prompt B 的 token

Mask 形状: 块对角矩阵 (block-diagonal)

可视化:
```
Prompt A (3 tokens): [A1, A2, A3]
Prompt B (2 tokens): [B1, B2]

拼接: [A1, A2, A3, B1, B2]

Attention Mask:
       A1   A2   A3   B1   B2
A1:   []  [ ]  [ ]  [ ]  [ ]
A2:   []  []  [ ]  [ ]  [ ]
A3:   []  []  []  [ ]  [ ]
B1:   [ ]  [ ]  [ ]  []  [ ]
B2:   [ ]  [ ]  [ ]  []  []

块对角结构:
┌─────┬─────┐
│ A A │     │
│ A A │     │
│ A A │     │
├─────┼─────┤
│     │ B B │
│     │ B B │
└─────┴─────┘
```

优势: 无 padding,所有计算都有意义

```python
# GPU 计算的每个 token 都是实际需要的!
# 没有 padding 浪费
```

---

### 5.9.3 Dynamic Scheduling: 动态替换完成的请求

场景: 某个 prompt 生成 `<eos>`

动作:
```
步骤 1: 检测到 Request A 完成
步骤 2: 立即从 batch 中移除 Request A
步骤 3: 用等待中的 Request C 替换
步骤 4: 重新构建 attention mask
```

目标: 保持 GPU 时刻满载

关键: Ragged batching 让替换成本低

```
传统 batching (需要重新 padding):
- 移除 Request A
- 重新计算最大长度
- 重新 padding 所有请求
- 成本: 高!

Continuous Batching (只需重建 mask):
- 移除 Request A 的 tokens
- 追加 Request C 的 tokens
- 重建 block-diagonal mask
- 成本: 低!
```

---

### 5.9.4 混合 Prefill 和 Decode: 最大化 throughput

挑战:
```
Decode 阶段的 prompt:
- 每次只加 1 个 token
- 快速,占用少量 GPU

新加入的 prompt:
- 需要 prefill 很多 tokens
- 慢,占用大量 GPU

如何平衡?
```

调度算法:
```
目标: 每个 batch 达到 m 个 token (memory budget)

步骤 1: 统计当前 decode 阶段的请求数
  decode_requests = 10
  decode_tokens = 10 × 1 = 10

步骤 2: 计算剩余空间
  remaining = m - decode_tokens
  remaining = 1000 - 10 = 990

步骤 3: 用 chunked prefill 加入新请求
  new_request_tokens = 100
  if remaining >= new_request_tokens:
    # 可以加入整个 request
    add_request(new_request)
  else:
    # 只 prefill 一个 chunk
    chunk_size = remaining
    add_chunk(new_request, chunk_size)
```

示例:
```
Memory budget: m = 1000

当前状态:
- 10 个 decode requests → 10 个 tokens

新请求:
- Request A: 100 tokens
- Request B: 200 tokens
- Request C: 500 tokens

调度:
- 加入 Request A: 10 + 100 = 110 tokens
- 加入 Request B: 110 + 200 = 310 tokens
- 加入 Request C (chunk 1): 310 + 500 = 810 tokens
- 剩余: 1000 - 810 = 190 tokens
- 加入 Request C (chunk 2): 810 + 190 = 1000 tokens (满!)

GPU 利用率: 1000/1000 = 100% ```

---

### 5.9.5 完整的 Continuous Batching 流程图

```
步骤 1: 初始 batch
┌─────────────────────────────────────┐
│ Request A (已生成 50 tokens)        │
│ Request B (已生成 30 tokens)        │
│ Request C (已生成 20 tokens)        │
└─────────────────────────────────────┘

步骤 2: 某个请求完成
┌─────────────────────────────────────┐
│ Request A (生成 <eos>)             │
│ Request B (已生成 31 tokens)        │
│ Request C (已生成 21 tokens)        │
└─────────────────────────────────────┘

步骤 3: 移除完成的请求,加入新请求
┌─────────────────────────────────────┐
│ Request D (新,需要 prefill 100)     │
│ Request B (已生成 32 tokens)        │
│ Request C (已生成 22 tokens)        │
└─────────────────────────────────────┘

步骤 4: Chunked prefill + Decode
┌─────────────────────────────────────┐
│ Request D (prefill chunk 1: 70)     │
│ Request B (decode: +1 token)        │
│ Request C (decode: +1 token)        │
└─────────────────────────────────────┘

步骤 5: Forward pass
→ GPU 处理 70 + 1 + 1 = 72 tokens
→ 生成新 tokens

步骤 6: 循环回到步骤 2
```

---

### 5.9.6 Continuous Batching vs 传统方法对比

Static Batching:
```
优点:
   简单,易于实现
   适合固定长度的批处理

缺点:
   大量 padding (50-90%)
   吞吐量低
   GPU 利用率低
```

Dynamic Batching:
```
优点:
   动态调整
   比静态 batching 灵活

缺点:
   padding 仍然严重
   频繁的重新 padding
   难以优化
```

Continuous Batching (vLLM):
```
优点:
   无 padding
   GPU 利用率最高 (可达 95%+)
   吞吐量提升 3-5 倍
   支持动态调度

缺点:
   实现复杂
   需要动态管理 attention mask
   CPU 开销较高
```

---

## 5.10 vLLM 架构全景 
>  来源: [Berkeley EECS-2025-192 - Deconstructing vLLM](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf)
>
> 核心价值: 系统性理解 vLLM 的三层架构——Interface、Model Authoring、Runtime,为后续章节铺垫架构知识。
>
> 为什么重要:
> - 从"会用 vLLM"到"理解 vLLM"的关键转变
> - 调试问题、性能优化、扩展开发的基础
> - 为第6章 (KV Cache)、第7章 (调度)、第10章 (部署) 铺垫

### 5.10.1 vLLM 的三层架构

Layer 1: Interfaces (用户交互层)

```
User Request → OpenAI Server → API Server → LLMEngine
```

- LLMEngine: 核心引擎
  - 作用: 协调所有组件
  - 职责: 请求管理、资源分配、结果返回
  - 接口: `generate()`, `encode()`

- API Server: HTTP 服务
  - 作用: 提供 REST API
  - 职责: 请求路由、认证、限流
  - 协议: HTTP/REST

- OpenAI-Compatible Server: 标准接口
  - 作用: 兼容 OpenAI API
  - 职责: `/v1/chat/completions` 等接口
  - 价值: 零代码迁移

---

Layer 2: Model Authoring (模型抽象层)

```
LLMEngine → ModelExecutor → BlockManager + Scheduler
```

- ModelExecutor: 模型执行器
  - 作用: 执行模型 forward pass
  - 抽象: 支持不同模型架构
  - 接口: `execute_model()`, `profile()`
  - 详见: 第10章 Model Authoring

- BlockManager: 内存块管理
  - 作用: 管理 KV Cache 的 physical blocks
  - 职责: 分配、释放、迁移 blocks
  - 抽象: Physical vs Logical blocks
  - 详见: 第6章 PagedAttention 原理

- Scheduler: 请求调度器
  - 作用: 决定哪些请求可以执行
  - 策略: FIFO、Priority、SJF
  - 输出: Scheduled requests
  - 详见: 第7章 vLLM 的调度器实现

---

Layer 3: Runtime (运行时层)

```
Scheduler → CacheEngine → Worker (GPU)
```

- CacheEngine: KV 缓存引擎
  - 作用: 管理 KV Cache 的物理存储
  - 数据结构: Block table
  - 功能: Hash-based lookup
  - 详见: 第6章 内存管理深度剖析

- Worker: 工作进程
  - 作用: 在 GPU 上执行计算
  - 职责: 模型推理、kernel 执行
  - 通信: 与主进程通信

---

### 5.10.2 用户请求的完整流程

步骤 1: 用户发送请求

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama2",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

步骤 2: OpenAI Server 接收
- 解析请求
- 验证参数
- 转发给 API Server

步骤 3: API Server 处理
- 请求路由
- 限流检查
- 调用 LLMEngine.generate()

步骤 4: LLMEngine 调度
- 创建请求对象
- 提交给 Scheduler
- 等待调度结果

步骤 5: Scheduler 决策
- 检查资源 (GPU memory、compute)
- 选择可执行的请求
- 返回 scheduled requests

步骤 6: ModelExecutor 执行
- 准备 input data
- 调用 Worker.execute_model()
- 等待 GPU 返回结果

步骤 7: Worker 在 GPU 上执行
- 加载模型 weights
- 执行 PagedAttention kernels
- 返回 generated tokens

步骤 8: 结果返回
```
Worker → ModelExecutor → LLMEngine
  ↓
LLMEngine → API Server → OpenAI Server
  ↓
OpenAI Server → 用户
```

---

### 5.10.3 架构图

```
┌─────────────────────────────────────────────────┐
│              Layer 1: Interfaces               │
├─────────────────────────────────────────────────┤
│  OpenAI Server  →  API Server  →  LLMEngine    │
│  (HTTP)            (REST)         (Core)        │
└─────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────┐
│           Layer 2: Model Authoring             │
├─────────────────────────────────────────────────┤
│  ModelExecutor  ←  Scheduler  ←  BlockManager   │
│  (Execution)      (Policy)       (Memory)       │
└─────────────────────────────────────────────────┘
                        │
┌─────────────────────────────────────────────────┐
│             Layer 3: Runtime                    │
├─────────────────────────────────────────────────┤
│  CacheEngine  →  Worker  →  GPU Kernels         │
│  (KV Cache)      (Compute)    (CUDA)            │
└─────────────────────────────────────────────────┘
```

---

### 5.10.4 与后续章节的关联

第6章 KV Cache 优化:
- BlockManager 的详细实现 (6.3.2)
- CacheEngine 的内存管理 (6.3.3)
- PagedAttention 的核心创新 (6.3.2)

第7章 请求调度策略:
- Scheduler 的调度算法 (7.4)
- Iteration-level scheduling (7.4.2)
- CPU overheads 分析 (7.4.3)

第10章 生产环境部署:
- Interface 层部署模式 (10.2-10.4)
- Model Authoring 实战 (10.6)
- 性能分析与调优 (10.5)

---

### 5.10.5 实战: 启动 vLLM 并观察架构

启动 vLLM server:

```bash
vllm serve meta-llama/Llama-2-7b-hf \
  --port 8000 \
  --host 0.0.0.0
```

查看启动过程:

```
INFO:     Started server process
INFO:     Waiting for vLLM engine to initialize
INFO:     Initializing an LLM engine with config
INFO:     Loading model weights
INFO:     GPU memory: 15.50 GB
INFO:     Model loaded
```

发送请求:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-7b-hf",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

### 5.10.6 架构理解检查点

完成本章后,你应该能够:

- [ ] 解释 vLLM 的三层架构
- [ ] 描述用户请求的完整流程 (8步骤)
- [ ] 理解 LLMEngine、ModelExecutor、Worker 的职责
- [ ] 知道 BlockManager 和 Scheduler 的作用
- [ ] 理解 PagedAttention 在架构中的位置

---

##  章节检查清单

完成本章后,你应该能够:

- [ ] 解释 LLM 生成文本的两个阶段 (Prefill 和 Decode)
- [ ] 理解 Attention 机制的计算过程和复杂度
- [ ] 说明 KV Cache 如何将复杂度从 O(n²) 降到 O(n)
- [ ] 计算 KV Cache 的显存占用
- [ ] 解释 Chunked Prefill 的原理和应用场景
- [ ] 对比 Static Batching、Dynamic Batching 和 Continuous Batching
- [ ] 描述 vLLM 的三层架构
- [ ] 追踪用户请求在 vLLM 中的完整流程

---

##  动手练习

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

练习 5.3: 追踪 vLLM 请求流程

启动 vLLM server,发送一个请求,并观察日志:
1. 识别每个层级 (Interface、Model Authoring、Runtime) 的日志
2. 记录请求从进入到返回的时间
3. 找出 Scheduler、ModelExecutor、Worker 的日志

---

##  总结

关键要点:
- 训练vs推理: 训练是计算密集型,推理是内存带宽密集型
- LLM 推理分为 Prefill (计算密集) 和 Decode (带宽密集) 两个阶段
- Attention 是唯一让 token 交互的操作,复杂度为 O(n²)
- KV Cache 通过缓存历史 token 的 K、V,将复杂度降到 O(n)
- 内存碎片化: 传统KV Cache管理的60-80%内存浪费问题
- PagedAttention: 借鉴操作系统虚拟内存技术,将KV Cache分块管理
  - 解决内部碎片: 按需分配blocks
  - 解决外部碎片: 固定大小的blocks可任意重用
  - 支持内存共享: 相同前缀的请求可共享物理blocks
- Chunked Prefill 允许处理超长 prompt,避免显存溢出
- Continuous Batching 通过去除 padding 和动态调度,大幅提升 GPU 利用率
- vLLM 的三层架构 (Interface、Model Authoring、Runtime) 提供了清晰的抽象

下一章: 第6章 KV Cache 优化——深入理解 PagedAttention 和内存管理的更多细节。

---

参考资源:
- [Paged Attention from First Principles: A View Inside vLLM](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/) by Hamza Elshafie
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) - vLLM原始论文

有问题?加入 [第5章 Discord 频道](https://discord.gg/TODO) 讨论!
