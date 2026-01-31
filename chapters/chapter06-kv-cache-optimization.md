# 第6章: KV Cache优化

> **💰 成本影响** (基于行业数据)
> - **显存节省**: KV Cache优化可减少显存占用50-70%
> - **吞吐提升**: 在同样硬件上可服务2-3倍更多用户
> - **成本节省**: 典型场景从$0.002/token降到$0.001/token

## 简介

在第5章中,我们学习了 KV Cache 的基本原理——通过缓存历史 token 的 Key 和 Value 向量,避免重复计算,将复杂度从 O(n²) 降到 O(n)。但传统的 KV Cache 实现有一个致命问题:**内存碎片化**,导致 GPU 显存利用率只有 60-70%。

vLLM 的核心创新——**PagedAttention**,借鉴操作系统的虚拟内存机制,将显存利用率提升到 90-95%。这是 vLLM 论文被引用 2000+ 次的根本原因,也是 Prefix Caching、Continuous Batching 等高级特性的基础。

本章将深入讲解:
- 传统 KV Cache 的问题和局限性
- PagedAttention 的设计思想和实现细节
- Block allocation 和 eviction 策略
- Prefix Caching 的原理和性能提升
- 多种 KV Cache 优化技术 (GQA、量化、共享等)

**学完本章,你将理解为什么 vLLM 比其他推理框架快 3-5 倍。**

---

## 6.1 Transformer 回顾

### 6.1.1 注意力机制原理

**Attention 的本质**: 让每个 token 能够"看到"并"聚合"其他 token 的信息

```python
# 简化的 Attention 计算
def attention(Query, Key, Value):
    # 1. 计算相似度
    scores = Query @ Key.T  # [seq_len, seq_len]

    # 2. 归一化
    attn_weights = softmax(scores / sqrt(d_k))

    # 3. 加权求和
    output = attn_weights @ Value  # [seq_len, d_v]

    return output
```

**物理意义**:
```
Token "bank":
Query: "我是名词,我想找与金融相关的上下文"
Key:   "我是名词,我可以被金融相关的查询找到"
Value: "我的具体语义内容是'银行'"

Attention("bank", "The money in the ___ was stolen"):
→ "bank" 的 Query 与 "money" 的 Key 匹配度高
→ "bank" 聚合了 "money" 的语义
→ 理解为"银行"而不是"河岸"
```

---

### 6.1.2 K、V、Q 是什么

**三个投影矩阵**: Wq、Wk、Wv

```
输入: x (每个 token 的表示)

Query  (Q): x @ Wq  → "我想找什么?"
Key    (K): x @ Wk  → "我能提供什么?"
Value  (V): x @ Wv  → "我的实际内容"

例如:
Token: "apple"
Q: "我是水果,我想找与食物相关的上下文"
K: "我是水果,我可以被与食物相关的查询找到"
V: "我的语义是'苹果'"
```

**多头注意力** (Multi-Head Attention):
```
每个 head 学习不同的关系模式:
- Head 1: 语法关系 (主谓宾)
- Head 2: 语义关系 (同义词、反义词)
- Head 3: 指代关系 (he → John)
- Head 4: 时态关系 (is → 现在)
...

最终输出: 拼接所有 heads,再经过线性变换
```

---

### 6.1.3 为什么需要缓存

**生成过程的重复计算**:

```
步骤 1: "The capital of France is"
→ 计算 Token 0-6 的 K、V
→ 生成 "Paris"

步骤 2: "The capital of France is Paris"
→ 重新计算 Token 0-7 的 K、V ❌
→ Token 0-6 的 K、V 重复计算了!

步骤 3: "The capital of France is Paris and"
→ 重新计算 Token 0-8 的 K、V ❌❌
→ Token 0-7 的 K、V 又重复计算了!
```

**核心洞察**:
- 旧 token 的 K、V 在每次生成步骤中不变
- 只有新 token 的 K、V 是新的
- 缓存旧 K、V,只计算新 K、V → 大幅减少计算

**缓存的好处**:
```
无 KV Cache:
- 每个步骤: O(n²)
- 总复杂度: O(n³)

有 KV Cache:
- 第一个步骤: O(n²)
- 后续步骤: O(n)
- 总复杂度: O(n²)
```

---

## 6.2 KV Cache 原理

### 6.2.1 生成过程的重复计算问题

**问题场景**: 生成第 n+1 个 token

**朴素做法**:
```
已生成: "Hello, my name is John" (7 tokens)
目标:   生成第 8 个 token

朴素方法:
1. 将所有 8 个 tokens 重新输入模型
2. 重新计算所有 token 的 K 和 V
3. 只使用最后一个 token 的输出
```

**复杂度分析**:
```
生成第 1 个 token:  O(1²)
生成第 2 个 token:  O(2²)
生成第 3 个 token:  O(3²)
...
生成第 n 个 token:  O(n²)

总复杂度: O(1² + 2² + ... + n²) = O(n³)
```

**可视化浪费**:
```
第 2 步: 重新计算 Token 1 的 K、V ❌ 浪费
第 3 步: 重新计算 Token 1, 2 的 K、V ❌ 浪费
...
第 n 步: 重新计算 Token 1, 2, ..., n-1 的 K、V ❌ 浪费
```

---

### 6.2.2 KV Cache 的核心思想

**核心洞察**: 旧 token 的 K、V 已经计算过,缓存起来!

**做法**:
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

**效果**: 避免重复计算
- ✅ 每个 token 的 K、V 只计算一次
- ✅ 大幅减少计算量

**代价**: 显存占用 O(n)
- ❌ 需要存储所有历史 token 的 K、V
- ❌ 序列越长,显存占用越大

---

### 6.2.3 如何减少计算量

**无 KV Cache 的计算**:
```python
for step in range(num_tokens):
    # 每次重新计算所有 tokens
    all_tokens = tokens[:step+1]
    K, V = model.compute_kv(all_tokens)  # O((step+1)²)
    output = model.generate(K, V)
```

**有 KV Cache 的计算**:
```python
# Prefill: 计算第一个 token
K, V = model.compute_kv(tokens[:1])  # O(1²)
cache = {'K': K, 'V': V}

for step in range(1, num_tokens):
    # 只计算新 token
    new_K, new_V = model.compute_kv(tokens[step:step+1])  # O(1)
    # 复用缓存
    K = torch.cat([cache['K'], new_K], dim=1)
    V = torch.cat([cache['V'], new_V], dim=1)
    cache = {'K': K, 'V': V}
    output = model.generate(K, V)
```

**加速比**:
```
序列长度 n = 1000:
- 无 KV Cache: ~1,000,000,000 次运算
- 有 KV Cache: ~1,000,000 + 999 × 1,000 = ~2,000,000 次运算
- 加速比: ~500x
```

---

### 6.2.4 图解 KV Cache 工作流程

**时间线可视化**:

```
步骤 1 (Prefill):
输入: "The capital of"
Tokens: [t0,   t1,   t2]
计算:  [K0,V0 K1,V1 K2,V2]
缓存:  ✓✓✓
输出:  "France"

步骤 2 (Decode):
输入: "The capital of France"
Tokens: [t0,   t1,   t2,   t3]
复用:  [K0,V0 K1,V1 K2,V2] ✓✓✓ 从缓存读取
计算:  [K3,V3]           ✓✓ 新计算
缓存:  [K0,V0 K1,V1 K2,V2 K3,V3]
输出:  " is"

步骤 3 (Decode):
输入: "The capital of France is"
Tokens: [t0,   t1,   t2,   t3,   t4]
复用:  [K0,V0 K1,V1 K2,V2 K3,V3] ✓✓✓✓ 从缓存读取
计算:  [K4,V4]                   ✓✓ 新计算
缓存:  [K0,V0 K1,V1 K2,V2 K3,V3 K4,V4]
输出:  "Paris"
```

**显存占用增长**:
```
每个 token 的 KV Cache:
= 2 × num_layers × num_heads × head_dim × bytes

Llama-2-7B:
= 2 × 32 × 32 × 128 × 2 bytes
= 524,288 bytes
≈ 0.5 MB/token

1000 tokens:
= 1000 × 0.5 MB
= 500 MB

10000 tokens:
= 10000 × 0.5 MB
= 5000 MB = 5 GB
```

---

## 6.3 KV Cache 实现

### 6.3.1 朴素实现方式

**连续内存分配**:
```python
class NaiveKVCache:
    def __init__(self, max_batch_size, max_seq_len, hidden_dim):
        # 预分配连续内存
        self.cache_k = torch.zeros(
            max_batch_size,
            num_layers,
            num_heads,
            max_seq_len,
            head_dim,
            dtype=torch.float16,
            device='cuda'
        )
        self.cache_v = torch.zeros(
            max_batch_size,
            num_layers,
            num_heads,
            max_seq_len,
            head_dim,
            dtype=torch.float16,
            device='cuda'
        )

    def append(self, batch_idx, layer_idx, new_k, new_v):
        # 追加新 token 的 K、V
        seq_len = self.seq_lens[batch_idx]
        self.cache_k[batch_idx, layer_idx, :, seq_len:seq_len+1, :] = new_k
        self.cache_v[batch_idx, layer_idx, :, seq_len:seq_len+1, :] = new_v
        self.seq_lens[batch_idx] += 1
```

**问题**:
```
1. 必须预先知道 max_batch_size 和 max_seq_len
   - 实际场景: 无法预测
   - 保守估计: 浪费大量内存
   - 激进估计: 可能溢出

2. 内存碎片化
   - Request A: [████████] 1000 tokens → 分配 1000
   - Request B: [████] 500 tokens → 分配 500
   - Request A 完成 → 释放 1000
   - Request C 需要 800 → 无法使用 Request A 的空间!
     (Request B 占据了中间位置)

3. GPU 利用率低
   - 内存碎片化导致大量空间无法使用
   - 实际利用率只有 60-70%
```

---

### 6.3.2 PagedAttention 原理 ⚡️ (vLLM 的核心)

> **💡 深度来源**: [Berkeley EECS-2025-192](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf)
>
> **核心洞察**: PagedAttention 借鉴操作系统的虚拟内存机制,将 KV Cache 分成固定大小的 pages,实现高效的内存管理。
>
> **为什么重要**:
> - vLLM 最核心的创新 (论文引用 2000+)
> - 内存利用率从 60-70% 提升到 90-95%
> - Prefix Caching 的底层基础

---

#### 6.3.2.1 传统 KV Cache 的问题

**连续内存分配的缺陷**:
```
Request 1: [████████] 1000 tokens → 连续分配 1000 token 空间
Request 2: [████] 500 tokens → 连续分配 500 token 空间
Request 1 完成 → 释放 1000 tokens
Request 3 需要 800 tokens → 无法使用 Request 1 的空间 (碎片化!)
```

**内存碎片化**:

- **External fragmentation**: 请求之间的小空隙无法利用
  ```
  GPU Memory: [Req1: 1000][空隙: 200][Req2: 500][空隙: 300]
  Request 3 需要 800 tokens → 失败! (空隙不够大)
  ```

- **Internal fragmentation**: 预分配的固定大小可能浪费
  ```
  预分配 2048 tokens → 实际使用 1000 tokens → 浪费 1048 tokens
  ```

**静态内存分配的问题**:
- 必须预先知道最大 batch size 和最大序列长度
- 无法动态调整内存使用
- GPU 利用率低 (大量内存浪费)

---

#### 6.3.2.2 PagedAttention 的设计思想

**灵感来源: OS 虚拟内存**
```
OS Virtual Memory:  Pages (4KB) + Page Table
vLLM KV Cache:      Blocks (16 tokens) + Block Table
```

**核心概念**:
- **Logical blocks**: 逻辑上的连续序列 (用户视角)
- **Physical blocks**: GPU 内存中的实际块 (系统视角)
- **Block table**: 映射关系 (logical → physical)

**工作原理**:
```
Request: [token1-16][token17-32][token33-48][...]
Logical:  Block 0      Block 1       Block 2
Physical: Block 15     Block 7       Block 23
         (分散在物理内存中,但逻辑上连续)
```

**关键优势**:
- 不需要连续内存
- Physical blocks 可以分散在 GPU 内存任意位置
- 逻辑上连续,物理上分散

**类比**:
```
传统 KV Cache:
→ 像连续的数组
→ 必须找到足够大的连续空间
→ 容易碎片化

PagedAttention:
→ 像链表
→ 每个节点 (block) 独立分配
→ 通过指针 (block table) 连接
→ 内存利用率高
```

---

#### 6.3.2.3 Block Allocation 策略

**预分配策略**:
```python
# vLLM 的启动时分配
def allocate_at_startup():
    # 计算可用 GPU 内存
    gpu_memory = get_gpu_memory()
    # 预分配 90% 给 KV Cache (保留 10% 给模型 weights)
    num_blocks = (gpu_memory * 0.9) / BLOCK_SIZE
    # 创建 block pool
    block_pool = BlockPool(num_blocks)
    return block_pool
```

**动态分配算法**:
```python
def allocate_blocks(request, num_tokens):
    num_blocks = ceil(num_tokens / BLOCK_SIZE)  # 16 tokens/block
    for i in range(num_blocks):
        block = find_free_block()
        if block is None:
            # 内存不足,触发 eviction
            trigger_eviction_policy()
            block = find_free_block()
        request.blocks.append(block)
    return request.blocks
```

**Block 的大小选择**:
- 默认: 16 tokens/block
- 为什么是 16?
  - 太小 (如 8): block table 太大,管理开销高
  - 太大 (如 32): internal fragmentation 严重
  - 16 是经验最优值 (平衡开销和浪费)

**内存利用率对比**:
```
传统方法:
Request 1: 1000 tokens → 分配 1000 (连续)
Request 2: 500 tokens → 分配 500 (连续)
Request 3: 800 tokens → 需要连续 800 → 失败!
内存利用率: (1000 + 500) / 2048 = 73%

PagedAttention:
Request 1: 1000 tokens → 63 blocks
Request 2: 500 tokens → 32 blocks
Request 3: 800 tokens → 50 blocks (分散使用碎片空间)
内存利用率: (1000 + 500 + 800) / 2048 = 91%
```

---

#### 6.3.2.4 Block Eviction 策略

**LRU (Least Recently Used)**:
```python
class LRU_Eviction:
    def __init__(self):
        self.access_time = {}  # block_id → timestamp

    def evict(self, num_blocks):
        # 按访问时间排序
        sorted_blocks = sorted(
            self.access_time.items(),
            key=lambda x: x[1]  # 按时间升序
        )
        # 驱逐最久未使用的 blocks
        return [block[0] for block in sorted_blocks[:num_blocks]]
```
- 适用场景: 大多数请求具有时间局部性
- 优势: 简单,有效
- 劣势: 不考虑访问频率

**LFU (Least Frequently Used)**:
```python
class LFU_Eviction:
    def __init__(self):
        self.access_count = {}  # block_id → count

    def evict(self, num_blocks):
        # 按访问频率排序
        sorted_blocks = sorted(
            self.access_count.items(),
            key=lambda x: x[1]  # 按频率升序
        )
        # 驱逐访问频率最低的 blocks
        return [block[0] for block in sorted_blocks[:num_blocks]]
```
- 适用场景: 某些 prefix 被频繁复用 (如系统提示词)
- 优势: 保留热点数据
- 劣势: 冷启动时效果差

**vLLM 的混合策略**:
```python
class HybridEviction:
    def evict(self, num_blocks):
        # Prefix cache blocks: 使用 LFU
        # (系统提示词等,被频繁复用)
        prefix_blocks = self.get_prefix_blocks()
        prefix_evict = lfu_evict(prefix_blocks, num_blocks // 2)

        # Decode blocks: 使用 LRU
        # (新生成的 tokens,时间局部性)
        decode_blocks = self.get_decode_blocks()
        decode_evict = lru_evict(decode_blocks, num_blocks // 2)

        return prefix_evict + decode_evict
```
- 优势: 兼顾 cache hit rate 和内存效率
- 结果: 优于单一策略

---

#### 6.3.2.5 Memory Manager 实现

**CacheEngine 的核心职责**:
```python
class CacheEngine:
    def __init__(self, block_size, num_gpu_blocks):
        self.block_size = block_size  # 16 tokens
        self.num_gpu_blocks = num_gpu_blocks
        self.free_blocks = set(range(num_gpu_blocks))
        self.block_table = {}  # {request_id: [block_ids]}
        self.hash_table = {}  # {block_hash: block_id}  # For prefix caching

    def allocate(self, request_id, num_blocks):
        """分配 blocks 给请求"""
        if len(self.free_blocks) < num_blocks:
            raise OutOfMemory(f"Need {num_blocks}, "
                            f"only {len(self.free_blocks)} free")
        blocks = list(self.free_blocks)[:num_blocks]
        self.free_blocks.difference_update(blocks)
        self.block_table[request_id] = blocks
        return blocks

    def free(self, request_id):
        """释放请求的 blocks"""
        blocks = self.block_table.pop(request_id)
        self.free_blocks.update(blocks)

    def get_block_hash(self, block_id):
        """计算 block 的 hash (用于 prefix caching)"""
        block_data = self.get_block_data(block_id)
        # 使用 SHA256 或自定义快速 hash
        return hash(block_data.tobytes())

    def check_prefix_cache(self, request_id, block_hashes):
        """检查 prefix cache hit"""
        cached_blocks = []
        for h in block_hashes:
            if h in self.hash_table:
                cached_blocks.append(self.hash_table[h])
            else:
                break  # 第一个 miss,后续无法使用
        return cached_blocks
```

---

#### 6.3.2.6 PagedAttention vs 传统方案对比

| 维度 | 连续内存 | PagedAttention |
|------|---------|----------------|
| **内存利用率** | 60-70% | 90-95% |
| **碎片化** | 严重 | 轻微 |
| **Prefix Caching** | 困难 | 容易 (hash-based) |
| **实现复杂度** | 简单 | 中等 |
| **性能开销** | 无 | 轻微 (block table lookup) |
| **适用场景** | 单请求、短序列 | 多请求、长序列、生产环境 |

**性能开销分析**:
- Block table lookup: O(1) hash table
- 额外内存: block_table (每个请求 ~1KB)
- 相比收益 (+30% 内存利用率),开销可忽略

---

#### 6.3.2.7 真实案例分析

**案例 1: ChatGPT 风格对话**
```
系统提示词: 500 tokens ("You are a helpful assistant...")
用户输入: 50 tokens
模型输出: 100 tokens

传统方法:
  - 每个请求需要 650 tokens 连续空间
  - 系统提示词每次重新计算
  - 内存利用率: ~65%

PagedAttention + Prefix Caching:
  - 系统提示词: 32 blocks (cached)
  - 100 个请求共享这 32 个 blocks
  - 每个请求只需要: 用户输入 4 blocks + 输出 7 blocks
  - 内存利用率: ~92%
```

**案例 2: 长文档摘要**
```
文档长度: 10,000 tokens
摘要长度: 200 tokens
并发数: 10 个请求

传统方法:
  - 需要 10 × 10,200 = 102,000 tokens 连续空间
  - 24GB GPU 只能处理 ~2 个并发请求
  - 内存利用率: ~60%

PagedAttention:
  - 动态分配 blocks
  - 可以处理 ~5 个并发请求
  - 内存利用率: ~88%
  - 吞吐量提升: 2.5x
```

---

#### 6.3.2.8 实战配置

**启动 vLLM 时启用 PagedAttention** (默认启用):
```bash
vllm serve meta-llama/Llama-2-7b-hf \
  --block-size 16 \              # Block 大小 (默认: 16)
  --gpu-memory-utilization 0.9 \  # GPU 内存利用率
  --max-num-batched-tokens 8192  # 最大 batch tokens
```

**监控 block 使用情况**:
```python
from vllm import LLM

llm = LLM(model="meta-llama/Llama-2-7b-hf")

# 获取 block allocator 统计
stats = llm.llm_engine.cache_engine.get_stats()
print(f"Free blocks: {stats['num_free_blocks']}")
print(f"Used blocks: {stats['num_used_blocks']}")
print(f"GPU utilization: {stats['gpu_utilization']:.2%}")
```

---

#### 6.3.2.9 性能监控

**关键指标**:
```python
# Block 使用率
block_utilization = used_blocks / total_blocks

# Cache hit rate (Prefix Caching)
cache_hit_rate = cache_hits / total_requests

# Memory fragmentation
fragmentation = 1 - (largest_free_block / total_free_blocks)
```

**告警阈值**:
```
Block 利用率 > 95% → 考虑增加 GPU 或减小 batch size
Cache hit rate < 30% → Prefix Caching 效果不佳
Fragmentation > 20% → 可能需要调整 block size
```

---

#### 6.3.2.10 总结: PagedAttention 的核心价值

**关键成就**:
1. ✅ 内存利用率从 60-70% 提升到 90-95%
2. ✅ 支持 Prefix Caching (相同 prompt 只计算一次)
3. ✅ 动态内存分配 (不需要预知序列长度)
4. ✅ 支持更高的并发和更长的序列

**适用场景**:
- ✅ 多租户 SaaS (大量并发请求)
- ✅ 长序列生成 (文档摘要、长对话)
- ✅ 共享 prompt (系统提示词、RAG 场景)

**权衡**:
- ⚠️ 轻微的内存开销 (block table)
- ⚠️ 实现复杂度增加
- ✅ 但收益远大于成本

---

## 6.4 KV Cache 优化技术

### 6.4.1 Multi-Query Attention vs Multi-Head Attention

**Multi-Head Attention (MHA)**:
```
每个 head 有独立的 K、V:
Head 0: K0, V0
Head 1: K1, V1
Head 2: K2, V2
...
Head 31: K31, V31

KV Cache 大小:
= 32 heads × 2 × seq_len × head_dim × bytes
```

**Multi-Query Attention (MQA)**:
```
所有 heads 共享 K、V:
Heads 0-31: 共享 K, V

KV Cache 大小:
= 1 × 2 × seq_len × head_dim × bytes
= MHA 的 1/32!
```

**对比**:

| 维度 | MHA | MQA |
|------|-----|-----|
| **KV Cache 大小** | 大 (32x) | 小 (1x) |
| **模型质量** | 高 | 略低 |
| **推理速度** | 慢 | 快 |
| **适用场景** | 追求质量 | 追求速度 |

---

### 6.4.2 Grouped-Query Attention (GQA)

**折中方案**:
```
32 个 heads,分成 8 组,每组共享 K、V:
Group 0 (Heads 0-3): 共享 K0, V0
Group 1 (Heads 4-7): 共享 K1, V1
Group 2 (Heads 8-11): 共享 K2, V2
...
Group 7 (Heads 28-31): 共享 K7, V7

KV Cache 大小:
= 8 groups × 2 × seq_len × head_dim × bytes
= MHA 的 1/4, MQA 的 8x
```

**为什么 GQA 是最佳选择**:
- ✅ 接近 MHA 的质量
- ✅ 接近 MQA 的速度
- ✅ Llama-3、Mistral 等现代模型采用

---

### 6.4.3 Shared KV Cache

**跨请求共享**:
```
Request A: "System: You are assistant. User: Hello"
Request B: "System: You are assistant. User: Hi"

共享部分: "System: You are assistant."
→ 只计算一次 KV Cache
→ Request A 和 B 都复用
```

**实现**:
```python
class SharedKVCache:
    def __init__(self):
        self.global_cache = {}  # {token_seq_hash: (K, V)}

    def get_or_compute(self, tokens):
        hash = compute_hash(tokens)
        if hash in self.global_cache:
            return self.global_cache[hash]
        else:
            K, V = model.compute_kv(tokens)
            self.global_cache[hash] = (K, V)
            return K, V
```

---

### 6.4.4 量化 KV Cache

**FP16 → INT8**:
```
FP16: 每个元素 2 bytes
INT8: 每个元素 1 byte

KV Cache 大小减半!
```

**量化方法**:
```python
def quantize_kv(kv_cache):
    # FP16 → INT8
    scale = kv_cache.abs().max() / 127
    kv_int8 = (kv_cache / scale).round().char()
    return kv_int8, scale

def dequantize_kv(kv_int8, scale):
    # INT8 → FP16
    return kv_int8.float() * scale
```

**质量影响**:
- ✅ 大多数场景下精度损失可接受
- ⚠️ 复杂推理任务可能受影响
- 💡 建议: 先实验,再决定是否使用

---

## 6.5 KV Cache 的代价

### 6.5.1 显存占用分析

**Llama-2-7B 的 KV Cache**:
```
单层单头的 KV Cache:
= 2 × seq_len × head_dim × bytes
= 2 × 4096 × 128 × 2
= 2,097,152 bytes
≈ 2 MB

所有层所有头:
= 2 MB × 32 layers × 32 heads
= 2,048 MB
≈ 2 GB (seq_len = 4096)
```

**不同模型的 KV Cache 大小**:

| 模型 | 层数 | Heads | Head Dim | 4K tokens | 8K tokens | 16K tokens |
|------|------|-------|----------|----------|----------|-----------|
| Llama-2-7B | 32 | 32 | 128 | 0.5 GB | 1 GB | 2 GB |
| Llama-2-13B | 40 | 40 | 128 | 0.8 GB | 1.6 GB | 3.2 GB |
| Llama-2-70B | 80 | 64 | 128 | 2 GB | 4 GB | 8 GB |

**并发请求的显存需求**:
```
10 个并发请求,每个 4K tokens:
- Llama-2-7B: 10 × 0.5 GB = 5 GB
- Llama-2-70B: 10 × 2 GB = 20 GB

加上模型权重:
- Llama-2-7B (13 GB) + KV (5 GB) = 18 GB → 单张 A100 (40GB) ✅
- Llama-2-70B (140 GB) → 需要模型并行
```

---

### 6.5.2 序列长度限制

**GPU 显存限制序列长度**:
```
A100 40GB:
- 模型权重: 13 GB (Llama-2-7B FP16)
- 剩余: 27 GB
- KV Cache: 27 GB / 0.5 GB per 4K = 216K tokens
- 单请求: 216K tokens
- 10 个并发: 21.6K tokens each
```

**实际考虑**:
- ⚠️ Prefill 阶段需要临时显存
- ⚠️ Decode 阶段需要额外显存
- ⚠️ 留一些 buffer 避免OOM

---

### 6.5.3 权衡: 计算vs显存

**有 KV Cache**:
```
✅ 优点:
  - 大幅减少计算
  - 降低延迟

❌ 缺点:
  - 占用大量显存
  - 限制并发数和序列长度
```

**无 KV Cache**:
```
✅ 优点:
  - 节省显存
  - 支持更长序列

❌ 缺点:
  - 计算量大
  - 延迟高
```

**最佳实践**:
```
短序列 (< 2K tokens): 不需要 KV Cache
中等序列 (2K-8K): KV Cache
长序列 (> 8K): KV Cache + 量化 + 分块处理
```

---

## 6.6 实战对比

### 6.6.1 无 KV Cache vs 有 KV Cache

**性能测试**:
```
模型: Llama-2-7B
硬件: A100 40GB
序列长度: 1024 tokens

无 KV Cache:
- TTFT: 200ms
- TBT: 50ms/token
- 总时间 (生成 100 tokens): 200ms + 100 × 50ms = 5.2s

有 KV Cache:
- TTFT: 200ms (相同)
- TBT: 5ms/token (10x 快!)
- 总时间 (生成 100 tokens): 200ms + 100 × 5ms = 0.7s

加速比: 5.2s / 0.7s = 7.4x
```

---

### 6.6.2 性能提升量化分析

**不同序列长度的加速比**:
```
序列长度 = 100:
  无 KV Cache: ~100ms
  有 KV Cache: ~100ms
  加速比: 1x (太短,没优势)

序列长度 = 1000:
  无 KV Cache: ~10s
  有 KV Cache: ~1.5s
  加速比: 6.7x

序列长度 = 10000:
  无 KV Cache: ~1000s
  有 KV Cache: ~8s
  加速比: 125x!
```

**结论**: 序列越长,KV Cache 的优势越明显

---

### 6.6.3 vLLM 的 KV Cache 实现

**关键特性**:
1. ✅ PagedAttention (高内存利用率)
2. ✅ Prefix Caching (跨请求复用)
3. ✅ 自动 eviction (LRU/LFU)
4. ✅ 动态 block 分配

**使用示例**:
```python
from vllm import LLM, SamplingParams

# vLLM 自动启用 PagedAttention
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Prefix Caching 自动生效
prompts = [
    "System: You are assistant. User: Hello",
    "System: You are assistant. User: Hi",
]
# 第二个请求会复用第一个的 system prompt KV Cache

outputs = llm.generate(prompts)
```

---

## 6.7 Prefix Caching ⭐⭐⭐

> **💡 核心洞察**: 重复的 prompt (如系统提示词) 只需要计算一次,后续请求直接复用 KV Cache。
>
> **🎯 性能提升**: ChatGPT 风格对话场景可提升 2-5 倍吞吐量。
>
> **来源**: vLLM 核心特性之一,已在生产环境大规模验证。

### 6.7.1 什么是 Prefix Caching

**定义**: 跨请求复用相同 prompt 的 KV Cache

**核心问题**: 重复 prompt 的计算浪费
```
Request 1: "System: You are helpful. User: What is AI?"
Request 2: "System: You are helpful. User: Tell me a joke"
Request 3: "System: You are helpful. User: How are you?"

问题: "System: You are helpful." 计算了 3 次! ❌
```

**典型场景**:
- 系统提示词 ("You are a helpful assistant...")
- 多轮对话的上下文
- RAG 场景的固定知识 prefix

**为什么叫"Prefix"**:
- Cache 的是 prompt 部分 (即序列的 prefix)
- 生成的部分 (decode 阶段) 因人而异,无法复用

---

### 6.7.2 Prefix Caching 的核心思想

**传统 KV Cache**: 单次请求内复用
- Token 0 的 KV 被 token 1, 2, 3...复用
- 但请求结束后,Cache 被清空

**Prefix Caching**: 跨请求复用
- 请求 1: 计算完整 prompt 的 KV → Cache
- 请求 2: 检测到相同 prefix → 直接复用 → 跳过计算
- 请求 3、4、5...: 同请求 2

**类比**:
```
传统 Cache: 函数内的 memoization
Prefix Caching: 全局 distributed cache (如 Redis)
```

---

### 6.7.3 vLLM 的实现: Hash-based KV Cache

**挑战**: 如何检测两个请求的 prefix 是否相同?

**方案 1: 字符串比较** (Naive)
- 每次比较 prompt 文本
- 问题: 慢!而且语义相同的 token 可能来自不同文本

**方案 2: vLLM 的 Hash-based 方法** ⭐
- 对每个 Block 的 KV Cache 计算 Hash
- Hash 相同的 Block 被认为内容相同

**Hash 算法**:
```python
def compute_block_hash(block_kv):
    """
    输入: Block 的 KV tensor
    输出: 固定长度的 hash 值
    实现: SHA256 或自定义快速 hash
    """
    # 方法 1: SHA256 (准确但慢)
    import hashlib
    return hashlib.sha256(block_kv.tobytes()).hexdigest()

    # 方法 2: 快速 hash (vLLM 可能使用的)
    # 简单的 XOR 或 rolling hash
    return fast_hash(block_kv)
```

**Cache Hit 检测流程**:
1. 新请求到来
2. 计算 prompt tokens 对应的 logical blocks
3. 查询 hash table: 是否已有这些 blocks 的 KV?
4. 如果 hit: 直接引用已有 physical blocks
5. 如果 miss: 分配新的 physical blocks 并计算

---

### 6.7.4 Prefix Caching 的工作流程

**首次请求 (Cold Path)**:
```
1. 用户发送 prompt (含系统提示词)
2. vLLM 计算所有 tokens 的 KV Cache
3. 将 KV Cache 分成 blocks,计算每个 block 的 hash
4. 存储到 cache engine (hash table)
5. 返回结果
```

**后续请求 (Warm Path)**:
```
1. 用户发送相同系统提示词的新请求
2. vLLM 计算 blocks 的 hash
3. **Cache Hit!**: 发现已有对应的 KV Cache
4. 直接引用已有 blocks,跳过 prefill 计算
5. 只需计算用户输入的新 tokens
6. 返回结果 (快得多!)
```

**部分 Hit 场景**:
```
系统提示词: hit ✅
用户输入: miss ❌

→ 复用系统提示词的 KV
→ 只计算用户输入部分
→ 仍然有加速效果
```

---

### 6.7.5 性能提升分析

**理论加速比**:
```
假设:
- 系统提示词长度 = P tokens
- 用户输入长度 = U tokens

无 Prefix Caching:
  每次计算 P + U

有 Prefix Caching:
  首次: P + U
  后续: U

加速比 ≈ (P + U) / U = 1 + P/U
```

**实际案例**:
```
场景 1: 系统提示词 200 tokens,用户输入 50 tokens
  加速比 = (200 + 50) / 50 = **5 倍**

场景 2: 系统提示词 1000 tokens (RAG 场景),用户输入 20 tokens
  加速比 = (1000 + 20) / 20 = **51 倍** (极端 case)

场景 3: 无系统提示词
  加速比 = 1x (无效果)
```

**内存开销**:
- Hash table 存储: 每个 block ~32 bytes hash
- KV Cache 存储: 原本就需要,不算额外开销

**考虑因素**:
- Cache expiration 时间
- Memory pressure
- 至少保留 system prompt 的 breakpoint

---

### 6.7.6 vLLM 配置

**启用 Prefix Caching** (v0.6.0+):
```bash
vllm serve meta-llama/Llama-2-7b-hf \
  --enable-prefix-caching \
  --max-num-seqs 128
```

**监控 Cache Hit Rate**:
```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_prefix_caching=True
)

# 生成多个请求
for prompt in prompts:
    llm.generate(prompt)

# 获取 cache 统计
stats = llm.llm_engine.cache_engine.get_prefix_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Total tokens cached: {stats['total_tokens']}")
print(f"Tokens served from cache: {stats['cached_tokens']}")
```

---

### 6.7.7 实战案例

**案例 1: Chatbot 服务**
```
场景:
- 系统提示词: 500 tokens
- 用户输入: 平均 50 tokens
- 每分钟 1000 个请求

无 Prefix Caching:
  - 每个请求计算 550 tokens
  - 总计算: 1000 × 550 = 550K tokens/分钟

有 Prefix Caching:
  - 首个请求: 550 tokens
  - 后续 999 个请求: 每个 50 tokens
  - 总计算: 550 + 999 × 50 = 50.5K tokens/分钟
  - 加速比: 550K / 50.5K = **10.9x**
```

**案例 2: RAG 应用**
```
场景:
- 知识库 prefix: 2000 tokens
- 用户问题: 平均 30 tokens
- 每分钟 500 个请求

无 Prefix Caching:
  - 总计算: 500 × 2030 = 1,015K tokens/分钟

有 Prefix Caching:
  - 总计算: 2030 + 499 × 30 = 17K tokens/分钟
  - 加速比: 1,015K / 17K = **59.7x**
```

---

### 6.7.8 最佳实践

**1. 识别可缓存的 Prefix**
```
✅ 适合缓存:
  - 系统提示词
  - 固定的知识库内容
  - 多轮对话的历史
  - 共享的上下文

❌ 不适合缓存:
  - 完全随机的输入
  - 每次都不同的用户查询
```

**2. 合理设置 Cache 大小**
```
太小:
  - 频繁 eviction
  - Cache hit rate 低

太大:
  - 占用过多显存
  - 影响并发能力

建议:
  - 根据实际 hit rate 调整
  - 目标: 70-90% hit rate
```

**3. 监控和调优**
```python
# 定期检查 cache 效果
def monitor_prefix_cache(llm):
    stats = llm.get_cache_stats()

    if stats['hit_rate'] < 0.5:
        print("⚠️  Cache hit rate 低于 50%,考虑优化:")
        print("  1. 增加系统提示词长度")
        print("  2. 检查是否有可共享的 prefix")
        print("  3. 调整 cache size")

    print(f"Cache hit rate: {stats['hit_rate']:.2%}")
    print(f"Memory usage: {stats['memory_used']}/{stats['memory_total']}")
```

---

## ✅ 章节检查清单

完成本章后,你应该能够:

- [ ] 解释传统 KV Cache 的内存碎片化问题
- [ ] 描述 PagedAttention 的设计思想 (借鉴 OS 虚拟内存)
- [ ] 对比 Logical blocks 和 Physical blocks
- [ ] 理解 Block allocation 和 eviction 策略
- [ ] 计算 KV Cache 的显存占用
- [ ] 对比 MHA、MQA、GQA 的 KV Cache 大小
- [ ] 解释 Prefix Caching 的工作原理
- [ ] 计算 Prefix Caching 的加速比
- [ ] 配置 vLLM 启用 Prefix Caching
- [ ] 监控 cache hit rate 并优化

---

## 📚 动手练习

**练习 6.1**: 计算 KV Cache 显存占用

Llama-2-7B 的配置:
- 层数: 32
- Attention heads: 32
- Head dimension: 128
- 数据类型: FP16 (2 bytes)
- Block size: 16 tokens

问题:
1. 单个 block 的 KV cache 大小是多少?
2. 100 个 blocks 需要多少显存?
3. 如果启用 Prefix Caching,缓存 1000 个 blocks,总显存占用是多少?

**练习 6.2**: 对比 PagedAttention 和传统方法

假设有以下请求序列:
```
Request A: 1000 tokens → 完成
Request B: 500 tokens → 进行中
Request C: 需要 800 tokens → 新请求
GPU 总显存: 2048 tokens
```

任务:
1. 传统方法能否处理 Request C? 为什么?
2. PagedAttention 如何处理这个场景?
3. 计算两种方法的内存利用率

**练习 6.3**: Prefix Caching 加速比计算

场景:
- 系统提示词: 800 tokens
- 用户输入: 平均 40 tokens
- 每小时 10,000 个请求

任务:
1. 计算无 Prefix Caching 的总计算量
2. 计算有 Prefix Caching 的总计算量
3. 计算加速比
4. 如果系统提示词增加到 2000 tokens,加速比是多少?

---

## 🎯 总结

**关键要点**:
- 传统 KV Cache 遭受内存碎片化,利用率只有 60-70%
- PagedAttention 借鉴 OS 虚拟内存,将利用率提升到 90-95%
- Block allocation 和 eviction 策略是 PagedAttention 的核心
- Prefix Caching 通过跨请求复用,可带来 2-50x 加速
- GQA 是质量与速度的最佳平衡
- vLLM 自动启用 PagedAttention 和 Prefix Caching

**下一章**: 第7章 请求调度策略——理解 vLLM 如何高效调度多个请求。

---

**有问题?加入 [第6章 Discord 频道](https://discord.gg/TODO) 讨论!**
