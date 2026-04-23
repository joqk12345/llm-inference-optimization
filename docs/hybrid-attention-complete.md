# 第2章 新增内容：混合注意力机制

## 2.1.6 混合注意力机制：Attention + 状态空间模型的崛起

当序列长度持续增长时，纯 Attention 的二次复杂度成为刚性瓶颈，混合模型（Hybrid Models）正在从研究走向生产。

### 背景

Attention 面临两个根本性限制：

（1）**KV Cache 线性增长**：每个生成的 token 都会追加新的 Key 和 Value 向量，当序列扩展到数十万 token 时，KV Cache 成为 GPU 显存的最大消耗者。

（2）**Prefill 阶段二次复杂度**：对于 128k token 以上的长 prompt，首 token 时间（Time to First Token，TTFT）急剧增加，有时甚至导致推理不可用。

这些限制并非理论担忧，而是来自真实工作负载的刚性需求：

- **RAG 场景**：用户查询搭配多个检索文档，prompt 长度轻松从几千 token 膨胀到数万甚至数十万。
- **Agent 循环**：模型生成输出、调用工具、获取结果、再将结果融入上下文——每轮循环都扩展序列长度。
- **推理 trace**：要求模型"逐步思考"或生成中间推理过程，会把所有中间 token 留在上下文里。

### 状态空间模型的演进

State Space Models（SSM）有着悠久的控制理论与动力学系统历史，近年来被重新引入 LLM 领域。表 2-3 给出了 SSM 的演进历程。

**表 2-3 状态空间模型演进历程**

| 阶段 | 代表模型 | 核心创新 | 局限性 |
|------|----------|----------|--------|
| 2021 | S4 | 固定维度隐状态，线性复杂度 | 选择性复制、上下文推理能力弱 |
| 2023 | Mamba-1 | 允许 A/B/C 矩阵随时间变化 | GPU 利用率低，无法充分利用 Tensor Core |
| 2024 | Mamba-2 | 矩阵形式化，可高效实现 | 理论上与线性注意力等价 |

关键洞察来自 Mamba-2 论文：SSM 实际上可以形式化为从输入序列到输出序列的矩阵变换，并且与**线性注意力**（Linear Attention）具有等价性。

线性注意力由 Katharopoulos 等人在 2020 年提出，其核心思想是用核特征映射的线性点积来近似 softmax，从而打破二次复杂度瓶颈。近年来两个值得关注的变体是：

- **Lightning Attention**：被 Minimax-Text-01 采用
- **Gated Delta Net**：被 Qwen3-Next 采用

### 混合模型格局

混合模型的核心思路很清晰——保留 Attention 的建模能力，同时插入 Mamba 或线性注意力层来提升长序列效率。

vLLM V1 目前已支持多种混合模型（详见第 6 章），包括 NVIDIA Nemotron、Gemma 3、Llama 4 等。这些模型表明：混合模型不再是实验性小众，而是跨厂商的活跃设计选择。

### vLLM 的状态管理演进

- **V0**：KV Cache 和 Mamba 状态分开管理，用户需要手动配置 `max_num_seqs`，设置过高会导致显存溢出（OOM），过低则会降低并发能力。
- **V1**：统一分配器管理 KV Cache 和 Mamba 状态，支持前缀缓存、KV 传输、Prefill/Decode 分离，并可通过 `torch.compile` 和调度优化获得性能提升。

### 决策：什么时候你需要关注混合模型

（1）你的业务涉及长上下文（RAG、Agent、推理 trace），且纯 Attention 已经触及显存或延迟瓶颈。

（2）你愿意接受新架构的学习曲线，换取长序列场景的效率优势。

（3）你需要在新一代模型上部署，而这些模型本身采用混合架构。

### 指标口径

- 显存：KV Cache 与 Mamba 状态在长序列下的占比对比
- 延迟：TTFT 在不同序列长度下的变化曲线
- 吞吐：有效 tokens/s 在混合层与纯 Attention 层的变化

> **一句话总结**：混合注意力不是"更先进的 Attention"，而是针对长序列场景的架构演进——当你的瓶颈从"算不动"变成"存不起"和"等不起"时，它是一个值得认真评估的方向。

---

# 第6章 新增内容：混合模型的状态管理

## 6.6 混合模型的状态管理

> **核心洞察**：当 Attention 与 Mamba/线性注意力混合部署时，状态管理不再只是 KV Cache 的问题，而是两种不同状态机制的统一协调问题。

前面几节我们讨论了纯 Attention 架构下的 KV Cache 管理。但当你使用混合模型（Hybrid Models）时，系统需要同时管理两类状态：

- **Attention 层**：Paged KV Cache（分页管理的 Key-Value 缓存）
- **Mamba/线性注意力层**：固定大小的隐状态（Hidden State）

理解这两种状态的差异，是掌握混合模型推理的基础。

### 6.6.1 Attention 与 Mamba 的状态对比

**Attention 层状态（KV Cache）**：

特征：

- 状态大小随序列长度线性增长
- 每个新 token 追加新的 KV 向量
- 以固定大小 blocks 组织（默认 16 tokens/block，每块约 64 KiB）
- 支持分页管理和页面对齐

示例（基于 NVIDIA Nemotron-Nano-12B-v2）：

- 1 个 block：16 tokens，约 64 KiB
- 128K 序列：8,192 blocks

**Mamba 层状态**：

特征：

- 固定大小的隐状态，与序列长度无关
- 每个时间步原地更新（in-place update）
- 状态大小由模型结构决定（通常约 2.57 MiB/序列）

示例（基于 NVIDIA Nemotron-Nano-12B-v2）：

- 每层 Mamba 状态：约 2.57 MiB
- 总状态大小与序列长度无关

### 6.6.2 长上下文下的状态大小对比

关键洞察：**在长序列场景下，两类状态的大小关系会发生戏剧性变化**。表 6-4 给出了详细的对比数据。

**表 6-4 长序列下 Attention 与 Mamba 状态大小对比**

| 序列长度 | KV Cache 大小 | Mamba 状态大小 | KV/Mamba 比率 |
|----------|--------------|----------------|---------------|
| 1K tokens | 约 0.5 GB | 2.57 MB | 约 195 倍 |
| 4K tokens | 约 2 GB | 2.57 MB | 约 780 倍 |
| 16K tokens | 约 8 GB | 2.57 MB | 约 3100 倍 |
| 128K tokens | 约 64 GB | 2.57 MB | 约 25000 倍 |

> **注**：估算基于 NVIDIA Nemotron-Nano-12B-v2 配置，实际数值因模型而异。

**核心结论**：

（1）短序列：两者状态大小相近，Mamba 状态可能略大。

（2）**长序列（> 4K）**：KV Cache 远大于 Mamba 状态，128K 时可达 **200 倍以上**。

（3）这就是混合架构的吸引力所在：Mamba 用固定成本覆盖长距离依赖，Attention 只处理关键局部上下文。

### 6.6.3 vLLM V0 的混合状态管理

vLLM V0 对混合模型的支持是通过一个实用的变通方案实现的。

架构特点：

- KV Cache：使用 Block 分配器管理（与纯 Attention 相同）
- Mamba 状态：为每个活跃序列单独分配 tensor
- 用户配置：手动设置 `max_num_seqs` 参数，控制最大并发 Mamba 状态数

**V0 的问题**：

问题 1：用户需要猜测合适的 `max_num_seqs`

- 设置过高 → CUDA 显存溢出（OOM）
- 设置过低 → 并发受限，吞吐下降

问题 2：状态分配与 KV Cache 分配独立

- 无法共享内存预算
- 资源利用率不佳

问题 3：缺乏高级特性

- 不支持前缀缓存（Prefix Caching）
- 不支持 KV 传输
- 不支持 Prefill/Decode 分离

### 6.6.4 vLLM V1 的统一状态管理

vLLM V1 重构了混合模型支持，实现了**统一分配器**（Unified Allocator）。

架构特点：

- 同时管理 KV Cache blocks 和 Mamba 状态 pages
- 统一的内存池，自动协调分配
- 支持前缀缓存、KV 传输、Prefill/Decode 分离
- 可利用 `torch.compile` 优化

**KVCacheGroups 机制**：V1 将相同类型的层分组管理。

示例模型结构：

- Attention 层（A）：32 层
- Sliding Window Attention 层（SWA）：8 层
- Mamba 层（M）：12 层

KVCacheGroups：

- Group A：32 层共享
- Group SWA：8 层共享
- Group M：12 层共享

**页面对齐策略**：Mamba 的页面大小远大于 Attention blocks，为了统一管理：

（1）**Attention block 自动扩大**：直到与 Mamba page size 对齐。

（2）**Mamba page 轻微填充**：使页面大小完全相等。

（3）**共享底层 KVCacheTensors**：不同 group 共用物理存储。

对齐示例（NVIDIA Nemotron-Nano-12B-v2）：

- 原始 Attention block size：16 tokens
- 对齐后 Attention block size：64 tokens（或更大）
- Mamba page size：已是较大固定值
- 结果：所有 group 使用统一的页面大小

### 6.6.5 混合模型实战配置

**在 vLLM 中启用混合模型**：

```bash
vLLM serve NVIDIA/Nemotron-Nano-12B-v2 \
  --gpu-memory-utilization 0.9 \
  --enforce-eager  # 某些混合模型需要 eager 模式
```

**监控混合状态使用**：

```python
from vLLM import LLM

llm = LLM(model="NVIDIA/Nemotron-Nano-12B-v2")

# 获取混合状态统计
stats = llm.llm_engine.cache_engine.get_stats()
print(f"KV Cache blocks: {stats['num_used_blocks']}")
print(f"Mamba states: {stats.get('num_mamba_states', 'N/A')}")
```

### 6.6.6 混合模型决策清单

> **使用场景**：你的业务是否适合混合模型？

**表 6-5 混合模型适用场景判断**

| 条件 | 判断方法 | 建议 |
|------|----------|------|
| 序列长度经常 > 4K | 流量分析 | 认真评估混合模型 |
| 显存瓶颈明显 | KV Cache 占比 > 70% | 混合模型可显著缓解 |
| 使用支持混合的模型 | vLLM 支持列表 | 直接部署 |
| 短序列为主 (< 1K) | 流量分析 | 纯 Attention 可能足够 |

**评估步骤**：

（1）确认你的模型是否被 vLLM V1 支持。

（2）在真实流量下测试，对比混合模型与纯 Attention 的 TTFT 和显存占用。

（3）监控 Mamba 状态与 KV Cache 的实际大小比例。

（4）评估调度和运维复杂度是否可接受。

---

# 章节检查清单（新增内容）

完成本章后，你应能够：

- [ ] 解释传统 KV Cache 的内存碎片化问题
- [ ] 描述 PagedAttention 的设计思想（借鉴 OS 虚拟内存）
- [ ] 对比 Logical blocks 和 Physical blocks
- [ ] 理解 Block allocation 和 eviction 策略
- [ ] 计算 KV Cache 的显存占用
- [ ] 对比 MHA、MQA、GQA 的 KV Cache 大小
- [ ] 解释 Prefix Caching 的工作原理
- [ ] 计算 Prefix Caching 的加速比
- [ ] 配置 vLLM 启用 Prefix Caching
- [ ] 监控 cache hit rate 并优化
- [ ] 对比 Attention 状态（KV Cache）和 Mamba 状态的差异
- [ ] 解释 vLLM V1 如何实现统一状态管理
- [ ] 评估混合模型是否适合你的业务场景

---

# 关键要点（新增内容）

- 传统 KV Cache 容易遭受内存碎片化，有效显存利用率会明显下降（常见在 60-70% 量级，依工作负载而变）
- PagedAttention 借鉴 OS 虚拟内存，通过固定大小 block 管理缓解碎片化，有效利用率通常可提升到 90% 左右（同样依工作负载而变）
- Block allocation 和 eviction 策略是 PagedAttention 的核心
- Prefix Caching 通过跨请求复用，在高重复前缀场景可带来从 2 倍到数十倍的加速
- GQA 是质量与速度的最佳平衡
- vLLM 自动启用 PagedAttention 和 Prefix Caching
- **混合模型（Attention + Mamba/线性注意力）通过固定大小隐状态解决长序列下的 KV Cache 爆炸问题**
- **vLLM V1 通过统一分配器实现 KV Cache 和 Mamba 状态的协同管理**
