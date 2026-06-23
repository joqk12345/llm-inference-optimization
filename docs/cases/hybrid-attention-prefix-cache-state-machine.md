---
id: "docs-cases-hybrid-attention-prefix-cache-state-machine"
title: "Hybrid Attention Prefix Cache 状态机案例研究"
slug: "docs-cases-hybrid-attention-prefix-cache-state-machine"
date: "2026-06-23"
type: "case-study"
topics:
  - "case-studies"
  - "kv-cache"
  - "long-context-inference"
  - "hybrid-attention"
concepts:
  - "prefix-caching"
  - "hybrid-kv-cache"
  - "paged-attention"
  - "compressed-attention"
  - "disaggregated-serving"
tools:
  - "vllm"
  - "sglang"
architecture_layer:
  - "optimization-techniques"
  - "production-systems"
  - "frontier-and-ecosystem"
learning_stage: "advanced"
optimization_axes:
  - "memory"
  - "latency"
  - "throughput"
  - "cost"
  - "operability"
related:
  - "chapters-chapter06-kv-cache-optimization"
  - "chapters-chapter07-request-scheduling"
  - "chapters-chapter10-production-deployment"
  - "chapters-chapter11-advanced-topics"
  - "docs-cases-deepseek-v4-inference-engines"
  - "docs-cases-vllm-mooncake-store-agentic-serving"
references:
  - "https://arxiv.org/abs/2309.06180"
  - "https://arxiv.org/abs/2312.07104"
  - "https://arxiv.org/abs/2606.09079"
  - "https://arxiv.org/abs/2605.02568"
  - "https://arxiv.org/abs/2604.05887"
  - "https://arxiv.org/abs/2605.05219"
status: "draft"
display_order: 217
---
# Hybrid Attention Prefix Cache 状态机案例研究

## 先说结论

Prefix cache 命中率不是正确性指标。

PagedAttention 让 serving 系统可以把 token 历史切成 page，再用 block table 把 logical block 映射到 physical KV block。这个抽象解决了传统 KV cache 的碎片化和跨请求复用问题，也让 prefix caching 成为主流推理框架的基础能力。

但 hybrid attention 把这条规则推到了边界：当一个模型里同时存在 full attention、sliding window attention、compressed attention、sparse routing 和 on-disk prefix 时，"前缀 token 相同"不再自动推出"cache 可以直接复用"。

更准确的说法是：

**page hit 只说明查到了某段历史；safe reuse 还要求这段历史在每一种 cache 状态里都完整、对齐、可恢复。**

这就是 DeepSeek-V4、vLLM hybrid KV cache、SGLang RadixAttention / HiCache 一类系统共同暴露出来的问题：cache manager 正在从 block allocator 变成 heterogeneous state manager。

---

## 五种状态

Hybrid attention 下，serving 层不能再只维护一类 KV block。至少要区分五种状态：

| 状态 | 作用 | 复用条件 | 误判后果 |
|------|------|----------|----------|
| compressed KV | 已压缩完成的历史 KV，例如 CSA / HCA entry | 必须落在完整压缩边界上 | 半个压缩块被当成完整历史 |
| uncompressed tail | 最新 token 尚未凑齐压缩块的临时状态 | 通常不跨请求复用，只能重算或随请求推进 | prefix 末尾缺失，后续位置错位 |
| SWA window | sliding window attention 的近期窗口 | 当前窗口覆盖范围必须一致 | 命中 token 相同，但可见近邻不同 |
| indexer state | sparse routing 的 top-k / 选择索引 | 路径选择必须 bit-exact | routing 分叉，后续 attention 读错对象 |
| on-disk prefix | CPU / SSD / 远端存储里的长前缀 | 父链、block hash、压缩边界都要完整 | 查到盘上数据，但 GPU 上不可直接读 |

这五种对象的生命周期不一样。compressed KV append 后相对稳定；tail 会随着 decode 推进被压缩或丢弃；SWA window 每步滑动；indexer state 决定 sparse attention 看哪些历史块；on-disk prefix 还要经过读盘、搬运、补 tail、恢复窗口等步骤。

因此，prefix hit 的查询结果不能再是一个全局布尔值。正确接口应该更接近：

```text
full attention manager: 我能复用到 token 800
SWA manager: 我只能接受当前窗口内的 token 500
CSA manager: 按压缩边界我能复用到 token 768
indexer manager: 只有前 768 的 routing state 是完整的

最终可复用长度 = 所有 group 的交集，并向下对齐到完整块边界
```

这也是为什么 vLLM 这类系统会把 hybrid KV cache 拆成多个 cache group，再由 coordinator 求交集；SGLang 一类系统则把 prefix tree、SWA cache、HiCache storage 的命中策略拆开处理。

---

## 四层对齐

prefix hit 还要同时通过四层对齐：

1. **模型语义层**：压缩 attention 有自己的步长。例如 CSA 和 HCA 的压缩边界不同，prefix 只能在双方都形成完整 entry 的位置复用。
2. **kernel ABI 层**：attention backend 有 page size / tile size 约束。即便模型语义允许 128 token 一块，具体 kernel 也可能要求更大的 page block。
3. **多 cache group 层**：full、SWA、CSA、HCA、indexer 各自能命中的长度不同，最终只能取交集。
4. **byte 级硬件层**：RDMA、DMA buffer、RowMajor KV pool、SSD / CPU staging buffer 还会带来地址和长度对齐要求。

这里最容易写错的是把某个具体数字当成普遍规律。比如某个模型设置下 `lcm(4,128)=128` 可以解释压缩边界，但生产系统里的实际 page size 还会被 FlashAttention、MLA backend、CUDA Graph、RDMA 注册粒度继续放大。

所以更稳妥的表达是：

**kernel-friendly boundary = 模型压缩边界、backend ABI、多 cache group block size、硬件 DMA 对齐共同约束后的结果。**

---

## KV 离开显存之后

当 context 拉到 128K、256K、1M 级别，显存通常装不下所有可复用状态，cache 会进入 CPU DRAM、SSD、远端 KV store 或 RDMA 池。这时问题从"查不查得到"变成三道工程边界：

**写入边界**：哪些状态值得落盘？

- compressed KV 稳定，适合持久化。
- tail 是临时状态，通常重算更清楚。
- SWA window 写入量大、生命周期短，常见策略是在 Full SWA Caching、Periodic Checkpointing、Zero SWA Caching 之间取舍。
- indexer state 不是普通 metadata，漏掉会让 sparse routing 路径改变。

**读取边界**：读回是否在关键路径上？

从 SSD / CPU / 远端节点读回 KV，如果不能和 GPU 计算 overlap，可能比重算还慢。KVPR、CPU-GPU hybrid attention、near-storage attention 这类方向都在挑战"复用越多越好"这个默认偏好：真正目标是让 GPU idle time 最小，而不是让 cache hit rate 数字最大。

**身份边界**：cache key 是否带上了完整前文？

同一段 token 在不同前文下算出的 KV 不相同。RAG、agent、多文档拼接场景尤其危险。正确的 on-disk prefix key 不应只描述"这段 token 是什么"，还要描述"它是在什么父链和边界下算出来的"。

---

## Allocator 变成状态机

传统 allocator 的核心动作是 alloc / free：给新 token 找空 block，用完释放。Hybrid attention 之后，这个角色不够了。

一个更接近生产现实的 allocator 至少包含四层：

| 层级 | 要回答的问题 | 典型风险 |
|------|--------------|----------|
| admission | token 有没有资格进入可共享 cache | tail / partial page 被误放进 prefix cache |
| state table | 命中后按哪种状态恢复 | compressed KV、SWA、indexer、on-disk prefix 混用 |
| consistency | 查到和用到之间谁持有所有权 | TOCTOU、并发 eviction、offload race |
| kernel / compile | 这条恢复路径能不能进当前 backend | CUDA Graph 拓扑变化、page size 不满足 ABI |

还要叠加一条校验严格度：

- **路径选择必须 bit-exact**：indexer top-k、prefix block lookup、pin ownership 这类离散决策不能漂。
- **数值表示可以行为等价**：compressed entry rebuild、tail 重算、部分低精度恢复只要最终 attention 行为等价即可。

这两类不能混在一起。把所有路径都要求 bit-exact，会误伤正常的数值近似；把路径选择也降成近似等价，又会让 sparse routing 分叉。

---

## 三个场景

### 1. CPU / SSD 介质层

CPU DRAM 适合做中间层，吸收 periodic checkpoint 或 staging buffer；SSD 更适合存稳定 compressed KV 和长前缀。SWA 全量落 SSD 通常写放大严重，tail 多数重算，indexer state 需要和被路由的状态一起管理。

这里的核心判断不是"能不能放得下"，而是"读回延迟、写入放大、重算成本三者谁更便宜"。

### 2. RDMA 跨节点

跨节点场景多出一种中间状态：block 正在被远端拉回。posted transfer 完成之前，这段 block 不能被当作可复用状态。

同时，跨节点传输不能只传 KV value。SWA window、indexer state、压缩边界、父链 key 都属于恢复路径的一部分。漏掉 auxiliary state，命中看起来存在，模型行为却会直接崩。

### 3. Agent 长轨迹

agent workload 会频繁 fork / join：同一前缀分叉成多条候选，之后只保留一条。allocator 自己不知道哪条分支会留下，所以不能只靠 LRU 决定 cache 命运。

更合理的接口是让 agent 控制流显式调用 prefetch / evict：

- shared system prompt 的 compressed KV 长期保留；
- 每条 fork 的 SWA / tail 跟随分支生命周期；
- indexer state 需要按分支快照；
- on-disk prefix 可以进入跨 sandbox 共享池。

这也是为什么 agentic serving 会把 KV cache 从单实例内部状态，推成集群级、控制流可见的资产。

---

## 如何写进正文

这类内容不适合整段塞进基础章节。推荐分层放置：

- 第6章只写原则：prefix hit 不等于 safe reuse，cache manager 变成状态机。
- 第7章讨论调度：pin、TOCTOU、RDMA posted transfer、cache-aware scheduling。
- 第10章讨论部署：CPU / SSD / RDMA / shared storage 的介质取舍。
- 第11章讨论前沿：DeepSeek-V4、FlashMemory、StreamIndex、HybridKV、Sparse Prefix Caching 代表的不同方向。

最终要传达的不是某个框架的某个实现细节，而是一个稳定判断：

**Hybrid attention serving 的 cache 复用目标不是提高命中率，而是在正确状态边界内最大化 GPU 有效工作时间。**

