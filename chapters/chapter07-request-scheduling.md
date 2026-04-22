---
id: "chapters-chapter07-request-scheduling"
title: "第7章：请求调度策略"
slug: "chapters-chapter07-request-scheduling"
date: "2026-03-11"
type: "article"
topics:
  - "request-scheduling"
concepts:
  - "continuous-batching"
  - "prefill-decode-disaggregation"
  - "throughput-engineering"
tools:
  - "vLLM"
  - "sglang"
architecture_layer:
  - "optimization-techniques"
learning_stage: "core-techniques"
optimization_axes:
  - "throughput"
  - "latency"
  - "operability"
related:
  - "chapters-chapter05-llm-inference-basics"
  - "chapters-chapter06-kv-cache-optimization"
  - "chapters-chapter10-production-deployment"
references: []
status: "published"
display_order: 8
---
# 第7章：请求调度策略

> **💰 成本影响**（常见量级，强依赖流量分布/上下文长度/实现细节）
> - **吞吐**：连续批处理与更好的调度策略通常能显著提高有效 tokens/s
> - **尾延迟**：调度往往比 kernel 更直接影响 P95/P99（排队、抢占、长短请求混杂）
> - **单位成本**：吞吐提升与尾延迟变稳会减少为保 SLA 支付的“冗余税”

## 简介

在第5章中，我们学习了 Continuous Batching 的基本原理：动态组织请求，尽量减少 padding 与空转，让 GPU 更接近“持续有活干”的状态。

但从“知道原理”到“线上跑得稳”，中间隔着一个真正的核心：**调度**。调度器决定了：

- 谁先算、谁后算（公平性与业务优先级）
- 哪些请求可以一起算（吞吐）
- 什么时候该拆开算（尾延迟与抖动）
- 在当前 KV 容量与系统预算下,哪些请求能进入本轮执行（容量与 OOM 风险）

这里也要先和第6章划清边界:

- 第6章负责 KV block 的组织、分配/回收、碎片治理与 prefix 复用
- 第7章负责利用这些 KV 状态做调度决策: 谁先跑、谁等待、谁被抢占、token budget 怎么分

也就是说,第7章默认“KV 管理器已经存在”,重点讨论的是调度器如何消费这些状态来做系统级取舍。

**本章回答什么**：
- 在线推理为什么需要显式调度器
- Continuous Batching 为什么适合在线场景
- 调度器如何围绕 admission、iteration 和 priority 做系统级决策

**本章不回答什么**：
- 不重新展开 PagedAttention、Prefix Caching 等底层 KV 机制
- 不处理生产环境里的监控、灾备、发布与回滚流程

本章仍然按“书籍化”的框架来读：

- **背景**：为什么在线推理的瓶颈常常不是算力，而是排队与组织方式？
- **决策**：延迟优先还是吞吐优先？什么时候要抢占、什么时候要隔离？
- **落地**：你用哪些指标把调度效果量化出来（TTFT/TPOT、P95/P99、队列长度、OOM/重试率）？
- **踩坑**：长短请求混跑、上下文分布漂移、优先级反转，会如何把系统拖垮？

这就是**调度器 (Scheduler)** 的职责。调度器是 vLLM 的核心组件,决定了推理系统的性能上限。一个优秀的调度器可以:
- 在有限的 GPU 显存下服务更多请求
- 降低 P95 延迟
- 最大化 GPU 利用率
- 支持 PD 分离等高级特性

本章将深入讲解:
- 为什么需要调度,调度的目标是什么
- 基础调度策略 (FIFO、Static Batching)
- Continuous Batching 的原理和实现
- vLLM 的调度器实现 (迭代级调度、Overlap Scheduling)
- 高级调度策略 (优先级、SJF、自适应)
- PD 分离 (Prefill-Decode 分离) 的架构演进

**学完本章，你将能用指标语言讨论调度设计，并能把“调度问题”拆成可验证的工程改动。**

---

## 7.1 调度的必要性

**背景**：在线推理天然是“多租户 + 波动流量”。短请求、长请求、长上下文、峰值并发会同时存在。你不显式做调度，系统就会用排队与阻塞替你“隐式调度”，表现为 P95/P99 抖动、超时与成本上升。

**决策**：进入算法细节前，先明确你的优先级排序：

- 延迟优先还是吞吐优先？更关心 TTFT 还是 TPOT？
- 公平性与可预测性是否重要？是否允许长请求“霸占”资源？
- 是否存在业务优先级与隔离需求（付费用户、关键链路、后台任务）？

**落地**：把调度当成端到端闭环，而不是单一策略名词：

- 输入：到达时间、prompt 长度、目标输出长度、上下文长度、优先级、超时预算
- 约束：显存预算、KV 增长、prefill/decode 的资源占比
- 输出：每轮迭代选择哪些请求、分配多少 token、是否抢占/降级

**踩坑**：

- 只追吞吐，P95/P99 崩溃，最后被迫加冗余 GPU，单位成本反而更高。
- 长短请求混跑不隔离，短请求 TTFT 被长请求拖慢，用户感知很差。
- 只看 `GPU-Util`，忽略队列与重试率，表面“利用率高”但有效吞吐低。

**指标口径**（建议最少盯这些）：

- 体验：TTFT、TPOT、P95/P99、超时率、降级率
- 资源：KV 占用曲线、OOM/预警次数、GPU 利用率与功耗
- 队列：队列长度、排队时间、不同优先级的等待分布

### 7.1.1 为什么需要调度

**场景**：多个用户同时发送推理请求

```
时间线:
t=0ms:  User A 发送请求 (prompt: 100 tokens)
t=10ms: User B 发送请求 (prompt: 50 tokens)
t=20ms: User C 发送请求 (prompt: 200 tokens)

GPU 资源:
- 总显存: 40GB (A100)
- 模型占用: 13GB (Llama-2-7B)
- 剩余: 27GB

问题:
1. 三个请求如何排序?
2. 是否可以并行处理?
3. 如何避免长请求饿死短请求?
4. 如何最大化 GPU 利用率?
```

**没有调度器的问题**：
```
❌ 串行处理:
  A → B → C
  User C 等待时间过长 ( unfairness)

❌ 简单批处理:
  [A, B, C] 一起处理
  需要等待最慢的请求完成
  大量 padding 浪费

❌ 先来先服务:
  长请求阻塞短请求
  P95 延迟高
```

**调度器的价值**：
```
✅ 动态调整:
  根据请求长度和资源情况动态调度

✅ 公平性:
  避免长请求饿死短请求

✅ 高效性:
  最大化 GPU 利用率和吞吐量
```

---

### 7.1.2 服务质量 vs 吞吐量

**服务质量 (Quality of Service, QoS)**：
- **延迟**: TTFT (首字延迟)、TBT (字间延迟)
- **公平性**: 所有请求都能及时处理
- **可靠性**: 请求不超时、不丢失

**吞吐量 (Throughput)**：
- 单位时间内处理的请求数
- 单位时间内生成的 tokens 数

**权衡曲线**：
```
吞吐量
  ↑
  │     ╱
  │    ╱  ← 最大化吞吐 (牺牲延迟)
  │   ╱
  │  ╱ ← 最佳平衡点
  │ ╱
  │╱     ← 最低延迟 (牺牲吞吐)
  └────────────→ 延迟
```

**调度器的目标**：找到最佳平衡点

---

### 7.1.3 调度器的目标

**主要目标**：
1. ✅ **最小化延迟**: P50、P95、P99 延迟尽可能低
2. ✅ **最大化吞吐量**: 在给定硬件上服务更多用户
3. ✅ **公平性**: 避免长请求饿死短请求
4. ✅ **资源利用**: GPU 利用率尽可能高

**次要目标**：
- 简单性: 易于理解和调试
- 可扩展性: 支持分布式部署
- 鲁棒性: 容忍异常情况

**设计原则**：
```
优先级 1: 不超时 (SLA)
优先级 2: 低延迟 (用户体验)
优先级 3: 高吞吐 (成本效率)
优先级 4: 简单可靠 (运维成本)
```

---

## 7.2 基础调度策略

**背景**：基础策略（FIFO、静态批处理）看似“低级”，但它们是你做任何高级调度之前必须对齐的基线。很多团队的问题不是没有高级算法，而是连基础策略的成本结构都没算清楚。

**决策**：把基础策略作为对照组：

- FIFO 能满足 SLA 时，高复杂度策略的运维成本可能不划算。
- 静态 batch 的 padding 浪费巨大时，你不需要争论，应该进入动态/连续批处理。

**落地**：用同一负载做 A/B：相同请求分布、相同模型、相同上下文上限，只替换调度策略，输出同一套指标（TTFT/TPOT/P95/P99/tokens/s/单位成本）。

**踩坑**：

- 用离线批处理的结论套在线流量，得到“理论吞吐高但线上排队更严重”的反效果。
- 用平均值（P50）替代尾延迟（P95/P99），掩盖最关键的问题。

### 7.2.1 FIFO (First In First Out)

**原理**：按请求到达顺序处理

```python
class FIFOScheduler:
    def __init__(self):
        self.queue = []

    def add_request(self, request):
        self.queue.append(request)

    def schedule(self):
        if self.queue:
            return [self.queue.pop(0)]  # 返回第一个请求
        return []
```

**优点**：
- ✅ 实现简单
- ✅ 公平 (先来先服务)
- ✅ 无饥饿 (每个请求最终都会被处理)

**缺点**：
- ❌ 吞吐量低 (一次只处理一个请求)
- ❌ GPU 利用率偏低
- ❌ 长请求阻塞后续所有请求

**适用场景**：
- 单用户环境
- 低并发场景
- 对公平性要求高的场景

---

### 7.2.2 静态批处理 (Static Batching)

**原理**：将多个请求打包成一个固定大小的 batch

```python
class StaticBatchScheduler:
    def __init__(self, batch_size=8):
        self.batch_size = batch_size
        self.queue = []

    def add_request(self, request):
        self.queue.append(request)

    def schedule(self):
        if len(self.queue) >= self.batch_size:
            batch = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            return batch
        return []
```

**Padding 的问题**：
```
Batch 中的请求长度不一致:
Request A: 10 tokens
Request B: 50 tokens
Request C: 20 tokens

需要 padding 到最长:
Padded A: [pad×40][10 tokens]
Padded B: [50 tokens]
Padded C: [pad×30][20 tokens]

浪费(示意): (40 + 0 + 30) / 100 = 70% padding
```

**优点**：
- ✅ 提高吞吐量 (相比 FIFO)
- ✅ GPU 利用率提升

**缺点**：
- ❌ 大量 padding 浪费
- ❌ 短请求被长请求阻塞
- ❌ 无法动态调整

**适用场景**：
- 请求长度相近的场景
- 对延迟不敏感的离线批处理

---

### 7.2.3 优缺点分析

| 策略 | 吞吐量 | 延迟 | GPU 利用率 | 实现复杂度 | 适用场景 |
|------|-------|------|-----------|-----------|---------|
| **FIFO** | 低 | 最低 (单请求) | 偏低 | 简单 | 低并发 |
| **Static Batching** | 中 | 高 (等待 batch) | 中等 | 简单 | 离线批处理 |
| **Continuous Batching** | 高 | 低 | 通常更高 | 中等 | 生产环境 |

**结论**：Continuous Batching 在多数生产环境中是常见选择

---

## 7.3 动态批处理 (Continuous Batching)

**背景**：静态批处理在在线场景里会被两个现实打败：请求长度分布不一致导致 padding 浪费；请求到达是连续的，等待凑 batch 会牺牲延迟，不等又牺牲吞吐。Continuous Batching 的价值是把“等 batch”变成“持续重组”，让 GPU 更接近稳定负载。

**决策**：是否应该上 Continuous Batching，常取决于：

- 你是否真的有并发（否则批处理没有原料）。
- 你的尾延迟预算是否允许一定程度的重排与队列。
- 你的 KV 管理是否足够稳（否则 batch 越大越容易 OOM/碎片化）。

**落地**：把它当成三件事的组合，而不是一个开关：

- token 级迭代与重组（本轮哪些请求继续 decode）
- attention mask 组织（减少 padding）
- 基于 KV 状态的容量判断（本轮还能安全接纳哪些请求）

**踩坑**：

- batch 盲目变大，TPOT 变快但 TTFT 变慢，交互体验反而更差。
- 长上下文请求不隔离，短请求被“历史 KV”拖累，尾延迟飙升。
- 缺少保护策略（超时、抢占、最大生成步数），长请求会劣化全局。

**指标口径**：

- 分解指标：TTFT（prefill）与 TPOT（decode）分别看
- 端到端：P95/P99 与超时率比 P50 更重要
- 吞吐：有效 tokens/s（扣除失败/重试/降级）比理论值更真实

本节只解决原理层面的三个问题: 它为什么出现、核心机制是什么、什么时候值得用。真正的请求状态机、每轮调度循环、容量预留和 overlap 细节统一放到 7.4。

### 7.3.1 它在对抗什么浪费

静态批处理的问题不只是“有 padding”，而是它默认所有请求会在同一个整齐的矩形里一起前进。但在线推理并不是这样运作的:

- 请求到达时间不同
- prompt 长度不同
- decode 阶段每轮只新增很少 token
- 已完成请求会立刻留下空槽

于是浪费会同时出现在两处:

- **计算浪费**: padding 和空槽让 GPU 计算了很多没有价值的 token
- **排队浪费**: 新请求明明已经到了,却要等旧 batch 整体收尾

Continuous Batching 的价值,就是把这两类浪费一起压下去。

---

### 7.3.2 它由哪三部分组成

从原理上看,Continuous Batching 只有三件事:

1. **工作集重组**
每一轮都允许完成的请求退出、等待中的请求进入,而不是等整批结束。

2. **非矩形 batch 组织**
请求不必强行补齐到同一长度,而是按当前活跃 token 组织计算。

3. **prefill / decode 混合执行**
系统可以同时面对“老请求继续 decode”和“新请求开始 prefill”,调度器负责决定这一轮怎么配比。

如果把它压缩成一句话,就是:

```text
Continuous Batching = 不再等待一个固定 batch 完整结束,而是按轮持续重组当前活跃工作集。
```

---

### 7.3.3 什么时候值得用

不是所有系统都需要它。通常在下面这些条件同时出现时,它的价值才会很明显:

- 有持续并发,而不是偶发单请求
- 请求长度分布差异明显
- 你确实在为 padding、空槽和排队付出代价
- KV 管理已经足够稳定,不会因为 batch 扩大而更容易 OOM

---

### 7.3.4 这一节和 7.4 的关系

到这里为止,你应该已经知道:

- 它为什么比静态批处理更适合在线场景
- 它依赖哪几类机制才能成立
- 它什么时候可能真正带来收益

接下来的 7.4 不再重复讲这些原理,而是回答另一类问题:

- 请求对象在系统里如何流转
- 调度器每一轮到底检查什么
- 容量预留和 CPU-GPU overlap 如何作为配套机制挂到主循环上

---

## 7.4 vLLM 的调度器实现

**背景**：理解 vLLM 的调度器不是为了照抄实现，而是为了学习它如何把三类矛盾放到同一个系统里解决：迭代级调度、容量预留策略、以及 CPU 开销治理（避免 GPU 空转等待）。

**决策**：读这一节时建议关注接口与职责边界：

- “请求生命周期”与“调度决策”如何分层？
- 哪些信息是调度器必须知道的（长度、优先级、KV 状态）？
- 哪些信息应该由内存管理器维护（block 分配/回收/碎片）？

一个实用的读法是:

- 把第6章当成“KV 管理器提供的能力边界”
- 把第7章当成“调度器如何使用这些能力做 admission / iteration / priority 决策”

**落地**：在生产系统里改调度的稳妥方式：

- 先加观测：每轮迭代的选择、队列、KV 占用必须可见
- 再做小改动：一次只改一个策略点（优先级/抢占/chunk/阈值）
- 最后做回归：看 P95/P99、失败率与单位成本是否同时改善

**踩坑**：

- 只看吞吐，不看 CPU overhead 与尾延迟，上线后“更快但更不稳”。
- 把内存管理与调度强耦合，导致系统难以演进与排障。

从层次上看,这一节可以按三步读:

1. 先看请求对象和生命周期
2. 再看每轮调度主循环
3. 最后看围绕主循环的两类增强: 容量预留与 CPU-GPU overlap

### 7.4.1 请求生命周期管理

**状态机**：
```
                            ┌─────────────┐
                            │   Waiting   │  ← 等待调度
                            └──────┬──────┘
                                   │ schedule()
                                   ▼
                            ┌─────────────┐
                            │  Scheduled  │  ← 已调度,等待执行
                            └──────┬──────┘
                                   │ execute()
                                   ▼
                            ┌─────────────┐
                      ┌────→│   Running   │  ← 正在执行
                      │     └──────┬──────┘
                      │            │
                      │            │ generate token
                      │            ▼
                      │     ┌─────────────┐
                      │     │  Decoding   │  ← 生成中
                      │     └──────┬──────┘
                      │            │
                      │            │ complete / abort
                      │            ▼
                      │     ┌─────────────┐
                      └─────│  Finished   │  ← 完成/中断
                            └─────────────┘
```

**vLLM 的请求对象**：
```python
class Sequence:
    def __init__(self, request_id, prompt):
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_tokens = tokenize(prompt)
        self.output_tokens = []

        # 状态
        self.status = "waiting"  # waiting, running, finished

        # KV Cache
        self.block_table = []  # Physical blocks

        # 元数据
        self.arrival_time = time.time()
        self.start_time = None
        self.finish_time = None
```

---

### 7.4.2 请求进入时的容量预留

这一节讨论的是**调度视角的容量预留策略**,不是第6章那种底层 block 布局实现。你可以把它理解成:

- 第6章回答: block 最终怎么放进显存
- 这里回答: 调度器在请求刚进入系统时,应该保守预留多少未来空间

**预分配 (Pre-allocation)**：
```python
# 传统方法: 预分配最大空间
def allocate_max_space(request):
    max_tokens = request.max_new_tokens
    prompt_tokens = len(request.prompt_tokens)
    total = prompt_tokens + max_tokens
    # 预分配 total tokens 的空间
    return allocate_blocks(total)
```

**问题**：
- 浪费显存 (大多数请求不会达到 max_new_tokens)
- 限制并发数

**动态分配 (Dynamic Allocation)**：
```python
# vLLM 方法: 动态增长
def allocate_dynamic(request):
    # 初始分配: prompt + 少量 decode
    initial = len(request.prompt_tokens) + 16
    blocks = allocate_blocks(initial)

    # 动态增长
    while need_more_space(request):
        new_blocks = allocate_blocks(16)
        blocks.extend(new_blocks)

    return blocks
```

**优势**：
- 节省显存 (幅度依场景而定)
- 提高并发数
- 支持 max_new_tokens 很大的场景

---

### 7.4.3 迭代级调度 (Iteration-level Scheduling)

**定义**：每次迭代 (iteration) 重新调度一次

```python
class Scheduler:
    def schedule(self):
        """每次迭代调用"""
        scheduled = self._schedule()
        self.model_executor.execute_model(scheduled)
        self._process_outputs()

    def _schedule(self):
        """决定哪些请求可以执行"""
        scheduled = []

        # 1. 从 running 中选择
        for seq in self.running:
            if self._can_schedule(seq):
                scheduled.append(seq)

        # 2. 从 waiting 中选择
        for seq in self.waiting:
            if self._can_schedule(seq):
                scheduled.append(seq)
                self.running.append(seq)
                self.waiting.remove(seq)

        return scheduled

    def _can_schedule(self, seq):
        """检查是否有足够的资源"""
        # 1. 检查 KV Cache 空间
        required_blocks = estimate_blocks(seq)
        if len(self.free_blocks) < required_blocks:
            return False

        # 2. 检查 GPU 计算
        # (CUDA 支持并发,通常不是瓶颈)
        return True
```

**调度流程**：
```
每次迭代:
  1. 调度器决定哪些请求可以执行
  2. 准备输入数据
  3. 启动 GPU kernel
  4. GPU 执行推理
  5. 获取输出
  6. 更新请求状态
  7. 回到步骤 1
```

---

### 7.4.4 Overlap Scheduling (Mini-SGLang) ⚡️

> **💡 深度来源**：[Mini-SGLang Blog](https://lmsys.org/blog/2025-12-17-minisgl/) + [Berkeley EECS-2025-192](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf)
>
> **核心问题**：CPU overhead 可能导致 GPU 闲置 → Overlap Scheduling 是一种应对方式
>
> **性能影响**：可减少 GPU stalls,具体提升需基准测试

---

#### 7.4.4.1 CPU 开销导致 GPU 闲置问题

**Berkeley EECS-2025-192 的发现**：
- CPU 开销在某些场景中占据可观比例
- 主要来源:
  - Kernel launch (启动 GPU kernel)
  - Memory copy (CPU↔GPU 数据传输)
  - Synchronization (等待 GPU 完成)
  - Batch scheduling (决定哪些请求一起处理)

**问题**：
- vLLM 的迭代级调度是 **串行** 的:
  ```
  Step 1: CPU 调度下一批请求
  Step 2: CPU 准备输入数据
  Step 3: CPU 启动 GPU kernel
  Step 4: GPU 计算 (此时 CPU 闲置!)
  Step 5: CPU 等待 GPU 完成
  Step 6: 回到 Step 1
  ```
 - 结果: **GPU 利用率偏低**,可能出现 GPU stalls

**Nsight Systems 分析** (无 overlap):
```
Timeline:
CPU: |--Schedule1--|--Prepare2--|--Launch3--|
GPU:              |<--Compute1-->|    stalls    |
```
看到 GPU 有明显的闲置期 (stalls)

---

#### 7.4.4.2 Overlap Scheduling 设计思想

**核心洞察**：
- **CPU-GPU 并行执行**:
  - CPU 准备下一批请求时,GPU 正在计算当前批次
  - GPU 计算完成后,下一批请求已经 ready,立即开始
- **生产者-消费者模式**:
  - CPU: 生产者 (准备 batches)
  - GPU: 消费者 (执行 batches)

**对比**：
```
无 Overlap (vLLM 默认):
CPU: |--Schedule--|--Prepare--|
GPU:                 |--Compute--|<-stall->|--Compute--|

有 Overlap (Mini-SGLang):
CPU: |--Schedule1--|--Prepare2--|--Prepare3--|
GPU:                 |--Compute1-->|--Compute2-->|
```
GPU 持续运行,无闲置!

---

#### 7.4.4.3 实现机制

**架构设计**：
```python
class OverlapScheduler:
    def __init__(self):
        self.cpu_queue = Queue()  # CPU 准备的请求队列
        self.gpu_queue = Queue()  # GPU 待执行的队列
        self.cpu_thread = Thread(target=self._cpu_worker)
        self.gpu_thread = Thread(target=self._gpu_worker)

    def start(self):
        """启动 CPU 和 GPU 线程"""
        self.cpu_thread.start()
        self.gpu_thread.start()

    def _cpu_worker(self):
        """CPU 线程: 准备 batches"""
        while True:
            # 调度下一批请求
            scheduled = self._schedule_next_batch()

            # 准备输入数据
            inputs = self._prepare_inputs(scheduled)

            # 放入 GPU 队列
            self.gpu_queue.put(inputs)

    def _gpu_worker(self):
        """GPU 线程: 执行 batches"""
        while True:
            # 从队列获取 (阻塞等待)
            inputs = self.gpu_queue.get()

            # 执行推理
            outputs = self.model_executor.execute(inputs)

            # 处理输出
            self._process_outputs(outputs)
```

**关键优化**：
1. **Pipeline 深度**: 通常 2-3 个 batches 的 pipeline
2. **同步机制**: 使用条件变量避免 busy waiting
3. **内存管理**: 预分配 buffers 避免运行时分配

---

#### 7.4.4.4 性能提升

**吞吐量提升(示意)**：
```
无 Overlap:
- CPU 开销: 依实现而定
- GPU stalls: 可能存在
- 有效计算: 依负载而定

有 Overlap:
- CPU 与 GPU 并行化
- GPU stalls 通常减少
- 吞吐量可能提升
```

**延迟改善(示意)**：
```
P95 延迟通常可改善
- CPU 准备时间不再完全阻塞 GPU
- 请求更快开始处理
```

---

#### 7.4.4.5 vLLM 的实现状态

**当前状态** (v0.6.x):
- ✅ 支持 iteration-level scheduling
- ⚠️ overlap 支持程度与版本/配置相关

**如何启用** (实验性):
```python
from vLLM import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_overlap_schedule=True,  # 实验性功能
)
```

---

### 7.4.5 运行中的动态容量预留

> **💡 参考思路（经验口径）**：一些推理引擎会提供动态容量预留来减少“按 max_new_tokens 预留”的浪费
>
> **问题**：预留 max_new_tokens 的空间可能浪费内存
>
> **解决**：调度器根据实际使用情况动态调整预留大小

注意这里讨论的仍然是“调度器怎么做 admission 和增长策略”,不是“底层 block manager 如何摆放 physical blocks”。后者属于第6章。

**核心问题**：
```python
# 用户设置
max_new_tokens = 2048

# 传统做法: 预留 2048 tokens 的空间
reserved = 2048

# 实际情况: 很多请求生成的 tokens 少于上限
actual = 500

# 浪费: 2048 - 500 = 1548 tokens (示意)
```

**调度侧的动态容量预留**：
```python
class DynamicReservationManager:
    def __init__(self, initial_beta=0.5):
        """
        beta: 预留比例
              初始: 0.5 (预留 50% 的 max_new_tokens)
        """
        self.beta = initial_beta
        self.actual_usage_history = []

    def allocate(self, prompt_len, max_new_tokens):
        """按当前策略预留容量"""
        # 预留: prompt + (beta × max_new_tokens)
        reserved = int(prompt_len + self.beta * max_new_tokens)
        blocks = allocate_blocks(reserved)
        return blocks

    def on_token_generated(self, seq):
        """生成新 token 时调用"""
        # 检查是否需要扩展
        current_tokens = len(seq.output_tokens)
        max_tokens = seq.max_new_tokens

        if current_tokens > self.beta * max_tokens * 0.8:
            # 即将到达预留上限,扩展
            self._expand_reservation(seq)

    def on_request_complete(self, seq):
        """请求完成时调用"""
        # 记录实际使用情况
        actual_tokens = len(seq.output_tokens)
        max_tokens = seq.max_new_tokens
        usage_ratio = actual_tokens / max_tokens
        self.actual_usage_history.append(usage_ratio)

        # 只保留最近 100 个请求的历史
        if len(self.actual_usage_history) > 100:
            self.actual_usage_history.pop(0)

    def get_stats(self):
        """获取统计信息"""
        if not self.actual_usage_history:
            return {}

        return {
            'beta': self.beta,
            'avg_usage_ratio': sum(self.actual_usage_history) / len(self.actual_usage_history),
            'memory_saved_pct': (1 - self.beta) * 100  # 仅示意
        }
```

**工作流程**：
```
请求到来时:
  1. 用户请求: prompt=1000 tokens, max_new_tokens=2048

  2. 传统做法:
     预留: 1000 + 2048 = 3048 tokens 的 KV Cache

  3. 动态容量预留:
     预留: 1000 + (0.5 × 2048) = 1000 + 1024 = 2024 tokens
     (β=0.5,节省 33% 内存,示意)

请求进行中:
  1. 请求已生成 600 tokens
  2. 发现即将到达 max_new_tokens 的 30%
  3. 动态扩展预留: 1024 → 1433 tokens
  4. 如果 GPU 内存不足,等待其他请求完成

请求完成时:
  1. 请求在 600 tokens 时遇到 EOS
  2. 释放所有 KV Cache (1000 + 600 = 1600 tokens)
  3. 记录实际使用率: 600 / 2048 = 29.3%
  4. 更新 β: 0.5 → 0.35 (根据历史平均,示意)
  5. 下次请求只预留: 1000 + (0.35 × 2048) = 1716 tokens
```

**性能提升(示意)**：
```
内存节省(示意):
  场景        | 传统做法 | 动态管理 | 节省
  Chat (500)  | 3048     | 2024     | 33%
  RAG (800)   | 3048     | 2240     | 27%
  Code (1200) | 3048     | 2640     | 13%

吞吐量提升:
  更大的 batch size (因为内存节省)
  提升幅度需基准测试验证
```

---

## 7.5 高级调度策略

**背景**：当你具备 Continuous Batching 的基本能力后，真正把系统做成“可运营平台”的往往是高级策略：优先级、抢占、公平性与隔离。它们决定了系统在峰值与异常时是否会雪崩。

**决策**：高级策略不是越多越好，先做“业务必须”的那几条：

- 是否存在硬优先级（付费用户/关键路径）？
- 是否必须隔离（长上下文/低优先级任务是否要单独分池）？
- 是否允许抢占（强制让长请求让路）？

**落地**：高级策略落地更像制度：

- 明确超时预算与降级策略（否则抢占会变成无休止重试）
- 把配额写进系统（否则低优先级任务会挤爆资源）
- 把回滚写进发布流程（否则一次策略回归会直接带来事故）

**指标口径**：

- 分层看：不同优先级流量分别看 P95/P99 与超时率
- 看峰值：峰值时的抖动与雪崩才是高级策略的主要价值场景

### 7.5.1 优先级调度

**原理**：不同请求有不同优先级

```python
class PriorityScheduler:
    def __init__(self):
        # 多个队列,不同优先级
        self.queues = {
            'high': [],    # 高优先级 (VIP 用户)
            'normal': [],  # 正常优先级
            'low': [],     # 低优先级 (免费用户)
        }

    def add_request(self, request, priority='normal'):
        self.queues[priority].append(request)

    def schedule(self):
        # 优先处理高优先级队列
        if self.queues['high']:
            return [self.queues['high'].pop(0)]
        elif self.queues['normal']:
            return [self.queues['normal'].pop(0)]
        else:
            return [self.queues['low'].pop(0)]
```

**应用场景**：
- VIP 用户 vs 普通用户
- 付费用户 vs 免费用户
- 实时请求 vs 离线批处理

---

### 7.5.2 最短作业优先 (SJF)

**原理**：优先处理预计完成时间最短的请求

```python
class SJFScheduler:
    def schedule(self, pending_requests):
        # 按预计完成时间排序
        sorted_requests = sorted(
            pending_requests,
            key=lambda r: r.estimated_duration()
        )
        # 返回前 N 个
        return sorted_requests[:batch_size]
```

**优势**：
- ✅ 降低平均延迟
- ✅ 提高吞吐量

**劣势**：
- ❌ 可能饿死长请求
- ❌ 需要准确估计请求长度

**改进**：Shortest Remaining Time First (SRTF)
- 动态重新评估
- 考虑已执行的时间

---

### 7.5.3 轮询调度 (Round Robin)

**原理**：公平地轮转处理每个队列

```python
class RoundRobinScheduler:
    def __init__(self, time_slice=10):
        self.time_slice = time_slice  # 每个 queue 的时间片
        self.queues = {
            'queue1': [],
            'queue2': [],
            'queue3': [],
        }
        self.current_queue = 0
        self.timer = 0

    def schedule(self):
        # 时间片用完,切换到下一个队列
        if self.timer >= self.time_slice:
            self.current_queue = (self.current_queue + 1) % len(self.queues)
            self.timer = 0

        # 从当前队列取请求
        queue_name = list(self.queues.keys())[self.current_queue]
        if self.queues[queue_name]:
            self.timer += 1
            return [self.queues[queue_name].pop(0)]

        return []
```

**优势**：
- ✅ 绝对公平
- ✅ 无饥饿

**劣势**：
- ❌ 上下文切换开销
- ❌ 可能降低吞吐量

---

### 7.5.4 自适应调度

**原理**：根据系统状态动态调整调度策略

```python
class AdaptiveScheduler:
    def __init__(self):
        self.strategies = {
            'low_load': FIFOScheduler(),
            'high_load': ContinuousBatchScheduler(),
            'mixed': PriorityScheduler(),
        }
        self.current_strategy = None

    def schedule(self):
        # 监控系统状态
        load = self.get_system_load()
        queue_length = len(self.waiting_queue)

        # 根据状态选择策略
        if load < 0.3:
            self.current_strategy = self.strategies['low_load']
        elif load > 0.8:
            self.current_strategy = self.strategies['high_load']
        else:
            self.current_strategy = self.strategies['mixed']

        return self.current_strategy.schedule()
```

**优势**：
- ✅ 适应不同工作负载
- ✅ 自动优化

**挑战**：
- ⚠️ 策略切换开销
- ⚠️ 参数调优复杂

---

## 7.6 实战配置

**背景**：调度“实战配置”不是把参数抄一遍，而是把它变成一套可复现的实验：相同负载、相同指标口径、相同回归集。否则你很容易陷入“调了很多，没人能证明变好”的状态。

**决策**：在生产系统里，建议先固定三个东西再调参：

- 固定模型与 max context（否则任何结果都不可比）
- 固定负载分布（至少有代表性的压测脚本或回放流量）
- 固定指标看板（TTFT/TPOT/P95/P99/吞吐/失败率/单位成本）

**落地**：把调参分成两类：

- 安全相关：最大并发、最大生成长度、超时与重试（先做，避免事故）
- 性能相关：batch 策略、chunk 策略、优先级与抢占（后做，用 A/B 验证）

### 7.6.1 vLLM 调度参数调优

**关键参数**：
```bash
vLLM serve meta-llama/Llama-2-7b-hf \
  # Batch 相关
  --max-num-batched-tokens 8192 \        # 每次 iteration 最大 tokens
  --max-num-seqs 256 \                    # 最大并发请求数

  # Memory 相关
  --gpu-memory-utilization 0.9 \         # GPU 内存利用率
  --block-size 16 \                       # PagedAttention block 大小

  # 调度相关
  --max-paddings 256 \                    # 最大 padding 数量
  --schedule-policy "fcfs" \              # 调度策略 (fcfs/priority)
```

**调优建议**：
```
场景 1: 低延迟优先
  --max-num-batched-tokens 4096  # 减小 batch size
  --max-num-seqs 64              # 减少并发

场景 2: 高吞吐优先
  --max-num-batched-tokens 16384 # 增大 batch size
  --max-num-seqs 512             # 增加并发

场景 3: 混合工作负载
  --max-num-batched-tokens 8192  # 平衡
  --schedule-policy "priority"   # 启用优先级
```

---

### 7.6.2 不同场景的调度策略

**场景 1: Chatbot 服务**
```
特征:
  - 大量短请求
  - 用户敏感延迟

推荐配置:
  - Continuous Batching
  - 较小的 batch size (减少等待)
  - FIFO 优先 (公平性)

参数:
  --max-num-batched-tokens 4096
  --max-num-seqs 128
  --schedule-policy "fcfs"
```

**场景 2: RAG 应用**
```
特征:
  - 长 prompt (文档内容)
  - 短输出 (答案)
  - 高 Prefill 比例

推荐配置:
  - Prefix Caching (缓存文档)
  - 较大的 batch size (Prefill 阶段)
  - 优先级调度 (VIP 用户)

参数:
  --enable-prefix-caching
  --max-num-batched-tokens 16384
  --schedule-policy "priority"
```

**场景 3: 批量处理**
```
特征:
  - 离线任务
  - 不敏感延迟
  - 追求吞吐量

推荐配置:
  - 大 batch size
  - Static Batching (可以接受)
  - SJF 调度 (最小化平均完成时间)

参数:
  --max-num-batched-tokens 32768
  --max-num-seqs 512
```

---

## 7.7 Prefill-Decode 分离 (PD 分离) ⚠️ 技术评估中

**背景**：PD 分离的价值经常来自“资源重配”而不是“无条件吞吐提升”。当你的负载混杂（短 prompt 与长上下文、短输出与长输出、峰值波动），把 prefill 与 decode 拆开能降低互相干扰，进而改善尾延迟并减少为保 SLA 支付的冗余成本。

**决策**：不要把 PD 分离当作默认选项。它通常更适合：

- P95/P99 是硬指标，且峰值波动明显
- decode 阶段被带宽拖慢，prefill 阶段又吃算力
- 你愿意承担更高系统复杂度（KV 传输、两套扩缩容、更多故障模式）

**落地**：更稳妥的路线是“先评估、再拆分”：

- 先在单体服务里测清 prefill 与 decode 的时间占比（TTFT vs TPOT）
- 再在同机/同节点范围内做最小拆分（减少 KV 传输开销）
- 最后再考虑跨节点与异构硬件的资源重配

**踩坑**：

- KV 传输与序列化开销吃掉收益，整体更慢
- 两套队列与扩缩容策略叠加，故障定位更难
- 缺少端到端追踪时，只能看到“某段慢”，但不知道慢在哪里

**指标口径**：

- 必看：TTFT/TPOT 分别变化、P95/P99、超时率
- 必看：KV 传输带宽与耗时、prefill/decode 队列长度、降级次数

### 7.7.1 什么是 PD 分离

**Prefill 阶段**：并行处理 prompt,计算密集
- 输入: 整个 prompt
- 计算: 矩阵乘法为主
- 特点: 计算密集,可以并行

**Decode 阶段**：串行生成 token,内存带宽密集
- 输入: 每次一个新 token
- 计算: 内存读取为主
- 特点: 带宽密集,串行生成

**两种阶段的计算模式差异**：
```
Prefill:
  GPU 利用: 计算占比更高
  瓶颈: 算力 (FLOPS)
  最优 GPU: H100 (高算力)

Decode:
  GPU 利用: 带宽占比更高
  瓶颈: 内存带宽
  最优 GPU: A100 (高带宽,低成本)
```

**为什么需要分离?**
- 同一个硬件无法同时优化两种模式
- 分离后可以针对性优化
- 资源利用率可能提升 (依负载而定)

---

### 7.7.2 PD 分离的架构演进

**演进路径(概述)**：
- 学术与社区提出架构思路
- 部分框架尝试落地实现
- 逐步出现可运维的工程实践

---

### 7.7.3 PD 分离的技术优势

#### 架构图：传统融合 vs PD 分离

**传统融合架构（单体 GPU）**：

```
┌─────────────────────────────────────────────────────┐
│                   单 GPU 推理服务                     │
│  ┌─────────────────────────────────────────────┐   │
│  │           Prefill + Decode 混合处理          │   │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐  │   │
│  │  │ Request │───▶│ Prefill │───▶│ Decode  │  │   │
│  │  │   A     │    │         │    │         │  │   │
│  │  └─────────┘    └─────────┘    └─────────┘  │   │
│  │                      │              │        │   │
│  │              ┌──────▼──────┐ ┌─────▼────┐   │   │
│  │              │  KV Cache   │ │  KV Cache │   │   │
│  │              │  (共享显存)  │ │  (共享显存)│   │   │
│  │              └─────────────┘ └───────────┘   │   │
│  └─────────────────────────────────────────────┘   │
│                                                      │
│ 问题：两种 workload 抢同一 GPU，难以同时优化         │
└─────────────────────────────────────────────────────┘
```

**PD 分离架构（双池）**：

```
┌────────────────────────────────────────────────────────────────────┐
│                    PD 分离推理架构                                   │
│                                                                      │
│  ┌─────────────────┐           ┌─────────────────┐                 │
│  │  Prefill 池      │           │  Decode 池       │                 │
│  │  (H100 x N)      │           │  (A100 x M)      │                 │
│  │                 │           │                  │                 │
│  │  ┌───────────┐  │    KV     │  ┌───────────┐  │                 │
│  │  │ Prefill   │──┼───传输───▶│─▶│ Decode    │  │                 │
│  │  │ Worker 1  │  │           │  │ Worker 1  │  │                 │
│  │  └───────────┘  │   ~100GB/s│  └───────────┘  │                 │
│  │  ┌───────────┐  │  (RDMA)   │  ┌───────────┐  │                 │
│  │  │ Prefill   │──┤           │  │ Decode    │  │                 │
│  │  │ Worker 2  │  │           │  │ Worker 2  │  │                 │
│  │  └───────────┘  │           │  └───────────┘  │                 │
│  └────────┬────────┘           └────────┬────────┘                 │
│           │                              │                          │
│           │    ┌──────────────────┐      │                          │
│           └───▶│  请求队列 (调度器) │◀─────┘                          │
│                │  - 优先级队列      │                                │
│                │  - 负载均衡        │                                │
│                │  - KV 路由         │                                │
│                └──────────────────┘                                │
│                                                                      │
│ 优势：                                                               │
│  • Prefill 池：专注算力优化，TTFT 更稳定                              │
│  • Decode 池：专注带宽优化，TPOT 更稳定                               │
│  • 独立扩缩容：流量高峰可单独扩容                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**数据流详解**：

```
时间线：

Request A (prompt=1000 tokens)
│
├─ 0ms: 到达调度器
│
├─ 10ms: 进入 Prefill 队列
│
├─ 50ms: Prefill Worker 处理
│  ├─ 计算 prompt 的 KV Cache
│  └─ 150ms: 完成，输出首 token
│
├─ 200ms: KV Cache 通过 RDMA 传输到 Decode 池
│  └─ 传输时间 ≈ KV大小 / 带宽
│      (1000 tokens × 0.5MB / 100GB/s ≈ 5ms)
│
├─ 205ms: 进入 Decode 队列
│
├─ 210ms: Decode Worker 处理
│  ├─ 使用已缓存的 KV Cache
│  └─ 逐 token 生成
│
└─ 2000ms: 生成完成 (100 tokens, ~18ms/token)
```

**异构部署配置示例**：

```yaml
# Kubernetes 下的 PD 分离配置
# 注意：这需要支持跨节点 GPU 调度的编排系统

# Prefill 池：H100，专注算力
deployment:
  name: vLLM-prefill
  replicas: 4
  gpu: nvidia-h100-80gb
  resources:
    limits:
      nvidia.com/gpu: 1
      memory: "80Gi"
  env:
    VLLM_WORKER_TYPE: "prefill"
    VLLM_GPU_MEMORY_UTILIZATION: "0.95"

# Decode 池：A100，专注成本
deployment:
  name: vLLM-decode
  replicas: 8
  gpu: nvidia-a100-80gb
  resources:
    limits:
      nvidia.com/gpu: 1
      memory: "80Gi"
  env:
    VLLM_WORKER_TYPE: "decode"
    VLLM_GPU_MEMORY_UTILIZATION: "0.90"

# 调度器：负责路由和 KV 传输
deployment:
  name: vLLM-scheduler
  replicas: 2
  # 需要支持 RDMA 的网络
```

**资源隔离**：
```
无分离:
  长请求的 Prefill 阻塞短请求的 Decode
  → P99 延迟高

有分离:
  Prefill 和 Decode 独立调度
  → 长请求不影响短请求
```

**弹性扩展**：
```
高峰期:
  增加 Prefill Worker (新用户多)

稳定期:
  增加 Decode Worker (生成多)
```

**性能优化**：
```
Prefill Worker:
  - 大 batch size
  - 算子融合
  - Tensor Core 优化

Decode Worker:
  - 高带宽优化
  - KV Cache 优化
  - Speculative Decoding
```

---

### 7.7.4 vLLM 的 PD 分离实现

**架构设计**：
```python
# Prefill Worker
class PrefillWorker:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.cache_engine = CacheEngine()

    def process(self, requests):
        """处理 Prefill 阶段"""
        for req in requests:
            # 计算 prompt 的 KV Cache
            kv_cache = self.model.prefill(req.prompt)

            # 存储到 Cache Engine
            self.cache_engine.store(req.id, kv_cache)

        return kv_cache

# Decode Worker
class DecodeWorker:
    def __init__(self, model_path, prefill_worker_url):
        self.model = load_model(model_path)
        self.cache_engine = CacheEngine()
        self.prefill_worker = PrefillClient(prefill_worker_url)

    def process(self, requests):
        """处理 Decode 阶段"""
        for req in requests:
            # 从 Prefill Worker 获取 KV Cache
            kv_cache = self.prefill_worker.fetch(req.id)

            # 加载到本地 Cache Engine
            self.cache_engine.load(req.id, kv_cache)

            # 开始 Decode
            output = self.model.decode(kv_cache, req.max_tokens)

        return output
```

**通信机制**：KV Cache 的传输
```python
# 序列化 KV Cache
def serialize_kv_cache(kv_cache):
    """将 KV Cache 序列化为字节流"""
    import pickle
    return pickle.dumps(kv_cache)

# 反序列化 KV Cache
def deserialize_kv_cache(data):
    """从字节流恢复 KV Cache"""
    import pickle
    return pickle.loads(data)

# RPC 调用
prefill_worker.push_kv_cache(
    request_id=req.id,
    kv_cache_bytes=serialize_kv_cache(kv_cache)
)
```

**调度策略**：
```python
def schedule_for_pd(requests):
    """将请求分配到 Prefill 或 Decode Worker"""
    prefill_requests = []
    decode_requests = []

    for req in requests:
        if req.state == 'waiting':
            # 新请求 → Prefill
            prefill_requests.append(req)
        elif req.state == 'decoding':
            # 正在生成 → Decode
            decode_requests.append(req)

    return prefill_requests, decode_requests
```

---

### 7.7.5 SGLang 的 PD 分离实践

**RadixAttention**：统一的注意力抽象
```python
class RadixAttention:
    def forward(self, query, key, value, state):
        # 自动检测是 Prefill 还是 Decode
        if state.is_prefill:
            return self._prefill_forward(query, key, value)
        else:
            return self._decode_forward(query, key, value)
```

**自动分离**：无需手动配置
```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3-8B \
  --enable-pd-separation  # 自动启用
```

**生产经验**：稳定性、性能监控
```
关键指标:
  - Prefill Worker: GPU 利用率高
  - Decode Worker: 内存带宽利用率高
  - KV Cache 传输: 延迟尽量低

告警阈值:
  - Prefill 队列长度偏高: 考虑扩容
  - Decode 队列长度偏高: 考虑扩容
  - KV Cache 传输延迟偏高: 检查网络
```

---

### 7.7.6 PD 分离的挑战

**KV Cache 传输**：
```
问题: 网络开销和序列化
  - KV Cache 很大 (数百 MB 到数 GB)
  - 序列化/反序列化开销
  - 网络传输延迟

解决方案:
  - 使用共享存储 (NVLink、InfiniBand)
  - 压缩 KV Cache
  - 增量传输 (只传输新增部分)
```

**负载均衡**：
```
问题: Prefill 和 Decode 的速率匹配
  - Prefill 快: Decode 成为瓶颈
  - Decode 快: Prefill 成为瓶颈

解决方案:
  - 动态调整 Worker 数量
  - 自适应调度策略
  - 监控和自动扩缩容
```

**容错处理**：
```
问题: Worker 故障如何恢复
  - Prefill Worker 故障: 新请求无法处理
  - Decode Worker 故障: 正在生成的请求中断

解决方案:
  - 冗余部署 (多 Worker)
  - KV Cache 持久化
  - 自动故障转移
```

**复杂度增加**：
```
问题: 部署和运维的挑战
  - 需要管理两种 Worker
  - 配置更复杂
  - 调试更困难

解决方案:
  - 完善的监控体系
  - 自动化部署工具
  - 统一的日志和追踪
```

---

### 7.7.7 实战案例

**案例 1: 单机 GPU 的 PD 分离 (示意)**
```
硬件: 单机 4 × A100 40GB

部署:
  GPU 0-1: Prefill Worker (2 个)
  GPU 2-3: Decode Worker (2 个)

性能:
  吞吐量: 可能提升 (依负载而定)
  P95 延迟: 可能改善
```

**案例 2: 跨机器的 PD 分离部署 (示意)**
```
硬件:
  机器 A: 4 × H100 (Prefill)
  机器 B: 8 × A100 (Decode)

网络: InfiniBand (100 Gbps)

性能:
  吞吐量: 可能提升
  成本: 可能降低 (取决于硬件价格与利用率)
```

**案例 3: 异构 GPU (H100 + H200) 的实践 (示意)**
```
硬件:
  H100: Prefill (算力优化)
  H200: Decode (带宽优化,大内存)

性能:
  吞吐量: 可能提升
  支持更长序列 (取决于显存容量)
```

---

## ✅ 章节检查清单

完成本章后,你应该能够:

- [ ] 解释为什么需要调度器
- [ ] 对比 FIFO、Static Batching、Continuous Batching
- [ ] 解释 Continuous Batching 的核心机制与适用条件
- [ ] 理解 vLLM 的迭代级调度
- [ ] 解释 Overlap Scheduling 的原理和优势
- [ ] 配置 vLLM 的调度参数
- [ ] 针对不同场景选择合适的调度策略
- [ ] 理解 PD 分离的架构演进
- [ ] 设计 PD 分离的部署方案
- [ ] 评估 PD 分离的收益和挑战

---

## 📚 动手练习

**练习 7.1**：对比静态批处理和动态批处理

场景:
- 8 个请求,长度分别为: [10, 50, 20, 100, 30, 15, 80, 25] tokens
- 假设每个请求都生成 100 tokens

任务:
1. 计算 Static Batching 的 padding 数量
2. 计算 Continuous Batching 的 padding 数量
3. 比较两种方法的 GPU 利用率

**练习 7.2**：针对不同场景优化调度参数

场景:
- Chatbot: 100 个并发,平均 50 tokens,对延迟敏感
- RAG: 20 个并发,平均 2000 tokens prompt,对吞吐量敏感
- 批处理: 1000 个请求,离线任务,追求最快完成

任务:
1. 为每个场景设计调度策略
2. 选择合适的调度算法
3. 配置 vLLM 参数

**练习 7.3**：使用 vLLM 部署 PD 分离架构 ⭐

任务:
1. 设计一个 PD 分离的部署方案
2. 选择合适的硬件配置
3. 编写 docker-compose.yml
4. 评估性能提升和成本

---

## 🎯 总结

关键要点：
- 调度器是推理系统的核心,决定性能上限
- Continuous Batching 通过动态调整,消除 padding 浪费
- Overlap Scheduling 通过 CPU-GPU 并行,消除 GPU stalls
- 调度侧的动态容量预留可减少“按 max_new_tokens 全额预留”带来的浪费
- PD 分离在合适负载与工程能力下可能带来显著收益,但主要价值经常体现在尾延迟与资源重配,而不是“无条件数倍提升”
- 不同场景需要不同的调度策略

## 章节衔接

第6章和第7章合起来,你已经有了推理系统最核心的两块拼图: 一块是 KV 如何被存放和复用,另一块是请求如何被组织和调度。接下来进入第8章,主题会从“怎么更聪明地使用显存和算力”转向“怎么从模型表示本身继续压缩资源占用”,也就是量化技术。

---
