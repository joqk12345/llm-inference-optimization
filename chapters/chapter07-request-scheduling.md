# 第7章: 请求调度策略

> **💰 成本影响** (基于行业数据)
> - **吞吐提升**: Continuous Batching 可将吞吐量提升 3-10 倍
> - **延迟改善**: P95 延迟可降低 50-70%
> - **GPU 利用率**: 从 30-40% 提升到 80-90%

## 简介

在第5章中,我们学习了 Continuous Batching 的基本原理——通过动态调整 batch,消除 padding 浪费,让 GPU 时刻满载。但如何高效实现 Continuous Batching?如何决定哪些请求可以一起处理?如何平衡延迟和吞吐量?

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

**学完本章,你将能够设计并优化自己的推理调度系统。**

---

## 7.1 调度的必要性

### 7.1.1 为什么需要调度

**场景**: 多个用户同时发送推理请求

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

**没有调度器的问题**:
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

**调度器的价值**:
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

**服务质量 (Quality of Service, QoS)**:
- **延迟**: TTFT (首字延迟)、TBT (字间延迟)
- **公平性**: 所有请求都能及时处理
- **可靠性**: 请求不超时、不丢失

**吞吐量 (Throughput)**:
- 单位时间内处理的请求数
- 单位时间内生成的 tokens 数

**权衡曲线**:
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

**调度器的目标**: 找到最佳平衡点

---

### 7.1.3 调度器的目标

**主要目标**:
1. ✅ **最小化延迟**: P50、P95、P99 延迟尽可能低
2. ✅ **最大化吞吐量**: 在给定硬件上服务更多用户
3. ✅ **公平性**: 避免长请求饿死短请求
4. ✅ **资源利用**: GPU 利用率 >80%

**次要目标**:
- 简单性: 易于理解和调试
- 可扩展性: 支持分布式部署
- 鲁棒性: 容忍异常情况

**设计原则**:
```
优先级 1: 不超时 (SLA)
优先级 2: 低延迟 (用户体验)
优先级 3: 高吞吐 (成本效率)
优先级 4: 简单可靠 (运维成本)
```

---

## 7.2 基础调度策略

### 7.2.1 FIFO (First In First Out)

**原理**: 按请求到达顺序处理

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

**优点**:
- ✅ 实现简单
- ✅ 公平 (先来先服务)
- ✅ 无饥饿 (每个请求最终都会被处理)

**缺点**:
- ❌ 吞吐量低 (一次只处理一个请求)
- ❌ GPU 利用率低 (~30-40%)
- ❌ 长请求阻塞后续所有请求

**适用场景**:
- 单用户环境
- 低并发场景
- 对公平性要求高的场景

---

### 7.2.2 静态批处理 (Static Batching)

**原理**: 将多个请求打包成一个固定大小的 batch

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

**Padding 的问题**:
```
Batch 中的请求长度不一致:
Request A: 10 tokens
Request B: 50 tokens
Request C: 20 tokens

需要 padding 到最长:
Padded A: [pad×40][10 tokens]
Padded B: [50 tokens]
Padded C: [pad×30][20 tokens]

浪费: (40 + 0 + 30) / 100 = 70% padding!
```

**优点**:
- ✅ 提高吞吐量 (相比 FIFO)
- ✅ GPU 利用率提升 (~60-70%)

**缺点**:
- ❌ 大量 padding 浪费
- ❌ 短请求被长请求阻塞
- ❌ 无法动态调整

**适用场景**:
- 请求长度相近的场景
- 对延迟不敏感的离线批处理

---

### 7.2.3 优缺点分析

| 策略 | 吞吐量 | 延迟 | GPU 利用率 | 实现复杂度 | 适用场景 |
|------|-------|------|-----------|-----------|---------|
| **FIFO** | 低 | 最低 (单请求) | 30-40% | 简单 | 低并发 |
| **Static Batching** | 中 | 高 (等待 batch) | 60-70% | 简单 | 离线批处理 |
| **Continuous Batching** | 高 | 低 | 80-95% | 中等 | 生产环境 |

**结论**: Continuous Batching 是生产环境的最佳选择

---

## 7.3 动态批处理 (Continuous Batching)

### 7.3.1 问题: 静态批处理的浪费

**场景回顾**:
```
Batch 中有 3 个请求:
Request A: 10 tokens (已生成 100,还需 50)
Request B: 50 tokens (已生成 200,还需 10)
Request C: 20 tokens (已生成 150,还需 30)

问题 1: Request B 完成后
  → Batch 中还有 A 和 C
  → B 的位置空着 (padding 浪费)

问题 2: 新请求 D 到达
  → 必须等待整个 batch 完成
  → 延迟高

问题 3: Batch 中请求长度差异大
  → 大量 padding
  → GPU 计算浪费
```

**浪费量化**:
```
假设 batch_size = 8,每个请求平均生成 100 tokens

Static Batching:
- 需要等待所有 8 个请求完成
- P95 延迟 = 最慢请求的完成时间
- Padding 比例 = 50-70%
```

---

### 7.3.2 Continuous Batching 原理

**核心思想**:
1. **去掉 batch 维度**,用 attention mask 控制 token 交互
2. **动态替换完成的请求**,立即加入新请求
3. **混合 Prefill 和 Decode**,最大化 GPU 利用率

**Ragged Batching**:
```python
# 拼接所有请求的 tokens
tokens = [
    # Request A (3 tokens)
    A1, A2, A3,
    # Request B (2 tokens)
    B1, B2,
    # Request C (4 tokens)
    C1, C2, C3, C4,
]

# Attention mask: 块对角矩阵
mask = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # A1
    [1, 1, 0, 0, 0, 0, 0, 0, 0],  # A2
    [1, 1, 1, 0, 0, 0, 0, 0, 0],  # A3
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # B1
    [0, 0, 0, 1, 1, 0, 0, 0, 0],  # B2
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # C1
    [0, 0, 0, 0, 0, 1, 1, 0, 0],  # C2
    [0, 0, 0, 0, 0, 1, 1, 1, 0],  # C3
    [0, 0, 0, 0, 0, 1, 1, 1, 1],  # C4
]
```

**动态替换**:
```python
def continuous_batching_step(scheduled, running, completed):
    """
    scheduled: 等待调度的请求
    running: 正在运行的请求
    completed: 刚完成的请求
    """
    # 1. 移除完成的请求
    for req in completed:
        running.remove(req)

    # 2. 从等待队列中加入新请求
    num_slots = batch_size - len(running)
    for i in range(num_slots):
        if scheduled:
            new_req = scheduled.pop(0)
            running.append(new_req)

    # 3. 重新构建 attention mask
    mask = build_ragged_mask(running)

    return running, mask
```

---

### 7.3.3 图解工作流程

**迭代级调度 (Iteration-level Scheduling)**:

```
时间线 (每次迭代 ~10ms):

Iter 1:
  Running: [Req A (100→101), Req B (50→51), Req C (200→201)]
  GPU: 处理 3 个请求,各生成 1 个 token

Iter 2:
  Running: [Req A (101→102), Req B (52→53), Req C (201→202)]
  GPU: 处理 3 个请求,各生成 1 个 token

Iter 3:
  Req B 完成 (生成 <eos>)
  Running: [Req A (102→103), Req C (202→203)]
  空出 1 个 slot
  Scheduled: [Req D (新请求,需要 prefill)]
  Action: 用 Req D 替换 Req B
  Running: [Req A (103→104), Req C (203→204), Req D (prefill→1)]
  GPU: 处理 3 个请求 (decode + decode + prefill)

Iter 4:
  Running: [Req A (104→105), Req C (204→205), Req D (1→2)]
  GPU: 处理 3 个请求

Iter 5:
  Req A 和 Req D 同时完成
  Running: [Req C (205→206)]
  空出 2 个 slots
  Scheduled: [Req E, Req F]
  Action: 加入 Req E 和 Req F
  Running: [Req C (206→207), Req E (prefill→1), Req F (prefill→1)]
  GPU: 处理 3 个请求
```

**关键观察**:
- GPU 时刻保持满载 (3 个请求)
- 完成的请求立即被替换
- Prefill 和 Decode 混合处理
- 无 padding 浪费

---

### 7.3.4 性能提升分析

**吞吐量提升**:
```
假设:
- GPU 每次迭代可处理 1024 tokens
- 平均请求长度: 100 tokens

Static Batching:
- Batch size: 8
- 每个迭代: 8 个请求 (各 1 token)
- 每 100 个迭代完成一批 (8 个请求)
- 吞吐量: 8 requests / 100 iterations = 0.08 req/iter

Continuous Batching:
- 迭代 1-10: Req A-D (prefill)
- 迭代 11-20: Req E-H (decode)
- 迭代 21: Req A 完成,加入 Req I (prefill)
- 迭代 22-30: Req E-I (decode)
- ...
- 吞吐量: ~0.25 req/iter (3x 提升!)
```

**延迟改善**:
```
假设:
- 100 个请求排队
- Batch size: 8

Static Batching:
- 第 100 个请求需要等待 12 个 batch
- P95 延迟: 12 × 100 iterations = 1200 iterations

Continuous Batching:
- 第 100 个请求等待 ~3 个 batch
- P95 延迟: 3 × 100 iterations = 300 iterations

改善: 1200 / 300 = 4x
```

**GPU 利用率**:
```
Static Batching:
  - Padding: 50-70%
  - GPU 利用率: 30-50%

Continuous Batching:
  - Padding: 0-5%
  - GPU 利用率: 80-95%
```

---

## 7.4 vLLM 的调度器实现

### 7.4.1 请求生命周期管理

**状态机**:
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

**vLLM 的请求对象**:
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

### 7.4.2 预分配 vs 动态分配

**预分配 (Pre-allocation)**:
```python
# 传统方法: 预分配最大空间
def allocate_max_space(request):
    max_tokens = request.max_new_tokens
    prompt_tokens = len(request.prompt_tokens)
    total = prompt_tokens + max_tokens
    # 预分配 total tokens 的空间
    return allocate_blocks(total)
```

**问题**:
- 浪费显存 (大多数请求不会达到 max_new_tokens)
- 限制并发数

**动态分配 (Dynamic Allocation)**:
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

**优势**:
- 节省显存 (30-50%)
- 提高并发数
- 支持 max_new_tokens 很大的场景

---

### 7.4.3 迭代级调度 (Iteration-level Scheduling)

**定义**: 每次迭代 (iteration) 重新调度一次

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

**调度流程**:
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

### 7.4.4 Overlap Scheduling (Mini-SGLang) ⚡️ 2025 新增

> **💡 深度来源**: [Mini-SGLang Blog](https://lmsys.org/blog/2025-12-17-minisgl/) + [Berkeley EECS-2025-192](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf)
>
> **核心问题**: Berkeley 论文指出 CPU overhead 导致 GPU 闲置 → Overlap Scheduling 是解决方案
>
> **性能提升**: 消除 GPU stalls,提升吞吐量 20-30%

---

#### 7.4.4.1 CPU 开销导致 GPU 闲置问题

**Berkeley EECS-2025-192 的发现**:
- CPU 开销占推理时间的 **10-20%**
- 主要来源:
  - Kernel launch (启动 GPU kernel)
  - Memory copy (CPU↔GPU 数据传输)
  - Synchronization (等待 GPU 完成)
  - Batch scheduling (决定哪些请求一起处理)

**问题**:
- vLLM 的迭代级调度是 **串行** 的:
  ```
  Step 1: CPU 调度下一批请求
  Step 2: CPU 准备输入数据
  Step 3: CPU 启动 GPU kernel
  Step 4: GPU 计算 (此时 CPU 闲置!)
  Step 5: CPU 等待 GPU 完成
  Step 6: 回到 Step 1
  ```
- 结果: **GPU 利用率低**,有明显的 GPU stalls

**Nsight Systems 分析** (无 overlap):
```
Timeline:
CPU: |--Schedule1--|--Prepare2--|--Launch3--|
GPU:              |<--Compute1-->|    stalls    |
```
看到 GPU 有明显的闲置期 (stalls)

---

#### 7.4.4.2 Overlap Scheduling 设计思想

**核心思想**:
- **CPU-GPU 并行执行**:
  - CPU 准备下一批请求时,GPU 正在计算当前批次
  - GPU 计算完成后,下一批请求已经 ready,立即开始
- **生产者-消费者模式**:
  - CPU: 生产者 (准备 batches)
  - GPU: 消费者 (执行 batches)

**对比**:
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

**架构设计**:
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

**关键优化**:
1. **Pipeline 深度**: 通常 2-3 个 batches 的 pipeline
2. **同步机制**: 使用条件变量避免 busy waiting
3. **内存管理**: 预分配 buffers 避免运行时分配

---

#### 7.4.4.4 性能提升

**吞吐量提升**:
```
无 Overlap:
- CPU 开销: 15%
- GPU stalls: 10%
- 有效计算: 75%
- 吞吐量: 100 req/s

有 Overlap:
- CPU 开销: 5% (并行化)
- GPU stalls: 0% (无闲置)
- 有效计算: 95%
- 吞吐量: 126 req/s (1.26x 提升)
```

**延迟改善**:
```
P95 延迟降低 20-30%
- CPU 准备时间不阻塞 GPU
- 请求更快开始处理
```

---

#### 7.4.4.5 vLLM 的实现状态

**当前状态** (v0.6.x):
- ✅ 支持 iteration-level scheduling
- ⚠️ 部分支持 overlap (experimental)
- 🚧 未来版本会完全支持

**如何启用** (实验性):
```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_overlap_schedule=True,  # 实验性功能
)
```

---

### 7.4.5 Dynamic Memory Management (动态内存管理)

> **💡 来源**: SGLang v0.2 核心特性
>
> **问题**: 预留 max_new_tokens 的空间浪费大量内存
>
> **解决**: 根据实际使用情况动态调整预留大小

**核心问题**:
```python
# 用户设置
max_new_tokens = 2048

# 传统做法: 预留 2048 tokens 的空间
reserved = 2048

# 实际情况: 大多数请求只生成 500 tokens
actual = 500

# 浪费: 2048 - 500 = 1548 tokens (75% 浪费!)
```

**Dynamic Memory Management**:
```python
class DynamicMemoryManager:
    def __init__(self, initial_beta=0.5):
        """
        beta: 预留比例
              初始: 0.5 (预留 50% 的 max_new_tokens)
        """
        self.beta = initial_beta
        self.actual_usage_history = []

    def allocate(self, prompt_len, max_new_tokens):
        """动态分配内存"""
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
            'memory_saved_pct': (1 - self.beta) * 100
        }
```

**工作流程**:
```
请求到来时:
  1. 用户请求: prompt=1000 tokens, max_new_tokens=2048

  2. 传统做法:
     预留: 1000 + 2048 = 3048 tokens 的 KV Cache

  3. Dynamic Memory Management:
     预留: 1000 + (0.5 × 2048) = 1000 + 1024 = 2024 tokens
     (β=0.5,节省 33% 内存)

请求进行中:
  1. 请求已生成 600 tokens
  2. 发现即将到达 max_new_tokens 的 30%
  3. 动态扩展预留: 1024 → 1433 tokens
  4. 如果 GPU 内存不足,等待其他请求完成

请求完成时:
  1. 请求在 600 tokens 时遇到 EOS
  2. 释放所有 KV Cache (1000 + 600 = 1600 tokens)
  3. 记录实际使用率: 600 / 2048 = 29.3%
  4. 更新 β: 0.5 → 0.35 (根据历史平均)
  5. 下次请求只预留: 1000 + (0.35 × 2048) = 1716 tokens
```

**性能提升**:
```
内存节省:
  场景        | 传统做法 | 动态管理 | 节省
  Chat (500)  | 3048     | 2024     | 33%
  RAG (800)   | 3048     | 2240     | 27%
  Code (1200) | 3048     | 2640     | 13%

吞吐量提升:
  更大的 batch size (因为内存节省)
  实测: 1.5-2x throughput 提升
```

---

## 7.5 高级调度策略

### 7.5.1 优先级调度

**原理**: 不同请求有不同优先级

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

**应用场景**:
- VIP 用户 vs 普通用户
- 付费用户 vs 免费用户
- 实时请求 vs 离线批处理

---

### 7.5.2 最短作业优先 (SJF)

**原理**: 优先处理预计完成时间最短的请求

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

**优势**:
- ✅ 降低平均延迟
- ✅ 提高吞吐量

**劣势**:
- ❌ 可能饿死长请求
- ❌ 需要准确估计请求长度

**改进**: Shortest Remaining Time First (SRTF)
- 动态重新评估
- 考虑已执行的时间

---

### 7.5.3 轮询调度 (Round Robin)

**原理**: 公平地轮转处理每个队列

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

**优势**:
- ✅ 绝对公平
- ✅ 无饥饿

**劣势**:
- ❌ 上下文切换开销
- ❌ 可能降低吞吐量

---

### 7.5.4 自适应调度

**原理**: 根据系统状态动态调整调度策略

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

**优势**:
- ✅ 适应不同工作负载
- ✅ 自动优化

**挑战**:
- ⚠️ 策略切换开销
- ⚠️ 参数调优复杂

---

## 7.6 实战配置

### 7.6.1 vLLM 调度参数调优

**关键参数**:
```bash
vllm serve meta-llama/Llama-2-7b-hf \
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

**调优建议**:
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

> **💡 2025 年技术趋势**: PD 分离在 2025 年从概念快速演进为生产标准。vLLM、SGLang 等主流框架都已支持,几乎所有厂商都在采用这种架构。

### 7.7.1 什么是 PD 分离

**Prefill 阶段**: 并行处理 prompt,计算密集
- 输入: 整个 prompt
- 计算: 矩阵乘法为主
- 特点: 计算密集,可以并行

**Decode 阶段**: 串行生成 token,内存带宽密集
- 输入: 每次一个新 token
- 计算: 内存读取为主
- 特点: 带宽密集,串行生成

**两种阶段的计算模式差异**:
```
Prefill:
  GPU 利用: 计算 90%, 带宽 10%
  瓶颈: 算力 (FLOPS)
  最优 GPU: H100 (高算力)

Decode:
  GPU 利用: 计算 30%, 带宽 70%
  瓶颈: 内存带宽
  最优 GPU: A100 (高带宽,低成本)
```

**为什么需要分离?**
- 同一个硬件无法同时优化两种模式
- 分离后可以针对性优化
- 资源利用率提升 2-3 倍

---

### 7.7.2 PD 分离的架构演进

**2025 年初**: 概念提出
- 学术论文发表
- 社区开始讨论

**2025 年中**: vLLM、SGLang 等社区合作实现
- vLLM 添加 PD 分离支持
- SGLang 推出 RadixAttention

**2025 年底**: 成为生产标准架构
- 几乎所有厂商都在采用
- 最佳实践逐步完善

**从概念到生产只用了一年**

---

### 7.7.3 PD 分离的技术优势

**异构部署**:
```
Prefill Worker: H100 (算力优化)
  - 高 FLOPS
  - 处理新请求的 Prefill

Decode Worker: A100 (带宽优化)
  - 高内存带宽
  - 处理 Decode 阶段
  - 成本更低
```

**资源隔离**:
```
无分离:
  长请求的 Prefill 阻塞短请求的 Decode
  → P99 延迟高

有分离:
  Prefill 和 Decode 独立调度
  → 长请求不影响短请求
```

**弹性扩展**:
```
高峰期:
  增加 Prefill Worker (新用户多)

稳定期:
  增加 Decode Worker (生成多)
```

**性能优化**:
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

**架构设计**:
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

**通信机制**: KV Cache 的传输
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

**调度策略**:
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

**RadixAttention**: 统一的注意力抽象
```python
class RadixAttention:
    def forward(self, query, key, value, state):
        # 自动检测是 Prefill 还是 Decode
        if state.is_prefill:
            return self._prefill_forward(query, key, value)
        else:
            return self._decode_forward(query, key, value)
```

**自动分离**: 无需手动配置
```bash
python -m sglang.launch_server \
  --model meta-llama/Llama-3-8B \
  --enable-pd-separation  # 自动启用
```

**生产经验**: 稳定性、性能监控
```
关键指标:
  - Prefill Worker: GPU 利用率 >80%
  - Decode Worker: 内存带宽利用率 >70%
  - KV Cache 传输: 延迟 <10ms

告警阈值:
  - Prefill 队列长度 >100: 考虑扩容
  - Decode 队列长度 >500: 考虑扩容
  - KV Cache 传输延迟 >50ms: 检查网络
```

---

### 7.7.6 PD 分离的挑战

**KV Cache 传输**:
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

**负载均衡**:
```
问题: Prefill 和 Decode 的速率匹配
  - Prefill 快: Decode 成为瓶颈
  - Decode 快: Prefill 成为瓶颈

解决方案:
  - 动态调整 Worker 数量
  - 自适应调度策略
  - 监控和自动扩缩容
```

**容错处理**:
```
问题: Worker 故障如何恢复
  - Prefill Worker 故障: 新请求无法处理
  - Decode Worker 故障: 正在生成的请求中断

解决方案:
  - 冗余部署 (多 Worker)
  - KV Cache 持久化
  - 自动故障转移
```

**复杂度增加**:
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

**案例 1: 单机 GPU 的 PD 分离**
```
硬件: 单机 4 × A100 40GB

部署:
  GPU 0-1: Prefill Worker (2 个)
  GPU 2-3: Decode Worker (2 个)

性能:
  吞吐量: 1.8x 提升 (相比无分离)
  P95 延迟: 降低 40%
```

**案例 2: 跨机器的 PD 分离部署**
```
硬件:
  机器 A: 4 × H100 (Prefill)
  机器 B: 8 × A100 (Decode)

网络: InfiniBand (100 Gbps)

性能:
  吞吐量: 2.5x 提升
  成本: 降低 30% (A100 比 H100 便宜)
```

**案例 3: 异构 GPU (H100 + H200) 的实践**
```
硬件:
  H100: Prefill (算力优化)
  H200: Decode (带宽优化,大内存)

性能:
  吞吐量: 3x 提升
  支持更长序列 (H200 141GB 内存)
```

---

## ✅ 章节检查清单

完成本章后,你应该能够:

- [ ] 解释为什么需要调度器
- [ ] 对比 FIFO、Static Batching、Continuous Batching
- [ ] 描述 Continuous Batching 的工作流程
- [ ] 理解 vLLM 的迭代级调度
- [ ] 解释 Overlap Scheduling 的原理和优势
- [ ] 配置 vLLM 的调度参数
- [ ] 针对不同场景选择合适的调度策略
- [ ] 理解 PD 分离的架构演进
- [ ] 设计 PD 分离的部署方案
- [ ] 评估 PD 分离的收益和挑战

---

## 📚 动手练习

**练习 7.1**: 对比静态批处理和动态批处理

场景:
- 8 个请求,长度分别为: [10, 50, 20, 100, 30, 15, 80, 25] tokens
- 假设每个请求都生成 100 tokens

任务:
1. 计算 Static Batching 的 padding 数量
2. 计算 Continuous Batching 的 padding 数量
3. 比较两种方法的 GPU 利用率

**练习 7.2**: 针对不同场景优化调度参数

场景:
- Chatbot: 100 个并发,平均 50 tokens,对延迟敏感
- RAG: 20 个并发,平均 2000 tokens prompt,对吞吐量敏感
- 批处理: 1000 个请求,离线任务,追求最快完成

任务:
1. 为每个场景设计调度策略
2. 选择合适的调度算法
3. 配置 vLLM 参数

**练习 7.3**: 使用 vLLM 部署 PD 分离架构 ⭐

任务:
1. 设计一个 PD 分离的部署方案
2. 选择合适的硬件配置
3. 编写 docker-compose.yml
4. 评估性能提升和成本

---

## 🎯 总结

**关键要点**:
- 调度器是推理系统的核心,决定性能上限
- Continuous Batching 通过动态调整,消除 padding 浪费
- Overlap Scheduling 通过 CPU-GPU 并行,消除 GPU stalls
- Dynamic Memory Management 通过动态分配,节省 30-50% 内存
- PD 分离是 2025 年的生产标准,带来 2-3x 性能提升
- 不同场景需要不同的调度策略

**下一章**: 第8章 量化技术——如何通过量化节省显存并提升推理速度。

---

**有问题?加入 [第7章 Discord 频道](https://discord.gg/TODO) 讨论!**
