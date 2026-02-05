# 第9章: 投机采样

> **💰 成本影响** (经验区间,需基准测试验证)
> - **速度提升**: 生成速度通常可提升,幅度依模型与实现而定
> - **成本降低**: 同样时间内输出增加,单位 token 成本通常下降
> - **适用场景**: 长文本生成 (文章、代码、报告) 等

## 简介

在前面的章节中,我们学习了 GPU 基础、KV Cache 优化、调度策略、量化技术等核心技术。这些技术主要关注**单次生成**的效率优化。

但自回归生成有一个根本限制:**只能逐个生成 token**。即使 GPU 很快,生成 100 个 tokens 也需要 100 次 forward pass。对于长文本生成 (文章、代码、报告),这成为了一个明显的瓶颈。

**投机采样 (Speculative Sampling)** 是打破这个限制的关键技术。它的核心思想是:**用一个小模型快速"猜测"后面的 tokens,然后用大模型"验证"这些猜测**。如果猜测正确,就节省了计算;如果猜测错误,就丢弃结果重新计算。

这种"投机-验证"的机制,在合适的模型组合与接受率下可显著提升生成速度,且精度损失通常可控。本章将深入讲解:
- 为什么自回归生成慢,投机采样如何解决
- 投机采样的原理和流程
- 草稿模型的选择策略
- Eagle 3 的实战应用 (需确认兼容性与版本支持)
- vLLM Speculators 的端到端训练

**学完本章,你将能够使用投机采样在合适负载下提升长文本生成速度。**

---

## 9.1 生成加速的基本思路

### 9.1.1 为什么自回归生成慢

**自回归生成的限制**:
```
每个 token 的生成都依赖于之前的 tokens:

Token 0:
  输入: [t0]
  输出: [t0, t1]

Token 1:
  输入: [t0, t1]
  输出: [t0, t1, t2]

Token 2:
  输入: [t0, t1, t2]
  输出: [t0, t1, t2, t3]

...

Token n:
  输入: [t0, t1, ..., tn]
  输出: [t0, t1, ..., tn, t(n+1)]

问题:
  - 每个 token 需要 1 次 forward pass
  - 无法并行 (因为依赖关系)
  - 生成 100 tokens = 100 次 forward pass
```

**性能瓶颈**:
```
单个 forward pass 的时间:
  - 取决于模型规模、硬件与实现
  - 生成越长,总延迟越明显
```

---

### 9.1.2 并行化生成的挑战

**为什么不能直接并行?**
```
错误的并行尝试:
  同时生成 token 1-10
  → 输出: [t0, t1', t2', ..., t10']

问题:
  - token 1' 依赖于 t0,但没有 t0 的上下文
  - token 2' 依赖于 t0, t1',但 t1' 可能错误
  - 错误会累积
  - 最终输出不连贯
```

**因果约束**:
```
Token i 只能依赖于 tokens 0, 1, ..., i-1
Token i 不能依赖于 tokens i+1, i+2, ...

这是 LLM 的基本设计
保证了输出的连贯性
```

---

### 9.1.3 投机执行的概念

**来自 CPU 的灵感**:
```
CPU 分支预测:
  CPU 遇到条件分支 (if-else)
  → 先"猜测"会走哪个分支
  → 提前执行猜测的分支
  → 验证猜测是否正确
  → 如果正确,节省时间
  → 如果错误,丢弃结果重新执行

关键洞察:
  - 猜测命中率通常较高,但依模型与任务而定
  - 即使有一定错误率,整体仍可能加速
```

**应用到 LLM 生成**:
```
传统方式:
  Token 0 → Token 1 → Token 2 → ... → Token 100
  (100 次 forward pass)

投机采样:
  Token 0 → [猜测 Token 1-10]
  → 验证 Token 1-10
  → 如果正确,节省了 9 次 forward pass
  → 如果错误,重新计算
```

---

## 9.2 投机采样原理

### 9.2.1 核心思想: 小模型先行

**两模型架构**:
```
主模型 (Main Model):
  - 大模型 (如 Llama-2-70B)
  - 准确但慢
  - 负责验证草稿

草稿模型 (Draft Model):
  - 小模型 (如 Llama-2-7B)
  - 快但不够准确
  - 负责生成草稿
```

**工作流程**:
```
步骤 1: 草稿模型快速生成
  输入: [t0]
  草稿模型: 生成 [t1, t2, ..., t10] (快速)
  时间: 1 次 forward pass (小模型)

步骤 2: 主模型并行验证
  输入: [t0, t1, t2, ..., t10]
  主模型: 并行验证所有 tokens (1 次 forward pass)
  输出: [t0, t1', t2', ..., t10']

步骤 3: 比对结果
  如果 t1' == t1, t2' == t2, ..., t10' == t10:
    → 全部正确! ✅
    → 接受草稿,节省时间
  否则:
    → 找到第一个不匹配的 token (如 t3)
    → 丢弃 t3-t10
    → 从 t2 重新开始
```

---

### 9.2.2 草稿模型 (Draft Model)

**草稿模型的作用**:
```
主模型和草稿模型的关系:
  主模型: "老师" - 准确但慢
  草稿模型: "助教" - 快速但不完美

草稿模型的要求:
  ✅ 速度快 (小模型,如 7B)
  ✅ 接受率尚可 (依任务而定)
  ✅ 与主模型架构相同或兼容
  ❌ 不需要完美 (可以接受一些错误)
```

**草稿模型的类型**:
```
类型 1: 小型号模型
  - Llama-2-7B 作为 Llama-2-70B 的草稿
  - 快速,但接受率较低

类型 2: 量化后的主模型
  - Llama-2-70B-INT4 作为 Llama-2-70B-FP16 的草稿
  - 快速,且架构完全相同

类型 3: 专门训练的草稿模型
  - 针对"预测主模型输出"优化
  - 接受率最高
  - 训练成本高
```

---

### 9.2.3 验证过程

**并行验证**:
```python
def verify_draft(main_model, prompt, draft_tokens):
    """
    主模型并行验证草稿 tokens

    Args:
        main_model: 主模型
        prompt: 输入 prompt (t0)
        draft_tokens: 草稿 tokens (t1, t2, ..., tn)

    Returns:
        verified_tokens: 验证通过的 tokens
        reject_position: 第一个不匹配的位置
    """
    # 拼接 prompt 和草稿 tokens
    full_input = [prompt] + draft_tokens

    # 主模型一次 forward pass
    outputs = main_model.generate(full_input)

    # 逐个比对
    for i in range(len(draft_tokens)):
        if outputs[i] == draft_tokens[i]:
            verified_tokens.append(draft_tokens[i])
        else:
            # 第 i 个 token 不匹配
            return verified_tokens, i

    # 全部匹配
    return verified_tokens, -1
```

**为什么可以并行验证?**
```
关键洞察: Attention 机制允许并行计算

传统生成:
  Token 0 → Token 1 → Token 2
  (串行,因为依赖关系)

投机采样验证:
  输入: [t0, t1, t2, ..., t10]
  → Attention 可以并行计算所有 tokens 的输出
  → 串行依赖关系体现在 attention mask 上
  → 一次 forward pass 验证所有 tokens
```

---

### 9.2.4 图解完整流程

```
时间线:

┌─────────────────────────────────────────────────┐
│ Step 1: 草稿模型生成                              │
├─────────────────────────────────────────────────┤
│ Input:  [t0]                                    │
│ Draft Model: 快速生成                         │
│ Output: [t0, t1, t2, t3, t4, t5, t6, t7, t8]    │
│ Time:   示意 (小模型,1 次 forward pass)        │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 2: 主模型验证                                │
├─────────────────────────────────────────────────┤
│ Input:  [t0, t1, t2, t3, t4, t5, t6, t7, t8]       │
│ Main Model: 并行验证                           │
│ Output: [t0, t1, t2, t3', t4', t5', t6', t7', t8']  │
│ Time:   示意 (大模型,1 次 forward pass)         │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 3: 比对结果                                  │
├─────────────────────────────────────────────────┤
│ Compare:                                         │
│   t1 == t1' ✅                                │
│   t2 == t2' ✅                                │
│   t3 == t3' ❌ (不匹配!)                       │
│ Result:                                         │
│   Accept: [t1, t2] (验证通过)                  │
│   Reject: [t3'-t8'] (丢弃)                      │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│ Step 4: 继续生成                                  │
├─────────────────────────────────────────────────┤
│ Current: [t0, t1, t2]                           │
│ Draft Model: 生成 [t3, t4, ..., t12]             │
│ Main Model: 验证                               │
│ ...                                             │
└─────────────────────────────────────────────────┘

总时间(示意):
  无投机采样: 随 token 数线性增长
  投机采样: 取决于草稿模型速度与接受率
```

---

## 9.3 投机采样变体

### 9.3.1 Speculative Decoding

**原始投机采样方法**:
```
核心特点:
  - 使用小模型或量化模型作为草稿
  - 固定长度的 speculation (如 4-8 tokens)
  - 一次 forward pass 验证

优点:
  ✅ 实现简单
  ✅ 开销小
  ✅ 适用性广

缺点:
  ❌ 草稿模型可能接受率低
  ❌ 固定长度不够灵活
  ❌ 失败时浪费计算
```

**适用场景**:
```
✅ 长文本生成 (文章、代码)
✅ 高吞吐量场景
⚠️ 需要选择合适的草稿模型
```

---

### 9.3.2 Assisted Decoding

**核心思想**: 辅助解码
```
与 Speculative Decoding 类似
但强调"辅助"而非"投机"

区别:
  - Speculative Decoding: 强调"猜测"
  - Assisted Decoding: 强调"辅助验证"
```

---

### 9.3.3 Lookahead Decoding

**前瞻解码**:
```
核心特点:
  - 使用多个小草稿模型
  - 每个草稿模型"预测"不同长度的 tokens
  - 选择最长的正确序列

示例:
  草稿模型 1: 预测 4 tokens
  草稿模型 2: 预测 8 tokens
  草稿模型 3: 预测 12 tokens

  验证结果:
    草稿 1: 4/4 正确 ✅
    草稿 2: 6/8 正确 ❌
    草稿 3: 10/12 正确 ❌

  接受: 草稿 1 的 4 tokens
  最远验证: 草稿 3 的前 10 个 tokens
```

**优势**:
```
✅ 多个草稿模型提高接受率
✅ 自适应 speculation 长度
❌ 计算复杂度高
```

---

### 9.3.4 Eagle 系列: Eagle、Eagle 2、Eagle 3 ⭐

**Eagle 3** (来源: NVIDIA Model Optimizer + SGLang):
```
基于投机采样的训练 checkpoint
使用 NVIDIA Model Optimizer 进行 QAT 训练
支持多种草稿模型策略
在 SGLang 中可直接使用
与 vLLM、SGLang 的集成
```

**Eagle 系列演进**:

**Eagle (初始版本)**:
```
特点:
  - 基础投机采样
  - 单层草稿模型
  - 固定 speculation 长度

性能(示意):
  - 加速比: 依接受率与实现而定
  - 接受率: 依任务与模型而定
```

**Eagle 2 (改进版本)**:
```
特点:
  - 改进训练策略
  - 更好的 acceptance rate
  - 支持 multi-layer 草稿

性能(示意):
  - 加速比: 依接受率与实现而定
  - 接受率: 依任务与模型而定
```

**Eagle 3 (常见选择)**:
```
特点:
  - QAT 训练优化
  - 支持更多主模型
  - SGLang 深度集成

性能(示意):
  - 加速比: 依接受率与实现而定
  - 接受率: 依任务与模型而定
  - 稳定性: 依训练与部署情况而定
```

---

### 9.3.5 方法对比

| 方法 | 草稿模型 | 加速比 | 接受率 | 复杂度 | 推荐度 |
|------|---------|-------|--------|--------|--------|
| **Speculative Decoding** | 小型/量化 | 依实现而定 | 依任务而定 | 中 | ⭐⭐⭐⭐ |
| **Assisted Decoding** | 小型/量化 | 依实现而定 | 依任务而定 | 中 | ⭐⭐⭐⭐ |
| **Lookahead Decoding** | 多个小型 | 依实现而定 | 依任务而定 | 高 | ⭐⭐⭐ |
| **Eagle 3** | QAT 训练 | 依实现而定 | 依任务而定 | 中 | ⭐⭐⭐⭐⭐ |

---

### 9.3.6 如何选择合适的变体

**决策树**:
```
问题 1: 你有 Eagle 3 checkpoint 吗?
  是 → 可优先试用 Eagle 3
  否 → 问题 2

问题 2: 你有兼容的小模型吗?
  是 → Speculative Decoding
  否 → 问题 3

问题 3: 你愿意训练自定义草稿模型吗?
  是 → Eagle 3 (vLLM Speculators)
  否 → Lookahead Decoding 或 Assisted Decoding

问题 4: 你的场景是什么?
  长文本生成 → Eagle 3 或 Speculative Decoding
  代码生成 → Eagle 3 或定制草稿模型
  低延迟需求 → 不推荐投机采样
```

---

## 9.4 草稿模型选择

### 9.4.1 小型号模型

**示例**:
```
主模型: Llama-2-70B
草稿模型: Llama-2-7B

优点:
  ✅ 速度快 (小模型 forward pass 快)
  ✅ 架构兼容 (都是 Llama 架构)
  ✅ 无需训练

缺点:
  ❌ 接受率可能较低
  ❌ 草稿质量差
  ❌ 加速比有限
```

**适用场景**:
```
✅ 快速原型验证
✅ 没有专门训练的草稿模型
❌ 追求极致性能
```

---

### 9.4.2 量化后的主模型

**示例**:
```
主模型: Llama-2-70B-FP16
草稿模型: Llama-2-70B-INT4

优点:
  ✅ 架构完全相同 (量化不改变架构)
  ✅ 接受率通常更高
  ✅ 无需训练

缺点:
  ❌ 量化后仍然可能较慢
  ❌ 显存占用仍较大

适用场景:
✅ 有足够显存
✅ 追求较好接受率
❌ 显存受限 (考虑小模型)
```

---

### 9.4.3 专门训练的草稿模型

**示例**:
```
主模型: Llama-2-70B
草稿模型: 专门训练的 7B 模型

训练目标:
  预测主模型的输出
  优化 acceptance rate
  快速推理

优点:
  ✅ 接受率通常最高
  ✅ 推理速度快
  ✅ 针对性优化

缺点:
  ❌ 训练成本高
  ❌ 实现复杂度高
  ❌ 需要大量训练数据
```

**训练流程**:
```
步骤 1: 收集训练数据
  - 使用主模型生成大量 outputs
  - 收集 (prompt, output) 对

步骤 2: 训练草稿模型
  - 输入: prompt
  - 目标: 主模型的 output
  - 损失函数: Cross-entropy
  - 优化: AdamW

步骤 3: 验证
  - 测试 acceptance rate
  - 调优草稿模型
```

---

### 9.4.4 选择标准

| 标准 | 小型模型 | 量化模型 | 专门训练 |
|------|---------|---------|---------|
| **训练成本** | 无 | 无 | 高 |
| **推理速度** | 快 | 中 | 快 |
| **显存占用** | 小 | 大 | 小 |
| **接受率** | 中 | 中~高 | 高 |
| **实现复杂度** | 低 | 低 | 高 |
| **加速比** | 依实现而定 | 依实现而定 | 依实现而定 |
| **推荐度** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 9.5 性能分析

### 9.5.1 理论加速比

**理想情况(示意)**:
```
无投机采样:
  总时间随 token 数线性增长

投机采样:
  取决于 speculation_len、接受率与草稿模型开销
  接受率越高,越可能加速
```

**关键洞察**:
```
加速比随 speculation_len 与接受率增加而上升,
但会受到草稿模型开销与验证成本的抵消。
```

---

### 9.5.2 实际加速比影响因素

**关键因素**:

**1. Acceptance Rate (接受率)**:
```
Acceptance Rate = 验证通过的 tokens / 总投机 tokens

影响:
  - 接受率越高,加速潜力越大
  - 接受率过低,可能不划算

影响因素:
  - 草稿模型质量
  - 任务难度
  - 训练数据匹配度
```

**2. Speculation Length (投机长度)**:
```
投机长度: 每次投机多少 tokens

权衡:
  - 太短: 加速不明显
  - 太长: 失败风险高,浪费计算
  - 需要结合接受率与硬件约束调优
```

**3. 草稿模型速度**:
```
草稿模型越快 → 投机开销越小

理想情况:
  草稿模型时间 << 主模型时间
  投机开销可忽略
```

**4. 验证开销**:
```
验证并行化程度:
  - GPU 内存限制: 无法验证太长的序列
  - 并行度: 单次验证的 tokens 数量

实际考虑:
  - 投机长度受 GPU 内存限制
  - 需通过基准测试确定合理长度
```

---

### 9.5.3 什么时候投机采样有效

**最佳场景**:
```
✅ 长文本生成
   - 文章生成
   - 代码生成
   - 报告生成

✅ 高吞吐场景
   - 批量处理多个请求
   - 离线批处理

✅ 简单到中等难度任务
   - 故事续写
   - 代码补全
   - 翻译

✅ 草稿模型训练数据匹配
   - 草稿模型见过类似的输入
   - 领域对齐 (如代码生成模型用于代码生成)
```

**性能数据(示意)**:
```
长文本生成:
  通常收益更明显

短文本对话:
  收益可能有限
```

---

### 9.5.4 什么时候会失败

**失败场景**:
```
❌ 短文本生成
  投机开销 > 收益
  示例: 短文本生成
  投机开销可能接近或超过收益

❌ 高难度任务
  草稿模型无法预测
  Acceptance rate 较低
  大量计算浪费

❌ 不匹配的领域
  草稿模型是通用模型,任务是专业领域 (如医学、法律)
  草稿模型质量差
  Acceptance rate 较低

❌ 资源受限
  GPU 显存不足以加载主模型 + 草稿模型
  无法并行验证

❌ 延迟敏感的实时应用
  需要稳定的延迟
  投机采样导致延迟抖动
```

**问题诊断**:
```
如果投机采样没有加速,检查:
  1. Acceptance rate 是否太低?
  2. Speculation length 是否合适?
  3. 草稿模型是否匹配任务?
  4. GPU 显存是否足够?
```

---

## 9.6 实战: vLLM 投机采样

### 9.6.1 配置投机采样

**vLLM 的支持情况**:
```
注意: vLLM 对投机采样的支持随版本演进

替代方案:
  - 使用支持投机采样的框架
  - 或采用外部草稿模型/验证器方案
```

**未来支持** (概念性示例):
```python
# 未来可能的 API (设计稿)
from vllm import LLM
from vllm.speculative import SpeculativeConfig

# 配置投机采样
config = SpeculativeSpeculativeDecoding(
    draft_model="meta-llama/Llama-2-7b-hf",
    speculation_len=8,
    verify_mode="parallel",
)

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_config=config,
)
```

---

### 9.6.2 选择合适的草稿模型

**选择标准**:
```
场景 1: 快速实验
  → 使用现成的小模型

场景 2: 生产环境 (追求性能)
  → 使用量化主模型
  → 或使用已验证的草稿模型 (如果有兼容 checkpoint)

场景 3: 自定义需求
  → 训练专门的草稿模型
```

**兼容性检查**:
```python
def check_compatibility(main_model, draft_model):
    """检查主模型和草稿模型的兼容性"""
    # 1. 架构兼容性
    assert main_model.arch == draft_model.arch, \
        "架构不兼容"

    # 2. Vocab size
    assert main_model.vocab_size == draft_model.vocab_size, \
        "Vocab size 不匹配"

    # 3. Max position embeddings
    assert main_model.max_pos >= draft_model.max_pos, \
        "Max position 不匹配"

    print("✅ 兼容性检查通过")
```

---

### 9.6.3 性能基准测试

**测试代码**:
```python
import time
from vllm import LLM, SamplingParams

def benchmark_speculative_decoding(llm, prompts, num_iterations=10):
    """测试投机采样性能"""
    latencies = []
    token_counts = []

    for i in range(num_iterations):
        start = time.time()

        # 生成 (with speculative decoding if available)
        outputs = llm.generate(prompts)

        end = time.time()

        # 统计
        latencies.append(end - start)
        token_counts.append(sum(len(o.outputs) for o in outputs))

    # 计算
    avg_latency = sum(latencies) / len(latencies)
    avg_tokens = sum(token_counts) / len(token_counts)
    throughput = avg_tokens / avg_latency

    print(f"Average Latency: {avg_latency:.3f}s")
    print(f"Average Tokens: {avg_tokens:.1f}")
    print(f"Throughput: {throughput:.2f} tokens/s")

    return {
        "avg_latency": avg_latency,
        "throughput": throughput,
    }
```

---

### 9.6.4 调优技巧

**技巧 1: 调整投机长度**:
```python
# 测试不同的投机长度
for spec_len in [4, 8, 16, 32]:
    config = SpeculativeConfig(
        speculation_len=spec_len,
    )
    stats = benchmark_speculative_decoding(llm, prompts, config)
    print(f"Spec Length {spec_len}: {stats['throughput']:.2f} tokens/s")

# 选择最佳长度
```

**技巧 2: 监控 Acceptance Rate**:
```python
def monitor_acceptance_rate(llm, prompts):
    """监控接受率"""
    total_speculated = 0
    total_accepted = 0

    for prompt in prompts:
        # 投机生成
        draft_tokens = draft_model.generate(prompt, n=8)
        # 验证
        verified, reject_pos = verify_draft(main_model, prompt, draft_tokens)

        total_speculated += len(draft_tokens)
        total_accepted += len(verified)

    acceptance_rate = total_accepted / total_speculated
    print(f"Acceptance Rate: {acceptance_rate:.2%}")

    return acceptance_rate
```

**技巧 3: 优化 Batch Size**:
```
更大的 batch size → 更好的 GPU 利用率
但更大的 batch → 可能降低 acceptance rate

建议:
  从 batch_size = 32 开始
  逐步调整找到最佳值
```

---

## 9.7 实战: Eagle 3 with SGLang ⚠️ 需确认兼容性

> **💡 工业界实践** (来源: 官方与社区分享)
>
> **核心洞察**: Eagle 3 提供了可直接使用的投机采样 checkpoint,在部分框架中可集成,实际性能提升需基准测试确认。

### 9.7.1 什么是 Eagle 3

**NVIDIA 官方训练**: 使用 NVIDIA Model Optimizer
**QAT 优化**: 量化感知训练提升精度
**即用型 checkpoint**: 无需自己训练草稿模型
**SGLang 原生支持**: 开箱即用
**性能保证**: 仍需结合自身负载验证

---

### 9.7.2 Eagle 3 vs 自训练草稿模型

| 维度 | Eagle 3 | 自训练草稿模型 |
|------|---------|---------------|
| **精度优势** | QAT 训练优化,接受率更高 | 需要精心设计 |
| **Numerical 稳定性** | 更好 (训练优化) | 可能不稳定 |
| **成本优势** | 无需自己训练草稿模型 | 训练时间和资源 |
| **维护优势** | 官方或社区支持 | 需要自己维护 |
| **灵活性** | 固定模型和配置 | 可调整训练参数 |
| **适用场景** | 生产环境、快速部署、追求稳定性 | 研究、自定义需求 |

---

### 9.7.3 在 SGLang 中使用 Eagle 3

**安装 SGLang**:
```bash
pip install "sglang[all]"
```

**下载 Eagle 3 checkpoint**:
```bash
# 从 Hugging Face 下载
# 支持的主模型: Llama、GPT 等系列
# 例如: Llama-3-70B-Eagle3
```

**配置 speculative decoding**:
```python
import sglang as sgl

# 配置 Eagle 3 作为草稿模型
model = sgl.launch_server(
    model_path="path/to/main/model",
    speculative_algorithm="Eagle",
    speculative_draft_model_path="path/to/eagle3",
    speculative_max_tokens=8,  # 投机长度
)
```

**性能调优**:
```python
# 调整 speculative_max_tokens
for max_tokens in [4, 8, 16, 32]:
    model = sgl.launch_server(
        model_path="path/to/main/model",
        speculative_algorithm="Eagle",
        speculative_draft_model_path="path/to/eagle3",
        speculative_max_tokens=max_tokens,
    )
    stats = benchmark(model)
    print(f"Max Tokens {max_tokens}: {stats['throughput']:.2f} tps")

# 监控 acceptance rate
acceptance_rate = monitor_acceptance_rate(model, test_prompts)
print(f"Acceptance Rate: {acceptance_rate:.2%}")

# 优化 batch size
for batch_size in [16, 32, 64]:
    model = sgl.launch_server(
        model_path="path/to/main/model",
        speculative_algorithm="Eagle",
        speculative_draft_model_path="path/to/eagle3",
        speculative_max_tokens=8,
        batch_size=batch_size,
    )
    stats = benchmark(model)
    print(f"Batch Size {batch_size}: {stats['throughput']:.2f} tps")
```

---

### 9.7.4 性能基准测试

**测试环境**:
- 需根据自身硬件与模型配置确定

**性能指标**:
```
建议关注:
  - 生成速度变化
  - Acceptance rate
  - TTFT / TBT 变化
  - Throughput 变化

不同场景表现:
  - 短文本: 可能收益有限
  - 长文本: 通常收益更明显
```

---

### 9.7.5 Eagle 3 的限制和注意事项

**模型支持**:
```
仅支持特定的主模型
需要检查兼容性列表
```

**硬件要求**:
```
需要足够显存同时加载主模型和草稿模型
硬件越强,可用的 speculation 长度与并行度越高
```

**适用场景**:
```
✅ 适合长文本生成
✅ 适合高吞吐场景
⚠️ 短文本收益有限
❌ 不适合延迟敏感的实时应用
```

---

### 9.7.6 Eagle 系列演进

**Eagle** (初始版本):
```
基础投机采样
单层草稿模型
```

**Eagle 2** (改进版本):
```
改进训练策略
更好的 acceptance rate
```

**Eagle 3** (当前最佳):
```
QAT 训练优化
支持更多主模型
SGLang 深度集成
```

**未来方向**:
```
支持更多模型架构
动态草稿长度
与其他优化技术结合 (如 PD 分离)
```

---

### 9.7.7 实战: vLLM Speculators - 端到端 Eagle 3 训练 ⭐💡

> **💡 技术趋势** (来源: 官方与社区资料)
>
> **核心洞察**: vLLM Speculators 提供端到端的草稿模型训练流程,覆盖离线数据生成、模型训练与推理部署。

**什么是 vLLM Speculators**:
```
vLLM 官方的投机采样训练库
支持端到端 Eagle 3 训练 pipeline
开源解决方案 (不同于 NVIDIA 的闭源 checkpoint)
与 vLLM 推理引擎无缝集成
```

**核心特性**:
```
1. Offline 数据生成:
   - 使用 vLLM 生成 hidden states
   - 支持大规模数据集生成
   - 智能 batch sampling 提升效率

2. 训练能力:
   - 单层草稿模型训练
   - 多层草稿模型训练
   - 支持 MoE 和 non-MoE verifiers
   - FlexAttention 高效 attention 计算

3. 模型支持:
   - Llama 系列: 3.1, 3.2, 3.3 (8B-70B)
   - Qwen3: 8B, 14B, 32B
   - Qwen3 MoE: 235B-A22B
   - GPT-OSS: 20B, 120B
```

**vs NVIDIA Eagle 3 对比**:
```
开源 vs 闭源:
  - vLLM Speculators: 完全开源,可自定义训练
  - NVIDIA Eagle 3: 官方 checkpoint,开箱即用

灵活性:
  - vLLM: 可调整训练参数和数据
  - NVIDIA: 固定模型和配置

适用场景:
  - vLLM: 研究、自定义需求、学习目的
  - NVIDIA: 生产环境、快速部署、追求稳定性
```

**完整训练流程**:

**步骤 1: 环境准备**:
```bash
pip install vllm-devtools  # 包含 speculators 训练工具
```

**步骤 2: 离线数据生成**:
```bash
python -m vllm.speculators.generate_hidden_states \
  --model-path meta-llama/Llama-3.1-8B \
  --dataset-path your_dataset.jsonl \
  --output-path hidden_states_output \
  --max-model-len 4096 \
  --batch-size 32
```

**步骤 3: 训练草稿模型**:
```bash
python -m vllm.speculators.train \
  --base-model-path meta-llama/Lama-3.1-8B \
  --hidden-states-path hidden_states_output \
  --output-path eagle3_draft_model \
  --num-layers 1 \
  --use-flex-attention
```

**步骤 4: 推理部署**:
```python
from vllm import LLM
from vllm.speculators import SpeculativeDecoder

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    speculative_decoder=SpeculativeDecoder(
        draft_model_path="eagle3_draft_model",
        max_tokens=8,
    ),
)

# 使用
outputs = llm.generate(prompts)
```

---

## ✅ 章节检查清单

完成本章后,你应该能够:

- [ ] 解释为什么自回归生成慢
- [ ] 描述投机采样的核心思想
- [ ] 对比不同的投机采样变体
- [ ] 选择合适的草稿模型
- [ ] 计算理论加速比
- [ ] 判断投机采样是否适合你的场景
- [ ] 使用 SGLang 部署 Eagle 3
- [ ] 使用 vLLM Speculators 训练草稿模型
- [ ] 调优投机采样性能

---

## 📚 动手练习

**练习 9.1**: 对比有无投机采样的性能

任务:
1. 使用 SGLang 测试 Llama-3-70B 的生成速度
2. 启用/禁用投机采样
3. 测量 TTFT、吞吐量、acceptance rate

**练习 9.2**: 调优投机采样参数

任务:
1. 测试不同的 speculation_len (4, 8, 16, 32)
2. 找到最佳的投机长度
3. 分析 acceptance rate 的变化

**练习 9.3**: 使用 vLLM Speculators 训练草稿模型 ⭐

任务:
1. 准备训练数据集
2. 使用 generate_hidden_states 生成数据
3. 使用 train 命令训练草稿模型
4. 验证草稿模型效果

---

## 🎯 总结

**关键要点**:
- 投机采样通过"猜测-验证"机制打破自回归生成的串行限制
- Eagle 3 是一种可用方案,需结合兼容性与基准测试评估
- 草稿模型的选择是性能的关键
- 长文本生成是投机采样的最佳场景
- vLLM Speculators 提供了端到端的训练支持

**下一章**: 第10章 生产环境部署——将推理系统部署到生产环境。

---

**有问题?加入 [第9章 Discord 频道](https://discord.gg/TODO) 讨论!**
