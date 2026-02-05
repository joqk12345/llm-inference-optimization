# 第8章: 量化技术

> **💰 成本影响** (经验区间,需基准测试验证)
> - **显存节省**: 权重量化通常可显著降低显存占用
> - **成本降低**: 同样模型可能在更小/更便宜的 GPU 上运行
> - **精度损失**: 现代量化技术精度损失通常可控
> - **硬件效率**: 低精度推理可能更快,但取决于硬件与内核实现
> - **极端压缩**: 更低 bit 的量化可大幅压缩,但需要更严格的验证

## 简介

在前面的章节中,我们学习了 GPU 基础、KV Cache 优化、调度策略等技术。但这些技术都有一个共同的限制:**模型太大,显存不够**。以 70B 级模型为例,其 FP16 权重占用量通常在百 GB 级别,常常超过单卡显存容量,从而推动量化与并行部署需求。

**量化 (Quantization)** 是解决这个问题的重要手段。通过降低模型权重的数值精度,我们可以:
- 将大模型权重占用压缩到原来的较小比例 (示意)
- 在单张 GPU 上运行更大的模型
- 在合适硬件上提升推理速度
- 降低硬件成本 (更小的 GPU 或更少的 GPU)

但量化也带来挑战: 精度损失、训练推理不一致、实现复杂度等。本章将深入讲解量化的原理、方法和实战技巧,帮助你安全地应用量化技术。

**学完本章,你将能够选择合适的量化方案,并在精度与性能之间做出可解释的权衡。**

---

## 8.1 量化基础

### 8.1.1 什么是量化

**定义**: 将高精度的数值表示转换为低精度表示

```
FP32 (32位浮点):
  符号 (1 bit) + 指数 (8 bits) + 尾数 (23 bits)
  表示范围: ±3.4×10³⁸
  精度: 约 7 位十进制数字

INT8 (8位整数):
  符号 (1 bit) + 数值 (7 bits)
  表示范围: -128 到 127
  精度: 整数
```

**直观理解**:
```
FP32 权重: [0.23456789, -1.2345678, 0.00001234]
                ↓ 量化
INT8 权重:  [0, -1, 0]  (损失了精度!)
```

**核心问题**: 如何在降低精度的同时,保持模型性能?

---

### 8.1.2 为什么量化能节省显存

**计算模型显存占用**:

```
模型总显存 = 模型权重 + KV Cache + 激活值 + 开销

模型权重 = 参数量 × 每参数字节数

Llama-2-7B (70亿参数,仅用于粗略估算):
  FP32: 70B × 4 bytes = 280 GB
  FP16: 70B × 2 bytes = 140 GB
  INT8: 70B × 1 byte  = 70 GB
  INT4: 70B × 0.5 byte= 35 GB
```

**量化效果(示意)**:
```
FP16 → INT8:
  权重占用: 140GB → 70GB (权重层面节省 50%)
  速度: 可能提升 (依硬件与内核实现)
  精度: 损失通常可控

FP16 → INT4:
  权重占用: 140GB → 35GB (权重层面节省 75%)
  速度: 可能提升
  精度: 需根据任务验证
```

**关键优势**:
- ✅ 显存占用减半或更多
- ✅ 推理速度提升
- ✅ 可以在更小的 GPU 上运行
- ✅ 降低硬件成本

---

### 8.1.3 精度 vs 性能的权衡

**权衡曲线**:
```
精度
  ↑
  │     ╱
  │    ╱  ← 量化导致精度下降
  │   ╱
  │  ╱
  │ ╱     ← 最佳平衡点
  │╱_______
  └────────────→ 性能 (显存、速度)
```

**不同量化的效果**:

| 格式 | 权重占用(相对) | 速度 | 精度损失 | 适用场景 |
|------|---------------|------|---------|---------|
| **FP32** | 高 | 慢 | 低 | 训练 |
| **FP16** | 中 | 中 | 低 | 推理标准 |
| **BF16** | 中 | 中 | 低 | 推理标准 |
| **INT8** | 低 | 通常更快 | 低~中 | 生产推理 |
| **INT4** | 很低 | 通常更快 | 中 | 极限压缩 |
| **FP8** | 低 | 通常更快 | 低~中 | 未来方向 |
| **FP4** | 很低 | 潜在更快 | 中~高 | 研究中 |

**选择原则**:
```
追求精度: FP16/BF16
追求性价比: INT8
追求极限压缩: INT4 (QAT)
未来方向: FP8/FP4
```

---

### 8.1.4 为什么量化有效: 模型的冗余性

**核心洞察**: 深度学习模型有大量冗余,量化不会破坏关键信息

**为什么冗余?**
1. **过参数化**: 模型参数远超需要
2. **分布式表示**: 信息分散在多个参数中
3. **鲁棒性**: 小的扰动不影响整体性能

**实验证据(概述)**:
```
多项研究与工程实践表明:
  - LLM 对一定程度的量化具有容忍性
  - INT8 往往可在较小精度损失下工作
  - INT4 的效果更依赖于算法与数据
```

**直观理解**:
```
神经网络: 不是所有参数都重要
  重要参数: 决定模型核心能力 (不能量化)
  冗余参数: 只贡献微小影响 (可以量化)

量化: 保留重要信息,丢弃冗余细节
  类比: 图像压缩 (JPEG 保留视觉关键信息)
```

---

## 8.2 量化方法分类

### 8.2.1 PTQ (Post-Training Quantization)

**定义**: 训练后量化,无需重新训练

**流程**:
```
1. 训练 FP32/FP16 模型
2. 收集校准数据 (Calibration Dataset)
3. 计算量化参数 (Scale、Zero Point)
4. 量化权重
5. (可选) 微调恢复精度
```

**常见方法**:
- **GPTQ**: Gradient-based Post-Training Quantization
- **AWQ**: Activation-aware Quantization
- **bitsandbytes**: 简单易用的 INT8 量化
- **SpQR**: 混合精度量化

**优点**:
- ✅ 快速 (几分钟到几小时)
- ✅ 无需完整训练周期
- ✅ 适合快速部署

**缺点**:
- ❌ 可能有一定精度损失
- ❌ 对极端值敏感
- ❌ 需要校准数据集

**适用场景**:
```
✅ 快速原型验证
✅ 不具备训练资源
✅ 模型已训练好,只需要部署
❌ 精度要求极高 (考虑 QAT)
```

---

### 8.2.2 QAT (Quantization-Aware Training) ⭐

**定义**: 量化感知训练,在训练时模拟量化

**核心思想**:
```
训练时:
  前向传播: Fake Quantization (模拟量化)
    → FP32 权重 → Fake Quant → INT8 模拟值 → 计算
  反向传播: STE (Straight-Through Estimator)
    → 梯度跳过量化步骤,直接传给 FP32 权重

推理时:
  导出 INT8 权重 → 直接 INT8 推理
```

**Fake Quantization 原理**:
```python
def fake_quantize(x, scale, zero_point):
    """
    模拟量化,但保持可微分性
    """
    # 前向: 模拟量化
    x_quant = torch.round(x / scale) + zero_point
    x_quant = torch.clamp(x_quant, 0, 255)

    # 反向: STE (直通估计器)
    # 梯度直接跳过 round 操作
    x_dequant = (x_quant - zero_point) * scale

    return x_dequant

# STE 实现
class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()  # 前向: 正常量化

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # 反向: 直接传递梯度
```

**优点**:
- ✅ 精度损失最小
- ✅ Train-Infer 一致性好
- ✅ 适合 RL 训练和高精度场景

**缺点**:
- ❌ 需要完整训练周期
- ❌ 计算成本高
- ❌ 实现复杂度高

**适用场景**:
```
✅ 需要最佳精度
✅ RL 训练 (需要 train-infer 一致)
✅ PTQ 精度损失不可接受
✅ 大规模模型 (100B+ 参数)
❌ 只需要推理 (用 PTQ 更快)
```

---

### 8.2.3 QLoRA vs Native Quantized Training vs QAT

| 方法 | 目的 | 适用场景 | 优缺点 |
|------|------|---------|--------|
| **QLoRA** | 降低 LoRA 微调的训练内存 | 参数高效微调 | ✅ 节省训练内存<br>❌ 只用于微调,不用于推理 |
| **Native Quantized Training** | 端到端低精度训练 | 研究和新算法 | ✅ 极致显存节省<br>❌ 实现极复杂<br>❌ 稳定性差 |
| **QAT** | 改善量化推理精度 | 生产级量化部署 | ✅ 最佳精度<br>✅ Train-Infer 一致<br>❌ 需要完整训练周期 |

**关系图**:
```
训练阶段:
  Full Precision Training
    ↓
  LoRA Fine-tuning (QLoRA: 量化版)
    ↓
  QAT (训练时模拟量化)

推理阶段:
  PTQ (训练后直接量化)
  QAT (训练导出的量化模型)
```

---

### 8.2.4 量化方法选择决策树

**决策流程**:

```
问题 1: 你需要训练还是只需推理?

只需要推理:
  → 问题 2: 对精度要求多高?

  精度要求不高 (可接受 1-2% 损失):
    → PTQ (INT8/INT4)
    → 快速、简单

  精度要求高 (损失 <0.5%):
    → QAT (INT8)
    → 或保持 FP16

需要训练 (RL、微调):
  → 问题 3: 训练资源如何?

  资源有限:
    → QLoRA (量化版 LoRA)
    → 节省训练内存

  资源充足:
    → QAT (INT8/INT4)
    → 最佳精度和一致性

  极致压缩:
    → Native Quantized Training
    → 实验性,风险高
```

**场景推荐**:
```
场景 1: 快速部署 → PTQ
场景 2: 精度要求高 → QAT
场景 3: 需要微调 → QLoRA 或 QAT
场景 4: RL 训练 → QAT (必须保证一致性)
```

---

## 8.3 常用量化格式

### 8.3.1 FP32 (32位浮点) - 训练标准

**表示**:
```
1 bit  符号
8 bits 指数
23 bits 尾数

范围: ±3.4×10³⁸
精度: ~7 位十进制数字
```

**特点**:
- ✅ 精度最高
- ✅ 训练稳定
- ❌ 显存占用大 (280GB for 70B)
- ❌ 推理速度慢

**用途**: 模型训练

---

### 8.3.2 FP16/BF16 (16位浮点) - 推理常用

**FP16 (半精度浮点)**:
```
1 bit  符号
5 bits 指数
10 bits 尾数

范围: ±65504
精度: ~3 位十进制数字
```

**BF16 (Brain Float 16)**:
```
1 bit  符号
8 bits 指数 (与 FP32 相同)
7 bits 尾数

范围: ±3.4×10³⁸ (与 FP32 相同)
精度: ~2 位十进制数字
```

**对比**:
| 格式 | 范围 | 精度 | 稳定性 | 推荐度 |
|------|------|------|--------|--------|
| **FP16** | 小 | 高 | 一般 (可能下溢) | ⭐⭐⭐ |
| **BF16** | 大 | 中 | 好 (不易下溢) | ⭐⭐⭐⭐⭐ |

**推荐**: BF16 (范围与 FP32 相同,更稳定)

---

### 8.3.3 INT8 (8位整数) - 经典量化

**表示**:
```
有符号 INT8:
  范围: -128 到 127
  精度: 整数

无符号 UINT8:
  范围: 0 到 255
  精度: 整数
```

**量化公式**:
```python
# Affine 量化
Q = round(R / S) + Z
R = (Q - Z) * S

其中:
  R: 原始实数
  Q: 量化后的整数
  S: Scale (缩放因子)
  Z: Zero Point (零点偏移)
```

**优点**:
- ✅ 权重占用显著下降
- ✅ 推理速度可能提升
- ✅ 精度损失通常可控
- ✅ 硬件支持较好 (依平台而定)

**缺点**:
- ❌ 需要校准数据集
- ❌ 极端值处理

**推荐度**: ⭐⭐⭐⭐⭐ (生产环境标准)

---

### 8.3.4 INT4 (W4A16) ⭐

**表示**:
```
INT4 权重:
  范围: -8 到 7 (有符号)
  或: 0 到 15 (无符号)

FP16 激活:
  保持 16 位浮点
```

**为什么 W4A16?**
```
权重量化到 INT4:
  节省 75% 显存

激活保持 FP16:
  避免精度累积误差
  硬件实现简单
```

**优点**:
- ✅ 权重占用显著下降
- ✅ 速度可能提升
- ✅ 精度损失可控需验证
- ✅ 硬件支持逐步完善

**缺点**:
- ❌ 常需更强的校准或训练手段保证精度
- ❌ 实现复杂度高

**推荐度**: ⭐⭐⭐⭐ (极限压缩首选)

---

### 8.3.5 FP4 vs INT4

**FP4 (4位浮点)**:
```
2 bit 指数
2 bit 尾数

范围: ±6
精度: 极低
```

**对比**:

| 维度 | INT4 | FP4 |
|------|------|-----|
| **表示范围** | 窄 (-8 到 7) | 更宽 (±6 浮点) |
| **精度** | 整数 | 浮点 |
| **稳定性** | 高 | 低 |
| **性能** | 快 | 理论更快 |
| **硬件支持** | 广泛 | 依赖新一代硬件 |
| **生态** | 成熟 | 发展中 |
| **推荐度** | ⭐⭐⭐⭐ | ⭐⭐⭐ (未来) |

**选择建议**:
- **当前**: INT4 (生态成熟,稳定可靠)
- **未来**: FP4 (理论性能更高,需新一代硬件支持)

---

### 8.3.6 FP8 / NVFP4: 未来方向

**FP8 (8位浮点)**:
```
E4M3 (4 bit 指数, 3 bit 尾数):
  用于训练
  范围: ±448
  精度: 类似 FP16

E5M2 (5 bit 指数, 2 bit 尾数):
  用于推理
  范围: ±57344
  精度: 低于 E4M3
```

**NVFP4 (NVIDIA FP4)**:
```
新一代硬件支持
更优的硬件加速
与 Tensor Core 深度集成
```

**硬件支持**:
- 新一代 GPU 通常提供 FP8 支持
- FP4/FP8 的硬件支持仍在演进
- 早期架构可能仅能通过软件模拟

**性能潜力(示意)**:
```
FP8 vs FP16:
  权重占用更低
  速度可能提升
  精度需验证

FP4 vs INT4:
  理论速度可能更快
  精度与稳定性需验证
  生态仍在发展
```

---

### 8.3.7 AWQ / GPTQ: 流行的 INT4 格式

**AWQ (Activation-aware Quantization)**:
```
原理:
  基于激活值的重要性来量化权重
  重要的权重保持高精度

步骤:
  1. 收集激活值统计
  2. 计算权重重要性
  3. 非均匀量化 (重要权重精度高)

优点:
  ✅ 精度优于 GPTQ
  ✅ 速度快

缺点:
  ❌ 需要校准数据
  ❌ 实现复杂
```

**GPTQ (Gradient-based Post-Training Quantization)**:
```
原理:
  基于梯度的二阶信息
  迭代量化权重

步骤:
  1. 计算海森矩阵近似
  2. 迭代量化权重
  3. 最小化量化误差

优点:
  ✅ 不需要校准数据
  ✅ 精度好
  ✅ 开源工具成熟

缺点:
  ❌ 量化速度慢
  ❌ 内存占用高
```

**对比**:
| 特性 | AWQ | GPTQ |
|------|-----|------|
| **精度** | 更好 | 好 |
| **速度** | 快 | 慢 |
| **校准数据** | 需要 | 不需要 |
| **工具支持** | vLLM, AutoGPTQ | AutoGPTQ, llama.cpp |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

**推荐**:
- 生产环境: AWQ (更快、精度更好)
- 研究/离线: GPTQ (不需要校准数据)

---

## 8.4 流行的量化框架

### 8.4.1 vLLM 量化支持

**支持的格式**:
- ✅ AWQ (推荐)
- ✅ GPTQ
- ✅ bitsandbytes (INT8)
- ✅ FP8 (实验性)

**使用示例**:
```python
from vllm import LLM, SamplingParams

# AWQ 量化模型
llm = LLM(
    model="TheBloke/Llama-2-7b-AWQ",
    quantization="awq",
    max_model_len=4096,
)

# GPTQ 量化模型
llm = LLM(
    model="TheBloke/Llama-2-7b-GPTQ",
    quantization="gptq",
    max_model_len=4096,
)

# 推理
prompts = ["Hello, my name is", "The future of AI is"]
sampling_params = SamplingParams(temperature=0.8)
outputs = llm.generate(prompts, sampling_params)
```

**KV Cache 量化**:
```python
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="awq",
    kv_cache_dtype="int8",  # KV Cache 量化到 INT8
)
```

**PagedAttention + 量化**:
```
优势:
  ✅ 内存利用率高 (PagedAttention)
  ✅ 显存占用低 (量化)
  ✅ 两者协同,效果叠加
```

---

### 8.4.2 SGLang INT4 推理 ⭐

**Marlin 内核支持**:
```
Marlin: 专为 INT4 设计的高效推理内核
  - Bit packing: 8 个 INT4 值打包到 1 个 INT32
  - 高效解包: 位运算 (>> 4 和 & 0xF)
  - 计算和 IO 重叠: 解包近零开销
  - MoE 算子深度融合
```

**W4A16 高效推理**:
```python
# 启动 SGLang INT4 推理
python -m sglang.launch_server \
  --model-path /path/to/llama-2-7b-gptq \
  --quantization marlin \  # 使用 Marlin 内核
  --context-length 4096 \
  --tp 1 \  # Tensor parallelism
  --host 0.0.0.0 \
  --port 8000
```

**Bit Packing 原理**:
```python
def pack_int4(values):
    """
    将 8 个 INT4 值打包到 1 个 INT32

    输入: [v0, v1, v2, v3, v4, v5, v6, v7]
           每个 vi ∈ [-8, 7]

    输出: 1 个 INT32
    """
    packed = 0
    for i, v in enumerate(values):
        packed |= (v & 0xF) << (4 * i)
    return packed

def unpack_int4(packed):
    """
    从 1 个 INT32 解包 8 个 INT4 值
    """
    values = []
    for i in range(8):
        v = (packed >> (4 * i)) & 0xF
        # 转换为有符号 INT4
        if v >= 8:
            v -= 16
        values.append(v)
    return values
```

**MoE 算子深度融合**:
```python
# 动态调整 MoE block size
def dynamic_moe_align_block_size(block_size):
    """
    根据 GPU 架构动态调整 MoE block size
    - 优化内存访问模式
    - 减少 kernel 启动次数
    - Gating 部分融合为单一内核
    """
    # 根据具体架构与基准测试调整
    if gpu_arch == "high_end":
        return 64
    elif gpu_arch == "mid_range":
        return 128
    else:
        return 32
```

**性能 Benchmark(示意)**:
```
Llama-2-7B INT4 vs FP16:
  权重占用显著下降
  速度可能提升
  精度需基准测试评估
```

---

### 8.4.3 NVIDIA Model Optimizer ⭐

**QAT 训练支持**:
```python
import torch
import torch.nn as nn
from modelopt.torch import quantization as mtq

# 定义模型
model = MyModel()

# 配置量化
config = mtq.GPTQConfig(
    weights="int4",
    activations="int8",
)

# 量化模型
mtq.replace_quantizers(model, config)

# 训练 (模拟量化)
optimizer = torch.optim.AdamW(model.parameters())
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

**Megatron-LM 集成**:
```
大规模分布式训练:
  - Tensor Parallelism
  - Pipeline Parallelism
  - 量化感知训练
  - 混合精度 (FP8 + INT4)
```

**MXFP4 / NVFP4 格式支持**:
```python
# NVIDIA 原生 FP4 量化
from modelopt.torch.quantization import NVFP4Quantizer

quantizer = NVFP4Quantizer()
model = quantizer.quantize(model)
```

**Fake Quantization 实现**:
```python
class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, qmin, qmax):
        # 前向: 量化 + 反量化
        x_quant = x / scale + zero_point
        x_quant = torch.clamp(x_quant, qmin, qmax)
        x_quant = torch.round(x_quant)
        x_dequant = (x_quant - zero_point) * scale
        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        # STE: 梯度直接传递
        return grad_output, None, None, None
```

---

### 8.4.4 AutoGPTQ / llama.cpp

**AutoGPTQ**:
```python
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

# 加载量化模型
model = AutoGPTQForCausalLM.from_quantized(
    "TheBloke/Llama-2-7b-GPTQ",
    use_safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-7b-GPTQ")

# 推理
input_ids = tokenizer("Hello, world!", return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
```

**llama.cpp (CPU 推理)**:
```bash
# 量化模型
./llama-cli \
  --model /path/to/llama-2-7b.gguf \
  --prompt "Hello, world!" \
  --n-predict 100

# GGUF 格式支持:
# - Q2_K: 2-bit 量化
# - Q3_K: 3-bit 量化
# - Q4_K: 4-bit 量化 (最常用)
# - Q5_K: 5-bit 量化
# - Q8_0: 8-bit 量化
```

**对比**:
| 工具 | GPU 推理 | CPU 推理 | 量化格式 | 易用性 |
|------|---------|---------|---------|--------|
| **vLLM** | ✅ | ❌ | AWQ, GPTQ | ⭐⭐⭐⭐⭐ |
| **SGLang** | ✅ | ❌ | GPTQ (Marlin) | ⭐⭐⭐⭐ |
| **AutoGPTQ** | ✅ | ❌ | GPTQ | ⭐⭐⭐ |
| **llama.cpp** | ❌ | ✅ | GGUF | ⭐⭐⭐⭐ |

---

## 8.5 KV Cache 量化

### 8.5.1 为什么量化 KV Cache

**KV Cache 在长上下文中可能成为主要显存占用**:

```
Llama-2-7B (序列长度 4096,示意):
  模型权重: 约十余 GB
  KV Cache: 数 GB
  激活值: 数 GB
  ─────────────────────
  总计: 依实现与配置而定

Llama-2-7B (序列长度 32768,示意):
  KV Cache 占用可能显著增加
  ─────────────────────
  总计: 可能超过单卡显存容量
```

**长上下文场景尤其重要**:
```
序列长度越长,KV Cache 越大(示意):
  4K tokens:   数 GB
  8K tokens:   约翻倍
  16K tokens:  进一步增加
  32K tokens:  显著增加
  64K tokens:  可能成为主要占用
```

---

### 8.5.2 KV Cache 量化方法

**INT8 KV Cache**:
```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_cache_dtype="int8",  # KV Cache 量化到 INT8
)

# 显存节省(示意):
# FP16: 数 GB
# INT8: 约减半
```

**动态量化 vs 静态量化**:
```python
# 静态量化 (推荐):
kv_cache_dtype="int8"
# 预先计算好 scale 和 zero point
# 速度快,精度好

# 动态量化:
kv_cache_dtype="dynamic_int8"
# 每次计算时动态量化
# 更灵活,但略慢
```

**Per-token 量化**:
```python
# 每个 token 独立量化
# 精度更高,但开销更大

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_cache_dtype="int8_per_token",
)
```

---

### 8.5.3 精度与速度平衡

**精度损失评估**:
```python
import torch
from vllm import LLM

# FP16 基线
llm_fp16 = LLM(model="meta-llama/Llama-2-7b-hf")
output_fp16 = llm_fp16.generate(prompts)

# INT8 KV Cache
llm_int8 = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_cache_dtype="int8"
)
output_int8 = llm_int8.generate(prompts)

# 计算相似度
from sklearn.metrics import accuracy_score
similarity = accuracy_score(
    tokenize(output_fp16),
    tokenize(output_int8)
)
print(f"Similarity: {similarity:.4f}")  # > 0.98
```

**性能提升**:
```
显存:
  FP16: 19 GB
  INT8: 13 GB (节省 6GB)

吞吐量:
  FP16: 40 requests/s
  INT8: 45 requests/s (略快)
```

**生产环境注意事项**:
```
✅ 推荐:
  - 长序列 (>8K tokens)
  - 高并发场景
  - 显存紧张

⚠️ 谨慎:
  - 短序列 (<2K tokens)
  - 精度敏感任务
  - 需要极致性能

❌ 不推荐:
  - 序列长度 <1K (节省有限)
  - 精度要求极高
```

---

## 8.6 实战: 量化部署

### 8.6.1 使用 vLLM 加载量化模型

**AWQ/GPTQ 模型加载**:
```python
from vllm import LLM, SamplingParams

# AWQ 量化
llm_awq = LLM(
    model="TheBloke/Llama-2-7b-AWQ",
    quantization="awq",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

# GPTQ 量化
llm_gptq = LLM(
    model="TheBloke/Llama-2-7b-GPTQ",
    quantization="gptq",
    max_model_len=4096,
)

# 生成
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100,
)

prompts = ["Write a story about AI", "Explain quantum computing"]
outputs = llm_awk.generate(prompts, sampling_params)
```

**性能对比测试**:
```python
import time

def benchmark(llm, prompts, num_iterations=100):
    latencies = []
    throughputs = []

    for i in range(num_iterations):
        start = time.time()
        outputs = llm.generate(prompts)
        end = time.time()

        latencies.append(end - start)
        throughputs.append(len(prompts) / (end - start))

    avg_latency = sum(latencies) / len(latencies)
    avg_throughput = sum(throughputs) / len(throughputs)

    return {
        "avg_latency": avg_latency,
        "p95_latency": sorted(latencies)[int(len(latencies) * 0.95)],
        "avg_throughput": avg_throughput,
    }

# 对比
fp16_stats = benchmark(llm_fp16, prompts)
awq_stats = benchmark(llm_awq, prompts)

print(f"FP16:  {fp16_stats}")
print(f"AWQ:   {awq_stats}")
print(f"Speedup: {awq_stats['avg_throughput'] / fp16_stats['avg_throughput']:.2f}x")
```

**精度损失评估**:
```python
from datasets import load_dataset
from evaluate import load

# 加载评估数据集
dataset = load_dataset("truthfulqa", "validation")
metric = load("truthfulness")

# FP16 基线
outputs_fp16 = llm_fp16.generate(dataset["question"][:100])
score_fp16 = metric.compute(
    references=dataset["correct_answer"][:100],
    predictions=outputs_fp16
)

# AWQ 量化
outputs_awq = llm_awq.generate(dataset["question"][:100])
score_awq = metric.compute(
    references=dataset["correct_answer"][:100],
    predictions=outputs_awq
)

print(f"FP16 Accuracy: {score_fp16['accuracy']:.4f}")
print(f"AWQ  Accuracy:  {score_awq['accuracy']:.4f}")
print(f"Accuracy Drop: {score_fp16['accuracy'] - score_awq['accuracy']:.4f}")
```

---

### 8.6.2 使用 SGLang 部署 INT4 模型 ⭐

**W4A16 推理配置**:
```bash
# 安装 SGLang
pip install "sglang[all]"

# 启动 INT4 推理服务
python -m sglang.launch_server \
  --model-path TheBloke/Llama-2-7b-GPTQ \
  --quantization marlin \        # 使用 Marlin 内核
  --context-length 4096 \
  --tp 1 \                       # Tensor parallelism
  --host 0.0.0.0 \
  --port 8000

# 测试
curl http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "sampling_params": {
      "temperature": 0.8,
      "max_new_tokens": 100
    }
  }'
```

**Marlin 内核启用**:
```python
# SGLang 自动检测并启用 Marlin 内核
# 如果检测到 GPTQ 格式的 INT4 权重,自动使用 Marlin

# 验证是否使用 Marlin
import sglang as sgl

# 查看内核信息
print(sgl.kernels.get_active_kernel())
# 输出: "marlin_int4" ✅
```

**性能 Benchmark**:
```python
import time
import requests

def benchmark_sglang(num_requests=1000):
    latencies = []

    for i in range(num_requests):
        start = time.time()
        response = requests.post(
            "http://localhost:8000/generate",
            json={
                "text": "Write a short story about AI.",
                "sampling_params": {
                    "temperature": 0.8,
                    "max_new_tokens": 100,
                }
            }
        )
        end = time.time()

        latencies.append(end - start)

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
    throughput = num_requests / sum(latencies)

    return {
        "avg_latency": avg_latency,
        "p95_latency": p95_latency,
        "throughput": throughput,
    }

stats = benchmark_sglang()
print(f"Average Latency: {stats['avg_latency']:.3f}s")
print(f"P95 Latency: {stats['p95_latency']:.3f}s")
print(f"Throughput: {stats['throughput']:.2f} req/s")
```

---

### 8.6.3 生产环境注意事项

**模型格式选择**:
```
生产环境 (稳定性优先):
  → AWQ (精度更好,速度快)

实验/研究:
  → GPTQ (开源工具成熟)

极致压缩:
  → INT4 QAT (最佳精度)
```

**硬件要求**:
```
INT8:
  - 需要硬件支持 INT8 运算加速
  - 数据中心 GPU 与部分消费级 GPU 可用

INT4:
  - 需要专用内核支持 (如 Marlin 等)
  - 不同硬件支持程度不同

FP8:
  - 需要新一代硬件支持
  - 老硬件可能仅支持软件模拟
```

**监控指标**:
```python
# 关键指标
metrics = {
    # 显存
    "memory_used": get_gpu_memory_used(),
    "memory_fragmentation": get_fragmentation(),

    # 性能
    "throughput": get_throughput(),
    "p50_latency": get_p50_latency(),
    "p95_latency": get_p95_latency(),
    "p99_latency": get_p99_latency(),

    # 精度
    "perplexity": get_perplexity(),
    "accuracy": get_accuracy(),

    # 量化特有
    "quantization_error": get_quant_error(),
}

# 告警阈值
if metrics["memory_used"] > 0.95 * total_memory:
    print("⚠️  显存接近上限,考虑降低 batch size")

if metrics["quantization_error"] > 0.05:
    print("⚠️  量化误差过大,考虑 QAT")

if metrics["p95_latency"] > sla_target:
    print("⚠️  P95 延迟超标,考虑优化")
```

---

## 8.7 量化进阶: INT4 QAT 实战 ⚠️ 工程案例

> **💡 案例来源**: 社区与工程团队的公开实践分享
>
> **核心成果**: 通过 INT4 QAT 在保证可用精度的前提下显著压缩模型权重,降低部署复杂度

### 8.7.1 什么是 QAT

**Fake Quantization 原理**:
```python
class FakeInt4QuantizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        # 前向: 模拟 INT4 量化
        # 1. 归一化
        x_norm = x / scale

        # 2. 量化到 [-7, 7] (INT4 范围)
        x_quant = torch.clamp(torch.round(x_norm), -7, 7)

        # 3. 反量化
        x_dequant = x_quant * scale

        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        # STE: 梯度直接传递 (跳过 round 操作)
        return grad_output, None
```

**STE (Straight-Through Estimator) 原理**:
```
问题: round 操作不可导
  y = round(x)  # dy/dx = 0 (除了 0 点)

STE 解决方案:
  前向: y = round(x)
  反向: ∂L/∂x = ∂L/∂y (直接传递梯度)

直觉:
  梯度的期望是正确的
  虽然单个样本的梯度有误差
  但大量样本的平均梯度准确
```

**Train-Infer 一致性的重要性**:
```
训练时: Fake Quantization
  → 模型"看到"量化的噪声
  → 学习适应这种噪声

推理时: True Quantization
  → 实际 INT4 权重
  → 与训练时一致

如果不一致:
  训练时 FP16,推理时 INT4
  → 模型没见过量化噪声
  → 性能崩溃
```

**消融实验: QAT vs PTQ 的精度差异**:
```
PTQ (训练后量化):
  - 精度损失: 通常更明显
  - PPL 上升: 可能较高

QAT (量化感知训练):
  - 精度损失: 通常更小
  - PPL 上升: 通常更低

结论: 在精度敏感场景中,QAT 往往优于 PTQ
```

---

### 8.7.2 INT4 QAT 完整 Pipeline

**Stage 1: QAT 训练 (模拟量化)**:
```python
class QATModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.quantizer = FakeInt4QuantizationSTE()
        self.scales = {}  # 每层的 scale

    def forward(self, x):
        # 1. 普通 FP16 前向传播
        x = self.model.embed_tokens(x)

        # 2. 每层量化
        for layer_name, layer in self.model.layers.items():
            # 计算量化 scale (per-group max absolute value)
            weight = layer.weight
            scale = weight.abs().max() / 7
            self.scales[layer_name] = scale

            # Fake quantization
            weight_quant = self.quantizer(weight, scale)

            # 前向传播 (使用量化后的权重)
            x = layer(x, weight=weight_quant)

        return x

    def backward(self, loss):
        # 反向传播: STE 自动传递梯度
        loss.backward()
```

**Stage 2: 权重转换 (真量化)**:
```python
def convert_to_int4(model_fp16, scales):
    """
    将 FP16 权重转换为 INT4

    Args:
        model_fp16: BF16/FP16 权重 (训练后)
        scales: 每层的量化 scale

    Returns:
        model_int4: INT4 权重
    """
    model_int4 = {}

    for layer_name, layer in model_fp16.layers.items():
        weight = layer.weight
        scale = scales[layer_name]

        # 真量化
        weight_quant = torch.clamp(torch.round(weight / scale), -7, 7)
        weight_quant = weight_quant.char()  # 转换为 INT8 (INT4 打包)

        # 转换为 Marlin 格式
        weight_marlin = marlin_pack(weight_quant)

        model_int4[layer_name] = {
            'weight': weight_marlin,
            'scale': scale,
        }

    return model_int4
```

**Stage 3: W4A16 推理**:
```python
# SGLang 加载 INT4 权重
python -m sglang.launch_server \
  --model-path /path/to/int4_model \
  --quantization marlin \
  --context-length 4096

# 高效推理 (INT4 权重 × BF16 激活)
# 1. Bit packing: 8 个 INT4 → 1 个 INT32
# 2. 解包: 位运算 (>> 4 和 & 0xF)
# 3. 计算: INT4 权重 × BF16 激活
# 4. 生成: BF16 输出
```

---

### 8.7.3 训练端实现

**Fake Quantization 和 STE 实现**:
```python
class _FakeInt4QuantizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 动态量化: per-group max absolute value
        group_size = 128
        x = x.view(-1, group_size)

        # 计算 scale
        scale = x.abs().max(dim=1, keepdim=True) / 7

        # 模拟 INT4 的 [-7, 7] 范围
        x_quant = torch.clamp(torch.round(x / scale), -7, 7)

        # 记录 scale (用于后续真量化)
        ctx.save_for_backward(scale)

        # 返回反量化结果 (保持可微分)
        return (x_quant * scale).view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: 梯度直接传递
        scale, = ctx.saved_tensors
        return grad_output

def apply_fake_quantization(model):
    """对模型应用 fake quantization"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 替换为量化版本
            module.weight = _FakeInt4QuantizationSTE.apply(module.weight)
```

**权重更新和格式适配**:
```python
def restore_weights_before_loading(model):
    """
    在加载权重前恢复原始权重

    问题: PyTorch 加载权重的机制
    → 加载后修改权重可能失效

    解决: 使用 register_buffer
    """
    for name, module in model.named_modules():
        if hasattr(module, 'weight_fp16'):
            # 注册为 buffer (不会被 optimizer 更新)
            module.register_buffer('weight_fp16', module.weight_fp16)

def process_weights_after_loading(model):
    """
    加载权重后处理

    1. 恢复原始 FP16 权重
    2. 应用 fake quantization
    3. 转换为 Marlin 格式
    """
    for name, module in model.named_modules():
        if hasattr(module, 'weight_fp16'):
            # 恢复 FP16 权重
            weight = module.weight_fp16

            # 应用 fake quantization
            weight_quant = fake_quantize(weight)

            # 转换为 Marlin 格式
            weight_marlin = marlin_pack(weight_quant)

            # 保存
            module.register_buffer('weight_marlin', weight_marlin)
            del module.weight_fp16  # 释放显存
```

**消融实验: QAT 的必要性**:
```
实验 1: QAT INT4 训练 + BF16 rollout
  - 训练: Fake Quantization (INT4)
  - Rollout: BF16 权重
  - 结果: 误差仍高 (模型不适应量化)

实验 2: 不启用 QAT + 直接 INT4 rollout
  - 训练: BF16 权重
  - Rollout: 直接 INT4 权重
  - 结果: 误差震荡上升 (崩溃)

实验 3: QAT INT4 训练 + INT4 rollout (正确)
  - 训练: Fake Quantization (INT4)
  - Rollout: INT4 权重
  - 结果: 误差收敛,与 BF16 baseline 接近

**结论**: 训练和推理必须同时启用量化!
```

---

### 8.7.4 推理端实现

**SGLang W4A16 推理**:
```python
# Bit packing: 8 个 INT4 值打包到 1 个 INT32
def marlin_pack(int4_weights):
    """
    将 INT4 权重打包为 Marlin 格式

    Args:
        int4_weights: [N, M] INT4 张量

    Returns:
        packed: [N//2, M] INT32 张量 (8 个 INT4 → 1 个 INT32)
    """
    # 重排和打包
    int4_weights = int4_weights.reshape(-1, 16)  # 每 16 个一组
    packed = torch.zeros(int4_weights.shape[0], int4_weights.shape[1] // 4,
                        dtype=torch.int32, device=int4_weights.device)

    for i in range(16):
        packed[:, i//4] |= (int4_weights[:, i] & 0xF) << (4 * (i % 4))

    return packed

# 高效解包: 位运算
def marlin_unpack(packed):
    """
    从 Marlin 格式解包 INT4 权重

    Args:
        packed: [N//2, M] INT32 张量

    Returns:
        int4_weights: [N, M] INT4 张量
    """
    int4_weights = torch.zeros(packed.shape[0] * 2, packed.shape[1] * 4,
                                dtype=torch.int8, device=packed.device)

    for i in range(8):
        int4_weights[i::8] = ((packed[i//4, ::8] >> (4 * (i % 4))) & 0xF).to(torch.int8)
        if int4_weights[i::8].min() >= 8:
            int4_weights[i::8] -= 16  # 转换为有符号

    return int4_weights
```

**计算和 IO 重叠,解包近零开销**:
```
优化策略:
  1. 预取下一批权重 (Prefetch)
  2. 当前批计算时,异步解包下一批
  3. 使用 CUDA stream 并行化

效果(示意):
  - 解包开销通常可被隐藏
  - 具体比例依实现与硬件而定
```

**MoE 算子深度融合**:
```python
def dynamic_moe_align_block_size(num_experts):
    """
    动态调整 MoE block size

    目标:
    - 优化内存访问模式
    - 减少 kernel 启动次数
    - Gating 部分融合为单一内核
    """
    if num_experts <= 8:
        return 64
    elif num_experts <= 32:
        return 128
    else:
        return 256

# 融合 Gating 和 Expert kernel
def fused_moe_kernel(gate, experts, block_size):
    """
    融合 MoE 的 Gating 和 Expert 计算

    避免多次 kernel 启动:
    旧: 1 (gate) + N (experts) = N+1 次
    新: 1 (融合) = 1 次
    """
    # 单一 kernel 完成 gating 和 routing
    return torch.ops.fused_moe(gate, experts, block_size)
```

---

### 8.7.5 实战案例: 超大模型的量化压缩 (示意)

**案例 1: 超大 MoE 模型**:
```
配置:
  - 参数量: 百亿到千亿级 (MoE 架构)
  - 原始大小: 百 GB 以上 (BF16,示意)
  - 量化: INT4 QAT
  - 目标硬件: 单节点高显存 GPU

结果(示意):
  - 量化后大小显著下降
  - 可能需要模型并行或结合 KV Cache 量化

精度:
  - 训练趋势可与高精度基线接近
  - 需通过基准评估验证
```

**案例 2: 大模型推理部署**:
```
配置:
  - 参数量: 百亿级
  - 多节点部署 (原始)

双节点 (BF16):
  - 受限于跨节点带宽
  - 通信成为瓶颈
  - Rollout 效率低

单节点 (INT4 QAT):
  - 减少跨节点通信
  - 显存占用显著下降
  - Rollout 效率通常提升

性能对比(示意):
  - 精度: 依任务与校准策略而定
  - 速度: 依硬件与内核实现而定
  - 显存: 量化通常显著节省权重占用
```

---

### 8.7.6 QAT 的适用场景

**✅ 推荐**:
- 大规模 RL 训练 (100B+ 参数)
- 需要单节点部署超大模型
- 需要 train-infer 一致性
- PTQ 精度损失不可接受

**⚠️ 注意**:
- 训练成本较高 (需要完整微调周期)
- 实现复杂度较高 (需要理解 QAT、STE、格式转换)

**❌ 不推荐**:
- 小规模模型 (成本不值得)
- 只需要推理不需要微调 (用 PTQ 更快)

---

## 8.8 精度对齐: Train vs Inference ⚠️ 工业界实践

> **💡 工业界实践** (来源: 2025"青稞"AI嘉年华 - 朱立耕@NVIDIA)
>
> **核心洞察**: 低精度训练不稳定的根本原因往往不是低精度本身,而是训练和推理使用的算子精度不对齐。
>
> **大团队的做法**: Train 和 Inference 的算子在同一个大的 wrapper 里维护,精度问题就不是问题。
>
> **开源社区的问题**: Train 和 Inference 是两帮人做,算子没对齐导致 accuracy 不稳定。

### 8.8.1 精度不对齐的问题

**典型场景**:
```
训练时:
  - 自定义 kernel (如自己写的 Flash Attention)
  - FP32/FP16 数值处理
  - 特定的算法实现

推理时:
  - 社区优化的 kernel (如 SGLang 的 Flash Attention)
  - INT8/INT4 量化
  - 不同的数值处理

结果: Numerical gap 导致 accuracy 不稳定
  - Training loss spike
  - 最终 accuracy 掉点
```

**表现**:
```
症状:
  - 训练时 loss 正常下降
  - 部署到推理框架后性能崩溃
  - PPL 明显上升
  - 生成质量明显下降

原因:
  - 训练和推理算子不对齐
  - 数值精度不同
  - 算法实现差异
```

---

### 8.8.2 为什么精度不对齐?

**开发团队分离**:
```
Training Team:
  - 关注收敛速度
  - 自定义优化
  - 快速迭代

Inference Team:
  - 关注推理速度
  - 社区优化
  - 兼容性

问题: 两个团队没有协同
```

**优化目标不同**:
```
Training:
  - 最大化训练吞吐
  - 支持分布式训练
  - 容错和检查点

Inference:
  - 最大化推理吞吐
  - 最小化延迟
  - 量化压缩

冲突: 优化方向不同,实现有差异
```

**实现细节差异**:
```
Flash Attention:
  - 训练版本: 某种数值简化
  - 推理版本: 另一种优化
  - 结果: 输出有微小差异

Attention Mask:
  - 训练: 布尔 mask
  - 推理: 浮点 mask (为了兼容性)
  - 结果: 精度累积误差
```

**测试场景不同**:
```
Training:
  - 合成数据 (随机输入)
  - 快速验证
  - 不覆盖 edge case

Inference:
  - 真实数据
  - 各种 edge case
  - 长序列、特殊字符

问题: 训练没覆盖的场景,推理暴露问题
```

---

### 8.8.3 如何确保精度对齐

**方法 1: 统一算子库** (推荐)
```python
# 统一的 Attention wrapper
class UnifiedAttention:
    def __init__(self, use_quantization=False):
        self.use_quantization = use_quantization
        self.kernel = get_attention_kernel(use_quantization)

    def forward(self, q, k, v):
        # 训练和推理使用同一套算子
        return self.kernel(q, k, v)

# 训练时
attn = UnifiedAttention(use_quantization=True)  # Fake quant
output = model(input)

# 推理时
attn = UnifiedAttention(use_quantization=True)  # True quant
output = model(input)
```

**方法 2: 数值对齐测试**
```python
def test_numerical_alignment():
    """测试训练和推理算子的数值对齐"""
    # 生成相同输入
    x = torch.randn(1, 512, 4096)

    # 训练算子
    train_output = training_attention(x)

    # 推理算子
    infer_output = inference_attention(x)

    # 比较输出差异
    abs_error = (train_output - infer_output).abs().max()

    assert abs_error < 1e-5, f"不对齐! 最大误差: {abs_error}"

# CI/CD Pipeline
def ci_pipeline():
    """自动检测精度 regression"""
    for commit in recent_commits:
        if not test_numerical_alignment():
            print(f"❌ Commit {commit} 导致精度不对齐")
            return False

    print("✅ 所有 commit 精度对齐")
    return True
```

**方法 3: 端到端验证**
```python
def end_to_end_validation():
    """端到端验证: 训练后直接在推理框架中测试"""
    # 1. 训练模型
    model = train_model()

    # 2. 导出权重
    weights = model.state_dict()

    # 3. 加载到推理框架
    inference_model = load_for_inference(weights)

    # 4. 比较输出
    train_output = model.generate(test_input)
    infer_output = inference_model.generate(test_input)

    # 5. 检查差异
    diff = (train_output - infer_output).abs().max()
    if diff > threshold:
        print(f"⚠️  发现精度 regression: {diff}")
        return False

    return True
```

---

### 8.8.4 不同任务对精度的敏感度

**LLM**: 离散采样,对低精度容忍度高
```
为什么 LLM 对量化友好?
  - 输出是离散的 (token IDs)
  - Temperature 引入随机性
  - 小的量化误差被采样掩盖

证据(概述):
  - INT8 量化: 损失通常较小
  - INT4 量化 (QAT): 损失通常可控
  - 结论: LLM 对量化具有一定容忍度
```

**Diffusion**: 连续空间采样,误差累积严重
```
为什么 Diffusion 对量化敏感?
  - 输出是连续的 (像素值)
  - 多步采样 (50-1000 steps)
  - 每步误差累积

证据(概述):
  - 低精度下误差可能累积
  - 需要更精细的校准与修正
  - 往往倾向使用更高精度格式

结论: Diffusion 模型至少使用 FP8
```

**对比**:
| 任务类型 | 推荐格式 | 精度损失 | 说明 |
|---------|---------|---------|------|
| **LLM** | INT4/INT8 | 低~中 | 离散采样,容忍度高 |
| **Diffusion** | FP8/FP16 | 低 | 连续采样,误差累积 |
| **Recommendation** | INT8 | 低 | 类似 LLM |
| **RL** | INT4 (QAT) | 中 | 需要 train-infer 一致 |

---

### 8.8.5 低精度的软件抽象复杂度

**BF16/FP16**: 一个 tensor 就是一个数据
```python
weight = torch.randn(4096, 4096, dtype=torch.float16)
# 简单、直观
```

**FP8**: 一个 weight 变成 3 个 tensor
```python
weight_fp8 = torch.randn(4096, 4096, dtype=torch.float8_e4m3)
scale = weight_fp8.abs().max() / 127  # 缩放因子
weight_meta = {"dtype": "fp8_e4m3", "scale": scale}  # 元数据

# 软件复杂度大幅增加
# 需要同时管理 3 个对象
```

**FP4**: 需要 padding、pack 等操作
```python
# FP4 打包: 2 个 FP4 → 1 byte
weight_fp4_packed = pack_fp4(weight_fp4)  # 自定义格式

# PyTorch 最少 1 byte
# 需要特殊处理

# 软件生态需要大规模演进
```

**挑战**: 用户心智负担大
```
问题:
  - 如何平衡收益和复杂度?
  - 抽象应该在哪里?
  - 用户需要理解底层细节吗?

方向:
  - 框架自动处理 (vLLM、SGLang)
  - 用户友好 API
  - 渐进式优化
```

---

### 8.8.6 低精度训练的稳定性问题

**常见症状**:
```
症状 1: 训练到一半 loss 炸了
  - 前 1000 steps: loss 正常下降
  - Step 1001: loss 突然暴涨
  - Step 1002: NaN

症状 2: 同样 task 高精度没问题,低精度直接起飞
  - FP32: 收敛正常
  - FP8: loss 不下降

症状 3: 高精度 accuracy 挺好,低精度瞬间掉 3-4 个点
  - FP32: 85% accuracy
  - FP8: 81% accuracy (掉 4 个点)
```

**根本原因**: (张明星@清华)
```
不全是精度问题,而是算法没调好

常见问题:
  - Loss control 没做好
  - Data mixing 不合理
  - Curriculum learning 缺失
  - LR schedule 不适合低精度
```

**解决方向**:
```
1. 把各种"内科" (张明星语) 检查得更细
   - Gradient clipping
   - Weight decay
   - Learning rate warmup
   - Batch size 调整

2. 不要上来就搞很难的题目,从简单开始
   - Curriculum learning
   - 从简单 task 开始
   - 逐步增加难度

3. 低精度可能引入噪声,反而有助于收敛
   - Kimi K2 的 INT4 经验
   - 噪声有助于泛化
   - 但需要控制噪声水平
```

---

### 8.8.7 从历史看精度演进 (朱立耕@NVIDIA)

**FP32 → FP16**: 见过类似问题,最终解决
```
2016-2018 年:
  问题: FP16 训练不稳定
  解决: 混合精度训练、Loss scaling
  现状: 完全成熟,工业标准
```

**FP16 → BF16**: 见过类似问题,最终解决
```
2020-2022 年:
  问题: FP16 范围小,容易下溢
  解决: BF16 (与 FP32 相同范围)
  现状: 完全成熟,广泛使用
```

**BF16 → FP8**: 现在是过渡期阵痛
```
2023-2025 年:
  问题: FP8 训练稳定性
  解决: 正在解决中...
  预期: 1-2 年内成熟
```

**结论**:
```
随着算法 stabilize 和 config 摸清,问题可以解决
低精度收益还是很大的,值得投入
```

---

## 8.9 量化技术总结与展望

### 8.9.1 量化技术演进路线

```
2020-2021:
  FP32/FP16 标准
  INT8 量化开始流行

2022-2023:
  BF16 广泛采用
  INT4 量化成熟 (GPTQ、AWQ)
  PTQ 为主流

2024-2025:
  FP8 逐步进入实践
  QAT 受重视
  PTQ + QAT 混合方案

2025-2026 (预期):
  FP4/NVFP4 持续推进
  统一量化框架
  端到端优化
```

---

### 8.9.2 不同场景的最佳实践

**场景 1: 快速部署**
```
推荐: PTQ (INT8/INT4)
工具: vLLM + AWQ
流程:
  1. 选择预训练模型
  2. AWQ/GPTQ 量化
  3. 部署到 vLLM
时间: 小时级 (依数据与硬件而定)
```

**场景 2: 生产环境 (精度要求高)**
```
推荐: QAT (INT8)
工具: NVIDIA Model Optimizer
流程:
  1. 准备训练数据
  2. QAT 训练 (几个 epoch)
  3. 导出 INT8 权重
  4. 部署
时间: 天级 (依数据与硬件而定)
```

**场景 3: 极限压缩 (100B+ 参数)**
```
推荐: QAT (INT4)
工具: SGLang + Marlin
流程:
  1. QAT 训练 (完整微调)
  2. 转换为 Marlin 格式
  3. SGLang 部署
  4. 精度验证
时间: 周级 (依模型与资源而定)
```

---

### 8.9.3 未来发展方向: FP4、NVFP4

**新一代硬件对 FP4/FP8 的支持**:
```
硬件特性:
  - 更高的算力与带宽
  - 对低精度格式的原生支持

性能潜力:
  - 速度可能提升
  - 显存占用进一步下降

挑战:
  - 软件生态不成熟
  - 需要新的量化算法
  - 精度对齐问题
```

**时间表**:
```
时间表(示意):
  - FP8: 持续成熟
  - FP4/NVFP4: 逐步推进
  - 更低精度格式: 仍在探索
```

---

### 8.9.4 算法和系统的 co-design (张博涵@浙大)

**核心观点**:
```
不是系统等算法成熟
不是算法等系统优化
需要同步螺旋式上升
```

**例子**:
```
算法进步:
  - 新的量化方法
  - 更好的 fake quantization
  - 更稳定的训练技巧

系统进步:
  - 更快的量化 kernel
  - 更好的硬件支持
  - 更成熟的工具链

两者协同:
  - 算法指导系统设计
  - 系统约束推动算法创新
  - 共同演进
```

**启示**:
```
不要等待"完美"的算法
算法和系统要一起迭代
小步快跑,快速验证
```

---

## ✅ 章节检查清单

完成本章后,你应该能够:

- [ ] 解释量化的基本原理和公式
- [ ] 计算量化后的显存占用
- [ ] 对比 PTQ 和 QAT 的优缺点
- [ ] 选择合适的量化格式 (INT8/INT4/FP8)
- [ ] 使用 vLLM 部署量化模型
- [ ] 使用 SGLang 部署 INT4 模型
- [ ] 实现 fake quantization
- [ ] 进行精度对齐测试
- [ ] 诊断量化精度问题
- [ ] 根据场景选择量化方案

---

## 📚 动手练习

**练习 8.1**: 对比不同量化格式的性能和精度

任务:
1. 加载 Llama-2-7B 的 FP16、INT8、INT4 版本
2. 测量显存占用、推理速度、精度
3. 绘制对比图表

**练习 8.2**: 量化 Llama-3-70B 并测试 (使用 vLLM + AWQ)

任务:
1. 下载 Llama-3-70B-AWQ 模型
2. 使用 vLLM 加载并测试
3. 对比 FP16 和 INT4 的性能

**练习 8.3**: 使用 SGLang 部署 INT4 模型并 benchmark ⭐

任务:
1. 安装 SGLang
2. 启动 INT4 推理服务
3. 进行性能 benchmark
4. 评估精度损失

**练习 8.4**: (进阶) 实现简单的 fake quantization ⭐

任务:
1. 实现 FakeInt4QuantizationSTE 类
2. 在小型模型上测试
3. 验证梯度正确传递

---

## 🎯 总结

**关键要点**:
- 量化通过降低精度节省显存和提升速度
- PTQ 快速但可能有精度损失,QAT 精度高但成本高
- INT4 是当前极限压缩的首选 (75% 节省)
- KV Cache 量化对长序列很重要
- 训练和推理必须精度对齐才能保证稳定性
- FP4/FP8 是未来方向 (需新一代硬件支持)

**下一章**: 第9章 投机采样——通过推测解码加速生成。

---

**有问题?加入 [第8章 Discord 频道](https://discord.gg/TODO) 讨论!**
