# SGLang INT4 QAT案例研究 - 端到端RL实践

**来源**: SGLang RL Team, InfiXAI Team, Ant Group, slime Team, RadixArk Team
**日期**: 2026-01-26
**主题**: 使用INT4 QAT将1TB模型压缩到单张H200 GPU的完整实践

---

## 核心洞察

### 为什么需要INT4 QAT

**问题**:
- ~1TB规模的大型模型(如Kimi K2)需要多节点部署
- 跨节点通信成为瓶颈,rollout效率低下
- BF16全精度训练成本极高

**解决方案 - INT4 QAT**:
- **显存压缩**: 通过权重量化和低比特量化,~1TB模型可压缩至单张H200(141GB)
- **训练-推理一致性**: 训练时使用QAT塑造INT4友好的权重分布,推理时使用W4A16(INT4权重,BF16激活)
- **单节点效率**: 避免跨节点通信瓶颈,显著提升rollout效率

**关键数据**:
```
模型规模: ~1TB (K2-like)
压缩后: 单张H200 (141GB)
压缩比: ~7倍
通信瓶颈: 完全消除
```

---

## 技术概览

### 端到端QAT Pipeline

```
┌─────────────────────────────────────────────────────┐
│  Stage 1: QAT训练阶段                                │
│  ────────────────────────                           │
│  训练端保持BF16主权重                                 │
│  前向传播: 插入fake quantization模拟量化噪声         │
│           - 权重 "离散化" 为INT4                      │
│           - 立即恢复                                  │
│           - 引入量化误差                              │
│  反向传播: 使用STE(直通估计器)绕过量化的不可微性     │
│           - 定义rounding的导数为1                    │
│           - 梯度直接传递到BF16主权重                  │
└───────────────────┬─────────────────────────────────┘
                    │
                    ↓ 权重转换
┌─────────────────────────────────────────────────────┐
│  Stage 2: 权重转换阶段                               │
│  ─────────────────────                              │
│  导出收敛的BF16权重                                  │
│  执行真正的量化: BF16 → INT4                         │
│  转换为推理引擎格式(如Marlin)                        │
└───────────────────┬─────────────────────────────────┘
                    │
                    ↓ RL rollout
┌─────────────────────────────────────────────────────┐
│  Stage 3: RL Rollout阶段                            │
│  ─────────────────────                              │
│  SGLang加载INT4权重                                  │
│  运行高效W4A16推理(INT4权重 × BF16激活)              │
│  生成的经验数据回流到Stage 1                         │
│  形成自洽的闭环                                      │
└─────────────────────────────────────────────────────┘
```

---

## 关键策略选择

### 1. 为什么选择INT4 (W4A16)而不是FP4?

**对比表**:

| 维度 | INT4 (W4A16) | FP4 |
|------|-------------|-----|
| **硬件支持** | ✅ 广泛(Blackwell之前的GPU) | ⚠️ 需要Blackwell |
| **生态成熟度** | ✅ 成熟的Marlin内核 | ⚠️ 较新 |
| **动态范围** | ✅ 足够(1×32 scale granularity) | ⚠️ 有限 |
| **精度稳定性** | ✅ 充分验证 | ⚠️ 正在验证 |
| **性能** | ✅ 优化良好 | ✅ 理论更高 |
| **维护成本** | ✅ 低 | ⚠️ 中等 |

**决策**: 选择INT4作为工业界"足够好"的量化标准,在性能、风险和维护成本之间取得理性平衡。

**未来**: 计划在NVIDIA Blackwell GPU上探索FP4 RL。

### 2. Fake Quantization + STE

**训练策略**:
- **维护BF16主权重**: 保持高精度存储和更新
- **前向传播**: 模拟量化噪声,迫使模型"学习"适应低精度表示
- **反向传播**: 使用STE确保梯度无损传递

**优势**:
- 最大化低精度训练的收敛性和稳定性
- 训练-推理严格对齐

---

## 训练端实现

### Fake Quantization和STE实现

**核心逻辑位置**: `megatron/core/extensions/transformer_engine.py`

**_FakeInt4QuantizationSTE类**:
```python
class _FakeInt4QuantizationSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # 动态量化: 基于per-group max absolute value
        # 模拟INT4的[-7, 7]范围和clipping
        # 但仍在BF16中计算,只注入量化误差
        return fake_quantize(x)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: 梯度直接通过,保持训练连续性
        return grad_output
```

**关键点**:
1. **动态量化**: 基于per-group max absolute value
2. **范围模拟**: INT4的[-7, 7]范围
3. **计算精度**: 物理计算仍在BF16
4. **梯度传递**: 反向传播时STE确保梯度无损

### Fake Quantization消融实验

**实验设计**: 验证QAT的必要性,研究train-infer精度不匹配的影响

**两种不对称场景**:
1. **QAT INT4训练 + BF16 rollout**
2. **不启用QAT训练 + 直接INT4 rollout**

**评估指标**: Logprob Abs Diff (log概率的绝对差异,衡量train-infer不一致性)

**实验1: QAT INT4训练 + BF16 rollout (红线)**

```
观察: 即使使用高精度BF16推理,误差仍然显著高于baseline

原因分析:
- QAT已经通过"补偿"使权重适应INT4量化噪声
- 如果在推理时移除量化,这种补偿反而变成扰动
- 导致分布偏移(distribution shift)
```

**实验2: 不启用QAT训练 + 直接INT4 rollout (红线)**

```
观察: 对应传统PTQ设置,误差随训练步数震荡并上升

原因分析:
- 模型在训练期间从未见过量化噪声
- 将权重压缩到INT4导致严重的信息损失
- 相对训练时特征分布发生偏移
```

**结论**:

> **训练端fake quantization和推理端real quantization必须同时启用**
>
> 只有当训练时的模拟噪声与推理时的真实量化严格对齐时,才能:
> - 抑制train-infer不匹配
> - 避免分布偏移
> - 保持误差接近baseline
> - 真正实现端到端低精度RL训练闭环

---

## 权重更新阶段

### 权重流转和动态格式适配

**问题**: QAT训练输出标准格式权重(类似Hugging Face),而SGLang的Marlin内核需要特殊打包和排列的权重。

**解决机制**: `restore_weights_before_loading`

```python
# 权重更新前的安全恢复
def restore_weights_before_loading():
    # 使用缓存的_original_shapes元数据
    # 将内存中的Marlin权重格式恢复(resize)回原始shape
    # 防止shape不匹配的运行时错误
    # 实现标准权重格式和Marlin格式的平滑切换
```

**动态权重管理**: `compressed_tensors_moe.py`

```python
def process_weights_after_loading():
    # 权重加载完成后自动运行
    # 调用gptq_marlin_moe_repack和marlin_moe_permute_scales等算子
    # 将标准权重转换为高度优化的Marlin格式
    # 最大化推理时的内存访问和计算效率
```

### 权重更新时的量化

**核心算子**: `int4_block_quantize`

```python
def int4_block_quantize(weights, group_size):
    # 1. 计算per-group scales
    scales = compute_per_group_scales(weights, group_size)

    # 2. 将高精度浮点映射到INT4整数域[-7, 7]
    int4_weights = quantize_to_int4(weights, scales)

    # 3. Bit packing (位打包)
    # PyTorch缺乏原生INT4 dtype,使用位运算技巧
    # 将8个INT4值紧密打包到一个INT32整数中
    # (8 × 4 bits = 32 bits)
    packed = pack_int4_to_int32(int4_weights)

    # 4. 返回"推理格式"
    return packed, scales
```

**内存优化**:
- 原始: 8个INT4值需要8字节
- 打包后: 8个INT4值只需1个INT32(4字节)
- **节省**: 50%的额外空间

---

## 推理阶段

### SGLang W4A16推理

**1. 最小化打包和近零开销解包**

```
存储:
- SGLang将两个4位值打包到一个字节中
- 相比BF16节省75%内存

推理:
- Triton内核使用位运算(>> 4 和 & 0xF)高效解包
- 计算和IO的重叠使解包几乎零开销
```

**2. MoE算子的深度融合**

**内存优化**:
```python
# 动态moe_align_block_size
def moe_align_block_size(token_count, expert_distribution):
    # 根据当前token数和专家分布选择block_size
    # 将同一专家的token分组和对齐
    # 提高带宽利用率
```

**计算融合**:
```python
# 高性能Marlin INT4实现
# 融合gating部分为单一高性能内核
# 避免重复的kernel启动和中间读写
```

**兼容性**:
- 支持主流格式: GPTQ, AWQ
- 支持对称和非对称模式

---

## INT4 QAT RL结果

### 训练端结果

**测试数据集**: dapo-math-17k
**测试模型**: Qwen3-235B-A22B, Kimi-K2-Thinking

**Raw-Reward对比**:

| 配置 | Qwen3-235B-A22B | Kimi-K2-Thinking |
|------|-----------------|------------------|
| BF16 train → BF16 infer | 稳定增长 | 稳定增长 |
| BF16 train → FP8 infer | 稳定增长 | 稳定增长 |
| **BF16 train → INT4 infer** | **稳定增长** | **稳定增长** |

**观察**:
- "BF16 train → INT4 infer"配置实现稳定的Raw-Reward增长
- 增长趋势与前两者基本一致
- 证明了该方法的有效性

### 评估端结果

**基准测试**: aime-2024
**频率**: 每10个训练步运行一次评估

**AIME评分轨迹**:

| 配置 | 斜率 | 峰值分数 | 与BF16对齐度 |
|------|------|----------|-------------|
| BF16 train → BF16 infer | 基准 | 基准 | 100% |
| BF16 train → FP8 infer | 接近 | 接近 | 高度对齐 |
| **BF16 train → INT4 infer** | **接近** | **接近** | **高度对齐** |

**结论**:

> **"BF16 train → INT4 infer"方案**:
> - 评估分数呈稳定上升趋势
> - 斜率和峰值分数与BF16、FP8高度重叠
> - 低比特量化不损害核心表达能力
> - 在大幅节省计算的同时保持(甚至匹配)全精度的泛化性能

### Train-Infer Gap分析

**测试模型**: Qwen3-30B, Qwen3-235B
**Y轴**: 训练端和推理端输出的logprob绝对差异(越低一致性越强)

**结果对比**:

| 配置 | Logprob Abs Diff | 一致性 |
|------|------------------|--------|
| BF16 baseline | 最低 | 基准 |
| FP8 | 较高 | 一般 |
| **INT4 (QAT)** | **几乎重叠BF16** | **优秀** |

**INT4 QAT显著低于FP8**,证实:
- 有效避免"BF16 train → FP8 infer"模式的精度损失
- 实现与全精度无差别的train-infer行为

**假设的两个原因**:

1. **截断误差抑制**:
   - 训练端fake quantization将权重约束到INT4范围
   - 有效减少并行matmul中非确定性累加顺序导致的浮点舍入误差
   - (即"小数加到大数"的精度损失)

2. **高精度计算**:
   - 推理使用W4A16,全程依赖BF16 Tensor Cores
   - 计算精度与训练高度对齐

### Rollout加速对比

**Qwen3-235B-A22B rollout性能**:

| 配置 | 每步延迟 | 性能层级 |
|------|----------|----------|
| BF16 baseline | 基准 | 慢 |
| FP8 | 显著加速 | 快 |
| **INT4** | **轻微优于FP8** | **快** |

**观察**: INT4和FP8都显著加速,但差距不大

**原因分析**:
- **硬件限制**: NVIDIA H系列GPU缺乏原生INT4 Tensor Cores
- **计算路径**: W4A16仍使用BF16 Tensor Cores进行计算
- **优势**: 大幅降低内存带宽压力
- **限制**: 无法获得原生FP8 Tensor Cores的计算提升

**Kimi-K2-Thinking rollout性能**:

| 场景 | 配置 | 性能 |
|------|------|------|
| **双节点** | FP8 | 受限于跨节点带宽 |
| **双节点** | INT4 | 受限于跨节点带宽 |
| **单节点** | **INT4** | **消除通信瓶颈** |

**关键洞察**:

> **在当前硬件下,INT4 QAT的主要价值是通过VRAM压缩实现高效的单节点rollout**
>
> - 通过将模型大小减半,可以加载~1TB规模的模型到单机VRAM
> - 消除昂贵的跨节点通信
> - 大幅降低rollout时间

---

## 总结与未来工作

### 验证的关键结论

1. **精度复现**:
   - 在slime复现中观察到相同的INT4 QAT精度优势
   - 匹配BF16 baseline

2. **效率提升**:
   - Rollout吞吐量显著提升
   - 验证了低比特量化在RL中的价值

3. **单节点部署**:
   - ~1TB模型压缩到单张H200
   - 避免跨节点通信瓶颈
   - 显著提升rollout效率

### 未来工作

**1. 训练端效率优化**:
- **当前问题**: 添加QAT fake quantization引入额外计算开销,训练明显慢于BF16
- **目标**: 提出新优化解决训练端瓶颈,加速全pipeline
- **影响**: 部分抵消更快rollout的端到端收益

**2. 推理端FP4**:
- **时机**: 随着NVIDIA Blackwell普及
- **目标**: 积极探索FP4精度用于RL训练和推理
- **价值**: 进一步挖掘硬件潜力

**3. slime的尝试意义**:
- 展示了在开源生态中复现工业界前沿技术的可行性
- 为极低成本的大规模训练开辟新路径
- 帮助更多开发者深入理解QAT并推动实际应用

---

## 对量化章节的启示

### 可以添加的内容

**1. 在8.2节(量化方法分类)中**:
- 补充QAT vs PTQ的详细对比表
- 增加QLoRA、Native quantized training、QAT三者的对比
- 说明QAT的特殊价值: train-infer一致性

**2. 在8.3节(常用量化格式)中**:
- 增加INT4 (W4A16)格式详解
- 对比INT4 vs FP4 vs FP8
- 说明INT4的广泛硬件支持优势

**3. 在8.4节(流行的量化框架)中**:
- 增加SGLang的INT4推理支持
- 介绍Marlin内核
- 补充NVIDIA Model Optimizer

**4. 新增8.8节(量化进阶: INT4 QAT实战)**:

```markdown
#### 8.8.1 什么是QAT
- QAT原理: fake quantization + STE
- QAT vs PTQ: 什么时候用QAT?
- train-infer一致性的重要性

#### 8.8.2 INT4 QAT完整Pipeline
- Stage 1: QAT训练(fake quantization)
- Stage 2: 权重转换(真量化)
- Stage 3: W4A16推理

#### 8.8.3 训练端实现
- Fake Quantization和STE实现
- 权重更新和格式适配
- 动态权重管理机制

#### 8.8.4 推理端实现
- SGLang W4A16推理
- Bit packing和高效解包
- MoE算子深度融合

#### 8.8.5 实战案例: 1TB模型压缩到单H200
- Qwen3-235B-A22B实践
- Kimi-K2-Thinking实践
- 性能对比: BF16 vs FP8 vs INT4

#### 8.8.6 QAT的适用场景
- ✅ 需要train-infer一致性
- ✅ 大规模模型的RL训练
- ✅ 单节点部署超大模型
- ⚠️ 训练成本较高(需要完整微调周期)
```

**5. 在实战部分**:
- 增加QAT微调的完整代码示例
- 提供SGLang INT4部署模板
- 性能对比数据和benchmark

---

## 技术栈总结

### 工具和框架

| 工具/框架 | 用途 | 特点 |
|----------|------|------|
| **slime** | RL训练框架 | 支持INT4 QAT端到端 |
| **SGLang** | 推理引擎 | W4A16高效推理,Marlin内核 |
| **Megatron-LM** | 大规模训练 | QAT fake quantization实现 |
| **Marlin** | INT4推理内核 | 高性能内核,支持GPTQ/AWQ |
| **NVIDIA Model Optimizer** | 量化工具 | QAT训练支持 |

### 硬件支持

| GPU架构 | INT4支持 | FP4支持 | 推荐场景 |
|---------|---------|---------|----------|
| **Blackwell** | ✅ | ✅ | INT4/FP4 QAT |
| **Hopper (H100/H200)** | ✅ (软件) | ⚠️ | INT4 QAT |
| **Ampere (A100)** | ✅ (软件) | ❌ | INT4 QAT |
| **Ada (4090)** | ✅ (软件) | ❌ | INT4 QAT |

**共同点**: 都支持在常用GPU上完成QAT工作流和INT4推理

---

## 关键要点

### ✅ INT4 QAT的优势

1. **显存压缩**: ~1TB模型压缩到单H200(141GB),7倍压缩
2. **train-infer一致性**: 与BF16 baseline几乎无差异
3. **精度保持**: AIME评分与BF16高度对齐
4. **单节点效率**: 消除跨节点通信瓶颈
5. **硬件友好**: 广泛支持(Blackwell之前的GPU)
6. **生态成熟**: Marlin等高性能内核

### ⚠️ INT4 QAT的限制

1. **训练开销**: 比BF16训练慢,增加fake quantization计算
2. **硬件限制**: H系列GPU无原生INT4 Tensor Cores
3. **复杂度**: 需要理解QAT、STE、格式转换等
4. **工具链**: 依赖特定框架(slime, SGLang, Megatron-LM)

### 🎯 最佳实践

1. **适用场景**:
   - 超大规模模型(100B+参数)的RL训练
   - 需要单节点部署的场景
   - 对train-infer一致性要求高的应用

2. **不适用场景**:
   - 小规模模型(训练成本不值得)
   - 只推理不需要微调(用PTQ即可)
   - 硬件不支持INT4(但几乎所有现代GPU都支持)

3. **实施路径**:
   - 先验证PTQ是否足够(更简单)
   - 如果精度损失不可接受,考虑QAT
   - 选择合适的量化格式(INT4/FP4/NVFP4)
   - 使用成熟的工具链(slime + SGLang)

---

## 参考资源

### 官方文档

- [SGLang](https://github.com/sgl-project/sglang)
- [slime框架](https://github.com/slime-dev/slime)
- [Marlin内核](https://github.com/IST-DasLab/marlin)

### 相关博客

- SGLang RL Team: "Squeezing 1TB Model Rollout into a Single H200: INT4 QAT RL End-to-End Practice"
- Kimi K2 Team: "K2-Thinking Technical Report"
- LMSYS Org: "Fine-tune and deploy gpt-oss MXFP4: ModelOpt + SGLang"

### 社区资源

- Megatron-LM QAT examples
- SGLang W4A16 documentation
- Miles community recipes

---

**文档状态**: 待整合到量化章节
**优先级**: 高
**建议位置**: 第8章(量化技术)新增8.8节
**相关文档**: docs/cases/lmsys-qat-mxfp4-analysis.md (MXFP4 QAT案例)
