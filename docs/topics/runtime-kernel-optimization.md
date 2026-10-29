# 算子与运行时优化专题

本专题承接第11章原 11.5–11.7 节。它是 profile 指向运行时热点后的选读材料。CPU 参考代码验证数学关系，不代表生产 kernel，也不提供 GPU 加速结论。

## 编译优化

先固定模型、输入形状、dtype、设备和采样方式，对比 eager 与编译执行。把首次编译耗时、稳态耗时和形状变化引起的重新编译分开记录；不要通过吞掉编译异常来制造“已启用”的假象。

对框架管理的模型，应使用该版本公开的编译配置和启动日志，不能依赖 `llm_engine.model_runner.model` 等内部属性来二次包装。适配入口见 [版本卡](../cards/vllm-0.30.0.md)。回归失败时恢复原配置，并保留失败输入及日志。

## FlashAttention 的在线 softmax

设一个 query 行已经处理的最大 score 为 `m`，指数和为 `l`，未归一化的加权值累积为 `a`。新 key/value 块的 score 为 `s`，其更新如下：[CITE: flashattention-2022]

```text
m_new = max(m, max(s))
alpha = exp(m - m_new)
w = exp(s - m_new)
l_new = alpha * l + sum(w)
a_new = alpha * a + w @ V_block
```

初始化 `m=-∞, l=0, a=0`，处理全部有效块后输出 `a/l`。`l` 是平移后的指数和，不是 logsumexp。若维护的是已经归一化的输出 `O`，则每一步应使用 `(alpha*l*O + w@V_block)/l_new`，不能混用这两种累积方式。

参考实现位于 `code/chapter11/attention_reference.py`，采用单头、单序列的 Python 列表，支持普通 Attention 和含对角线的 causal self-attention，避免把广播与设备管理混入原理讲解。`tests/README.md` 中的测试覆盖多分块、非整除长度、不同 value 维度、大 score 和 causal mask。任意 mask 下的全屏蔽行、dropout、混合精度与 GPU kernel 不在此实现范围内。

在批量张量实现中，K 应交换最后两个维度，例如 `K.transpose(-2, -1)`；不能用三维张量的 `.T` 代替。统计量需保持正确的广播维度并位于同一设备。

该算法保留精确 dense Attention 的数学定义，浮点结果允许舍入差异。长序列完整 Attention 的算术复杂度仍为 `O(N²d)`；收益来自分块和减少 HBM 读写，不是把 Attention 改为线性计算。[CITE: flashattention-2022]

验证 GPU 实现时，先与可信基线比较输出，再按真实 prefill/decode 形状测显存峰值、kernel 时间和端到端延迟。后端选择应按锁定引擎版本执行，并从日志确认生效；本专题不提供未经执行验证的后端命令。

## 自定义算子的验证流程

1. 用 trace 定位热点，并估算该热点占端到端耗时的比例。
2. 写清张量形状、stride、dtype、mask 和输出语义，以标准实现作为数值基线。
3. 实现最小 kernel，覆盖空输入、尾块、非连续布局等实际支持的边界；不支持的输入应明确拒绝或回退。
4. 正确性通过后预热，使用设备计时或同步后的墙钟计时，分别报告首次调用和稳态结果。
5. 回到服务压测，确认质量、P95/P99、失败率和成本没有回归；保留原 kernel 作为退出路径。

Triton 的 host wrapper 需要配置 launch grid、输出缓冲和形状参数，不能用普通 Python 函数调用替代 kernel launch。可运行 GPU 实现应单独锁定依赖并在目标硬件验证后发布。
