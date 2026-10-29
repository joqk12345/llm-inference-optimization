# DeepSeek-V4.1-Flash KV 实现记录

- 编辑迁移日期：2026-10-29。
- 来源快照：原第6章引用的 vLLM 2026-10-07 实现分析，链接见文末。
- 验证等级：保留原稿的来源解读，本次编辑未重新核验上游，也没有 GPU 实测。
- 版本边界：以下参数、字节数和执行布局属于来源快照；执行前需锁定引擎 commit、模型 revision、硬件和完整配置。
- 维护方式：升级时另存新快照；参数失效时保留历史说明。替代路径为目标版本官方文档及第10章的受控实验流程。
- 返回[第6章](../../chapters/chapter06-kv-cache-optimization.md)。

## 来源快照

DeepSeek-V4.1-Flash 展示了一个重要趋势：KV Cache 优化不再只是把同一种 KV 状态压缩或分页，而是由模型结构直接决定状态的形态、复用边界和 prefill/decode 工作量。该模型采用 causal encoder-decoder（CED）结构，官方 vLLM 实现中观察到 decode 每个 token 激活约 16B 参数，而 prefill 每个 token 激活约 8B 参数。

### 6.9.1 两类 KV 状态

V4.1-Flash 同时维护两类 KV：

- **Global KV**：跨层共享、压缩保存，并使用 FP4 表示；官方 vLLM 文章给出的全局 KV 规模约为 890 bytes/token。
- **SWA KV**：不压缩，使用 FP8，覆盖 40 个层中每层最近 128 个位置的滑动窗口。

这两类状态的生命周期不同，因此不能简单地用一个“KV Cache 命中率”概括。Global KV 更适合跨请求、跨轮次复用；SWA KV 则带来更高的边界存储成本。vLLM 文章指出，如果在每个可能的 prefix hit 边界都保存 SWA KV，其存储量可能超过 Global KV 的 10 倍。

### 6.9.2 SWA bounded replay

一种直接做法是在 prefix hit 后重新计算 SWA KV，但精确重建会沿层级传播，代价较高。SWA bounded replay 的做法是只重放最后 128 个 token，并从 replay 起点裁剪滑动窗口。它牺牲 bit-exact 一致性换取更低的重算成本，质量影响需要通过目标任务评估确认。

vLLM 在两个位置使用这一机制：

1. **Encoder 侧**：只缓存 Global KV，prefix 命中后重算命中边界前的最后 128 个 token，以恢复 SWA KV。
2. **Decoder 侧**：每个 token 仍运行产生 Global KV 所需的层；后续层只处理请求的最后 128 个 token，从而跳过大部分长 prompt prefill。

这说明 prefix caching 的关键不只是“命中多少 token”，还包括命中后需要恢复哪类状态、恢复状态需要运行哪些层，以及近似重放是否满足质量要求。

### 6.9.3 CUDA Graph 与分段执行

裁剪后的后半段层计算量很小，如果使用 eager 执行，CPU kernel launch 开销可能抵消计算节省。由于前后两段的输入形状不同，不能简单地放进同一个 CUDA Graph。vLLM 的做法是把前半段和裁剪后的后半段分别捕获，并为后半段使用独立的 capture size。

这给工程实现一个通用启示：当模型结构让不同层组拥有不同的 token 形状时，应把图捕获、batch 组织和调度边界一起设计，而不是只在单个 kernel 上做优化。

### 6.9.4 结构变化与引擎执行

下表以 vLLM 2026-10-07 文章为实现观察，不把其中的硬件配置作为通用默认值。

| 结构变化 | 状态或计算变化 | 引擎处理 |
|---|---|---|
| CED 与跨层 KV 共享 | 不同层组需要处理的 prompt token 数不同 | 0～20 层处理完整 batch；21～39 层处理裁剪 batch，分别捕获图 |
| CSA2、FP4 与跨层共享 | 共同降低 Global KV 容量 | 使用匹配的 KV 格式和注意力 kernel；890 bytes/token 仅指 Global KV |
| 后续 indexer 层采用固定 16K 候选位置 | 索引评分可限制在候选集合 | sparse indexer kernel 直接评分候选，避免先扫描完整上下文再 mask |
| mHC 系数依赖移到前一子层已存在的 residual streams | 下一段系数计算可与当前 attention/FFN 重叠 | 小 batch 下使用旁路 CUDA stream，并保留汇合依赖 |
| Engram 使用 token n-gram 哈希查表 | 两张大型 FP8 表只读少量行，host lookup 延迟成为新因素 | 异步预取 CPU 表、共置 DP 副本共享 host 表，评估大页与内存容量 |

CSA2 在这里描述压缩稀疏注意力这一结构选择；具体压缩与索引算法应以模型报告为准。固定候选评分、融合 kernel 和旁路 stream 则属于实现优化。二者共同改变性能，但不能把某个 kernel 的加速归因于模型结构本身。

Engram 表也不属于 Global KV。容量规划应分别统计模型权重、Global KV、SWA KV、host 查找表和运行时缓冲，避免用 890 bytes/token 推导整个服务的内存需求。

### 6.9.5 结构化 KV 优化与 kernel 协同

V4.1-Flash 的推理优化还依赖一组与模型结构配套的 kernel：MegaAttention、Mega-mHC、Mega-Gate、DeepSelect，以及支持 NVFP4 KV 的 FlashMLA 路径。它们分别涉及稀疏注意力、mHC 融合、MoE 路由、稀疏 top-k 和压缩 KV 读取。

对这类模型，评估时至少要分开记录：

- Global KV 与 SWA KV 的字节数和生命周期；
- prefix hit 后的 replay token 数量；
- 每一层实际处理的 token 数量；
- CUDA Graph 覆盖率与 eager fallback 比例；
- KV 读取、索引、MoE 路由和 host lookup 的通信/访存时间。

不能只用总显存或总 tokens/s 判断优化是否有效。

### 6.9.6 部署策略启示

vLLM 的公开实测将低延迟场景与高吞吐场景分开处理：低延迟采用 TP4，高吞吐采用 DEP2，并利用 Attention DP 避免在多卡间重复复制共享 KV。多轮 Agent 负载还需要 session affinity，以保持 prefix-cache locality。

这些配置不是通用默认值。它们依赖具体 GPU、batch、上下文长度、请求轮次和通信域。正式部署时应分别压测：

1. 单请求短 prompt 的 TTFT；
2. 长 prompt prefix hit 后的 replay 成本；
3. 多轮 Agent 的跨请求 KV 复用；
4. TP、DP 和 EP 组合下的通信与显存；
5. NVFP4/FP8 KV 对质量和并发的影响。

复现 bounded replay 时还需记录引擎差异：本文介绍的 vLLM 重放最近 128 个位置；DeepInfer README 的默认保留后缀为 140 token，需要同时覆盖 SWA 窗口和最大回退长度。后缀配置不能跨引擎直接复制，详见 [DeepInfer 实现记录](deepinfer-runtime.md)。

本节依据 [vLLM 对 DeepSeek-V4.1-Flash 的实现分析](https://vllm.ai/blog/2026-10-07-deepseek-v41-flash)。其中的性能数字来自文章所述的特定 GPU、负载和 benchmark，不应直接当作本书的普遍性能承诺。
