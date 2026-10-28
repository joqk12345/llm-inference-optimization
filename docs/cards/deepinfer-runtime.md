# DeepInfer 专用运行时实现记录

- 编辑迁移日期：2026-10-29。
- 来源快照：原第11章记录的 2026-10-22 README 核对，来源链接见文末。
- 验证等级：保留原稿的来源解读，本次编辑未重新核验上游，也没有 GPU/NPU 实测。
- 版本边界：未记录不可变上游 commit，以下内容只作历史说明，不能视为可直接执行的适配保证。执行前锁定 commit、模型 revision、硬件、驱动、编译器及通信库。
- 维护方式：升级时另存快照；失效参数保留历史说明。替代路径为目标 commit 的官方示例及通用服务框架的受控基线。
- 返回[第11章](../../chapters/chapter11-advanced-topics.md)。相关状态管理见 [KV 实现记录](deepseek-v41-flash-kv.md)。

## 来源快照

DeepInfer 代表另一类推理引擎路线：它不是面向所有模型的通用服务层，而是围绕特定模型结构、硬件和大规模推理负载做深度协同优化。官方开源版本当前支持 DeepSeek-V4.1-Flash，并支持 NVIDIA SM100 GPU 与华为昇腾 950 NPU。它更适合作为“专用模型运行时”纳入引擎选型，而不是直接替换所有 vLLM 或 SGLang 部署。

**核心能力**：

- 针对模型结构提供 FlashMLA、DeepGEMM、DeepEP、TileKernels、DeepSelect 等优化 kernel；
- 支持 Attention DP、balanced CP、MoE EP 与 EPLB 等并行方式；
- 提供 DSpark 与 confidence-scheduled verification，用于草稿 token 和验证调度；
- 提供 SWA bounded replay，减少部分 decoder prefill 工作；
- 默认启用异步调度；结构化输出支持正则表达式、JSON Schema 和 EBNF。Batch-invariant inference 与确定性采样目前仅支持 NVIDIA 后端。

按官方 README（2026-10-22 核对），确定性采样要求同时设置 `DEEPINFER_BATCH_INVARIANT=1`、`DEEPINFER_ENABLE_MTP_VARLEN=0`，并在 `SamplingParams` 中传入固定的 `seq_seed`。这项保证针对相同输入 token 和采样参数，不应扩展为跨硬件、跨版本逐位一致。

DeepInfer 的默认 bounded replay 后缀配置为 `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`，配合 `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`；官方要求它覆盖 SWA 窗口及最大回退长度。这与 [KV 实现记录](deepseek-v41-flash-kv.md)介绍的 vLLM 128-token 窗口重放口径不同，不能把窗口长度直接当作所有引擎的保留 token 数。具体配置应随引擎 commit 记录。

**适用边界**：

DeepInfer 的公开版本更接近离线批量推理和专用运行时。官方明确列出的完整服务缺口包括 KV Cache 系统、VE/Prefill/Decode 分离，以及 prefill/decode 数据并行副本之间的负载均衡。因此，不能仅凭其 kernel 或单机 benchmark 结果推断它已经覆盖完整在线服务栈；这些能力仍需要结合外部缓存、编排、路由和服务治理系统补齐。

**环境门槛**：

官方构建路径要求 Linux、GCC 13 及以上、Python 3.12，以及 NVIDIA CUDA 13 或昇腾 CANN 9.2。当前版本对硬件和软件环境要求较新，不适合直接作为本书的默认入门环境。部署前应先确认 GPU/NPU 型号、驱动、编译器、CUDA/CANN、模型权重和通信库版本，随后再进行单节点离线推理和多卡扩展验证。

**建议的引入方式**：

1. 先以 DeepInfer 官方离线示例建立可重复基线，固定模型 revision、硬件、并行度、输入输出 token 分布和采样参数。
2. 将 DeepInfer 与 vLLM/SGLang 的比较限定在同一模型、同一 batch、同一上下文长度和同一质量验收条件下，不把不同引擎的默认参数直接横向比较。
3. 将指标拆成 prefill、decode、端到端延迟、吞吐、显存、通信、确定性和结构化输出成功率，避免只看 tokens/s。
4. 如果要用于在线服务，额外实现 KV Cache 管理、PD/VE 分离、跨副本路由、故障转移和降级策略，并把这些部分纳入整体 SLO。

**与本书其他章节的关系**：

- 第 6 章的 KV Cache 管理决定 DeepInfer 能否进入长上下文在线服务；
- 第 7 章的调度和 PD 分离用于补足其服务化边界；
- 第 8 章的低精度与 kernel 评估用于解释专用硬件路径的收益；
- 第 9 章的投机采样内容可与 DSpark/verification 机制做概念对照；
- 第 10 章的服务治理内容用于完成从离线运行时到生产系统的最后一层封装。

**实践练习 11.4**：在具备 SM100 或昇腾 950 的环境中，使用 DeepInfer 官方示例完成离线推理基线，记录引擎 commit、模型 revision、并行配置、prefill/decode 时间、吞吐、显存和结构化输出结果。仅 NVIDIA 后端增加确定性测试，按上文配置固定 seed、关闭变长 MTP，并分别验证重复运行与改变 batch 组成的输出；昇腾后端将该测试记录为“不适用”。再与同模型的另一种推理后端进行受控对比。没有对应硬件时，不生成性能结论。来源：[DeepInfer 官方 README](https://github.com/deepseek-ai/DeepInfer/blob/main/README.md)。
