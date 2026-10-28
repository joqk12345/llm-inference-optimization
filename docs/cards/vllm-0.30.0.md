# vLLM 0.30.0 执行适配卡

- 核查日期：2026-09-26。
- 上游仓库：[vllm-project/vllm](https://github.com/vllm-project/vllm)。本卡以 0.30.0 文档为执行目标，不把 `main` 分支的未来行为当作稳定 API。
- 适用范围：第4章环境、第6章 KV、 第8章量化、第9章投机采样、第10章生产实验和第11章高级部署。
- 验证等级：已核对 0.30.0 官方参数/功能文档；本仓库没有对应 GPU 环境的启动或性能实测。
- 版本边界：实验必须锁定 `vllm==0.30.0`、容器 digest 或 lockfile、模型与 tokenizer revision、GPU/驱动/CUDA 和完整启动参数。

## 0.30.0 对焦内容

上游 README 当前将 vLLM定位为高吞吐推理与 serving 引擎，明确列出 PagedAttention、continuous batching、chunked prefill、prefix caching、CUDA/HIP graphs、FP8/MXFP8/MXFP4/NVFP4/INT8/INT4 等量化、FlashAttention/FlashInfer/FlashMLA/Triton 等 attention 后端、n-gram/suffix/EAGLE/DFlash 投机解码，以及 prefill/decode/encode 分离能力。这些是能力地图，不是“所有模型、硬件和组合都支持”的保证。[CITE: vllm-0-30-0-upstream]

0.30.0 的 engine arguments 文档已将配置分为 Model、Load、Attention、Parallel、Cache、Compilation、Kernel、Observability 和 Scheduler 等组；生产材料应按公开参数和启动日志核验，不能依赖内部 Python 类路径。参数存在也不等于当前模型后端能执行。

## 贯穿实验的允许接口

本仓库实验只锁定两个变量：`--enable-prefix-caching` 与 `--no-enable-prefix-caching`。服务版本通过 `/version` 核对，模型通过 `/v1/models` 核对；其余环境字段必须从 manifest 和启动日志提供证据。

KV Cache 量化示例使用 0.30.0 文档列出的 `kv_cache_dtype` 选择。当前文档包含 `auto`、`bfloat16`、`float16`、FP8 变体、`int4_per_token_head`、`int8_per_token_head`、`nvfp4` 和 TurboQuant 相关选项；可用项会随平台和模型后端收缩，不能把列表当作兼容矩阵。[CITE: vllm-kv-cache-0-30-0]

投机采样应使用 0.30.0 的 Speculative Decoding 文档和对应示例（draft model、EAGLE、MLP、n-gram、suffix 等）核对参数；本书第9章先讲算法和证据口径，未把某个启动参数写成跨模型通用命令。

## 兼容性与升级规则

- `main`、nightly、Docker tag 和发行版下游打包不可互换；升级时复制本卡创建新版本。
- 任何参数弃用、默认值变化、模型支持变化、kernel fallback 或 metrics 名称变化，都要更新适配卡并重跑 smoke、质量和性能回归。
- 用 `vllm serve --help` 和目标版本文档确认参数，再保存实际命令；不能从旧卡片复制内部开关。
- 指标以目标实例 `/metrics` 的实际输出为准，dashboard 必须随版本回归。

## 来源与限制

官方仓库 README 和 0.30.0 文档支持能力和参数存在性；它们不支持本书任何固定吞吐、延迟、显存或成本数字。性能仍需在目标模型、硬件、负载、精度、并发和预热条件下实测。
