# vLLM 0.30.0 上游能力与参数

Sources:

- https://github.com/vllm-project/vllm
- https://docs.vllm.ai/en/v0.30.0/configuration/engine_args/

Evidence grade: A（上游仓库 README 与版本化官方文档）

Verified on: 2026-09-26

Supported claims: 上游 README 列出的 serving、KV、batching、量化、attention、speculative decoding、图优化和 disaggregated serving 能力；0.30.0 engine arguments 的配置分组和公开参数入口；0.30.0 文档提供 `vllm bench serve` 基准 CLI。

Limits: README 是能力地图而非逐模型兼容矩阵；参数存在不代表硬件、模型、后端组合可运行；不支持性能泛化。

Open questions: 本书目标模型与硬件上的启动、质量和性能回归尚未执行。
