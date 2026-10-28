# vLLM 0.30.0 KV Cache 接口

Source: https://docs.vllm.ai/en/v0.30.0/configuration/engine_args/

Source: https://docs.vllm.ai/en/v0.30.0/features/quantization/quantized_kvcache/

Evidence grade: A（版本化官方文档）

Verified on: 2026-09-26（接口核对；非 GPU 实测）

Supported claims: 0.30.0 文档公开的 KV cache dtype 和 scale 配置入口；具体选项与后端、模型支持相关。

Limits: 不外推到旧版本或 nightly；不提供吞吐、质量、显存或成本保证。

Open questions: 目标 GPU 上的启动、质量与性能验证尚需执行。
