# vLLM Production Metrics（稳定版文档）

Source: vLLM official documentation, “Production Metrics.”  
URL: https://docs.vllm.ai/en/stable/usage/metrics/

Supports:

- OpenAI-compatible server 在 `/metrics` 暴露 Prometheus 指标。
- 当前稳定文档列出 TTFT、TPOT、队列、prefill/decode、KV 使用率、请求数与 token counter 等指标名。
- 文档明确指标存在 deprecation policy，因此书中示例需要标注核对日期。

Does not support:

- 使用不存在或旧版命名的 `vLLM:*`/下划线指标。
- 把固定告警阈值当作所有模型和业务的默认值。
- 只凭缓存命中率判断缓存正确性或经济收益。

Owner: 第10章作者  
Purpose: 支撑 vLLM 可观测性示例  
Evidence grade: A  
Assumptions: 链接指向 stable，出版前仍需锁定具体版本  
Open questions: 最终出版版本应记录文档版本和核对日期  
Handoff: 第10章技术审校
