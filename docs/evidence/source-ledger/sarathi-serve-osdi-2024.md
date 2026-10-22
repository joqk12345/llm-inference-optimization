# Sarathi-Serve / Chunked Prefill（OSDI 2024）

Source: Amey Agrawal et al., “Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve.”  
URL: https://www.usenix.org/conference/osdi24/presentation/agrawal  
Published: OSDI 2024

Supports:

- 长 prefill 可能阻塞 decode 并放大 inter-token latency。
- chunked prefill 与 stall-free batching 用于控制 prefill 对 decode 的干扰，在论文设置中改善吞吐—延迟权衡。

Does not support:

- 任意固定 chunk size 或 batch token budget 都适用于线上负载。
- “batch 越大越好”或“chunking 必然改善尾延迟”。

Owner: 第7章作者  
Purpose: 支撑 chunked prefill 与调度干扰机制  
Evidence grade: A  
Assumptions: 使用 USENIX OSDI 2024 版本  
Open questions: 具体框架参数需按版本核对  
Handoff: 第7章技术审校
