# KIVI KV Cache 量化（ICML 2024）

Source: Zirui Liu et al., “KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.”  
URL: https://arxiv.org/abs/2402.02750  
Published: ICML 2024

Supports:

- KV Cache 量化与权重量化是不同问题；KIVI 针对 K/V 分布分别设计量化粒度。
- 论文报告的是 2-bit KIVI 在指定模型、实现和负载下的峰值内存、batch 与吞吐结果。

Does not support:

- 用 AWQ/GPTQ 权重量化论文推导 KV Cache 的 MMLU 损失。
- 任意 FP8/INT8/INT4 KV 方案具有固定速度或固定质量损失。
- 不检查 runtime kernel 支持就直接采用论文收益。

Owner: 第6章作者  
Purpose: 支撑 KV 专用量化的机制与边界  
Evidence grade: A  
Assumptions: 使用 arXiv v2  
Open questions: 需补目标推理引擎的实现与版本证据  
Handoff: 第6章技术审校
