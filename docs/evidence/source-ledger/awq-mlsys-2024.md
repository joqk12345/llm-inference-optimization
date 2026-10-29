# AWQ（MLSys 2024）

Source: Ji Lin et al., “AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.”  
URL: https://arxiv.org/abs/2306.00978  
Published: MLSys 2024

Supports:

- AWQ 是 activation-aware 的 weight-only PTQ 方法。
- 方法用离线激活统计识别 salient channels，并通过等价缩放降低量化误差；它不是简单地把重要权重永久保留为高精度混合格式。
- 论文的速度数字来自 TinyChat 和指定平台，不等价于任意 vLLM/AWQ 部署的固定速度。

Does not support:

- AWQ 在所有模型、任务和 kernel 上都比 GPTQ 更快或更准。
- AWQ 的 INT8/INT4 具有统一 MMLU 损失。
- “生产环境默认选择 AWQ”这一无条件结论。

Owner: 第8章作者  
Purpose: 支撑 AWQ 原理与实验外推边界  
Evidence grade: A  
Assumptions: 使用论文公开版本  
Open questions: 框架支持矩阵需要另引对应版本官方文档  
Handoff: 第8章技术审校
