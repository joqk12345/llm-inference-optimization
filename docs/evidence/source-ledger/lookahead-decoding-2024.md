# Lookahead Decoding（2024）

Source: Yichao Fu et al., “Break the Sequential Dependency of LLM Inference Using Lookahead Decoding.”  
URL: https://arxiv.org/abs/2402.02057

Supports:

- Lookahead Decoding 是不依赖辅助 draft model 或外部 datastore 的精确并行解码算法。
- 方法并行产生并验证 n-gram 候选，用更多可并行计算换取更少串行步骤。

Does not support:

- 把 Lookahead Decoding 描述成多个小草稿模型之间选择最长候选。
- 将论文中的最高加速数字外推到任意模型、任务或 GPU 数量。

Owner: 第9章作者  
Purpose: 修正 Lookahead Decoding 的机制描述  
Evidence grade: A  
Assumptions: 使用 arXiv:2402.02057  
Open questions: 如引用性能数字需补论文具体任务、GPU 数与配置  
Handoff: 第9章技术审校
