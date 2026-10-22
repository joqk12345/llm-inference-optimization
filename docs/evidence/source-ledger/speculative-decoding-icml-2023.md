# Speculative Decoding（ICML 2023）

Source: Yaniv Leviathan et al., “Fast Inference from Transformers via Speculative Decoding.”  
URL: https://arxiv.org/abs/2211.17192  
Published: ICML 2023

Supports:

- 算法通过近似模型提出多个 token，由目标模型并行验证。
- 在正确接受/拒绝采样规则下，可以保持目标模型的输出分布。
- 论文的 2–3 倍结果来自 T5-XXL/T5X 的指定设置，不能写成通用收益。

Does not support:

- 所有“小模型 + 大模型”组合天然兼容或一定加速。
- 投机解码主要改善 TTFT；其直接目标是减少串行 decode 步数。
- 接受率、验证成本和尾延迟可以忽略。

Owner: 第9章作者  
Purpose: 支撑投机解码正确性与收益边界  
Evidence grade: A  
Assumptions: 使用 arXiv v2  
Open questions: 另一篇独立 speculative sampling 工作需单独建卡  
Handoff: 第9章技术审校
