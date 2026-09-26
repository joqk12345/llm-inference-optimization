# EAGLE（2024）

Source: Yuhui Li et al., “EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty.”  
URL: https://arxiv.org/abs/2401.15077

Supports:

- EAGLE 在特征层进行自回归，并把 token 预测提前一位用于处理特征不确定性。
- 它属于需要训练专用 drafter 的投机采样路线。

Does not support:

- 用“单层草稿、固定 spec_len”概括整个 EAGLE 系列。
- 未阅读 EAGLE-2/EAGLE-3 原始来源就描述其具体演进与框架成熟度。
- EAGLE 对任意模型和负载都是最优方案。

Owner: 第9章作者  
Purpose: 支撑 EAGLE 初代机制  
Evidence grade: A  
Assumptions: 当前只覆盖初代 EAGLE 论文  
Open questions: EAGLE-2、EAGLE-3 需要分别建卡后再写版本演进  
Handoff: 第9章技术审校
