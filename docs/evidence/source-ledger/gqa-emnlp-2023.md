# Grouped-Query Attention（EMNLP 2023）

Source: Joshua Ainslie et al., “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.”  
URL: https://arxiv.org/abs/2305.13245  
Published: EMNLP 2023

Supports:

- MQA 使用单个 KV head；GQA 使用多于 1、少于 query head 数的 KV heads。
- 论文研究的是把已有 MHA checkpoint uptrain 为 MQA/GQA，并报告特定实验中的质量—速度折中。

Does not support:

- 对任意模型统一写成 MQA 损失 2.1 MMLU、GQA 损失 0.4 MMLU。
- 服务框架可以在不改变权重的情况下随意把一个既有 MHA 模型切换为 GQA。
- GQA 在所有模型上都是“最佳平衡”。

Owner: 第6章作者  
Purpose: 支撑 MHA/MQA/GQA 机制边界  
Evidence grade: A  
Assumptions: 使用 arXiv v3  
Open questions: 若引用具体质量与速度，需要补齐论文任务、模型和表号  
Handoff: 第6章技术审校
