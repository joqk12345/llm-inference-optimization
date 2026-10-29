# PagedAttention / vLLM（SOSP 2023）

Source: Woosuk Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention.”  
URL: https://arxiv.org/abs/2309.06180  
Published: SOSP 2023

Supports:

- PagedAttention 借鉴操作系统分页与虚拟内存思想，以 block 管理动态增长的 KV Cache。
- 论文系统通过减少碎片和冗余复制来提高可用于 batching 的 KV 容量。
- 论文报告的吞吐提升有明确对照系统和实验条件，不能脱离论文设置写成 vLLM 的固定收益。

Does not support:

- 任意工作负载都达到固定的 90% 或更高显存利用率。
- PagedAttention 自动等价于 Prefix Caching 已启用。
- 仅凭 block 化就能推出某个端到端加速倍数。

Owner: 第6章作者  
Purpose: 支撑 PagedAttention 的机制与原始实验边界  
Evidence grade: A  
Assumptions: 使用论文版本 arXiv:2309.06180v1  
Open questions: 正文如引用吞吐数字，需补论文表号和完整实验条件  
Handoff: 第6章技术审校
