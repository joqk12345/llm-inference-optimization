# Orca Iteration-Level Scheduling（OSDI 2022）

Source: Gyeong-In Yu et al., “Orca: A Distributed Serving System for Transformer-Based Generative Models.”  
URL: https://www.usenix.org/conference/osdi22/presentation/yu  
Published: OSDI 2022

Supports:

- 自回归生成具有多 iteration 特征；request-level 静态 batch 会让新请求等待并让已完成请求受批次约束。
- Orca 提出 iteration-level scheduling，并结合 selective batching。

Does not support:

- 把 Orca 的论文速度数字外推为所有 continuous batching 实现的固定收益。
- 仅凭“连续批处理”四个字推断具体框架的抢占、公平性或 token budget 行为。

Owner: 第7章作者  
Purpose: 支撑迭代级调度的机制起点  
Evidence grade: A  
Assumptions: 使用 USENIX 开放论文版本  
Open questions: vLLM/SGLang 的当前实现需另引版本化官方文档或源码  
Handoff: 第7章技术审校
