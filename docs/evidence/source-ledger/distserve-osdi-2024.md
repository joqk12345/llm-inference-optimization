# DistServe / Prefill-Decode Disaggregation（OSDI 2024）

Source: Yinmin Zhong et al., “DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving.”  
URL: https://arxiv.org/abs/2401.09670  
Published: OSDI 2024

Supports:

- Prefill 与 decode 具有不同资源特征，混部会产生干扰并耦合两阶段资源规划。
- PD 分离需要同时处理 KV 传输、放置、并行策略和 TTFT/TPOT SLO。
- 论文性能数字绑定其模型、集群、SLO 和对照系统。

Does not support:

- PD 分离在所有负载上都优于聚合部署。
- 固定指定 H100 做 prefill、A100 做 decode 就是普适最优配置。
- 忽略 KV 传输和两套队列的复杂度。

Owner: 第7章作者  
Purpose: 支撑 PD 分离的机制与工程边界  
Evidence grade: A  
Assumptions: 使用 arXiv:2401.09670  
Open questions: 如引用 goodput 数字需补实验表号与完整 SLO  
Handoff: 第7章技术审校
