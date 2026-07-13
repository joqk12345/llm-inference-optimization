# Roofline Model（2009）

Source: Samuel Williams, Andrew Waterman, David Patterson, “Roofline: An Insightful Visual Performance Model for Multicore Architectures.”  
URL: https://crd.lbl.gov/assets/pubs_presos/parlab08-roofline-talk.pdf

Supports:

- 可达到的性能上界由峰值计算能力与内存带宽乘算术强度两者中的较小值约束。
- Roofline 用于判断优化应该优先面向计算还是数据移动。

Does not support:

- 不经过 shape、dtype、kernel 和内存层级测量就给任意算子分配固定算术强度。
- 仅用 `nvidia-smi` 的 GPU 利用率判断 compute-bound 或 memory-bound。

Owner: 第3章作者  
Purpose: 支撑 Roofline 机制与使用边界  
Evidence grade: A  
Assumptions: 使用 LBNL/伯克利原始材料  
Open questions: 可补正式期刊版本 DOI  
Handoff: 第3章技术审校
