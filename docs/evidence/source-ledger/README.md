# 技术证据账本

本目录为正文中的引用标记提供可追溯 source card；标记格式为 `CITE` 加卡片 slug。每张卡只记录原始来源能够支持的结论、实验条件以及不能外推的部分。

使用规则：

- 正文中的数字和关键机制必须指向具体 card，不能只写“论文/官方 benchmark”。
- 论文报告的速度或质量只在其模型、硬件、实现和负载条件内成立。
- `Evidence grade: A` 表示原始论文、官方文档、源码或可复现实验，不表示其中结论可以无条件泛化。
- 若卡片的 `Open questions` 未解决，正文只能使用机制性结论，不能使用具体性能数字。

当前卡片：

- `pagedattention-sosp-2023`
- `gqa-emnlp-2023`
- `awq-mlsys-2024`
- `gptq-2022`
- `kivi-icml-2024`
- `speculative-decoding-icml-2023`
- `lookahead-decoding-2024`
- `eagle-2024`
- `roofline-2009`
- `orca-osdi-2022`
- `sarathi-serve-osdi-2024`
- `distserve-osdi-2024`
- `vllm-production-metrics-stable`
- `nvidia-h100-product-spec`
- `vllm-installation-gpu-stable`
- `nvidia-container-toolkit-install`

Owner: 技术主编  
Purpose: 让章节级技术论断可追溯、可审计  
Evidence grade: A（索引本身；具体等级见各卡）  
Assumptions: 正文继续使用 source-card slug 作为内部引用标记  
Open questions: 后续如何从卡片自动生成章末参考文献  
Handoff: 引用审校负责人
