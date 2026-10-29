# LLM推理性能优化

本仓库提供两类阅读入口：

主阅读路径：
- [前言与项目说明](/README)
- [主阅读路径目录](/SUMMARY)（四部分，11章）

补充导航与可视化：
- [自动知识目录](/generated/summary)
- [学习路径总览](/visualizations/path-overview)
- [知识架构总览](/visualizations/architecture-overview)
- [十年可用性审稿与重构方案](/docs/book-longevity-review)

如果你只想按书的主线阅读，从 [主阅读路径目录](/SUMMARY) 开始；如果你想按主题跳读，再结合自动知识目录和可视化总览。整条主线会先建立硬件与推理基础，再进入 KV 管理、请求调度、量化与投机采样，最后收束到生产部署与运行治理，再延伸到高级专题。

实战入口：[前缀缓存贯穿实验](/code/chapter10/README)，按固定负载、基线、单变量优化和回归保留证据。当前提供客户端与 CPU 检查，真实 GPU 结果待补。
