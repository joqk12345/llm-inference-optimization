---
id: "docs-faq"
title: "Frequently Asked Questions"
slug: "docs-faq"
date: "2026-03-11"
type: "reference"
topics:
  - "community"
  - "reference-materials"
concepts: []
tools:
  - "vllm"
  - "docker"
architecture_layer:
  - "production-systems"
learning_stage: "orientation"
optimization_axes:
  - "operability"
related:
  - "readme"
  - "chapters-chapter04-environment-setup"
  - "appendix-b-troubleshooting"
references: []
status: "published"
display_order: 202
---
# Frequently Asked Questions

## 阅读与费用

### 仓库内容可以免费阅读吗？

仓库中的正文和配套代码可直接阅读，使用条件见 [LICENSE](../LICENSE)。本仓库未提供已上线的 Pro 会员购买、折扣或退款入口；旧稿中的相关承诺已移除。

### 需要深度学习基础吗？

需要 Python 和基本编程能力。第3章介绍 GPU，第4章介绍环境，第5章介绍推理机制。高级专题涉及分布式系统、模型结构和数值方法，可以在完成基础章节后按需阅读。

### 需要 GPU 吗？

阅读、CPU 参考算法和 HTTP 协议回归测试不需要 GPU。真实模型服务和性能实验需要满足目标模型、精度、上下文、并发及运行时要求的硬件，不能用固定显存下限保证所有示例可运行。先完成第3章容量估算和第4章环境检查。

## 版本与验证

### 本书使用什么 vLLM 版本？

当前贯穿实验的执行目标为 `vllm==0.30.0`，详见[版本适配卡](cards/vllm-0.30.0.md)。该卡记录了文档核对范围，仓库尚无对应真实 GPU 性能结果。其他案例保留各自的来源日期和版本边界，不能混用参数或默认值。

### 哪些代码可以运行？

- 第3章：显存预算、GPU 监控和瓶颈诊断示例，GPU 部分需对应环境。
- 第4章：环境检查脚本。
- 第9章：投机采样 CPU 参考实现。
- 第10章：前缀缓存实验客户端，需连接模型服务才能执行完整实验。
- 第11章：Attention CPU 数值参考实现。

在根目录运行 `python3 -m unittest discover -s tests -v` 执行 CPU 回归。覆盖范围与限制见[测试说明](../tests/README.md)。通过 CPU 测试不代表 GPU 后端、模型质量或性能已验证。

### 能用于其他模型或硬件吗？

可以迁移诊断方法，但容量公式、状态结构、低精度 kernel 和参数支持需逐项核对。第11章及案例材料包含异构硬件讨论，不能理解为所有后端都已实测。多模态内容在第11章，核心优化技术在第6—9章。

## 阅读路径与实验

### 从哪里开始？

1. 第1—2章：优化目标、诊断方法与技术地图。
2. 第3—5章：硬件、环境与推理机制。
3. 第6—9章：KV、调度、量化与投机采样。
4. 第10章：生产部署与运行治理。
5. 第11章：按业务需要选择高级专题。

完整目录见 [SUMMARY](../SUMMARY.md)。第5章之后可先阅读[贯穿实验](../code/chapter10/README.md)，将基线与质量门槛带入后续章节。学习时间取决于基础和实验环境，本仓库不承诺固定完成周期。

### 如何证明一次优化有效？

固定环境和负载，只改变一个变量，同时检查质量、错误率、SLO、goodput 与成本，保留失败记录并重复 A/B。默认贯穿实验仅允许比较前缀缓存开关；调度、量化和投机需要另行设计控制变量与质量评测。记录模板见[附录C](../appendix-c-benchmarks-roi.md)。

## 反馈与贡献

### 遇到问题或想参与怎么办？

通过 [GitHub Issues](https://github.com/joqk12345/llm-inference-optimization/issues)提交章节与段落、预期行为、复现命令和环境信息，删除凭证及私有数据。贡献方式见[贡献指南](../CONTRIBUTING.md)。本仓库暂未提供有效的 Discord 邀请或固定 Office Hours 公告，请以仓库中的实际公告为准。
