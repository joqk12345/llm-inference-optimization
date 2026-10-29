---
id: "adr-0001-stable-core-and-versioned-adapters"
title: "采用稳定内核与版本化适配层"
slug: "adr-0001-stable-core-and-versioned-adapters"
date: "2026-04-26"
type: "plan"
topics:
  - "book-architecture"
  - "editorial-review"
concepts:
  - "inference-fundamentals"
  - "operability"
tools: []
architecture_layer:
  - "foundations"
  - "production-and-operations"
learning_stage: "orientation"
optimization_axes:
  - "latency"
  - "cost"
  - "operability"
related:
  - "docs-book-longevity-review"
  - "readme"
references: []
status: "published"
display_order: 210
---
# ADR-0001：采用稳定内核与版本化适配层

- **状态**：已接受
- **日期**：2026-04-26
- **决策人**：编辑与维护者

## 背景

LLM 推理的模型、硬件、框架 API、指标名称和部署方式变化很快。若把这些变化直接写进主叙事，书的概念价值会被版本细节稀释，读者也难以判断一个性能数字是否仍适用。当前书稿还存在同一主题跨章节重复、章节 6 编号缺口以及本地绝对链接等维护问题。

本书的长期目标是成为“性能工程的诊断与决策教材”，而不是某一年某个 runtime 的参数手册。

## 决策

采用“**稳定内核 + 版本化适配层**”的编辑架构：

1. 主章节优先讲与框架无关的机制、问题分解、实验方法、权衡和运营闭环。
2. vLLM、SGLang、硬件、模型和版本特性放入适配卡、案例或 playbook，并记录验证日期、版本/commit、硬件、命令、限制和替代路径。
3. 所有性能数字必须附带模型、输入/输出长度、并发、硬件、精度、runtime 版本、warm-up、分位数和质量口径；否则只作为示意或定性结论。
4. 保留现有 11 章 URL，采用渐进迁移，避免一次性重命名造成链接断裂。
5. 每次适配层更新不得改变稳定章节的概念定义；过期内容应归档，而不是悄悄覆盖历史结论。

## 取舍

- **优点**：降低版本漂移；提高跨硬件、跨模型迁移能力；便于读者复现实验和做回归；旧链接保持兼容。
- **代价**：需要维护卡片元数据和验证记录；正文不再提供“一个命令适配所有环境”的短答案；迁移会经历一段重复期。
- **不选择的方案**：不按年份重写整本书，也不把单一框架作为所有概念的定义来源。

## 执行边界

- 稳定章节：目标是解释“为什么”和“如何证明”。
- 适配卡：目标是解释“在某版本、某硬件上怎么做”。
- 案例研究：目标是记录“在明确负载下发生了什么”，不得外推为普遍保证。
- 市场与趋势材料：必须标注日期、来源和证据等级，不进入核心推理链。

详细章节迁移和验收标准见：[十年可用性审稿与重构方案](../book-longevity-review.md)。

## 验收

后续重构以以下条件为门槛：稳定内核可以独立完成 baseline → profile → 优化 → 质量/性能回归；适配材料没有未说明版本的默认值；文档构建、结构 lint、引用 lint 和死链检查均通过。
