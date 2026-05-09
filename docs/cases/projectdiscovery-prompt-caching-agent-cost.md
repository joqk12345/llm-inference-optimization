---
id: "docs-cases-projectdiscovery-prompt-caching-agent-cost"
title: "ProjectDiscovery Prompt Caching 案例研究 - 多步 Agent 成本优化"
slug: "docs-cases-projectdiscovery-prompt-caching-agent-cost"
date: "2026-05-09"
type: "case-study"
topics:
  - "case-studies"
  - "production-deployment"
  - "advanced-systems"
  - "kv-cache"
concepts:
  - "cost-optimization"
  - "kv-cache"
  - "prefix-caching"
  - "agent-infrastructure"
  - "observability"
tools: []
architecture_layer:
  - "production-systems"
learning_stage: "production"
optimization_axes:
  - "cost"
  - "latency"
  - "operability"
  - "quality"
related:
  - "chapters-chapter10-production-deployment"
  - "chapters-chapter11-advanced-topics"
  - "docs-refs"
references:
  - "https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching"
status: "published"
display_order: 217
---
# ProjectDiscovery Prompt Caching 案例研究 - 多步 Agent 成本优化

**来源**: ProjectDiscovery Blog - https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching  
**发布日期**: 2026年4月10日  
**主题**: 在安全测试 Agent 平台 Neo 中,通过 prompt caching breakpoints、动态内容后移和稳定模板把多步任务的 LLM 成本显著降下来。

---

## 先说结论

ProjectDiscovery 这篇文章很适合放进本书第 10 章和第 11 章,因为它把“围绕 KV Cache 设计 Agent”写成了一套可执行的 prompt 结构工程:

- 静态系统提示词用长 TTL 缓存,跨用户共享。
- 静态工具定义排序并缓存,动态工具后置。
- 会话历史用滑动窗口 breakpoint,每一步只处理新增部分。
- 工作记忆、runtime context、skills 等动态内容从 prefix 中移走,放到尾部。
- provider routing 保持缓存 locality,避免同一 prompt 被分散到互不共享缓存的供应商路径。

它的核心判断是:

**Agent 成本不是线性增长,多步任务会不断重发历史上下文。Prompt caching 是把这条成本曲线压下来的结构性手段。**

ProjectDiscovery 报告的结果包括:cache hit rate 从 7% 提升到 84%,整体 LLM 成本节省 59%,后续优化阶段达到 66%-70% 的节省。工程上应该把这些数字视为该平台、该模型、该供应商计费体系下的生产结果,不能直接外推到所有 Agent 系统。

---

## 为什么多步 Agent 特别适合缓存

传统聊天往往只有一两轮,system prompt 较短。Agent 系统不同:

- 一次任务可能有 20-40+ 次 LLM step。
- 每一步会重发 system prompt、工具定义、历史消息和工具结果。
- 多 agent 架构会把这个开销继续放大。
- conversation 随 step 线性增长,但累计输入成本会接近二次增长。

ProjectDiscovery 的 Neo 平均任务包含 26 个步骤和 40 次工具调用,system prompt 超过 20K tokens。对这种负载来说,只优化模型单价不够,必须减少重复 prefix 的重复计算和重复计费。

---

## 三个 Breakpoint

文章基于 Anthropic prompt caching 的 `cache_control` 机制,最多使用 4 个 breakpoint,实际主要使用 3 个。

### BP1: 静态系统提示词

第一个 breakpoint 标记最后一个静态 system message。关键点不是“系统提示词能缓存”这么简单,而是:

- 要跳过 Working Memory、Relevant Skills、Runtime Context 这类动态 system message。
- 静态系统提示词要在不同用户、不同线程、不同请求之间 byte-identical。
- TTL 使用 1 小时,而不是默认 5 分钟,让同类 agent 的系统提示词在业务高峰期持续 warm。

这对应本书第 10 章的生产成本治理:共享静态前缀的价值来自“跨用户复用”,不是单个 session 内的小优化。

### BP3: 静态工具定义

文章先讲 BP3,因为工具定义的位置会影响整体缓存链。做法是:

- 静态工具排在前面,动态 per-user subagent 或用户定制工具排在后面。
- 静态工具内部保持稳定排序,例如按名称排序。
- 标记最后一个静态工具定义,形成 `[system prompt -> BP1] [static tools -> BP3]` 的共享 cache chain。

这个细节值得写进书里:工具定义越多,越不能随意调整顺序。动态增删工具虽然看起来灵活,但会破坏 prefix cache。

### BP2: conversation sliding window

BP2 标记 conversation 中最后一个 tool result,让每一步只为 BP2 之后新增的消息付出完整处理成本。

文章还提到 Anthropic 的一个实现约束:如果 cache breakpoint 前有超过 20 个 content blocks,并且你修改了更早的内容,可能无法命中缓存。ProjectDiscovery 的处理方式是在每 18 个 blocks 加中间 breakpoint,支持更长的多步对话,退化时也尽量保持 partial caching。

对本书来说,这里的重点是:

**不要只设计一个“缓存点”,要设计能随任务增长而滑动的缓存结构。**

---

## 最大收益: 把动态内容移出 Prefix

文章里最重要的经验是 relocation trick。原始结构把 working memory、skills context、runtime context 放在 BP1 和 BP3 中间。由于这些内容每一步都变,它会让后面的工具定义和对话历史缓存全部失效。

优化方式是:

1. 从 system messages 中找出动态段落。
2. 把它们从 prefix 中移除。
3. 合并成一个尾部 user message。
4. 用类似 `<system-reminder>` 的包裹方式告诉模型这是运行时上下文,不是用户的新任务。

结构变化如下:

```
错误结构:
  static system -> BP1
  dynamic working memory
  static tools -> BP3
  conversation -> BP2

更好结构:
  static system -> BP1
  static tools -> BP3
  conversation -> BP2
  dynamic runtime context at tail
```

ProjectDiscovery 报告这一项把 cache rate 从个位数直接推到约 74%。这也是最应该吸收到第 10 章 checklist 的实践:低 cache hit rate 时,先检查是否有动态内容夹在可缓存 prefix 中间。

---

## 小优化也有工程价值

文章后半部分列了几项把 cache rate 从 74% 推到 84% 的小优化,它们适合直接变成生产 checklist。

### 稳定模板变量

系统提示词里的 `{{current_datetime}}`、`{{env_vars_list}}`、`{{task_workspace}}` 不应该在静态 prompt 渲染阶段展开成真实值。静态 prefix 应保留稳定 placeholder,真实值通过尾部 Runtime Context 注入。

### 冻结时间

时间不要每一步重新取当前秒级时间。更稳的做法是:

- 每个 task run 开始时冻结一次时间。
- 输出 date-only 或任务粒度时间。
- 避免每一步都因为 timestamp 改变而破坏 tail cache。

### Provider routing 保持缓存局部性

如果同一模型请求有时走 Anthropic Direct,有时走 Bedrock 或 Vertex,即使 prompt 完全相同也可能无法共享缓存。ProjectDiscovery 的做法是优先走 Anthropic Direct,只有故障时 fallback。

这对多供应商架构是一个提醒:provider routing 不只是 SLA 和价格策略,也会影响 prompt cache locality。

### Tool message part-level marking

并行工具调用可能在 SDK wire level 展开成多个 tool response。如果把 cache marker 放在整个 tool message 上,可能一次消耗多个 breakpoint slots。文章的处理是只标记最后一个 tool message 的最后一个 content part。

这说明 prompt caching 的边界不只在“逻辑消息数组”,还在 SDK 实际发出的 wire format。生产系统要用真实请求日志验证 cache markers 的位置。

---

## 应该加入书里的工程清单

适合放进第 10 章的 checklist:

1. 静态 system prompt 与静态 tools 是否 byte-identical。
2. 动态内容是否全部后移到尾部。
3. 工具定义是否稳定排序,动态工具是否后置。
4. 时间、路径、环境变量是否使用稳定 placeholder。
5. conversation 是否有 sliding-window cache breakpoint。
6. provider routing 是否破坏缓存局部性。
7. 并行工具调用是否意外消耗多个 breakpoint slots。
8. 指标是否按 stream/task 拆分,而不是只看全局平均 cache rate。

---

## 这篇案例最想留下的判断

ProjectDiscovery 的经验说明,Agent 成本优化不是简单地“打开 prompt caching”。真正有效的是围绕缓存机制重排 prompt 结构:

- 静态内容尽量前置并共享。
- 动态内容尽量后置并局部化。
- 会话历史用滑动窗口缓存。
- 工具定义保持确定性。
- provider 和 SDK wire format 都纳入缓存设计。

这和本书前面对 KV Cache / Prefix Caching 的系统判断是一致的:缓存不是附加优化,而是 Agent 基础设施的架构约束。
