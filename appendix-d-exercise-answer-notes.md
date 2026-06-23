---
id: "appendix-d-exercise-answer-notes"
title: "附录D: 练习参考提示"
slug: "appendix-d-exercise-answer-notes"
date: "2026-05-13"
type: "appendix"
topics:
  - "exercise-notes"
status: "draft"
display_order: 16
---

# 附录D 练习参考提示

本附录用于承接正文中不宜展开的练习参考提示。正式出版时，可根据编辑安排将其作为教师资源、配套代码说明或在线补充材料发布。

## D.1 第10章练习提示

### D.1.1 部署 vLLM 到 Kubernetes

参考答案应包含 Deployment、Service、GPU 资源声明、健康检查和模型配置。重点不是复制一份 YAML，而是解释这些配置如何共同保证服务可启动、可发现、可观测、可恢复。

### D.1.2 搭建监控系统

参考答案应包含 Prometheus scrape 配置、Grafana 仪表盘指标、告警阈值和排障路径。核心指标至少包括 TTFT、TPOT、请求失败率、GPU 利用率、KV Cache 使用率和队列长度。

### D.1.3 建立 ROI 监控

参考答案应说明如何从请求日志中提取 input tokens、output tokens、GPU 时间、模型名称和缓存命中状态，并计算单位 token 成本、单位请求成本和优化前后的节省金额。

## D.2 第11章练习提示

### D.2.1 搭建简单 Jupyter Agent

参考答案应包含 kernel 生命周期管理、代码执行、文件读写、错误捕获和超时控制。生产化时还需要加入沙箱、权限、资源配额和审计日志。

### D.2.2 异构硬件部署实验

参考答案应明确训练、prefill、decode、rollout 等工作负载的资源画像，并比较不同硬件组合下的吞吐、延迟、显存占用和单位成本。

### D.2.3 Context Engineering 实践

参考答案应围绕稳定前缀、上下文外部化、工具调用 schema、任务状态复述和错误轨迹保留展开。评估指标应包括 cache hit rate、平均上下文长度、工具调用成功率、任务完成率和单位任务成本。
