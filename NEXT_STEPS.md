# 当前编辑与验证任务

当前书稿为四部分、11章。阅读入口见 [SUMMARY](SUMMARY.md)，历史启动计划保存在仓库文件 `docs/plans/2025-01-26-launch-checklist-legacy.md`，不进入站点阅读路径。

## 已完成的内容整理

- 将第2章诊断与选型方法前置，趋势案例后置。
- 将模型专属 KV 和 DeepInfer 配置迁入带来源快照边界的实现记录。
- 将附录C改为实验与成本计算模板，移除无原始证据的跑分。
- 更新阅读路径、代码覆盖与社区说明。

## 后续需要真实环境完成

1. 按[贯穿实验](code/chapter10/README.md)在适配的 GPU 主机上运行交替 A/B，保存不可变环境、启动日志、请求级结果及失败记录。
2. 补充低前缀复用负载、服务器指标与 profiler trace，写出收益条件及失败分析；没有结果时不填性能数字。
3. 为调度、量化、投机分别设计控制变量与质量门槛，当前实验比较器只接受前缀缓存差异。
4. 为[模型 KV](docs/cards/deepseek-v41-flash-kv.md)与[DeepInfer](docs/cards/deepinfer-runtime.md)实现记录补不可变上游 commit、目标硬件、启动验证及失效说明。
5. 正文定稿后重新生成出版社 DOCX，完成分页和逐页视觉审阅。既有 `publish/` 文件是历史导出，不会随 Markdown 编辑自动更新。

## 提交前检查

```bash
npm run docs:build
python3 -m unittest discover -s tests -v
python3 scripts/wordcount.py --write
git diff --check
```

文档构建包含知识导航生成、结构与引用检查。CPU 回归不能替代真实模型质量、GPU 性能或生产故障演练。
