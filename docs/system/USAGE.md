# Knowledge System 使用说明

## 一次性构建

```bash
npm run knowledge:build
```

该命令会：

1. 注入或更新 frontmatter
2. 同步 `SUMMARY.md` 分组标题
3. 重建 `generated/graph.json`
4. 重建 `generated/*.md`
5. 重建 `.vitepress/knowledge-nav.json`
6. 执行结构校验

## 本地预览

```bash
npm run docs:dev
```

## 维护规则

- 不手工编辑 `generated/*`
- 不手工编辑 `.vitepress/knowledge-nav.json`
- 新增核心章节时必须同步更新 `SUMMARY.md`
- 新增次级知识文档时，应把文件加入脚本管理清单
