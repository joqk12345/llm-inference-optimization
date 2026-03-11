# Knowledge System Architecture

## Source of Truth

- Markdown 正文内容保留在原有内容文件中。
- 元数据保存在每个受管 Markdown 文件的 frontmatter 中。
- 主阅读顺序以 `SUMMARY.md` 为唯一来源。

## Control Plane

- `meta/content-schema.json`
- `meta/taxonomy.yaml`
- `meta/architecture-layers.yaml`
- `meta/learning-stages.yaml`
- `meta/optimization-axes.yaml`

## Build Plane

1. `knowledge:ingest` 为受管内容补齐或更新 frontmatter。
2. `knowledge:graph` 基于 `SUMMARY.md` 和受管内容生成 `generated/graph.json`。
3. `knowledge:generate` 生成自动目录、学习路径、架构图、引用索引。
4. `knowledge:vite-nav` 生成 `.vitepress/knowledge-nav.json`。
5. `knowledge:lint` 校验结构、元数据和引用关系。

## Presentation Plane

- `.vitepress/config.mts` 读取 `.vitepress/knowledge-nav.json`
- `visualizations/` 提供可视化模板页
- `generated/` 提供自动生成页面
