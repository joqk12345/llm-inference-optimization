import fs from 'node:fs'
import { defineConfig } from 'vitepress'

const knowledgeSidebar = fs.existsSync('.vitepress/knowledge-nav.json')
  ? JSON.parse(fs.readFileSync('.vitepress/knowledge-nav.json', 'utf8'))
  : []

const repoName = process.env.GITHUB_REPOSITORY?.split('/')[1]
const isUserOrOrgPagesSite = repoName?.endsWith('.github.io')
const base = process.env.VITEPRESS_BASE
  ?? (process.env.GITHUB_ACTIONS === 'true' && repoName && !isUserOrOrgPagesSite
    ? `/${repoName}/`
    : '/')

export default defineConfig({
  base,
  lang: 'zh-CN',
  title: 'LLM推理性能优化',
  description: 'AI Knowledge Graph System for LLM inference optimization',
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: [/^\.\/LICENSE$/, /^\.\.\/LICENSE$/, /^\.\/\.\.\/LICENSE$/, /^\.\/index$/],
  themeConfig: {
    search: {
      provider: 'local'
    },
    nav: [
      { text: '前言', link: '/README' },
      { text: '目录', link: '/SUMMARY' },
      { text: '知识系统', link: '/generated/summary' },
      { text: '可视化', link: '/visualizations/' },
      { text: '参考资料', link: '/docs/refs' }
    ],
    sidebar: [
      {
        text: '知识系统（自动生成）',
        items: [
          { text: '自动目录', link: '/generated/summary' },
          { text: '学习路径', link: '/generated/path' },
          { text: '知识架构', link: '/generated/architecture' },
          { text: '引用索引', link: '/generated/references' }
        ]
      },
      ...knowledgeSidebar,
      {
        text: '可视化',
        items: [
          { text: '总览', link: '/visualizations/' },
          { text: '学习路径总览', link: '/visualizations/path-overview' },
          { text: '知识架构总览', link: '/visualizations/architecture-overview' }
        ]
      },
      {
        text: '次级知识区',
        items: [
          { text: '书籍元数据', link: '/docs/book-metadata' },
          { text: '内容摘要', link: '/docs/content-summary' },
          { text: 'FAQ', link: '/docs/faq' },
          { text: '参考资料汇总', link: '/docs/refs' },
          { text: '关键词', link: '/docs/keywords' },
          { text: '选题背景与特色', link: '/docs/topic-background' },
          { text: '市场需求与发展趋势', link: '/docs/market-analysis' },
          { text: '市场竞品对比分析', link: '/docs/market-comparison' },
          { text: 'Reader Success Stories', link: '/docs/success-stories' }
        ]
      },
      {
        text: '案例研究',
        items: [
          { text: 'Boaz Barak AI 经济学', link: '/docs/cases/boaz-barak-ai-economics' },
          { text: '青稞 AI Infra 2025', link: '/docs/cases/qingke-ai-infra-2025-analysis' },
          { text: 'SGLang INT4 QAT', link: '/docs/cases/sglang-int4-qat-rl-analysis' },
          { text: 'LMSYS MXFP4 QAT', link: '/docs/cases/lmsys-qat-mxfp4-analysis' }
        ]
      },
      {
        text: '系统说明',
        items: [
          { text: '系统架构', link: '/docs/system/ARCHITECTURE' },
          { text: '使用说明', link: '/docs/system/USAGE' },
          { text: '生成文件', link: '/docs/system/GENERATED_FILES' }
        ]
      }
    ]
  }
})
