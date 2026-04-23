#!/usr/bin/env python3
"""
生成可直接导入Word的HTML格式文档
使用方法: python export_to_word.py
"""

import os
import re
import html

# 配置
BOOK_TITLE = "LLM推理优化实战"
AUTHOR = "编著"

CHAPTERS = [
    ("preface.md", "前言"),
    ("content-summary.md", "内容简介"),
    ("chapter01-introduction.md", "第1章 重新理解推理这件事"),
    ("chapter02-technology-landscape.md", "第2章 技术全景与趋势"),
    ("chapter03-gpu-basics.md", "第3章 GPU基础"),
    ("chapter04-environment-setup.md", "第4章 环境搭建"),
    ("chapter05-llm-inference-basics.md", "第5章 LLM推理基础"),
    ("chapter06-kv-cache-optimization.md", "第6章 KV Cache优化"),
    ("chapter07-request-scheduling.md", "第7章 请求调度策略"),
    ("chapter08-quantization.md", "第8章 量化技术"),
    ("chapter09-speculative-sampling.md", "第9章 投机采样"),
    ("chapter10-production-deployment.md", "第10章 生产环境部署"),
    ("chapter11-advanced-topics.md", "第11章 高级话题"),
]

def read_markdown(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def md_to_html_simple(content):
    """简化版Markdown转HTML，保持基本格式"""
    lines = content.split('\n')
    html_lines = []
    in_code_block = False

    for line in lines:
        line = line.strip()

        # 跳过frontmatter
        if line == '---':
            continue

        # 处理代码块
        if line.startswith('```'):
            if not in_code_block:
                html_lines.append('<pre style="background:#f5f5f5;padding:10px;font-family:Courier New;font-size:10.5pt">')
                in_code_block = True
            else:
                html_lines.append('</pre>')
                in_code_block = False
            continue

        if in_code_block:
            html_lines.append(html.escape(line))
            continue

        # 处理标题
        if line.startswith('# '):
            title = line[2:]
            html_lines.append(f'<h1 style="text-align:center;font-size:16pt;font-weight:bold">{title}</h1>')
        elif line.startswith('## '):
            title = line[3:]
            html_lines.append(f'<h2 style="font-size:14pt;font-weight:bold;margin-top:20pt">{title}</h2>')
        elif line.startswith('### '):
            title = line[4:]
            html_lines.append(f'<h3 style="font-size:12pt;font-weight:bold;margin-top:15pt">{title}</h3>')
        elif line.startswith('#### '):
            title = line[5:]
            html_lines.append(f'<h4 style="font-size:11pt;font-weight:bold">{title}</h4>')
        elif line.startswith('##### '):
            title = line[6:]
            html_lines.append(f'<h5 style="font-size:10.5pt;font-weight:bold">{title}</h5>')
        # 处理表格
        elif line.startswith('|'):
            # 简化处理：跳过表格
            continue
        # 处理加粗
        elif line:
            # 处理加粗文本
            line = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
            # 处理行内代码
            line = re.sub(r'`(.+?)`', r'<code style="font-family:Courier New;font-size:10pt">\1</code>', line)
            html_lines.append(f'<p style="font-size:10.5pt;line-height:1.5">{line}</p>')

    return '\n'.join(html_lines)

def generate_html_doc():
    """生成完整HTML文档"""
    html_parts = []

    # HTML头部（兼容Word）
    html_parts.append('''<!DOCTYPE html>
<html xmlns:o="urn:schemas-microsoft-com:office:office"
xmlns:w="urn:schemas-microsoft-com:office:word"
xmlns="http://www.w3.org/TR/REC-html40">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: "Times New Roman", "宋体", serif; font-size: 10.5pt; line-height: 1.5; }}
h1 {{ text-align: center; font-size: 16pt; font-weight: bold; }}
h2 {{ font-size: 14pt; font-weight: bold; margin-top: 20pt; }}
h3 {{ font-size: 12pt; font-weight: bold; margin-top: 15pt; }}
h4 {{ font-size: 11pt; font-weight: bold; }}
h5 {{ font-size: 10.5pt; font-weight: bold; }}
pre {{ background: #f5f5f5; padding: 10px; font-family: "Courier New"; font-size: 10pt; }}
code {{ font-family: "Courier New"; font-size: 10pt; background: #f0f0f0; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #000; padding: 8px; text-align: left; }}
th {{ background: #f0f0f0; }}
blockquote {{ border-left: 3px solid #ccc; margin-left: 0; padding-left: 10px; color: #666; }}
</style>
</head>
<body>'''.format(title=BOOK_TITLE))

    # 添加书名
    html_parts.append(f'<h1>{BOOK_TITLE}</h1>')
    html_parts.append(f'<p style="text-align:center">{AUTHOR}</p>')

    # 处理每个章节
    for chapter_file, chapter_title in CHAPTERS:
        if os.path.exists(chapter_file):
            content = read_markdown(chapter_file)
            html_content = md_to_html_simple(content)
            html_parts.append(html_content)

    # HTML尾部
    html_parts.append('</body></html>')

    return '\n'.join(html_parts)

if __name__ == '__main__':
    html_content = generate_html_doc()

    output_file = 'LLM推理优化实战_清华格式.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f'HTML文档已生成: {output_file}')
    print('导入Word方法：')
    print('1. 打开Word')
    print('2. 文件 -> 打开')
    print('3. 选择 "所有文件" 类型')
    print(f'4. 选择 {output_file}')
    print('5. 文件 -> 另存为 .docx 格式')
