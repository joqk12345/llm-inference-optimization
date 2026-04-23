#!/usr/bin/env python3
"""完整转换markdown到HTML"""
import os
import re

def escape_html(text):
    """转义HTML特殊字符"""
    return (text.replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;'))

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return process_content(content)

def process_content(content):
    lines = content.split('\n')
    html = []
    in_code = False
    in_table = False
    table_data = []

    for line in lines:
        line = line.strip()

        # 跳过frontmatter
        if line == '---':
            continue

        # 代码块
        if line.startswith('```'):
            if in_code:
                html.append('</pre>')
                in_code = False
            else:
                html.append('<pre style="background:#f5f5f5;padding:10px;font-family:Courier New;font-size:9pt">')
                in_code = True
            continue

        if in_code:
            html.append(escape_html(line))
            continue

        # 表格
        if line.startswith('|'):
            if not in_table:
                in_table = True
                table_data = []
            # 跳过分隔行
            if re.match(r'^[\|\-\s:]+$', line):
                continue
            cells = [c.strip() for c in line.split('|')[1:-1]]
            table_data.append(cells)
            continue
        elif in_table and table_data:
            html.append(make_table(table_data))
            table_data = []
            in_table = False

        if not line:
            html.append('')
            continue

        # 标题
        if line.startswith('# '):
            title = line[2:]
            html.append(f'<h1 style="text-align:center;font-size:22pt;font-weight:bold;margin-top:24pt">{title}</h1>')
        elif line.startswith('## '):
            title = line[3:]
            html.append(f'<h2 style="font-size:16pt;font-weight:bold;margin-top:20pt">{title}</h2>')
        elif line.startswith('### '):
            title = line[4:]
            html.append(f'<h3 style="font-size:14pt;font-weight:bold;margin-top:16pt">{title}</h3>')
        elif line.startswith('#### '):
            title = line[5:]
            html.append(f'<h4 style="font-size:12pt;font-weight:bold;margin-top:12pt">{title}</h4>')
        elif line.startswith('##### '):
            title = line[6:]
            html.append(f'<h5 style="font-size:10.5pt;font-weight:bold;margin-top:8pt">{title}</h5>')
        # 引用
        elif line.startswith('> '):
            text = format_inline(line[2:])
            html.append(f'<blockquote style="border-left:3px solid #ccc;margin:10px 0;padding-left:15px;color:#666">{text}</blockquote>')
        # 列表
        elif line.startswith('- ') or line.startswith('* '):
            text = format_inline(line[2:])
            html.append(f'<li style="margin:5px 0">{text}</li>')
        # 有序列表
        elif re.match(r'^\d+\. ', line):
            m = re.match(r'^(\d+)\. (.*)', line)
            if m:
                num, text = m.groups()
                text = format_inline(text)
                html.append(f'<li style="margin:5px 0">{num}. {text}</li>')
        # 普通段落
        else:
            text = format_inline(line)
            html.append(f'<p style="font-size:10.5pt;line-height:1.6;margin:6px 0">{text}</p>')

    # 处理最后的表格
    if in_table and table_data:
        html.append(make_table(table_data))

    return '\n'.join(html)

def make_table(data):
    if not data:
        return ''
    html = ['<table style="border-collapse:collapse;width:100%;margin:10px 0">']
    # 表头
    html.append('<tr style="background:#e8e8e8">')
    for cell in data[0]:
        html.append(f'<th style="border:1px solid #000;padding:6px 10px;font-weight:bold;text-align:center">{cell}</th>')
    html.append('</tr>')
    # 数据行
    for row in data[1:]:
        html.append('<tr>')
        for cell in row:
            html.append(f'<td style="border:1px solid #000;padding:6px 10px">{cell}</td>')
        html.append('</tr>')
    html.append('</table>')
    return '\n'.join(html)

def format_inline(text):
    # 加粗 **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # 斜体 *text*
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # 行内代码 `code`
    text = re.sub(r'`(.+?)`', r'<code style="font-family:Courier New;font-size:9pt;background:#f0f0f0">\1</code>', text)
    return text

def main():
    chapters = [
        ('content-summary.md', '内容简介'),
        ('preface.md', '前言'),
        ('chapters/chapter01-introduction.md', '第1章'),
        ('chapters/chapter02-technology-landscape.md', '第2章'),
        ('chapters/chapter03-gpu-basics.md', '第3章'),
        ('chapters/chapter04-environment-setup.md', '第4章'),
        ('chapters/chapter05-llm-inference-basics.md', '第5章'),
        ('chapters/chapter06-kv-cache-optimization.md', '第6章'),
        ('chapters/chapter07-request-scheduling.md', '第7章'),
        ('chapters/chapter08-quantization.md', '第8章'),
        ('chapters/chapter09-speculative-sampling.md', '第9章'),
        ('chapters/chapter10-production-deployment.md', '第10章'),
        ('chapters/chapter11-advanced-topics.md', '第11章'),
    ]

    html_parts = ['''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>LLM推理优化实战</title>
<style>
body { font-family: "Times New Roman", "宋体", serif; font-size: 10.5pt; line-height: 1.6; }
h1 { text-align: center; font-size: 22pt; font-weight: bold; margin-top: 24pt; }
h2 { font-size: 16pt; font-weight: bold; margin-top: 20pt; }
h3 { font-size: 14pt; font-weight: bold; margin-top: 16pt; }
h4 { font-size: 12pt; font-weight: bold; margin-top: 12pt; }
h5 { font-size: 10.5pt; font-weight: bold; margin-top: 8pt; }
p { font-size: 10.5pt; line-height: 1.6; margin: 6px 0; }
pre { background: #f5f5f5; padding: 10px; font-family: Courier New; font-size: 9pt; margin: 8px 0; border-left: 3px solid #0066cc; }
code { font-family: Courier New; font-size: 9pt; background: #f0f0f0; padding: 1px 3px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th, td { border: 1px solid #000; padding: 6px 10px; font-size: 9.5pt; }
th { background: #e8e8e8; font-weight: bold; text-align: center; }
blockquote { border-left: 3px solid #ccc; margin: 10px 0; padding-left: 15px; color: #666; }
li { margin: 5px 0; }
b { font-weight: bold; }
i { font-style: italic; }
</style>
</head>
<body>
<h1 style="text-align:center;font-size:24pt">LLM推理优化实战</h1>
<p style="text-align:center">编著</p>
''']

    total_chars = 0
    for filepath, title in chapters:
        if os.path.exists(filepath):
            print(f"处理: {title}...")
            content = process_file(filepath)
            html_parts.append(content)
            total_chars += len(content)
            html_parts.append('<div style="page-break-after:always"></div>')

    html_parts.append('</body></html>')

    output = '\n'.join(html_parts)

    with open('LLM推理优化实战_完整版.html', 'w', encoding='utf-8') as f:
        f.write(output)

    print(f"完成! 总字符数: {total_chars:,}")

if __name__ == '__main__':
    main()
