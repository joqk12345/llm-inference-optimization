#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a Word document from manuscript markdown files
following Tsinghua University Press formatting requirements.
"""

import os
import re
from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsmap
from docx.oxml import OxmlElement

# ============================================================
# File ordering
# ============================================================
FRONT_MATTER_FILES = [
    "preface.md",
]

CHAPTER_FILES = [
    "chapters/chapter01-introduction.md",
    "chapters/chapter02-technology-landscape.md",
    "chapters/chapter03-gpu-basics.md",
    "chapters/chapter04-environment-setup.md",
    "chapters/chapter05-llm-inference-basics.md",
    "chapters/chapter06-kv-cache-optimization.md",
    "chapters/chapter07-request-scheduling.md",
    "chapters/chapter08-quantization.md",
    "chapters/chapter09-speculative-sampling.md",
    "chapters/chapter10-production-deployment.md",
    "chapters/chapter11-advanced-topics.md",
]

APPENDIX_FILES = [
    "appendix-a-tools-resources.md",
    "appendix-b-troubleshooting.md",
    "appendix-c-benchmarks-roi.md",
]

BACK_MATTER_FILES = [
    "docs/refs.md",
]

# Base directory is the parent of the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# ============================================================
# Font sizes (pt)
# ============================================================
SIZE_SAN_HAO = Pt(16)      # 三号
SIZE_XIAO_SAN = Pt(15)     # 小三号
SIZE_SI_HAO = Pt(14)       # 四号
SIZE_XIAO_SI = Pt(12)      # 小四号
SIZE_WU_HAO = Pt(10.5)     # 五号
SIZE_XIAO_WU = Pt(9)       # 小五号

# ============================================================
# Helper: add styled paragraph
# ============================================================
def add_styled_paragraph(doc, text, font_name_cn="宋体", font_name_en="Times New Roman",
                         size=SIZE_WU_HAO, bold=False, italic=False,
                         alignment=WD_ALIGN_PARAGRAPH.LEFT, first_line_indent=None,
                         left_indent=None, space_before=Pt(0), space_after=Pt(0),
                         line_spacing=1.5):
    p = doc.add_paragraph()
    p.alignment = alignment
    pf = p.paragraph_format
    pf.space_before = space_before
    pf.space_after = space_after
    pf.line_spacing = line_spacing
    if first_line_indent is not None:
        pf.first_line_indent = first_line_indent
    if left_indent is not None:
        pf.left_indent = left_indent

    run = p.add_run(text)
    set_run_font(run, font_name_cn, font_name_en, size, bold, italic)
    return p

# ============================================================
# Helper: set run font
# ============================================================
def set_run_font(run, font_name_cn="宋体", font_name_en="Times New Roman", size=SIZE_WU_HAO, bold=False, italic=False):
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.font.name = font_name_en
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn('w:eastAsia'), font_name_cn)

# ============================================================
# Inline markdown parser for bold/italic/code/strike
# ============================================================
def parse_inline(paragraph, text):
    """Parse inline markdown and add runs to paragraph."""
    pattern = re.compile(r'(\*\*\*(.*?)\*\*\*|\*\*(.*?)\*\*|\*(.*?)\*|`(.*?)`|~~(.*?)~~)')
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            pre = text[pos:m.start()]
            run = paragraph.add_run(pre)
            set_run_font(run)

        content = None
        bold = False
        italic = False
        code = False
        strike = False

        if m.group(2) is not None:
            content = m.group(2)
            bold = True
            italic = True
        elif m.group(3) is not None:
            content = m.group(3)
            bold = True
        elif m.group(4) is not None:
            content = m.group(4)
            italic = True
        elif m.group(5) is not None:
            content = m.group(5)
            code = True
        elif m.group(6) is not None:
            content = m.group(6)
            strike = True

        if content is not None:
            run = paragraph.add_run(content)
            if code:
                set_run_font(run, "Courier New", "Courier New", SIZE_XIAO_WU, bold=False)
            else:
                set_run_font(run, bold=bold, italic=italic)
            if strike:
                run.font.strike = True

        pos = m.end()

    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        set_run_font(run)

# ============================================================
# Remove YAML front matter
# ============================================================
def remove_yaml_front_matter(text):
    if text.startswith("---"):
        idx = text.find("---", 3)
        if idx != -1:
            return text[idx + 3:].lstrip("\n")
    return text

# ============================================================
# Parse a single markdown file into structured blocks
# ============================================================
def parse_markdown(text):
    lines = text.split("\n")
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Horizontal rule
        if re.match(r'^-{3,}\s*$', stripped):
            blocks.append({"type": "hr"})
            i += 1
            continue

        # Code block
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            blocks.append({"type": "code", "lang": lang, "content": "\n".join(code_lines)})
            i += 1
            continue

        # Table
        if stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            filtered = [l for l in table_lines if not re.match(r'^\|[-:\s|]+$', l.strip())]
            if filtered:
                rows = []
                for l in filtered:
                    cells = [c.strip() for c in l.strip("|").split("|")]
                    rows.append(cells)
                blocks.append({"type": "table", "rows": rows})
            continue

        # Heading
        heading_match = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if heading_match:
            level = len(heading_match.group(1))
            content = heading_match.group(2).strip()
            blocks.append({"type": "heading", "level": level, "content": content})
            i += 1
            continue

        # Blockquote
        if stripped.startswith(">"):
            quote_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            blocks.append({"type": "quote", "content": " ".join(quote_lines)})
            continue

        # List item
        list_match = re.match(r'^(\s*)([-*+]|\d+[.\)])\s+(.*)$', line)
        if list_match:
            indent = len(list_match.group(1))
            marker = list_match.group(2)
            content = list_match.group(3)
            i += 1
            while i < len(lines):
                if not lines[i]:
                    i += 1
                    continue
                if re.match(r'^\s*([-*+]|\d+[.\)])\s+', lines[i]):
                    break
                if lines[i].strip().startswith("#") or lines[i].strip().startswith("|") or lines[i].strip().startswith("```"):
                    break
                if len(lines[i]) - len(lines[i].lstrip()) > indent:
                    content += " " + lines[i].strip()
                    i += 1
                else:
                    break
            blocks.append({"type": "list_item", "marker": marker, "content": content, "indent": indent})
            continue

        # Regular paragraph
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            if not lines[i].strip():
                break
            if re.match(r'^#{1,6}\s', lines[i]):
                break
            if lines[i].strip().startswith("|") or lines[i].strip().startswith("```") or lines[i].strip().startswith(">"):
                break
            if re.match(r'^\s*([-*+]|\d+[.\)])\s+', lines[i]):
                break
            if re.match(r'^-{3,}\s*$', lines[i].strip()):
                break
            para_lines.append(lines[i].strip())
            i += 1
        blocks.append({"type": "paragraph", "content": " ".join(para_lines)})

    return blocks

# ============================================================
# Render blocks to docx
# ============================================================
def render_blocks(doc, blocks):
    for block in blocks:
        btype = block["type"]

        if btype == "heading":
            level = block["level"]
            content = block["content"]

            if level == 1:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                pf = p.paragraph_format
                pf.space_before = Pt(24)
                pf.space_after = Pt(18)
                pf.line_spacing = 1.5
                run = p.add_run(content)
                set_run_font(run, "黑体", "Times New Roman", SIZE_SAN_HAO, bold=True)

            elif level == 2:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                pf = p.paragraph_format
                pf.space_before = Pt(18)
                pf.space_after = Pt(12)
                pf.line_spacing = 1.5
                run = p.add_run(content)
                set_run_font(run, "黑体", "Times New Roman", SIZE_XIAO_SAN, bold=True)

            elif level == 3:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                pf = p.paragraph_format
                pf.space_before = Pt(12)
                pf.space_after = Pt(6)
                pf.line_spacing = 1.5
                run = p.add_run(content)
                set_run_font(run, "黑体", "Times New Roman", SIZE_SI_HAO, bold=True)

            elif level == 4:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                pf = p.paragraph_format
                pf.space_before = Pt(6)
                pf.space_after = Pt(6)
                pf.line_spacing = 1.5
                pf.first_line_indent = Cm(0.74)
                run = p.add_run(content)
                set_run_font(run, "黑体", "Times New Roman", SIZE_XIAO_SI, bold=True)

            elif level == 5:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                pf = p.paragraph_format
                pf.space_before = Pt(3)
                pf.space_after = Pt(3)
                pf.line_spacing = 1.5
                pf.first_line_indent = Cm(0.74)
                run = p.add_run(content)
                set_run_font(run, "黑体", "Times New Roman", SIZE_WU_HAO, bold=True)

            else:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                pf = p.paragraph_format
                pf.space_before = Pt(3)
                pf.space_after = Pt(3)
                pf.line_spacing = 1.5
                pf.first_line_indent = Cm(0.74)
                run = p.add_run(content)
                set_run_font(run, "黑体", "Times New Roman", SIZE_WU_HAO, bold=True)

        elif btype == "paragraph":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf = p.paragraph_format
            pf.space_before = Pt(3)
            pf.space_after = Pt(3)
            pf.line_spacing = 1.5
            pf.first_line_indent = Cm(0.74)
            parse_inline(p, block["content"])

        elif btype == "quote":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf = p.paragraph_format
            pf.space_before = Pt(3)
            pf.space_after = Pt(3)
            pf.line_spacing = 1.5
            pf.left_indent = Cm(1.0)
            pf.first_line_indent = Cm(0.74)
            run = p.add_run(block["content"])
            set_run_font(run, italic=True)

        elif btype == "list_item":
            marker = block["marker"]
            content = block["content"]
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf = p.paragraph_format
            pf.space_before = Pt(2)
            pf.space_after = Pt(2)
            pf.line_spacing = 1.5
            pf.left_indent = Cm(0.74)
            pf.first_line_indent = Cm(-0.37)
            run_marker = p.add_run(marker + " ")
            set_run_font(run_marker)
            parse_inline(p, content)

        elif btype == "code":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            pf = p.paragraph_format
            pf.space_before = Pt(6)
            pf.space_after = Pt(6)
            pf.line_spacing = 1.2
            pf.left_indent = Cm(0.74)
            run = p.add_run(block["content"])
            set_run_font(run, "Courier New", "Courier New", SIZE_XIAO_WU)

        elif btype == "table":
            rows = block["rows"]
            if not rows:
                continue
            num_cols = max(len(r) for r in rows)
            table = doc.add_table(rows=len(rows), cols=num_cols)
            table.style = 'Table Grid'
            for ri, row in enumerate(rows):
                for ci, cell_text in enumerate(row):
                    if ci < num_cols:
                        cell = table.cell(ri, ci)
                        cell.text = ""
                        p = cell.paragraphs[0]
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        pf = p.paragraph_format
                        pf.space_before = Pt(2)
                        pf.space_after = Pt(2)
                        run = p.add_run(cell_text)
                        if ri == 0:
                            set_run_font(run, "黑体", "Times New Roman", SIZE_XIAO_WU, bold=True)
                        else:
                            set_run_font(run, "宋体", "Times New Roman", SIZE_XIAO_WU)
            doc.add_paragraph()

        elif btype == "hr":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run("—" * 20)
            set_run_font(run, size=SIZE_WU_HAO)

# ============================================================
# Setup document defaults
# ============================================================
def setup_document_defaults(doc):
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = SIZE_WU_HAO
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.get_or_add_rFonts()
    rFonts.set(qn('w:eastAsia'), '宋体')

    sections = doc.sections[0]
    sections.top_margin = Cm(2.54)
    sections.bottom_margin = Cm(2.54)
    sections.left_margin = Cm(3.17)
    sections.right_margin = Cm(3.17)
    sections.page_height = Cm(29.7)
    sections.page_width = Cm(21.0)
    sections.orientation = WD_ORIENT.PORTRAIT

# ============================================================
# Add page numbers
# ============================================================
def add_page_number(section, fmt="{PAGE}"):
    """Add page number field to section footer."""
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.clear()

    run = p.add_run()
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')

    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = fmt

    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'end')

    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    set_run_font(run, "宋体", "Times New Roman", SIZE_WU_HAO)

# ============================================================
# Add TOC placeholder
# ============================================================
def add_toc(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("目  录")
    set_run_font(run, "黑体", "Times New Roman", SIZE_SAN_HAO, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = p.add_run("（请在此处右键点击 → 更新域 → 更新整个目录，以生成自动目录）")
    set_run_font(run, "宋体", "Times New Roman", SIZE_WU_HAO)

    # Insert TOC field
    p = doc.add_paragraph()
    run = p.add_run()
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')

    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = r' TOC \o "1-3" \h \z \u '

    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')

    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')

    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)

# ============================================================
# Create a new section
# ============================================================
def add_section_break(doc):
    """Add a new-page section break."""
    section = doc.add_section()
    return section

# ============================================================
# Main
# ============================================================
def main():
    doc = Document()
    setup_document_defaults(doc)

    # ========== Title Page ==========
    add_styled_paragraph(doc, "LLM推理性能优化", font_name_cn="黑体", font_name_en="Times New Roman",
                         size=Pt(22), bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                         space_before=Pt(120), space_after=Pt(12))
    add_styled_paragraph(doc, "从原理到生产环境的性能优化实战", font_name_cn="黑体", font_name_en="Times New Roman",
                         size=Pt(16), bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                         space_before=Pt(12), space_after=Pt(60))
    add_styled_paragraph(doc, "（待出版书稿）", font_name_cn="宋体", font_name_en="Times New Roman",
                         size=SIZE_SI_HAO, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                         space_before=Pt(60), space_after=Pt(12))

    doc.add_page_break()

    # ========== Front Matter Section (Roman numerals) ==========
    # Add TOC
    add_toc(doc)
    doc.add_page_break()

    # Process front matter files
    for rel_path in FRONT_MATTER_FILES:
        full_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full_path):
            print(f"Warning: file not found: {full_path}")
            continue
        print(f"Processing front matter: {rel_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            raw = f.read()
        text = remove_yaml_front_matter(raw)
        blocks = parse_markdown(text)
        render_blocks(doc, blocks)

    # Add section break for main content (Arabic numerals)
    main_section = add_section_break(doc)
    main_section.top_margin = Cm(2.54)
    main_section.bottom_margin = Cm(2.54)
    main_section.left_margin = Cm(3.17)
    main_section.right_margin = Cm(3.17)

    # ========== Main Content (Chapters) ==========
    for rel_path in CHAPTER_FILES:
        full_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full_path):
            print(f"Warning: file not found: {full_path}")
            continue
        print(f"Processing chapter: {rel_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            raw = f.read()
        text = remove_yaml_front_matter(raw)
        blocks = parse_markdown(text)
        render_blocks(doc, blocks)
        doc.add_page_break()

    # ========== Appendices ==========
    for rel_path in APPENDIX_FILES:
        full_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full_path):
            print(f"Warning: file not found: {full_path}")
            continue
        print(f"Processing appendix: {rel_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            raw = f.read()
        text = remove_yaml_front_matter(raw)
        blocks = parse_markdown(text)
        render_blocks(doc, blocks)
        doc.add_page_break()

    # ========== Back Matter (References) ==========
    for rel_path in BACK_MATTER_FILES:
        full_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full_path):
            print(f"Warning: file not found: {full_path}")
            continue
        print(f"Processing back matter: {rel_path}")
        with open(full_path, "r", encoding="utf-8") as f:
            raw = f.read()
        text = remove_yaml_front_matter(raw)
        blocks = parse_markdown(text)
        render_blocks(doc, blocks)

    # Save
    output_path = os.path.join(BASE_DIR, "publish", "manuscript.docx")
    doc.save(output_path)
    print(f"\nDocument saved to: {output_path}")
    print("\n注意事项：")
    print("1. 打开文档后，请在目录页右键点击 → 更新域 → 更新整个目录")
    print("2. 前言部分和正文部分已用分节符分隔，可分别设置页码格式（罗马数字/阿拉伯数字）")
    print("3. 如系统缺少 SimSun/黑体字体，Word 会自动替换，建议检查字体设置")
    print("4. 待发布文件统一存放在 publish/ 目录下")

if __name__ == "__main__":
    main()
