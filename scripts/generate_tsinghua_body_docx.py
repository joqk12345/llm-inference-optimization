#!/usr/bin/env python3
"""
Generate a DOCX manuscript body using the Tsinghua University Press style rules.

The converter is intentionally small and conservative: it handles the Markdown
features used by this repository and maps them to publisher-facing Word styles.
"""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
PUBLISH_DIR = ROOT / "publish"
OUTPUT = PUBLISH_DIR / "LLM推理优化实战_正文_清华体例.docx"

BOOK_TITLE = "LLM推理优化实战"
AUTHOR_LINE = "编著"

MANUSCRIPT_FILES = [
    ROOT / "content-summary.md",
    ROOT / "preface.md",
    ROOT / "chapters/chapter01-introduction.md",
    ROOT / "chapters/chapter02-technology-landscape.md",
    ROOT / "chapters/chapter03-gpu-basics.md",
    ROOT / "chapters/chapter04-environment-setup.md",
    ROOT / "chapters/chapter05-llm-inference-basics.md",
    ROOT / "chapters/chapter06-kv-cache-optimization.md",
    ROOT / "chapters/chapter07-request-scheduling.md",
    ROOT / "chapters/chapter08-quantization.md",
    ROOT / "chapters/chapter09-speculative-sampling.md",
    ROOT / "chapters/chapter10-production-deployment.md",
    ROOT / "chapters/chapter11-advanced-topics.md",
    ROOT / "appendix-a-tools-resources.md",
    ROOT / "appendix-b-troubleshooting.md",
    ROOT / "appendix-c-benchmarks-roi.md",
    ROOT / "appendix-d-exercise-answer-notes.md",
    ROOT / "generated/references.md",
]


def set_run_font(run, *, size: float, east_asia: str = "宋体", latin: str = "Times New Roman", bold: bool = False):
    run.font.name = latin
    run._element.rPr.rFonts.set(qn("w:eastAsia"), east_asia)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.bold = bold


def set_paragraph_font(paragraph, *, size: float, east_asia: str = "宋体", latin: str = "Times New Roman", bold: bool = False):
    for run in paragraph.runs:
        set_run_font(run, size=size, east_asia=east_asia, latin=latin, bold=bold or bool(run.bold))


def configure_styles(doc: Document) -> None:
    styles = doc.styles

    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.first_line_indent = Pt(21)
    normal.paragraph_format.space_after = Pt(3)

    heading_specs = {
        "Heading 1": (16, "黑体", WD_ALIGN_PARAGRAPH.CENTER, None),
        "Heading 2": (15, "黑体", WD_ALIGN_PARAGRAPH.LEFT, None),
        "Heading 3": (14, "黑体", WD_ALIGN_PARAGRAPH.LEFT, None),
        "Heading 4": (12, "黑体", WD_ALIGN_PARAGRAPH.LEFT, Pt(21)),
        "Heading 5": (10.5, "黑体", WD_ALIGN_PARAGRAPH.LEFT, Pt(21)),
    }
    for style_name, (size, font, alignment, left_indent) in heading_specs.items():
        style = styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), font)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.alignment = alignment
        style.paragraph_format.first_line_indent = None
        style.paragraph_format.left_indent = left_indent
        style.paragraph_format.space_before = Pt(9)
        style.paragraph_format.space_after = Pt(6)


def configure_sections(doc: Document) -> None:
    for section in doc.sections:
        section.top_margin = Cm(2.54)
        section.bottom_margin = Cm(2.54)
        section.left_margin = Cm(3.0)
        section.right_margin = Cm(2.6)
        section.footer_distance = Cm(1.4)


def add_page_number(section) -> None:
    footer = section.footer
    paragraph = footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run("- ")

    field_begin = OxmlElement("w:fldChar")
    field_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    field_end = OxmlElement("w:fldChar")
    field_end.set(qn("w:fldCharType"), "end")

    run = paragraph.add_run()
    run._r.append(field_begin)
    run._r.append(instr)
    run._r.append(field_end)
    paragraph.add_run(" -")
    set_paragraph_font(paragraph, size=10.5)


def strip_frontmatter(text: str) -> str:
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        if end != -1:
            return text[end + 4 :].lstrip()
    return text


def clean_markdown_text(text: str) -> str:
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1（\2）", text)
    text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    return text


def add_inline_runs(paragraph, text: str, *, size: float = 10.5, paragraph_bold: bool = False) -> None:
    text = clean_markdown_text(text)
    token_re = re.compile(r"(\*\*.*?\*\*|`.*?`)")
    pos = 0
    for match in token_re.finditer(text):
        if match.start() > pos:
            run = paragraph.add_run(text[pos : match.start()])
            set_run_font(run, size=size, bold=paragraph_bold)
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            set_run_font(run, size=size, east_asia="黑体", bold=True)
        else:
            run = paragraph.add_run(token[1:-1])
            set_run_font(run, size=9, latin="Courier New")
        pos = match.end()
    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        set_run_font(run, size=size, bold=paragraph_bold)


def add_body_paragraph(doc: Document, text: str, *, first_line_indent: bool = True, bold: bool = False) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.first_line_indent = Pt(21) if first_line_indent else None
    add_inline_runs(paragraph, text, paragraph_bold=bold)


def add_heading(doc: Document, raw_line: str) -> None:
    level = min(len(raw_line) - len(raw_line.lstrip("#")), 5)
    title = raw_line[level:].strip()
    paragraph = doc.add_paragraph(style=f"Heading {level}")
    paragraph.add_run(clean_markdown_text(title))
    specs = {
        1: (16, "黑体", WD_ALIGN_PARAGRAPH.CENTER, None),
        2: (15, "黑体", WD_ALIGN_PARAGRAPH.LEFT, None),
        3: (14, "黑体", WD_ALIGN_PARAGRAPH.LEFT, None),
        4: (12, "黑体", WD_ALIGN_PARAGRAPH.LEFT, Pt(21)),
        5: (10.5, "黑体", WD_ALIGN_PARAGRAPH.LEFT, Pt(21)),
    }
    size, font, alignment, left_indent = specs[level]
    paragraph.alignment = alignment
    paragraph.paragraph_format.first_line_indent = None
    paragraph.paragraph_format.left_indent = left_indent
    set_paragraph_font(paragraph, size=size, east_asia=font, bold=True)


def parse_table(lines: list[str]) -> list[list[str]]:
    rows = []
    for line in lines:
        if re.match(r"^\|[\s\-:|]+\|?$", line.strip()):
            continue
        cells = [clean_markdown_text(cell.strip()) for cell in line.strip().strip("|").split("|")]
        rows.append(cells)
    return rows


def add_table(doc: Document, rows: list[list[str]]) -> None:
    if not rows:
        return
    cols = max(len(row) for row in rows)
    table = doc.add_table(rows=len(rows), cols=cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    for row_idx, row in enumerate(rows):
        for col_idx in range(cols):
            cell = table.cell(row_idx, col_idx)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            text = row[col_idx] if col_idx < len(row) else ""
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if row_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.first_line_indent = None
            paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            add_inline_runs(paragraph, text, size=9, paragraph_bold=(row_idx == 0))
            if row_idx == 0:
                set_paragraph_font(paragraph, size=9, east_asia="黑体", bold=True)


def add_code_block(doc: Document, code_lines: list[str]) -> None:
    for line in code_lines:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.first_line_indent = None
        paragraph.paragraph_format.left_indent = Pt(21)
        paragraph.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        paragraph.paragraph_format.space_after = Pt(0)
        run = paragraph.add_run(line)
        set_run_font(run, size=9, latin="Courier New")


def process_markdown(doc: Document, path: Path) -> None:
    lines = strip_frontmatter(path.read_text(encoding="utf-8")).splitlines()
    i = 0
    in_code = False
    code_lines: list[str] = []
    unordered_counter = 0

    while i < len(lines):
        raw = lines[i].rstrip()
        line = raw.strip()

        if line.startswith("```"):
            unordered_counter = 0
            if in_code:
                add_code_block(doc, code_lines)
                code_lines = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue

        if in_code:
            code_lines.append(raw)
            i += 1
            continue

        if not line or line == "---":
            unordered_counter = 0
            i += 1
            continue

        if line.startswith("#"):
            unordered_counter = 0
            add_heading(doc, line)
            i += 1
            continue

        if line.startswith("|"):
            unordered_counter = 0
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            add_table(doc, parse_table(table_lines))
            continue

        if line.startswith(">"):
            unordered_counter = 0
            add_body_paragraph(doc, line.lstrip("> ").strip(), first_line_indent=False)
            i += 1
            continue

        bullet = re.match(r"^([-*+])\s+(.*)$", line)
        ordered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if bullet:
            unordered_counter += 1
            add_body_paragraph(doc, f"（{unordered_counter}）{bullet.group(2)}", first_line_indent=True)
        elif ordered:
            unordered_counter = 0
            add_body_paragraph(doc, f"{ordered.group(1)}. {ordered.group(2)}", first_line_indent=True)
        else:
            unordered_counter = 0
            add_body_paragraph(doc, line)
        i += 1


def add_cover(doc: Document) -> None:
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(120)
    run = title.add_run(BOOK_TITLE)
    set_run_font(run, size=22, east_asia="黑体", bold=True)

    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.paragraph_format.space_before = Pt(24)
    run = author.add_run(AUTHOR_LINE)
    set_run_font(run, size=14, east_asia="宋体")
    doc.add_page_break()


def add_toc_placeholder(doc: Document) -> None:
    paragraph = doc.add_paragraph(style="Heading 1")
    paragraph.add_run("目录")
    set_paragraph_font(paragraph, size=16, east_asia="黑体", bold=True)

    note = doc.add_paragraph()
    note.paragraph_format.first_line_indent = Pt(21)
    add_inline_runs(note, "请在 Word 中右键更新域，以生成正式目录。")
    doc.add_page_break()


def main() -> None:
    PUBLISH_DIR.mkdir(exist_ok=True)
    doc = Document()
    configure_styles(doc)
    configure_sections(doc)
    add_page_number(doc.sections[0])

    add_cover(doc)
    add_toc_placeholder(doc)

    for index, path in enumerate(MANUSCRIPT_FILES):
        if not path.exists():
            continue
        if index > 0:
            doc.add_page_break()
        process_markdown(doc, path)

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
