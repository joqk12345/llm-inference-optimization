#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path


def _clean_heading(line: str) -> str:
    # Remove editorial markers in headings for the clean TOC.
    # Examples: "⭐ 新增", "⭐⭐⭐ 2025新增"
    line = line.replace("⭐", "")
    # Remove warning/notes suffixes like: "⚠️ NVIDIA官方支持", "⚠️ 技术评估中".
    # For clean TOC, keep only the structural title.
    line = re.sub(r"\s*⚠️.*$", "", line)
    line = re.sub(r"\s*⚠.*$", "", line)
    # Remove standalone "新增" (and common variants), then normalize whitespace.
    line = re.sub(r"\b2025\s*新增\b", "", line)
    line = re.sub(r"\b新增\b", "", line)
    # Remove common standalone emoji markers if they appear in headings.
    line = re.sub(r"[💡💰📌✅❌🚧]", "", line)
    line = re.sub(r"[ \t]{2,}", " ", line).rstrip()
    # Fix common spacing around punctuation after stripping markers.
    line = re.sub(r"\s+([：:])", r"\1", line)
    return line


def simplify_toc(text: str) -> str:
    out: list[str] = []
    lines = text.splitlines()

    def emit(line: str) -> None:
        # Avoid multiple blank lines.
        if line == "" and (not out or out[-1] == ""):
            return
        out.append(line)

    stop_patterns = [
        re.compile(r"^##\s+完整统计\s*$"),
        re.compile(r"^##\s+V2\+V3融合版主要变化\s*$"),
    ]

    for line in lines:
        if any(p.match(line) for p in stop_patterns):
            break

        if line.startswith("## ") or line.startswith("### ") or line.startswith("#### "):
            emit(_clean_heading(line.rstrip()))
            emit("")

    # Trim trailing blank line.
    while out and out[-1] == "":
        out.pop()

    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a simplified TOC by removing level-3 items (e.g., 1.1.1).")
    parser.add_argument(
        "--in",
        dest="in_path",
        default="docs/table-of-contents-v7.0-detailed.md",
        help="Input TOC markdown (detailed).",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="docs/table-of-contents-v7.0-clean.md",
        help="Output TOC markdown (clean).",
    )
    args = parser.parse_args()

    src = Path(args.in_path)
    dst = Path(args.out_path)
    detailed = src.read_text(encoding="utf-8")
    body = simplify_toc(detailed)

    header = [
        "# LLM推理优化实战 - 目录（v7.0·简化版：去掉三级标题）",
        "",
        f"**更新日期**：{date.today().isoformat()}",
        f"**来源**：`{src.as_posix()}`",
        "",
        "说明：本版本保留到“章/节（例如 1.1、2.3）”，移除“三级条目（例如 1.1.1）”与大段说明文字。",
        "",
        "---",
        "",
    ]

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(header) + body, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
