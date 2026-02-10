#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ChapterStats:
    file: str
    bytes: int
    lines: int
    chars: int
    non_ws: int
    cjk: int
    words: int


def _is_cjk_unified_ideograph(ch: str) -> bool:
    # Basic CJK Unified Ideographs block. This keeps the count simple and stable.
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FFF


def _count_lines(text: str) -> int:
    # Similar to wc -l, but count last line even without trailing newline.
    if not text:
        return 0
    return text.count("\n") + (0 if text.endswith("\n") else 1)


def _run_git(args: list[str]) -> str | None:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None


def _discover_chapters(root: Path, glob: str) -> list[Path]:
    paths = sorted(root.glob(glob))

    # Prefer natural order: chapter01, chapter02, ...
    def key(p: Path) -> tuple[int, str]:
        m = re.search(r"chapter(\d+)", p.name)
        if m:
            return (int(m.group(1)), p.name)
        return (10**9, p.name)

    return sorted(paths, key=key)


def _compute_stats(path: Path, *, display_path: str) -> ChapterStats:
    b = path.read_bytes()
    text = b.decode("utf-8")
    bytes_ = len(b)
    lines = _count_lines(text)
    chars = len(text)
    non_ws = sum(0 if ch.isspace() else 1 for ch in text)
    cjk = sum(1 for ch in text if _is_cjk_unified_ideograph(ch))
    words = len(text.split())
    return ChapterStats(
        # Keep repo-relative paths for stable diffs across machines/CI.
        file=display_path,
        bytes=bytes_,
        lines=lines,
        chars=chars,
        non_ws=non_ws,
        cjk=cjk,
        words=words,
    )


def _sum_stats(rows: Iterable[ChapterStats]) -> ChapterStats:
    total = ChapterStats(file="TOTAL", bytes=0, lines=0, chars=0, non_ws=0, cjk=0, words=0)
    for r in rows:
        total = ChapterStats(
            file="TOTAL",
            bytes=total.bytes + r.bytes,
            lines=total.lines + r.lines,
            chars=total.chars + r.chars,
            non_ws=total.non_ws + r.non_ws,
            cjk=total.cjk + r.cjk,
            words=total.words + r.words,
        )
    return total


def _render_markdown(rows: list[ChapterStats], total: ChapterStats, meta: dict) -> str:
    sha = meta.get("git", {}).get("sha") or "unknown"
    commit_date = meta.get("git", {}).get("commit_date") or "unknown"
    generated_at = meta.get("generated_at") or "unknown"

    header = [
        "# Chapter 1-11 字数统计",
        "",
        f"- Commit: `{sha}`",
        f"- Commit date: `{commit_date}`",
        f"- Generated at (UTC): `{generated_at}`",
        "",
        "说明：",
        "- `non_ws` = 去除空白后的字符数（更接近“字数”口径）",
        "- `cjk` = 汉字数量（U+4E00..U+9FFF）",
        "- `words` = 按空白分词的词数（对中文不敏感，仅作参考）",
        "",
    ]

    cols = ["file", "bytes", "lines", "chars", "non_ws", "cjk", "words"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{r.file}`",
                    str(r.bytes),
                    str(r.lines),
                    str(r.chars),
                    str(r.non_ws),
                    str(r.cjk),
                    str(r.words),
                ]
            )
            + " |"
        )
    lines.append(
        "| "
        + " | ".join(
            [
                f"**{total.file}**",
                str(total.bytes),
                str(total.lines),
                str(total.chars),
                str(total.non_ws),
                str(total.cjk),
                str(total.words),
            ]
        )
        + " |"
    )

    return "\n".join(header + lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute word/char counts for chapters.")
    parser.add_argument(
        "--glob",
        default="chapters/chapter??-*.md",
        help="Glob pattern for chapter markdown files (relative to repo root).",
    )
    parser.add_argument(
        "--out-md",
        default="docs/word-counts.md",
        help="Output markdown path (relative to repo root).",
    )
    parser.add_argument(
        "--out-json",
        default="docs/word-counts.json",
        help="Output json path (relative to repo root).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write outputs to files. Without this flag, prints markdown to stdout.",
    )
    args = parser.parse_args()

    root = Path(".").resolve()
    chapter_paths = _discover_chapters(root, args.glob)
    # Default scope: chapter01..chapter11 (as requested). If you want more,
    # pass a different --glob and/or edit this filter.
    filtered = []
    for p in chapter_paths:
        m = re.search(r"chapter(\d+)", p.name)
        if not m:
            continue
        n = int(m.group(1))
        if 1 <= n <= 11:
            filtered.append(p)
    chapter_paths = filtered
    if not chapter_paths:
        raise SystemExit(f"No chapter files matched glob: {args.glob}")

    rows = []
    for p in chapter_paths:
        rel = p.relative_to(root).as_posix()
        rows.append(_compute_stats(p, display_path=rel))
    total = _sum_stats(rows)

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    sha = os.environ.get("GITHUB_SHA") or _run_git(["rev-parse", "HEAD"]) or "unknown"
    commit_date = _run_git(["show", "-s", "--format=%cI", "HEAD"]) or "unknown"

    meta = {
        "generated_at": generated_at,
        "git": {"sha": sha, "commit_date": commit_date},
        "chapters": [asdict(r) for r in rows],
        "total": asdict(total),
    }

    md = _render_markdown(rows, total, meta)

    if not args.write:
        print(md, end="")
        return 0

    out_md = root / args.out_md
    out_json = root / args.out_json
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_md.write_text(md, encoding="utf-8")
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
