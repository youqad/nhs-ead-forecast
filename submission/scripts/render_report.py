"""Render report.md -> report.pdf.

Enforces the contest's 1000-word cap before rendering. Two rendering
backends are tried in order: weasyprint (preferred, supports proper
typography), then a minimal HTML write as fallback for sandbox environments
without cairo / pango.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Ensure WeasyPrint can find pango/cairo on macOS where Homebrew installs them
# under /opt/homebrew/lib. Set before importing weasyprint.
if sys.platform == "darwin":
    _existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    _brew_lib = "/opt/homebrew/lib"
    if _brew_lib not in _existing.split(":"):
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = (
            f"{_brew_lib}:{_existing}" if _existing else _brew_lib
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--template", default="submission/report.md")
    p.add_argument("--out", default="submission/report.pdf")
    p.add_argument("--max-words", type=int, default=1000)
    args = p.parse_args()

    md_text = Path(args.template).read_text(encoding="utf-8")

    # Word count excluding fenced code, image directives, and table separators.
    body = re.sub(r"```[\s\S]*?```", "", md_text)
    body = re.sub(r"^\s*\|[-:|\s]+\|\s*$", "", body, flags=re.MULTILINE)
    word_count = len(re.findall(r"\b[\w'-]+\b", body))
    if word_count > args.max_words:
        sys.exit(f"report exceeds {args.max_words}-word cap: {word_count}")
    print(f"word count: {word_count} / {args.max_words}")

    try:
        import markdown  # type: ignore
        from weasyprint import HTML  # type: ignore
    except ImportError as e:
        sys.exit(
            f"render dependencies missing ({e}). "
            f"run with markdown and weasyprint installed."
        )

    html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])
    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>"
        "@page{size:A4;margin:1.8cm;}"
        "body{font-family:'Times New Roman',Times,serif;font-size:12pt;line-height:1.4;margin:0;}"
        "p:first-of-type{font-size:9pt;white-space:nowrap;}"
        "h1{font-family:'Times New Roman',Times,serif;font-size:18pt;margin-top:0;}"
        "h2{font-family:'Times New Roman',Times,serif;font-size:14pt;margin-top:1em;}"
        "code{font-family:'SFMono-Regular',Menlo,Consolas,monospace;font-size:10pt;"
        "background:#f5f5f5;padding:0 3px;word-wrap:break-word;overflow-wrap:anywhere;}"
        "pre{font-family:'SFMono-Regular',Menlo,Consolas,monospace;font-size:9pt;"
        "background:#f5f5f5;padding:6px 8px;margin:0.5em 0;"
        "white-space:pre-wrap;word-wrap:break-word;overflow-wrap:anywhere;"
        "max-width:100%;box-sizing:border-box;}"
        "pre code{background:transparent;padding:0;font-size:inherit;}"
        "table{border-collapse:collapse;margin:0.5em 0;max-width:100%;word-wrap:break-word;}"
        "th,td{border:1px solid #aaa;padding:4px 8px;font-size:11pt;word-wrap:break-word;}"
        "th{background:#eee;}"
        "</style></head><body>" + html_body + "</body></html>"
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_doc).write_pdf(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
