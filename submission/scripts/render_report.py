"""Render report.md -> report.pdf.

Enforces the contest's 1000-word cap, then typesets via pandoc and pdfLaTeX
with the preamble in report_header.tex (XCharter text, matching newtx math,
booktabs tables). Requires pandoc and a TeX Live installation on PATH.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HEADER = Path(__file__).with_name("report_header.tex")
FIGURES = Path(__file__).with_name("figures")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--template", default="submission/report.md")
    p.add_argument("--out", default="submission/report.pdf")
    p.add_argument("--max-words", type=int, default=1000)
    args = p.parse_args()

    md_text = Path(args.template).read_text(encoding="utf-8")

    # Word count excluding fenced code, image directives, table separators,
    # and display/inline math (formulae are not prose).
    body = re.sub(r"```[\s\S]*?```", "", md_text)
    body = re.sub(r"^\s*\|[-:|\s]+\|\s*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"\$\$[\s\S]*?\$\$", "", body)
    body = re.sub(r"\$[^$\n]+\$", "", body)
    word_count = len(re.findall(r"\b[\w'-]+\b", body))
    if word_count > args.max_words:
        sys.exit(f"report exceeds {args.max_words}-word cap: {word_count}")
    print(f"word count: {word_count} / {args.max_words}")

    for tool in ("pandoc", "pdflatex"):
        if shutil.which(tool) is None:
            sys.exit(f"render dependency missing: {tool} not on PATH")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    template_dir = Path(args.template).parent
    # pdflatex runs in pandoc's temporary directory, so the logo directory
    # must be supplied as an absolute path at render time.
    graphics = tempfile.NamedTemporaryFile(
        "w", suffix=".tex", delete=False, encoding="utf-8"
    )
    graphics.write("\\graphicspath{{%s/}}\n" % FIGURES.resolve().as_posix())
    graphics.close()
    cmd = [
        "pandoc",
        args.template,
        "-o",
        args.out,
        "--pdf-engine=pdflatex",
        "--fail-if-warnings",
        "-H",
        str(HEADER),
        "-H",
        graphics.name,
        "--citeproc",
        "--csl",
        str(HEADER.with_name("ieee.csl")),
        "--bibliography",
        str(template_dir / "references.bib"),
        "-M",
        "link-citations=true",
        "-V",
        "geometry:margin=15mm",
        "-V",
        "fontsize=10pt",
        "-V",
        "colorlinks=true",
        "-V",
        "linkcolor=reportlink",
        "-V",
        "urlcolor=reportlink",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        Path(graphics.name).unlink(missing_ok=True)
    if res.stderr.strip():
        print(res.stderr.strip(), file=sys.stderr)
    if res.returncode != 0:
        sys.exit(f"pandoc failed:\n{res.stdout}")
    print(f"wrote {args.out}")

    if shutil.which("pdftotext") is None:
        print("pdftotext not found; skipping the PDF-side word-cap re-check")
        return
    text = subprocess.run(
        ["pdftotext", args.out, "-"], capture_output=True, text=True
    ).stdout
    pdf_words = len(text.split())
    print(f"pdf word count (pdftotext): {pdf_words} / {args.max_words}")
    if pdf_words > args.max_words:
        sys.exit("rendered PDF exceeds the word cap under a plain word count")


if __name__ == "__main__":
    main()
