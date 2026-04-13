"""Compile the v2 preprint markdown directly to PDF via pandoc + xelatex.

Applies minimal Unicode sanitization, writes a sanitized .md, then lets pandoc
drive xelatex with a --include-in-header preamble that defines the theorem
environment and uses a Nature-style layout.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "2d" / "preprint_2d_draft.md"
OUTDIR = ROOT / "docs" / "2d"
OUTDIR.mkdir(parents=True, exist_ok=True)
SANITIZED = OUTDIR / "preprint_2d_sanitized.md"
PDF = OUTDIR / "preprint_2d.pdf"
HEADER = OUTDIR / "preamble.tex"


UNICODE_REPLACEMENTS = [
    ("¹³C", r"$^{13}$C"),
    ("¹H", r"$^{1}$H"),
    ("¹", r"$^{1}$"),
    ("³", r"$^{3}$"),
    ("²", r"$^{2}$"),
    ("±", r"$\pm$"),
    ("×", r"$\times$"),
    ("≤", r"$\leq$"),
    ("≥", r"$\geq$"),
    ("→", r"$\rightarrow$"),
    ("∈", r"$\in$"),
    ("σ", r"$\sigma$"),
    ("λ", r"$\lambda$"),
    ("π", r"$\pi$"),
    ("δ", r"$\delta$"),
    ("α", r"$\alpha$"),
    ("∑", r"$\sum$"),
    ("≈", r"$\approx$"),
]


PREAMBLE = r"""
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{float}
\newtheorem{theorem}{Theorem}
\setlength{\parskip}{0.5em}
\setlength{\parindent}{0em}
"""


def main():
    md = SRC.read_text()
    for old, new in UNICODE_REPLACEMENTS:
        md = md.replace(old, new)
    # Replace lone ~ (tilde used in "~340k") with "$\sim$"
    md = md.replace(r"$\sim$", r"$\sim$")  # noop placeholder
    # Remove stray tildes that should be math
    md = md.replace(r"\sim$340 k", r"$\sim$340 k")
    SANITIZED.write_text(md)

    HEADER.write_text(PREAMBLE)

    cmd = [
        "pandoc",
        str(SANITIZED),
        "-o", str(PDF),
        "--pdf-engine=xelatex",
        "--include-in-header", str(HEADER),
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "colorlinks=true",
        "-V", "linkcolor=blue!60!black",
        "-V", "urlcolor=blue!60!black",
        "--wrap=preserve",
        "--resource-path", str(OUTDIR),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("PANDOC FAILED:")
        print(r.stderr[-4000:])
        return
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
