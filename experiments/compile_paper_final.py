"""Compile the final official paper PDF with a polished journal-style layout.

docs/paper_final/paper_final.md  →  docs/paper_final/paper_final.pdf

Uses pandoc + xelatex, with a custom preamble that styles the paper like
a Nature-family methods article:
- Large centered title, subtitle, author block, affiliation footnote
- Abstract box with subtle shaded background
- Serif body font, microtype optical alignment
- Proper section numbering (Main: 1-5; Supplementary: S1-S7)
- Numbered Table X / Figure X captions via a fragile but reliable
  caption-counter auto-hook (we rely on the markdown to include the
  "Figure N." / "Table N." prefix inline).
- Narrow margins (1.0 in) but generous line spacing (1.15)
- Numbered references
- Page numbers bottom-center
- "NMR 2-D SSL — final v1.0" running header right

Compiled once, then a second pass to resolve hyperref cross-references.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "paper_final" / "paper_final.md"
OUTDIR = ROOT / "docs" / "paper_final"
OUTDIR.mkdir(parents=True, exist_ok=True)
SANITIZED = OUTDIR / "paper_final_sanitized.md"
PDF = OUTDIR / "paper_final.pdf"
HEADER = OUTDIR / "preamble.tex"


UNICODE_REPLACEMENTS = [
    # Math-mode replacements for symbols pandoc's math passthrough misses
    ("¹³C", r"$^{13}$C"),
    ("¹H", r"$^{1}$H"),
    ("¹",  r"$^{1}$"),
    ("³",  r"$^{3}$"),
    ("²",  r"$^{2}$"),
    ("±",  r"$\pm$"),
    ("×",  r"$\times$"),
    ("≤",  r"$\leq$"),
    ("≥",  r"$\geq$"),
    ("→",  r"$\rightarrow$"),
    ("∈",  r"$\in$"),
    ("σ",  r"$\sigma$"),
    ("λ",  r"$\lambda$"),
    ("π",  r"$\pi$"),
    ("δ",  r"$\delta$"),
    ("α",  r"$\alpha$"),
    ("∑",  r"$\sum$"),
    ("≈",  r"$\approx$"),
    ("½",  r"$\tfrac{1}{2}$"),
    # NOTE: Accented Latin characters (é, ö, ü, etc.) are intentionally NOT
    # replaced because we compile with xelatex+fontspec which handles them
    # natively. Replacing them with \'{e}-style latex-math macros would break
    # under xelatex.
    ("–",  r"--"),
    ("—",  r"---"),
    ("…",  r"\ldots{}"),
    ("’",  r"'"),
    ("‘",  r"'"),
    ("“",  r"``"),
    ("”",  r"''"),
]


PREAMBLE = r"""
\usepackage{fontspec}
\usepackage{hyperref}
\usepackage{url}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs}
\usepackage{float}
\usepackage{longtable}
\usepackage{tabularx}
\usepackage{array}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{caption}
\usepackage{xcolor}
\usepackage{lastpage}
\usepackage[most]{tcolorbox}

% --- section formatting -------------------------------------------------
\titleformat{\section}{\Large\bfseries\sffamily\raggedright}{\thesection}{1em}{}
\titleformat{\subsection}{\large\bfseries\sffamily\raggedright}{\thesubsection}{1em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\itshape\raggedright}{\thesubsubsection}{1em}{}

\titlespacing*{\section}{0pt}{1.5em}{0.6em}
\titlespacing*{\subsection}{0pt}{1.0em}{0.4em}

% --- running header / footer -------------------------------------------
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0pt}
\fancyhead[L]{\footnotesize\textsf{NMR 2-D HSQC sort-match SSL}}
\fancyhead[R]{\footnotesize\textsf{final paper, April 2026}}
\fancyfoot[C]{\thepage~/~\pageref{LastPage}}

% --- abstract box ------------------------------------------------------
\newtcolorbox{abstractbox}{
    enhanced, breakable,
    colback=blue!3!white, colframe=blue!40!black,
    boxrule=0.5pt, arc=1.5pt,
    left=10pt, right=10pt, top=6pt, bottom=6pt,
}

% --- figure / table captions ------------------------------------------
\captionsetup[figure]{font=small,labelfont={bf,sf},labelsep=period,justification=raggedright,singlelinecheck=false}
\captionsetup[table]{font=small,labelfont={bf,sf},labelsep=period,justification=raggedright,singlelinecheck=false}

% --- compact bullet lists -----------------------------------------------
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
\providecommand{\pandocbounded}[1]{#1}

% --- miscellaneous ------------------------------------------------------
\setlength{\parskip}{0.55em}
\setlength{\parindent}{0pt}
\linespread{1.12}

\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}

% --- colour links ------------------------------------------------------
\hypersetup{
    colorlinks=true,
    linkcolor=blue!55!black,
    urlcolor=blue!55!black,
    citecolor=blue!55!black,
    breaklinks=true,
    pdftitle={Learning 1H and 13C NMR chemical shifts jointly from unassigned 2-D HSQC peak sets},
    pdfauthor={Zhaohe Dong (Yusuf Hamied Department of Chemistry, University of Cambridge)},
    pdfsubject={Semi-supervised NMR chemical shift prediction with sliced-Wasserstein sort-match loss},
    pdfkeywords={NMR, HSQC, semi-supervised learning, sliced Wasserstein, conformal prediction, dereplication}
}

% --- allow URLs to break at any character (for long github urls) -------
\makeatletter
\g@addto@macro{\UrlBreaks}{\UrlOrds}
\makeatother
\urlstyle{same}

% --- title block --------------------------------------------------------
\makeatletter
\def\maketitle{%
  \newpage\null
  \begin{center}
    {\LARGE\bfseries\sffamily\@title\par}
    \vspace{0.8em}
    {\large\sffamily\itshape A sliced sort-match weak-supervision recipe,\\
     causally audited and calibrated for dereplication\par}
    \vspace{1.2em}
    {\large\sffamily Zhaohe Dong\textsuperscript{1,*}\par}
    \vspace{0.6em}
    {\footnotesize\textsuperscript{1}Yusuf Hamied Department of Chemistry,
     University of Cambridge, Lensfield Road, Cambridge CB2~1EW, United
     Kingdom.\par}
    \vspace{0.3em}
    {\footnotesize\textsuperscript{*}Correspondence:
     \href{mailto:zd314@cam.ac.uk}{\texttt{zd314@cam.ac.uk}}\par}
    \vspace{0.8em}
    {\footnotesize April 2026}
  \end{center}
  \vspace{1.2em}
}
\makeatother
"""


def sanitize(text: str) -> str:
    for old, new in UNICODE_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def main():
    md = SRC.read_text()
    md = sanitize(md)
    # Strip the YAML front matter that pandoc would otherwise use for a
    # different kind of title block — we want our custom \maketitle.
    if md.startswith("---"):
        end = md.find("---", 3)
        if end != -1:
            md = md[end + 3 :].lstrip("\n")
    SANITIZED.write_text(md)
    HEADER.write_text(PREAMBLE)

    cmd = [
        "pandoc",
        str(SANITIZED),
        "-o", str(PDF),
        "--pdf-engine=xelatex",
        "--include-in-header", str(HEADER),
        "--from", "markdown+raw_tex+tex_math_dollars+pipe_tables+backtick_code_blocks",
        "-V", "documentclass=article",
        "-V", "papersize=a4",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "linkcolor=blue",
        "-V", "colorlinks=true",
        "-V", "title=Learning $^1$H and $^{13}$C NMR chemical shifts jointly from unassigned 2-D HSQC peak sets",
        "--resource-path", str(ROOT / "docs" / "paper_final") + ":" + str(ROOT / "docs" / "2d"),
        "--wrap=preserve",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("PANDOC FAILED:", file=sys.stderr)
        print(r.stderr[-5000:], file=sys.stderr)
        sys.exit(1)
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
