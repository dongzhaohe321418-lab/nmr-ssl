"""Build a Journal of Cheminformatics LaTeX submission package.

Produces docs/paper_final/jcheminf_submission/ containing:
    manuscript.tex       -- standalone pandoc-generated LaTeX
    preamble.tex         -- custom preamble (identical to local build)
    figures/*.pdf        -- all figures referenced by the paper
    README.txt           -- submission checklist + compile instructions
and zips it into docs/paper_final/jcheminf_submission.zip.

Follows the Journal of Cheminformatics (Springer Nature / BMC) manuscript
file guidelines:

    "LaTeX documents with figures and tables compressed into a .zip
    format. We will compile these into a PDF for peer review."
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_MD = ROOT / "docs" / "paper_final" / "paper_final_sanitized.md"
PREAMBLE = ROOT / "docs" / "paper_final" / "preamble.tex"
FIGURES_DIR = ROOT / "docs" / "2d" / "figures"
OUT_DIR = ROOT / "docs" / "paper_final" / "jcheminf_submission"
ZIP_PATH = ROOT / "docs" / "paper_final" / "jcheminf_submission.zip"
ARTIFACT_MAP = ROOT / "docs" / "paper_final" / "ARTIFACT_MAP.md"
COVER_LETTER_PDF = ROOT / "docs" / "paper_final" / "cover_letter.pdf"

FIGURES_NEEDED = [
    "fig_v4_headline.pdf",
    "fig_h_zero.pdf",
    "fig_label_sweep.pdf",
    "fig_wrong_struct_v4.pdf",
]


README = """\
Journal of Cheminformatics submission package
==============================================

Title:   Learning ^1H and ^13C NMR chemical shifts jointly from
         unassigned 2-D HSQC peak sets
Author:  Zhaohe Dong (zd314@cam.ac.uk)
         Yusuf Hamied Department of Chemistry,
         University of Cambridge, UK
Date:    April 2026

Article type:
    Research article (methodology)

Contents of this archive
------------------------

    manuscript.tex      Main LaTeX source. Self-contained: compiles
                        with xelatex to produce the full 19-page PDF
                        (main text + supplementary information).
    preamble.tex        Custom preamble loaded by manuscript.tex via
                        \\input{preamble.tex}.  Contains font, margin,
                        float-placement, abstract-box, and running-
                        header setup.  No non-standard packages.
    figures/            All figures referenced by the paper:
                        - fig_v4_headline.pdf   (Figure 1)
                        - fig_h_zero.pdf        (Figure 2)
                        - fig_label_sweep.pdf   (Figure 3)
                        - fig_wrong_struct_v4.pdf (Figure 4)

Compile instructions
--------------------

    xelatex manuscript.tex
    xelatex manuscript.tex          # second pass resolves cross-refs

Engine: xelatex (required for fontspec + Unicode chemical notation).
Packages: all from TeX Live 2024 / MacTeX 2024; no external .sty files.

Declarations
------------

All sections required by the journal submission guidelines are
included in the manuscript under the "Declarations" top-level header:

    Availability of data and materials
    Code availability
    Abbreviations
    Competing interests
    Funding
    Authors' contributions
    Acknowledgements
    Ethics approval and consent to participate
    Consent for publication

Data and code availability
--------------------------

All code, data filters, trained model checkpoints, per-seed result
JSON files, and figure-generation scripts are permanently archived at

    https://github.com/dongzhaohe321418-lab/nmr-ssl

under the MIT license (software) and CC-BY 4.0 (documentation).
The main experimental pipeline is reproduced by

    python3 experiments/run_option_b_master.py --seeds 0 1 2

(approximately 26 minutes on an Apple M4 Pro with MPS).

AI usage disclosure
-------------------

Code, figures, and manuscript drafts were developed with AI assistance
(Anthropic Claude Opus 4.6) under the author's direction. The author
verified every methodological choice, numerical result, and final
claim; every empirical number reported in this paper is reproducible
from the public code repository linked above.
"""


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("COMMAND FAILED:", " ".join(cmd), file=sys.stderr)
        print(r.stderr[-4000:], file=sys.stderr)
        sys.exit(1)
    return r


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    (OUT_DIR / "figures").mkdir()

    # Step 1: pandoc -> standalone .tex
    tex_raw = OUT_DIR / "manuscript_body.tex"
    cmd = [
        "pandoc",
        str(SRC_MD),
        "-o", str(tex_raw),
        "--standalone",
        "--from", "markdown+raw_tex+tex_math_dollars+pipe_tables+backtick_code_blocks",
        "-V", "documentclass=article",
        "-V", "papersize=a4",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "title=Learning $^1$H and $^{13}$C NMR chemical shifts jointly from unassigned 2-D HSQC peak sets",
        "--wrap=preserve",
    ]
    run(cmd)
    body = tex_raw.read_text()

    # Step 2: rewrite absolute figure paths to ./figures/
    body = body.replace(
        "/Users/ericdong/nmr-ssl/docs/2d/figures/",
        "figures/",
    )
    body = body.replace("../2d/figures/", "figures/")

    # Step 3: swap out pandoc's \usepackage hook so the custom preamble
    # is loaded in-line instead of via --include-in-header, and drop the
    # pandoc-inserted hyperref/xcolor blocks that duplicate ours. We
    # insert \input{preamble.tex} right after \documentclass's last
    # pandoc-default \usepackage.
    preamble_include = "\n% --- Journal of Cheminformatics submission preamble ---\n\\input{preamble.tex}\n"
    # Insert right before \begin{document}
    body = body.replace("\\begin{document}", preamble_include + "\n\\begin{document}", 1)

    manuscript = OUT_DIR / "manuscript.tex"
    manuscript.write_text(body)
    tex_raw.unlink()

    # Step 4: copy preamble
    shutil.copy(PREAMBLE, OUT_DIR / "preamble.tex")

    # Step 5: copy figures
    for fig in FIGURES_NEEDED:
        src = FIGURES_DIR / fig
        if not src.exists():
            print(f"MISSING: {src}", file=sys.stderr)
            sys.exit(1)
        shutil.copy(src, OUT_DIR / "figures" / fig)

    # Step 6: README
    (OUT_DIR / "README.txt").write_text(README)

    # Step 6b: ARTIFACT_MAP.md and cover_letter.pdf bundled with the zip
    if ARTIFACT_MAP.exists():
        shutil.copy(ARTIFACT_MAP, OUT_DIR / "ARTIFACT_MAP.md")
    if COVER_LETTER_PDF.exists():
        shutil.copy(COVER_LETTER_PDF, OUT_DIR / "cover_letter.pdf")

    # Step 7: zip it
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT_DIR.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(OUT_DIR.parent))

    total = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file())
    print(f"wrote {ZIP_PATH}")
    print(f"unpacked dir:  {OUT_DIR}")
    print(f"total size:    {total/1024:.1f} KiB")
    print("contents:")
    for p in sorted(OUT_DIR.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(OUT_DIR)}  ({p.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
