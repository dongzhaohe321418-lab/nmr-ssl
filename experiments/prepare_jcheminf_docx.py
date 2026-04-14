"""Build a Journal of Cheminformatics Word (.docx) manuscript.

The journal accepts either a LaTeX zip or a Word document with
figures and tables embedded in the body where they are referenced.
This script produces docs/paper_final/manuscript.docx from
paper_final.md via pandoc, after rewriting the raw LaTeX figure
environment and the absolute figure path into pandoc-friendly
markdown image syntax so docx export embeds the figures properly.

Pandoc docx output uses PNG for embedded images (docx has no native
PDF support), so we point at the PNG siblings of the vector figures.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "paper_final" / "paper_final.md"
FIGURES = ROOT / "docs" / "2d" / "figures"
OUT_DIR = ROOT / "docs" / "paper_final"
DOCX = OUT_DIR / "manuscript.docx"
WORK_MD = OUT_DIR / "paper_final_docx.md"


def sanitize_for_docx(text: str) -> str:
    # Strip YAML front matter (pandoc md->docx uses it for metadata but
    # the custom title-page hack in the markdown is LaTeX-specific).
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].lstrip("\n")

    # Replace the raw LaTeX figure environment for fig_wrong_struct_v4
    # with a plain markdown image so pandoc embeds it in the docx body.
    fig_wrong_struct_tex = re.compile(
        r"\\begin\{figure\}.*?"
        r"\\includegraphics\[[^\]]*\]\{[^}]*fig_wrong_struct_v4\.pdf\}"
        r"(?P<rest>.*?)"
        r"\\end\{figure\}",
        re.DOTALL,
    )

    def _replace_fig4(m: re.Match) -> str:
        return (
            "![Figure 4. Joint pass rates on the three wrong-candidate "
            "controls. Green bars are the joint pass rate for the correct "
            "structure; red bars are the joint pass rate for the wrong "
            "candidate. Numbers above each group are the correct-to-wrong "
            "discrimination ratio. The constitutional isomer control "
            "(same molecular formula, different connectivity) is the "
            "chemistry-meaningful number for natural-product "
            "dereplication.]"
            "(../2d/figures/fig_wrong_struct_v4.png)"
        )

    text = fig_wrong_struct_tex.sub(_replace_fig4, text)

    # Change the three pandoc markdown figures from .pdf to .png (docx
    # only embeds raster images).
    text = text.replace("fig_v4_headline.pdf", "fig_v4_headline.png")
    text = text.replace("fig_h_zero.pdf", "fig_h_zero.png")
    text = text.replace("fig_label_sweep.pdf", "fig_label_sweep.png")

    # Drop LaTeX-only directives that are meaningless in docx.
    text = re.sub(r"^\\clearpage\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\\FloatBarrier\s*$", "", text, flags=re.MULTILINE)

    # Convert \subsection*{X} raw latex to a markdown level-2 heading so
    # the Word navigator sees them.
    text = re.sub(
        r"\\subsection\*\{([^}]*)\}",
        lambda m: f"## {m.group(1)}",
        text,
    )

    return text


def main() -> None:
    md = SRC.read_text()
    md = sanitize_for_docx(md)

    # Write title, author, and abstract at the top in pandoc-metadata
    # form so the docx has a real title block.
    header = (
        "---\n"
        "title: |\n"
        "  Learning $^{1}$H and $^{13}$C NMR chemical shifts jointly\n"
        "  from unassigned 2-D HSQC peak sets\n"
        "author: |\n"
        "  Zhaohe Dong^1^\\*\n"
        "  \n"
        "  ^1^ Yusuf Hamied Department of Chemistry, University of\n"
        "  Cambridge, Lensfield Road, Cambridge CB2 1EW, United Kingdom.\n"
        "  \n"
        "  \\* Correspondence: zd314@cam.ac.uk\n"
        "date: April 2026\n"
        "---\n\n"
    )
    md = header + md
    WORK_MD.write_text(md)

    cmd = [
        "pandoc",
        str(WORK_MD),
        "-o", str(DOCX),
        "--from", "markdown+raw_tex+tex_math_dollars+pipe_tables+backtick_code_blocks+header_attributes",
        "--resource-path", str(FIGURES) + ":" + str(OUT_DIR),
        "--wrap=preserve",
        "--standalone",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("PANDOC FAILED:", file=sys.stderr)
        print(r.stderr[-4000:], file=sys.stderr)
        sys.exit(1)

    WORK_MD.unlink()
    print(f"wrote {DOCX} ({DOCX.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
