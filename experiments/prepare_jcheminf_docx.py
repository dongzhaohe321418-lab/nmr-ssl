"""Build a Journal of Cheminformatics Word (.docx) manuscript.

The journal accepts either a LaTeX zip or a Word document with
figures and tables embedded in the body where they are referenced.
This script produces docs/paper_final/manuscript.docx from
paper_final.md via pandoc, after rewriting raw LaTeX constructs
(figure environments, reference blocks, sectioning commands) into
pandoc-friendly markdown so the docx export renders everything
correctly with embedded figures, Office MathML equations, and a
properly structured Word navigation pane.

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
REF_DOCX = OUT_DIR / "reference.docx"


# --- Unicode replacements (reverse of compile_paper_final.py) --------
# The source .md has some LaTeX math-mode replacements; for docx we
# want the original unicode so pandoc's docx writer can handle them
# natively. We also convert some LaTeX constructs to markdown.

LATEX_TO_UNICODE = [
    (r"$\pm$", "±"),
    (r"$\times$", "×"),
    (r"$\leq$", "≤"),
    (r"$\geq$", "≥"),
    (r"$\rightarrow$", "→"),
    (r"$\approx$", "≈"),
    (r"$\sigma$", "σ"),
    (r"$\lambda$", "λ"),
    (r"\ldots{}", "…"),
]


def sanitize_for_docx(text: str) -> str:
    """Transform the markdown source to be docx-friendly."""

    # ---- 1. Strip YAML front matter ----
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            text = text[end + 3:].lstrip("\n")

    # ---- 2. Replace raw LaTeX figure environment for fig_wrong_struct_v4
    fig_wrong_struct_tex = re.compile(
        r"\\begin\{figure\}.*?"
        r"\\includegraphics\[[^\]]*\]\{[^}]*fig_wrong_struct_v4\.pdf\}"
        r"(?P<rest>.*?)"
        r"\\end\{figure\}",
        re.DOTALL,
    )

    def _replace_fig4(m: re.Match) -> str:
        return (
            "![**Figure 4. Joint pass rates on the three wrong-candidate "
            "controls.** Green bars are the joint pass rate for the correct "
            "structure; red bars are the joint pass rate for the wrong "
            "candidate. Numbers above each group are the correct-to-wrong "
            "discrimination ratio. The constitutional isomer control "
            "(same molecular formula, different connectivity) is the "
            "chemistry-meaningful number for natural-product "
            "dereplication.]"
            "(../2d/figures/fig_wrong_struct_v4.png)"
        )

    text = fig_wrong_struct_tex.sub(_replace_fig4, text)

    # ---- 3. Figures: switch .pdf → .png ----
    text = text.replace("fig_v4_headline.pdf", "fig_v4_headline.png")
    text = text.replace("fig_h_zero.pdf", "fig_h_zero.png")
    text = text.replace("fig_label_sweep.pdf", "fig_label_sweep.png")

    # ---- 4. Drop LaTeX-only directives ----
    text = re.sub(r"^\\clearpage\s*$", "\n---\n", text, flags=re.MULTILINE)
    text = re.sub(r"^\\FloatBarrier\s*$", "", text, flags=re.MULTILINE)

    # ---- 5. Convert \subsection*{X} to markdown ## heading ----
    text = re.sub(
        r"\\subsection\*\{([^}]*)\}",
        lambda m: f"## {m.group(1)}",
        text,
    )

    # ---- 6. Convert the raw LaTeX references block ----
    # The references section uses \begingroup...\endgroup with raw LaTeX
    # formatting (\url{}, \textit{}, \raggedright, \sloppy). We need to
    # convert this to plain markdown references.
    text = re.sub(r"\\begingroup\s*", "", text)
    text = re.sub(r"\\endgroup\s*", "", text)
    text = re.sub(r"\\raggedright\s*", "", text)
    text = re.sub(r"\\sloppy\s*", "", text)
    text = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}\s*", "", text)

    # \url{X} → X (plain URL)
    text = re.sub(r"\\url\{([^}]*)\}", r"\1", text)

    # \textit{X} → *X*
    text = re.sub(r"\\textit\{([^}]*)\}", r"*\1*", text)

    # \textbf{X} → **X**
    text = re.sub(r"\\textbf\{([^}]*)\}", r"**\1**", text)

    # \texttt{X} → `X`
    text = re.sub(r"\\texttt\{([^}]*)\}", r"`\1`", text)

    # \textsuperscript{X} → ^X^
    text = re.sub(r"\\textsuperscript\{([^}]*)\}", r"^(\1)", text)

    # NOTE: do NOT replace "---" with em-dash globally — it would
    # break pipe table separator rows "|---|---|---|". Pandoc's docx
    # writer converts "---" to em-dashes in prose context natively.

    # \noindent → (drop)
    text = re.sub(r"\\noindent\s*", "", text)

    # \vspace{...} → (drop)
    text = re.sub(r"\\vspace\{[^}]*\}", "", text)

    # \quad → space
    text = text.replace("\\quad", " ")

    # ---- 7. Handle remaining LaTeX math with $..$ ----
    # Pandoc handles $..$ natively for docx (converts to OMML).
    # But some display math blocks use $$ ... $$ — pandoc handles those too.
    # No action needed.

    # ---- 8. Clean up stray backslash-escapes from LaTeX ----
    # \, → thin space (pandoc may or may not handle)
    text = text.replace(r"\,", " ")
    # \! → nothing
    text = text.replace(r"\!", "")
    # i.e.\ → i.e.
    text = text.replace(r"i.e.\ ", "i.e. ")

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
        "subtitle: |\n"
        "  A sliced sort-match weak-supervision recipe,\n"
        "  causally audited and calibrated for dereplication\n"
        "author: |\n"
        "  Zhaohe Dong^1,\\*^\n"
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
        "--from",
        "markdown+raw_tex+tex_math_dollars+pipe_tables+"
        "backtick_code_blocks+header_attributes+superscript+subscript",
        "--resource-path", str(FIGURES) + ":" + str(OUT_DIR),
        "--wrap=preserve",
        "--standalone",
        "--reference-links",
    ]
    # Use custom reference doc if available (sets fonts, headers, footers)
    if REF_DOCX.exists():
        cmd.extend(["--reference-doc", str(REF_DOCX)])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("PANDOC FAILED:", file=sys.stderr)
        print(r.stderr[-4000:], file=sys.stderr)
        sys.exit(1)

    WORK_MD.unlink()
    print(f"wrote {DOCX} ({DOCX.stat().st_size/1024:.1f} KiB)")


if __name__ == "__main__":
    main()
