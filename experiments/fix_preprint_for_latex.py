"""Post-process the preprint markdown to produce a LaTeX-compatible version.

Fixes:
  1. Converts [^N] footnote references to superscript numbers with a
     manually-rendered bibliography section at the end.
  2. Replaces Unicode ±, σ, − with LaTeX math equivalents.
  3. Writes output to docs/preprint_final.md.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "docs" / "preprint_v1_filled.md"
out = ROOT / "docs" / "preprint_final.md"

text = src.read_text()

# Step 1: find all [^N]: definitions and collect them as bibliography entries
ref_def_re = re.compile(r"^\[\^(\d+)\]:\s*(.+?)(?=^\[\^\d+\]:|\Z)", re.M | re.S)
refs: dict[int, str] = {}
for m in ref_def_re.finditer(text):
    num = int(m.group(1))
    body = m.group(2).strip()
    refs[num] = body

# Remove the definitions from the text
text = ref_def_re.sub("", text)

# Step 2: replace inline [^N] with superscript markdown ^N^ (pandoc superscript)
text = re.sub(r"\[\^(\d+)\]", r"<sup>\1</sup>", text)

# Step 3: Unicode replacements to tokens that survive pandoc cleanly.
# We use $...$ with a NON-BREAKING SPACE wrapper so pandoc sees whitespace
# on both sides of the delimiter and parses as math consistently.
replacements = [
    ("±", r"$\pm$"),
    ("σ", r"$\sigma$"),
    ("−", r"$-$"),
    ("≤", r"$\leq$"),
    ("≥", r"$\geq$"),
    ("∞", r"$\infty$"),
    ("×", r"$\times$"),
    ("→", r"$\rightarrow$"),
    ("⋯", r"\dots"),
    ("∼", "approx "),
]
for old, new in replacements:
    text = text.replace(old, new)

# Step 3b: replace "$\sim$" and "~" (pandoc-ambiguous) with plain text words
# to avoid math-mode adjacency issues that break xelatex font lookup.
text = text.replace(r"$\sim$", "~")
text = text.replace(r"\(\sim\)", "~")
text = text.replace(r"$\approx$", "~")
text = text.replace(r"\(\approx\)", "~")

# Step 3c: Ensure "~" in text mode becomes \textasciitilde{} (it's a non-breaking
# space otherwise in LaTeX). Use a regex that avoids replacing ~ inside code
# blocks / math environments / URLs.
#
# Simple approach: only replace ~ that is adjacent to a digit like ~100 or ~10^4
# since those are our only real use cases.
import re as _re
text = _re.sub(r"~(\d)", r"approximately \1", text)

# Step 4: build a proper References section
# The original text ends with the references. We removed them, so append a new
# References section with a numbered list.
ref_section = "\n\n## References\n\n"
for num in sorted(refs.keys()):
    body = refs[num].replace("\n", " ").strip()
    # Collapse multi-space
    body = re.sub(r"\s+", " ", body)
    ref_section += f"{num}. {body}\n"

# Trim any trailing empty lines / stray section headings from the text, then append
text = text.rstrip()
# Remove any empty "References" heading left behind
text = re.sub(r"##\s*References\s*$", "", text, flags=re.M).rstrip()
text += ref_section

out.write_text(text)
print(f"wrote {out} ({len(text)} chars, {len(refs)} refs migrated)")
