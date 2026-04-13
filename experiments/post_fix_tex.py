"""Post-process preprint.tex to fix pandoc's broken math-delimiter escapes.

Pandoc sometimes converts `\(foo\)` in markdown output to literal `(foo)` in
the .tex file, killing the math mode. This script walks preprint.tex and
wraps known math symbols that appear outside math mode with proper `$...$`.
"""

from __future__ import annotations

import re
from pathlib import Path

TEX = Path.home() / "nmr-ssl" / "docs" / "preprint.tex"

text = TEX.read_text()

# Bad patterns pandoc produces when it escapes \( \): literal parentheses
# containing an unambiguous LaTeX math command. We wrap those in $...$.
#
# Examples:
#   (\pm)0.102    →  $\pm$0.102
#   (\sim)10^4    →  $\sim$10^4
#   (\sigma)=1    →  $\sigma$=1
#   (\leq)5       →  $\leq$5
#
# The pattern matches "(\CMD)" where CMD is one of our known math commands.
known_math_cmds = [
    "pm", "sim", "sigma", "leq", "geq", "infty", "times", "rightarrow",
    "approx", "cdot", "ldots", "dots",
]
pattern = re.compile(r"\(\\(" + "|".join(known_math_cmds) + r")\)")
text, n1 = pattern.subn(lambda m: f"${{\\{m.group(1)}}}$", text)

# Also handle pandoc's literal escapes like "\$\pm\$" which it sometimes
# produces when it doesn't want to interpret $ as math.
bad_dollars = re.compile(r"\\\$\\(" + "|".join(known_math_cmds) + r")\\\$")
text, n2 = bad_dollars.subn(lambda m: f"${{\\{m.group(1)}}}$", text)

# Unicode characters that slipped through (fallback)
unicode_fixes = {
    "±": r"$\pm$",
    "σ": r"$\sigma$",
    "−": r"$-$",
    "≤": r"$\leq$",
    "≥": r"$\geq$",
    "∞": r"$\infty$",
    "×": r"$\times$",
    "∼": r"$\sim$",
    "→": r"$\rightarrow$",
    "𝜎": r"$\sigma$",  # mathematical italic sigma
}
n3 = 0
for old, new in unicode_fixes.items():
    count = text.count(old)
    if count:
        text = text.replace(old, new)
        n3 += count

TEX.write_text(text)
print(f"post_fix_tex: fixed {n1} (cmd), {n2} (dollar), {n3} (unicode)")
