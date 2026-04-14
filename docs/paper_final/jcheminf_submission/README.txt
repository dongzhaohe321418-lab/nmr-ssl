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
                        \input{preamble.tex}.  Contains font, margin,
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
