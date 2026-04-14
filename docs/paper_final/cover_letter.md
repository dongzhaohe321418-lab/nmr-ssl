---
geometry: margin=0.75in
fontsize: 10.5pt
linestretch: 1.0
header-includes:
  - \setlength{\parskip}{0.25em}
  - \setlength{\parindent}{0pt}
---

\begin{flushright}
Zhaohe Dong \\
Yusuf Hamied Department of Chemistry \\
University of Cambridge \\
Lensfield Road, Cambridge CB2 1EW, UK \\
\texttt{zd314@cam.ac.uk} \\[0.3em]
April 2026
\end{flushright}

\vspace{0.5em}

\noindent To the Editors, *Journal of Cheminformatics*:

\vspace{0.3em}

Please find enclosed the manuscript **"Learning $^{1}$H and $^{13}$C
NMR chemical shifts jointly from unassigned 2-D HSQC peak sets"**,
which I am submitting as a research article (methodology) for
consideration in *Journal of Cheminformatics*.

Predicting $^{1}$H and $^{13}$C NMR chemical shifts from a molecular
graph is a core cheminformatics task. The general feasibility of
using unassigned 2-D HSQC peak sets as permutation-invariant set
supervision for joint $^{1}$H / $^{13}$C prediction was recently
established in concurrent work by **Jin et al., arXiv:2601.18524
(January 2026)**, via exact Hungarian bipartite matching on millions
of literature-extracted spectra. The present manuscript does not
claim priority on that formulation; it contributes four technical
and statistical advances that Jin et al. do not report on, all
specifically relevant to small-scale, chemistry-driven deployment.

First, a **Bonferroni-corrected split-conformal calibration** that
delivers a rigorous $94.8\%$ molecule-level coverage guarantee at
$\alpha = 0.05$, alongside per-atom coverage of $95.2\%$ / $96.7\%$.
We are not aware of a prior NMR chemical-shift predictor that ships
with a formal molecule-level uncertainty certificate.

Second, a **causal $^{1}$H-zeroing audit** that rules out gradient
leakage from the $^{13}$C head as the source of the weakly
supervised $^{1}$H signal: zeroing the $^{1}$H coordinate of every
HSQC training target collapses $^{1}$H error from $0.35$ ppm to
$4.69$ ppm, while $^{13}$C is unchanged. This localizes the
supervision at the HSQC target itself.

Third, a **sliced sort-match loss** — an $O(K\,n \log n)$
sliced-Wasserstein construction that replaces the $O(n^{3})$ exact
Hungarian update, batchable through `torch.sort`, with a
self-contained proof of the underlying 1-D sort-match theorem in
Supplementary S8.

Fourth, an **explicit constitutional-isomer discrimination
experiment**: a 3.5-fold correct-to-wrong discrimination ratio on
the hardest (same-formula, different-connectivity) control,
positioned as a natural-product dereplication ranker and
explicitly flagged as insufficient for standalone accept/reject.

On a 1,542-molecule NMRShiftDB2 filtered subset, with only $10\%$
atom-assigned $^{13}$C labels and HSQC peak lists for the remaining
$90\%$, the model reaches $4.53 \pm 0.11$ ppm $^{13}$C and
$0.35 \pm 0.02$ ppm $^{1}$H test MAE *without ever seeing an
atom-assigned $^{1}$H label*. The absolute accuracy sits behind
both NMRNet and Jin et al. by roughly the ratio of training-corpus
sizes, as expected; the contributions of this paper are orthogonal
to that scale gap.

This work fits *Journal of Cheminformatics* for three reasons.
First, the problem — extracting weak supervision from unassigned
literature spectra with statistical guarantees — is a
cheminformatics problem that would lose chemical context in a pure
machine-learning venue. Second, the entire pipeline is fully
traceable: code, dataset filters, trained checkpoints, conformal
quantile tables, per-seed result JSONs, and figure-generation
scripts are archived at the submission-frozen git tag
`v2.0-jin-revision` of
\texttt{https://github.com/dongzhaohe321418-lab/nmr-ssl} under the
MIT license (software) and CC-BY 4.0 (documentation), with an
explicit table-and-figure-to-script artefact map
(`docs/paper_final/ARTIFACT_MAP.md`). Third, honest limitations —
validation on NMRShiftDB2-derived HSQC rather than scraped
literature peak lists, absolute accuracy below both Jin et al.
and NMRNet, and the $22\%$ false-positive rate against
constitutional isomers — are reported up front in the manuscript
rather than buried.

The manuscript has not been published elsewhere and is not under
consideration at any other journal. I declare no competing
interests and received no external funding. The work involves no
human or animal subjects. Code and drafts were developed with AI
assistance (Anthropic Claude Opus 4.6) under my direction; each
headline number in the paper is traceable through the artefact
map to a specific script, seed, and result JSON at the
submission-frozen git tag.

Thank you for considering this submission.

\noindent Sincerely,\\
Zhaohe Dong \quad \texttt{zd314@cam.ac.uk}
