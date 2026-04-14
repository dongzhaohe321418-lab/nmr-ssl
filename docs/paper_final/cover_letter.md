---
geometry: margin=0.85in
fontsize: 11pt
linestretch: 1.0
header-includes:
  - \setlength{\parskip}{0.3em}
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
graph is a core cheminformatics task, but every state-of-the-art
predictor is trained on atom-assigned spectra, where each example
specifies which atom produced which peak. Atom assignment is the
slow, expensive part of NMR data curation, which is why most of the
spectral information in the published literature never enters a
training set. Two-dimensional HSQC peak lists, by contrast, appear
in nearly every modern organic-chemistry paper, pair directly-bonded
H–C groups, and carry no atom identity. This manuscript asks whether
such unassigned peak lists are rich enough to train a joint
$^{1}$H / $^{13}$C shift predictor, and answers in the affirmative.

The paper introduces a **sliced sort-match loss** — a
sliced-Wasserstein construction that turns a 2-D optimal-transport
problem between predicted and observed HSQC multisets into a small
number of differentiable 1-D sort problems — and uses it as a
semi-supervised objective on top of a four-layer graph isomorphism
network. On a 1,542-molecule NMRShiftDB2 subset, with only $10\%$ of
molecules providing atom-assigned $^{13}$C labels and the remaining
$90\%$ contributing HSQC peak lists, the model reaches
$4.53 \pm 0.11$ ppm $^{13}$C and $0.35 \pm 0.02$ ppm $^{1}$H test MAE
*without ever seeing an atom-assigned $^{1}$H label*. A causal
audit in which the $^{1}$H coordinate of every HSQC target is zeroed
confirms that the $^{1}$H head learns from the HSQC signal itself
rather than from gradient leakage through the $^{13}$C head. I wrap
the predictor with split-conformal calibration for both per-atom
and Bonferroni-corrected molecule-level guarantees, and demonstrate
a 3.5-fold discrimination ratio against constitutional isomers.

This work fits *Journal of Cheminformatics* because its target
problem — extracting weak supervision from unassigned literature
spectra — is a cheminformatics problem that would lose chemical
context in a pure machine-learning venue, and because the entire
pipeline is fully reproducible: code, dataset filters, trained
checkpoints, conformal quantile tables, per-seed result JSONs, and
figure-generation scripts are archived at
\texttt{https://github.com/dongzhaohe321418-lab/nmr-ssl} under MIT
(software) and CC-BY 4.0 (documentation). The full pipeline runs
deterministically in about 26 minutes on a consumer GPU, matching
the journal's standing requirement that published research be
reproducible by third parties without registration or restrictive
licensing. Honest limitations — validation on NMRShiftDB2-derived
rather than scraped literature HSQC data, absolute accuracy below
NMRNet, and a $22\%$ false-positive rate against constitutional
isomers — are reported up front.

The manuscript has not been published elsewhere and is not under
consideration at any other journal. I declare no competing interests
and received no external funding. The work involves no human or
animal subjects. Code and drafts were developed with AI assistance
(Anthropic Claude Opus 4.6) under my direction; every numerical
result is reproducible from the public repository.

Thank you for considering this submission.

\noindent Sincerely,\\
Zhaohe Dong \quad \texttt{zd314@cam.ac.uk}
