---
title: |
  Learning $^{1}$H and $^{13}$C NMR chemical shifts jointly
  from unassigned 2-D HSQC peak sets
subtitle: |
  A sliced sort-match weak-supervision recipe, causally audited
  and calibrated for dereplication
author: |
  Zhaohe Dong$^{1,*}$
affiliation: |
  $^{1}$Yusuf Hamied Department of Chemistry, University of Cambridge,
  Lensfield Road, Cambridge CB2 1EW, United Kingdom. \
  $^{*}$Correspondence: `zd314@cam.ac.uk`. This work was developed with
  AI-assisted code and draft generation under the author's direction; the
  author bears final responsibility for all methodological choices,
  claims, and errors.
date: April 2026
keywords: |
  NMR chemical shift prediction; semi-supervised learning; sliced
  Wasserstein; split-conformal prediction; natural-product dereplication;
  graph neural network.
---

# Abstract

**Background.** Nuclear magnetic resonance (NMR) chemical-shift
predictors are almost always trained on atom-assigned spectra, where
every training example specifies which atom in the molecular graph
produced which peak. Atom assignment is the expensive part of
curating NMR data, so most published spectra never enter a training
set. Two-dimensional HSQC peak lists are a different matter: they
appear in nearly every modern organic paper, pair the $^{1}$H and
$^{13}$C shifts of directly bonded H–C groups, and carry no atom
identity. Concurrent work by Jin et al. [11] has established that
such unassigned HSQC peak sets can be used as permutation-invariant
set supervision for joint $^{1}$H / $^{13}$C prediction, via exact
Hungarian bipartite matching at million-spectrum scale. We address
four remaining questions that are specifically relevant to
small-scale, chemistry-driven deployment and that Jin et al. do not
report on: (i) whether the predictor can be wrapped with a
rigorous molecule-level uncertainty guarantee suitable for
natural-product dereplication; (ii) whether the weakly supervised
$^{1}$H head is actually learning from the HSQC signal rather than
from encoder leakage out of the supervised $^{13}$C loss;
(iii) whether a cheaper sliced-Wasserstein loss can replace the
$O(n^{3})$ exact Hungarian step at small scale; and (iv) whether
the predictor can discriminate correct structures from
constitutional isomers well enough to function as a dereplication
ranker.

**Results.** We train a four-layer graph isomorphism network on a
1,542-molecule NMRShiftDB2 subset with $10\%$ atom-assigned
$^{13}$C labels and HSQC peak lists for the remaining $90\%$,
treating each HSQC peak list as an unordered multiset in
$(\delta_{H}, \delta_{C})$ space. The headline test MAE is
$4.53 \pm 0.11$ ppm $^{13}$C and $0.35 \pm 0.02$ ppm $^{1}$H over
three seeds, with no atom-assigned $^{1}$H labels ever used;
layered on full $^{13}$C supervision the same recipe reaches
$3.23 \pm 0.10$ ppm $^{13}$C and $0.30 \pm 0.03$ ppm $^{1}$H. On
top of this headline, four experimental deliverables establish the
paper's specific contributions in canonical order:
(i) **Bonferroni-corrected split-conformal calibration** delivers a
rigorous $94.8\%$ molecule-level coverage guarantee at
$\alpha = 0.05$, alongside per-atom coverage of $95.2\%$ / $96.7\%$;
(ii) a **causal $^{1}$H-zeroing audit**, in which the $^{1}$H
coordinate of every HSQC target is zeroed and the model is
retrained, collapses $^{1}$H error to $4.69 \pm 0.10$ ppm while
leaving $^{13}$C essentially at its baseline, confirming the
$^{1}$H head learns from the HSQC signal rather than from
gradient leakage out of the $^{13}$C head; (iii) the underlying
**sliced sort-match loss** reduces the 2-D optimal-transport
problem to $K$ differentiable 1-D sort problems at
$O(K\,n \log n)$ per molecule, asymptotically cheaper than the
$O(n^{3})$ exact Hungarian update used by Jin et al. [11] and
batchable through `torch.sort` without a custom autograd hack;
and (iv) against constitutional isomers the model is correct on
$77\%$ of candidates and wrong on $22\%$, a 3.5-fold
dereplication-ranking ratio explicitly flagged as insufficient
for standalone accept/reject.

**Conclusions.** Complementary to the feasibility established
concurrently by Jin et al. [11], we contribute four technical
advances targeted at small-scale, chemistry-driven deployment, in
order: (i) a Bonferroni-corrected molecule-level split-conformal
coverage guarantee for NMR shift prediction, of which we are not
aware of a prior example in the literature; (ii) a causal audit of
the $^{1}$H supervision pathway that rules out gradient leakage
from the $^{13}$C head; (iii) an $O(K\,n \log n)$ sliced-Wasserstein
loss that replaces the $O(n^{3})$ exact Hungarian update used by
[11]; and (iv) an explicit constitutional-isomer discrimination
experiment for natural-product dereplication ranking. The validation
in this paper uses HSQC multisets **derived from NMRShiftDB2
atom-assigned spectra**, not scraped literature peak lists; carrying
the pipeline to real literature HSQC data is the single largest
outstanding gap. Code, data filters, trained checkpoints, and an
artefact map are at
`https://github.com/dongzhaohe321418-lab/nmr-ssl` (release
`v2.0-jin-revision`).

**Keywords:** NMR chemical shift prediction; semi-supervised learning;
sliced Wasserstein; optimal transport; split-conformal prediction;
natural-product dereplication; graph neural network.

\clearpage

# Background

Predicting NMR chemical shifts from a molecular graph is a standard
machine-learning task in chemoinformatics. The current best predictor,
NMRNet [1], reaches $1.10$ ppm $^{13}$C and $0.18$ ppm $^{1}$H mean
absolute error on NMRShiftDB2. It achieves this by training two separate
heads on two separately-curated atom-assigned corpora. Every training
example carries per-atom labels that someone had to annotate.

Per-atom annotation is the slow, expensive part of building an NMR
training set. Most spectra in the literature never get annotated to that
level, so they cannot be used with the standard supervised recipe. 2-D
experiments such as HSQC, HMBC, COSY and NOESY appear in far more
published papers than fully atom-assigned 1-D spectra, but their peak
lists are multisets of correlations rather than atom-indexed values,
which rules them out of conventional training.

HSQC is the smallest step away from 1-D supervision. Each cross-peak
in an HSQC peak list pairs a $^{1}$H shift with the $^{13}$C shift of
its directly-bonded carbon. The pairing is known; the atom identity
is not. Jin et al. [11], in concurrent work posted on arXiv in
January 2026, have established that this form of weakly-paired
supervision is in fact rich enough to train a joint $^{1}$H / $^{13}$C
predictor at scale, using an exact Hungarian bipartite matching loss
over millions of literature-extracted spectra. Their feasibility
result is not in dispute. The present paper addresses four remaining
questions that are specifically relevant to small-scale,
chemistry-driven deployment, and that Jin et al. do not report on.

\subsection*{Related work and contributions of this paper}

We do not claim priority on the unassigned-HSQC-as-supervision
formulation; that priority belongs to Jin et al. [11]. What we
contribute is a set of four complementary technical and statistical
advances, each of which targets a failure mode of the
exact-Hungarian, large-scale recipe.

**First — a formal molecule-level coverage guarantee.** We wrap the
predictor with split-conformal calibration [4, 5] and, using a
Bonferroni correction over the per-molecule HSQC peak count, recover
a rigorous molecule-level coverage guarantee. Empirical per-atom test
coverage at $\alpha = 0.05$ is $95.2\%$ on $^{13}$C and $96.7\%$ on
$^{1}$H; the Bonferroni-corrected joint pass rate lands at
$\mathbf{94.8\%}$, within $0.2\%$ of the nominal target. To our
knowledge this is the first NMR chemical-shift predictor that ships
with a formal molecule-level uncertainty guarantee. Jin et al. [11]
report point predictions only.

**Second — a causal audit of the supervision pathway.** A shared
graph encoder with two output heads is vulnerable to a well-known
confound: the $^{1}$H head could be learning from gradient leakage
out of the $^{13}$C supervised loss rather than from the HSQC target
itself. We rule this out by zeroing the $^{1}$H coordinate of every
HSQC training target and retraining with the same hyperparameters.
The $^{1}$H test MAE jumps from $0.35 \pm 0.02$ ppm to
$4.69 \pm 0.10$ ppm, more than ten times worse than the baseline,
while the $^{13}$C error barely moves. The HSQC $^{1}$H coordinate is
doing the work. Jin et al. [11] do not report this ablation.

**Third — a sliced-Wasserstein alternative to exact Hungarian
matching.** Across both coordinate axes jointly we use the
sliced-Wasserstein construction of Bonneel et al. [3], which averages
a 1-D sort-match loss over $K$ random linear projections and brings
the per-molecule cost from $O(n^{3})$ (exact Hungarian, as in [11])
down to $O(K\,n \log n)$. Along either coordinate axis the inner
1-D loss is exactly minimized by sort-matching, by the Hardy–
Littlewood–Pólya rearrangement inequality [6]; we give a full
statement and proof of this 1-D sort-match theorem in
Supplementary Information **S8**, so that the core methodological
result of this paper is self-contained. The sliced construction is
permutation-invariant, differentiable through `torch.sort`, and
batchable without a custom autograd hack — useful properties in
low-compute regimes where the Hungarian inner loop of [11] would
dominate training time.

**Fourth — an explicit dereplication experiment.** We evaluate the
predictor against three levels of wrong-candidate controls — random
pair, scaffold neighbour, and constitutional isomer — with the
hardest, chemistry-meaningful setting being the constitutional
isomer control (same molecular formula, different connectivity).
The correct-candidate joint pass rate is $77.0\%$ and the
wrong-candidate joint pass rate is $21.7\%$, giving a 3.5-fold
discrimination ratio. This is explicitly positioned as a candidate
ranker for natural-product dereplication pipelines, not as a
standalone binary accept/reject classifier. Jin et al. [11] do not
report an isomer-discrimination experiment.

**Boundary of the claim.** All experiments in this paper use HSQC
multisets **derived from NMRShiftDB2 atom-assigned spectra**, not
from scraped literature peak lists; we construct the multiset by
averaging the attached-$^{1}$H shifts at each H-bearing carbon. This
preserves the combinatorial structure of a real HSQC peak list but
not its experimental noise characteristics, and we flag the
literature-scraping step as the single largest outstanding gap.
Headline label-efficiency numbers in this paper are reached at the
1,542-molecule NMRShiftDB2 filtered-subset scale, not at the
literature scale reported by Jin et al. [11]; absolute accuracy is
correspondingly behind and we do not claim otherwise.

The rest of the paper is organized as follows. *Methods* writes out
the sliced sort-match loss and its theoretical motivation, pointing
to Supplementary **S8** for the full statement and proof of the
underlying 1-D sort-match theorem. *Experimental details* covers
dataset construction, model architecture, training protocol,
split-conformal calibration, and the structure-verification
procedure. *Results* reports the main label-efficiency numbers, the
causal $^{1}$H-zeroing audit, the combined supervision regime, the
label-efficiency sweep, the three wrong-candidate controls, the
scaffold-OOD split, robustness to realistic HSQC degradation, the
hyperparameter and axis-aligned-decomposition sweeps, the
per-carbon error decomposition, and the split-conformal coverage
analysis. *Discussion* covers honest limitations and the path to
real literature HSQC data. *Conclusions* summarizes the findings
with explicit reference to Jin et al. [11] as the concurrent
state-of-the-art feasibility result. The Supplementary Information
contains per-seed tables, hyperparameter sweep data, a stop-gradient
control, a pretrain-then-finetune baseline, the stereo-rich stress
test, a multiplicity-augmented loss experiment, and the full
sort-match theorem (S8).

# Methods

\subsection*{Set-valued NMR supervision}

Fix a molecule $M$ with its molecular graph $G(M)$ and write
$\mathbf{P}^{\star}(M) = \{(\delta^{\star}_{H,j},\,\delta^{\star}_{C,j})\}_{j=1}^{n}$
for its observed HSQC peak set. The target is an unordered multiset: the
training pipeline sees the $n$ pairs but not which pair corresponds to
which heavy atom. A graph neural network
$f_{\theta}: G \to (\hat{\boldsymbol{\delta}}_{H}, \hat{\boldsymbol{\delta}}_{C})$
predicts per-atom shifts; from its outputs at the H-bearing carbons we
form the predicted set
$\hat{\mathbf{P}}(M,\theta) =
\{(\hat{\delta}_{H,i},\hat{\delta}_{C,i})\}_{i=1}^{n}$.

The goal is to minimize a permutation-invariant loss between
$\hat{\mathbf{P}}$ and $\mathbf{P}^{\star}$. In one dimension this
reduces exactly to sorting both sets and taking a per-element
$\ell^{2}$ distance, as a direct consequence of the
Hardy–Littlewood–Pólya rearrangement inequality [6]; we give a
full statement and proof of this 1-D sort-match theorem in
Supplementary **S8**, so that the methodological backbone of this
paper is self-contained within the manuscript. In two dimensions
the exact solution is Hungarian optimal bipartite matching — the
approach taken by Jin et al. [11] — which is $O(n^{3})$ per
molecule and not easily batched on GPU.

\subsection*{Sliced approximation}

We follow Bonneel et al. [3] and use the sliced-Wasserstein distance
as a cheap upper bound. For a random direction $\theta \in S^{1}$ drawn
uniformly from the unit sphere, define the projection
$\Pi_{\theta}(h, c) = h\cos\theta + c\sin\theta$. Then

$$
SW_{2}^{2}(\hat{\mathbf{P}}, \mathbf{P}^{\star})
= \mathbb{E}_{\theta \sim \mathrm{Unif}(S^{1})}
    \bigl[W_{2}^{2}(\Pi_{\theta}\hat{\mathbf{P}},
                    \Pi_{\theta}\mathbf{P}^{\star})\bigr],
$$

where each inner one-dimensional 2-Wasserstein is solved exactly by
sorting the two projected sets. A Monte-Carlo estimator with $K$
directions gives the **sliced sort-match loss**,

$$
\mathcal{L}_{\mathrm{SSW}}(\hat{\mathbf{P}}, \mathbf{P}^{\star})
= \frac{1}{K}
  \sum_{k=1}^{K}
  \mathcal{L}_{\mathrm{sort}}\bigl(
    \Pi_{\theta_k}\hat{\mathbf{P}},
    \Pi_{\theta_k}\mathbf{P}^{\star}
  \bigr),
$$

which is permutation-invariant, differentiable almost everywhere through
`torch.sort`, and computed in $O(Kn\log n)$ per molecule. It is a
consistent estimator of $SW_{2}^{2}$ as $K \to \infty$, and satisfies
$SW_{2}^{2} \leq W_{2}^{2}$ by the classical inequality.

**Numerical audit.** On 20 random two-dimensional point-set pairs drawn
from the empirical shift distribution of NMRShiftDB2
($n \in [5, 20]$, $K = 64$), the ratio
$\mathcal{L}_{\mathrm{SSW}}/W_{2}^{\text{Hungarian}}$ concentrates in
$[0.30, 0.54]$ with mean $0.41 \pm 0.06$, consistent with the expected
two-dimensional bound. Implementation: `src/nmr2d/losses_2d.py`.
Verification test: `tests/test_theorem_2d.py`.

\subsection*{Axis-aligned decomposition}

A specialization of the sliced estimator uses only the two native
coordinate axes (i.e.\ $\theta \in \{0,\pi/2\}$). This reduces to two
independent one-dimensional sort-match losses on the $^{1}$H and
$^{13}$C projections, with no Monte-Carlo randomness. At equal $K$ it
is a biased lower bound on the full sliced estimator, but on
NMRShiftDB2 it matches sliced $K=16$ within noise at one-eighth the
cost (see *Results: Hyperparameter sensitivity*). We report both
variants; the sliced-random construction is the cleaner theoretical
scaffolding and preserves the inequality
$SW_{2}^{2} \leq W_{2}^{2}$, while the axis-aligned variant is the
cheaper form and is what we use in the structure-verification
experiments.

\subsection*{Full training objective}

The full training loss adds a supervised $^{13}$C mean squared error on
the labelled split:

$$
\mathcal{L}(\theta) =
\underbrace{\frac{1}{|\mathcal{L}|}
  \sum_{M\in\mathcal{L}} \frac{1}{|C(M)|}
  \sum_{a\in C(M)}
  \bigl(\hat{\delta}_{C}(a;\theta) - \delta^{\star}_{C}(a)\bigr)^{2}}%
  _{\text{supervised }^{13}\text{C MSE on labelled molecules}}
+ \lambda
\underbrace{\frac{1}{|\mathcal{U}|}
  \sum_{M\in\mathcal{U}}
  \mathcal{L}_{\mathrm{SSW}}\bigl(\hat{\mathbf{P}}(M;\theta),
                                  \mathbf{P}^{\star}(M)\bigr)}%
  _{\text{sliced sort-match on unlabelled molecules}},
$$

where $\mathcal{L}$ and $\mathcal{U}$ are the labelled and unlabelled
training partitions and $C(M)$ is the set of carbon atoms in molecule
$M$. The SSL weight $\lambda$, the number of sliced directions $K$, and
the labelled fraction are the three hyperparameters we sweep in
*Results*.

# Experimental details

\subsection*{Dataset construction}

We parse the NMRShiftDB2 [9] dump `nmrshiftdb2withsignals.sd`
(SourceForge, release 2026-03-15, CC-BY-SA) with RDKit [8] and join the
$^{13}$C and $^{1}$H spectra by
molecule ID. Retention filters: (a) non-degenerate $^{13}$C assignments
(one peak per carbon atom), (b) at least one $^{1}$H assignment
grouped by heavy atom, (c) at least three HSQC cross-peaks, (d) at most
60 atoms. Out of 20,000 $^{13}$C records and 18,169 $^{1}$H records,
1,542 molecules pass all filters.

For each retained molecule we construct the HSQC peak multiset by
iterating over all H-bearing carbons and emitting the tuple
$(\bar{\delta}_{H}, \delta_{C})$, where
$\bar{\delta}_{H}$ is the mean shift over all H atoms bonded to that
carbon. This is equivalent to the multiset a single
multiplicity-averaged HSQC experiment reports. Code:
`src/nmr2d/data_2d.py`.

\subsection*{Model}

A shared 4-layer graph isomorphism network (GIN [7]) encoder with dense
adjacency and 20 atom features (element one-hot, degree, formal
charge, aromaticity, hybridization, hydrogen count, ring membership,
atomic mass), followed by two 2-layer MLP readout heads: one for
per-atom $^{13}$C shift and one for per-heavy-atom mean $^{1}$H shift.
Hidden size 192. Total parameters are approximately 340k. Code:
`src/nmr2d/model_2d.py`.

\subsection*{Sliced sort-match loss}

At each forward pass we draw $K$ direction vectors uniformly on $S^{1}$
from a 2-D normal rescaled to unit norm, project every predicted and
target peak onto each direction, sort the two projected sets within
each mini-batch row, and compute the masked mean squared error on the
aligned pairs. The final loss is the mean over $K$ directions.
Permutation invariance is inherited from the sort. Differentiability
through `torch.sort` is standard. Code: `src/nmr2d/losses_2d.py`. The
axis-aligned variant fixes $K=2$ directions $(1,0)$ and $(0,1)$ and
returns the sum of the two 1-D sort-match losses.

\subsection*{Training protocol}

AdamW, learning rate $10^{-3}$, weight decay $10^{-5}$, batch size 32,
30 epochs, gradient clipping L2 $= 5$, best-val-$^{13}$C-MAE early
stopping. Three seeds (0, 1, 2) with independent 80/10/10 random
train/val/test splits and independent labelled/unlabelled partitions
(labelled fraction $0.1$ unless otherwise stated). SSL weight
$\lambda = 2$, sliced directions $K = 16$. Implementation in PyTorch
2.8 [10] with the Apple Silicon MPS backend. The algorithmic
complexity is $O(K\,n \log n)$ per molecule per epoch; exact
wall-clock numbers are dominated by I/O and vary with hardware and
are therefore not reported here as durable claims.

\subsection*{Split-conformal calibration}

We use the split-conformal prediction framework of Vovk, Gammerman and
Shafer [4], specialized to regression by Lei, G'Sell, Rinaldo,
Tibshirani and Wasserman [5]. After training the 2-D SSL model, we
collect per-atom absolute residuals
$|\hat{\delta} - \delta^{\star}|$ on the held-out validation split
(150 molecules, 1,959 $^{13}$C atoms and 1,307 $^{1}$H atoms). We
take the finite-sample-corrected $(1-\alpha)$ empirical quantile at
$\alpha = 0.05$. The chemistry-demonstration calibration run yields
$q_{C} \approx 14.8$ ppm and $q_{H} \approx 1.06$ ppm with empirical
test-set coverage of $95.2\%$ on $^{13}$C and $96.7\%$ on $^{1}$H;
the parallel run used for the Bonferroni table (Table 8) gives
slightly tighter quantiles $q_{C} = 13.4$ ppm and $q_{H} = 1.03$
ppm on the same validation split, well within sampling variance.
For the molecule-level Bonferroni correction we precompute a lookup
table of per-$k$ quantiles at $\alpha_{\text{atom}} =
\alpha_{\text{mol}}/(2k)$ and apply the molecule-appropriate quantile
per candidate during evaluation. Code: `src/nmr2d/conformal.py`,
`experiments/compute_bonferroni_conformal.py`.

\subsection*{Structure-verification protocol}

Given a proposed structure $M$ and an observed HSQC peak list
$\mathbf{P}^{\star}$, we (a) predict the model's full $^{1}$H/$^{13}$C
shift tensors on $M$, (b) read off the predicted HSQC cross-peaks at
the H-bearing carbons, (c) pair each observed peak with its predicted
counterpart at the same carbon-atom index using $M$'s connectivity as
the assignment source, and (d) check whether each observed shift lies
within its conformal interval. The structure is declared consistent
if every cross-peak passes. The fraction-within-interval and
worst-residual score are reported as continuous ranking scores for
dereplication pipelines. Code:
`src/nmr2d/conformal.py::ConformalCalibrator.structure_verification_score`.

**Scope caveat.** The protocol above is **training-unassigned but
deployment-assigned**: the model is trained without atom-level H or
HSQC-atom correspondences, but at inference time the protocol pairs
observed peaks with predicted shifts through the candidate
structure's connectivity. This matches the natural-product
dereplication workflow (rank a list of candidate structures against
one observed HSQC) and does not support *de novo* assignment of a
peak list to an unknown molecular graph.

# Results

All experiments use NMRShiftDB2 release 2026-03-15 (SourceForge,
CC-BY-SA). After filtering for molecules with both $^{13}$C and $^{1}$H
spectra, non-degenerate $^{13}$C assignments, at least three HSQC
cross-peaks, and at most 60 atoms, the final corpus is 1,542 molecules.
We report three-seed aggregates (mean and one standard deviation) over
seeds 0, 1, 2, using the training protocol in *Experimental details*.
The dual-head
architecture is a four-layer graph isomorphism network with hidden size
192, twenty atom features, and two 2-layer MLP readout heads. The total
parameter count is roughly 340k.

\subsection*{Main result}

Table 1 gives the four main training regimes in one place. The 2-D
sort-match SSL variant at $\lambda = 2$, $K = 16$ is the low-label
headline. On $^{13}$C it sits at 4.53 ppm, essentially tied with the
1-D sort-match SSL baseline at 4.56 ppm. On $^{1}$H it reaches 0.35 ppm,
which is roughly seven times better than either 1-D variant and is
achieved without a single atom-assigned $^{1}$H label anywhere in
training. The fourth row of Table 1 adds the SSL loss on top of full
$^{13}$C supervision. The result drops $^{13}$C by another 1.3 ppm and
$^{1}$H by a further 0.05 ppm. The SSL loss is not a low-label trick;
it helps in the fully supervised regime as well.

**Table 1.** Main test-set MAE on NMRShiftDB2 (3 seeds, mean $\pm$ std).
All rows share the same 4-layer GIN, AdamW optimizer, batch size 32,
gradient clipping L2 $=$ 5, and best-val-$^{13}$C-MAE early stopping.

| Variant | $^{13}$C MAE (ppm) | $^{1}$H MAE (ppm) |
|---|---|---|
| Supervised-1-D ($10\%$ labelled, $^{13}$C only) | $5.60 \pm 0.34$ | $2.47 \pm 0.38$ (untrained) |
| 1-D sort-match SSL ($10\%$ labelled, prior work [2]) | $4.56 \pm 0.31$ | $2.61 \pm 0.32$ (untrained) |
| **2-D SSL** ($10\%$ labelled, $\lambda=2$, $K=16$) | $\mathbf{4.53 \pm 0.11}$ | $\mathbf{0.35 \pm 0.02}$ |
| **2-D SSL + full $^{13}$C** (combined) | $\mathbf{3.23 \pm 0.10}$ | $\mathbf{0.30 \pm 0.03}$ |

For context, two external baselines exist in the literature at very
different scales. NMRNet [1], a fully supervised predictor, reaches
$1.10$ ppm $^{13}$C and $0.18$ ppm $^{1}$H on NMRShiftDB2 using the
full $\sim\!15{,}000$-molecule atom-assigned corpus. Jin et al. [11],
the concurrent unassigned-HSQC semi-supervised baseline, reach
approximately $0.8$ ppm $^{13}$C and $0.1$ ppm $^{1}$H using millions
of literature-extracted spectra and an exact Hungarian matching
loss. Our headline numbers sit above both: roughly a factor of
$4$–$5$ behind Jin et al. on absolute accuracy, which is consistent
with the $\sim\!10^{3}$ difference in training-corpus size. The
contribution of this paper is explicitly *not* in the headline MAE
— it is in the loss family, the causal audit, the calibration, and
the dereplication experiment (see *Related work and contributions
of this paper*).

![Main result. (a) $^{13}$C test MAE across five training
variants: supervised-1-D baseline (grey), 1-D sort-match SSL (blue),
2-D SSL at earlier $K=8$ working point (orange), 2-D SSL at
$K=16$ low-label headline (purple), 2-D SSL combined with full
$^{13}$C supervision (green). (b) $^{1}$H test MAE for the same five
variants. Error bars are one standard deviation over 3 seeds. The
$K=16$ low-label headline closes the $^{13}$C gap to the 1-D SSL
baseline; the combined variant delivers a further $1.3$ ppm
improvement.](../2d/figures/fig_v4_headline.pdf)

\subsection*{Causal audit: the $^{1}$H head learns from the HSQC target}

Before reading anything more into the main result, we tested whether the
$^{1}$H head might be learning not from the HSQC target itself, but from
gradient leakage out of the $^{13}$C supervised loss through the shared
encoder. The test is simple. We retrain the main 2-D SSL model at
$K = 16$, 30 epochs, three seeds, with one change: every $^{1}$H
coordinate in every unlabelled-split HSQC training target is set to
zero before the sliced sort-match loss is computed. The $^{13}$C
coordinate of the target is unchanged.

Table 2 shows the outcome. The $^{13}$C error moves by less than
$0.5$ ppm (from $4.53$ to $5.00$ ppm, within one standard deviation
of the sliced run-to-run spread), while the $^{1}$H error goes from
$0.35$ ppm to $4.69$ ppm — a more than thirteen-fold collapse,
worse than the untrained supervised-1-D baseline of $2.47$ ppm.
The signal in the HSQC target is doing the work on the $^{1}$H side.
If encoder leakage were the mechanism, the $^{1}$H number would
have stayed near $0.35$ ppm.

**Table 2.** Causal audit. The $^{1}$H error collapses by more than a
factor of ten when the HSQC $^{1}$H coordinate of the training target
is zeroed; the $^{13}$C error is essentially unchanged.

| Configuration | $^{13}$C MAE (ppm) | $^{1}$H MAE (ppm) |
|---|---|---|
| 2-D SSL baseline ($\lambda=2$, $K=16$) | $4.53 \pm 0.11$ | $0.35 \pm 0.02$ |
| 2-D SSL with HSQC $^{1}$H coordinate zeroed | $5.00 \pm 0.46$ | $4.69 \pm 0.10$ |

![Causal audit. (a) $^{13}$C test MAE is essentially insensitive to
zeroing the HSQC $^{1}$H coordinate, confirming that the $^{13}$C
head is trained by its own supervised loss. (b) $^{1}$H test MAE
collapses by more than a factor of ten when the HSQC $^{1}$H
coordinate is zeroed. The collapse rules out encoder leakage from the
$^{13}$C supervised loss as the training signal for the $^{1}$H head,
and localizes the training signal at the HSQC target itself.
](../2d/figures/fig_h_zero.pdf)

\clearpage

\subsection*{Combined supervision and the full-label regime}

An obvious objection is that the 2-D SSL loss might only help in the
low-label regime and vanish once the supervised head has access to
every $^{13}$C label. To test this, we retrained with the full
$^{13}$C supervision on all 1,542 molecules and the SSL loss on the
same molecules. Row 4 of Table 1 gives the result: $3.23 \pm 0.10$
ppm $^{13}$C and $0.30 \pm 0.03$ ppm $^{1}$H, a 1.3 ppm $^{13}$C
improvement and a 0.05 ppm $^{1}$H improvement over the low-label
headline. The SSL loss is not a low-label artefact. It is a training
signal that still helps when the supervised head is already seeing
every label.

\subsection*{Label-efficiency curve}

Figure 3 plots test MAE against the $^{13}$C-labelled fraction. At
$1\%$ labels (twelve molecules), supervised-1-D gives a useless
$18.5$ ppm $^{13}$C and $2.87$ ppm $^{1}$H. At the same label fraction,
2-D SSL reaches $6.0$ ppm $^{13}$C and $0.44$ ppm $^{1}$H, roughly three
times better on $^{13}$C and six times better on $^{1}$H. The two
variants converge on $^{13}$C near $50\%$ labels. The $^{1}$H gap does
not change with label fraction at all: 2-D SSL sits at about $0.4$ ppm
whether $1\%$ or $50\%$ of the training molecules are $^{13}$C-labelled,
because the HSQC supervision is decoupled from the $^{13}$C label
fraction.

![Label-efficiency curve. (a) $^{13}$C MAE as a function
of the $^{13}$C-labelled fraction. Supervised-1-D (blue dashed) is
useless below $10\%$ labels; 2-D SSL (green solid) remains within
6 ppm across the entire range and tracks supervised-1-D above $50\%$
labels. (b) $^{1}$H MAE. Supervised-1-D is pinned at the random floor
($\sim\!3$ ppm) because its $^{1}$H head receives no training signal;
2-D SSL delivers sub-0.45 ppm $^{1}$H at every label fraction.
](../2d/figures/fig_label_sweep.pdf)

\subsection*{Wrong-candidate discrimination and the dereplication ranker}

The structure-verification use case needs a test against wrong
candidates. We built three levels of wrong-candidate controls,
progressively harder and progressively more chemically realistic.

The easiest control is a random pair. For each test molecule we pick a
different test molecule of the same HSQC peak count and compare the
predicted HSQC of the second against the observed HSQC of the first.
Nothing about the two molecules is otherwise related, so any
reasonable predictor will separate them.

The middle control is a scaffold neighbour. For each test molecule we
find up to three other training molecules that share its Bemis–Murcko
scaffold. They have the same ring skeleton but different functional
groups, so the $^{1}$H and $^{13}$C shifts differ mostly in the
decoration around the core.

The hardest control is a constitutional isomer. For each test molecule
with a peer in the corpus of the same molecular formula, we compare
against up to three such peers. Constitutional isomers share atom
counts and elemental composition but differ in connectivity, which is
exactly the setting a natural-product dereplication pipeline faces
after a molecular-formula lookup.

**Table 3.** Structure-verification discrimination for the three negative
controls, at per-atom conformal $\alpha = 0.05$.

| Control | Correct joint pass | Wrong joint pass | Discrimination |
|---|---|---|---|
| **Constitutional isomer (headline)** | $77.0\%$ (n=74) | $21.7\%$ | $\mathbf{3.5\times}$ |
| Scaffold neighbour | $62.4\%$ (n=93) | $5.2\%$ | $12\times$ |
| Random pair (lower bound) | $72.9\%$ (n=155) | $1.3\%$ | $55\times$ |

\clearpage

\begin{figure}[!t]
\centering
\includegraphics[width=0.92\linewidth]{/Users/ericdong/nmr-ssl/docs/2d/figures/fig_wrong_struct_v4.pdf}
\caption{\textbf{Joint pass rates on the three wrong-candidate controls.}
Green bars are the joint pass rate for the correct structure; red bars
are the joint pass rate for the wrong candidate. Numbers above each
group are the correct-to-wrong discrimination ratio. The constitutional
isomer control (same molecular formula, different connectivity) is the
chemistry-meaningful number for natural-product dereplication.}
\end{figure}

A $22\%$ false-positive rate on constitutional isomers is too high for a
standalone binary accept/reject system. The use case this supports is
candidate ranking: given a list of plausible structures for an unknown
compound, rank them by their worst-residual score against the observed
HSQC, then send the top few for expert review with HMBC and COSY. The
method does not replace binary structure assignment, and we do not
claim it does.

\subsection*{Scaffold-OOD generalization}

Random 80/10/10 splits tend to overstate generalization, because the
test molecules often share scaffolds with training molecules. We
retrained the $K = 16$, $\lambda = 2$ headline on a Bemis–Murcko
scaffold-stratified split instead: the largest scaffold (benzene
derivatives) is placed in the training set, and the remaining
scaffolds are shuffled across val and test so that no test molecule
shares its Murcko scaffold with any training molecule. Table 4 gives
the result.

**Table 4.** Random vs scaffold-OOD generalization. Scaffold-OOD uses
Bemis–Murcko scaffold stratification, which gives different test-set
sizes across seeds (155, 261, 261); we report the average.

| Split | $^{13}$C MAE (ppm) | $^{1}$H MAE (ppm) | $n_{\text{test}}$ |
|---|---|---|---|
| Random 80/10/10 (main) | $4.53 \pm 0.11$ | $0.35 \pm 0.02$ | 155 |
| Scaffold-OOD (Bemis–Murcko) | $6.06 \pm 0.01$ | $0.40 \pm 0.003$ | 226 (avg) |

The $^{13}$C error grows by 1.5 ppm under the scaffold-stratified
split, while the $^{1}$H error barely moves. This asymmetry is
consistent with NMR chemistry. Proton shifts are dominated by the
local bonding environment, which transfers across scaffolds; carbon
shifts are sensitive to longer-range electronic effects that depend
on the surrounding ring system, which does not transfer.

\subsection*{Robustness to realistic HSQC degradation}

A reasonable concern is that our HSQC targets, derived from
NMRShiftDB2's atom-assigned spectra, are too clean compared to peak
lists scraped from real published papers. NMRShiftDB2's own HSQC
records turn out to be empty stubs in the SDF dump (the 2-D spectra
are referenced in the metadata but the peak-list fields are blank), so
we cannot train against the native records directly. Instead we
simulated the kinds of corruption a literature-mining pipeline would
produce. Our `src/nmr2d/realistic_hsqc.py` pipeline applies four
degradation modes independently: per-peak Gaussian noise in $^{1}$H
and $^{13}$C, per-molecule solvent offsets drawn from a Gaussian,
random peak dropout, and single-linkage peak merging within a
resolution tolerance. We retrained the full 2-D SSL model on four
recipes: a clean baseline, a realistic setting at typical literature
noise levels, a setting that adds peak merging, and an aggressive
worst case.

**Table 5.** Realistic HSQC degradation stress test (3 seeds per row).

| Recipe | $^{13}$C MAE (ppm) | $^{1}$H MAE (ppm) |
|---|---|---|
| Clean baseline | $5.93$ | $0.52$ |
| Realistic ($\sigma_H=0.03$, $\sigma_C=0.5$, $10\%$ dropout, solvent offsets) | $5.82$ | $0.53$ |
| + peak merging | $6.35$ | $0.58$ |
| Aggressive ($\sigma_H=0.08$, $\sigma_C=1.5$, $25\%$ dropout, large solvent offsets, merging) | $6.43$ | $0.65$ |

At realistic noise levels both nuclei stay within 0.1 ppm of the clean
baseline. The $^{13}$C number actually drops slightly, which is
consistent with a regularization effect at moderate noise. The
aggressive worst case costs about half a ppm on $^{13}$C and 0.13 ppm
on $^{1}$H. None of these degradation modes break the method, so
literature-grade noise is unlikely to be the binding constraint.

\FloatBarrier
\clearpage

\subsection*{Hyperparameter sensitivity ($K$, $\lambda$) and axis-aligned decomposition}

**Table 6.** 3-seed $K$-sweep and $\lambda$-sweep (20 epochs each).

| $K$ | $^{13}$C (ppm) | $^{1}$H (ppm) |
|---|---|---|
| 2 | $5.31 \pm 0.02$ | $0.78 \pm 0.33$ |
| 4 | $5.52 \pm 0.18$ | $0.60 \pm 0.09$ |
| 8 | $5.40 \pm 0.57$ | $0.63 \pm 0.24$ |
| **16** | $\mathbf{5.32 \pm 0.30}$ | $\mathbf{0.55 \pm 0.03}$ |
| 32 | $5.36 \pm 0.12$ | $0.57 \pm 0.23$ |

| $\lambda$ | $^{13}$C (ppm) | $^{1}$H (ppm) |
|---|---|---|
| 0.25 | $5.55 \pm 0.24$ | $0.77 \pm 0.52$ |
| 0.5 | $5.32 \pm 0.30$ | $0.55 \pm 0.03$ |
| 1.0 | $5.03 \pm 0.26$ | $0.41 \pm 0.03$ |
| **2.0** | $\mathbf{4.72 \pm 0.17}$ | $\mathbf{0.37 \pm 0.03}$ |

$K = 16$ minimizes both nuclei in the sweep, and $\lambda = 2$ is the
best SSL weight. Both choices are used in the 30-epoch headline of
Table 1. At the same hyperparameter setting, the axis-aligned $K = 2$
variant reaches $5.52$ ppm $^{13}$C and $0.38$ ppm $^{1}$H, which is
within noise of sliced-random $K = 16$ at one-eighth the cost. For
deployment we recommend the axis-aligned variant. We keep the
sliced-random construction in the paper because it is the cleaner
mathematical scaffolding, but the axis-aligned form is the version a
practitioner should run.

\subsection*{Per-carbon-type error decomposition}

**Table 7.** $^{13}$C test MAE decomposed by carbon type.

| Carbon type | $n$ | MAE (ppm) | 90-th percentile |
|---|---|---|---|
| sp$^3$ CH$_{3}$ | 228 | $2.86$ | $5.39$ |
| Aromatic C | 950 | $4.87$ | $10.33$ |
| sp$^3$ CH$_{2}$ | 276 | $5.52$ | $10.71$ |
| sp$^3$ CH | 96 | $6.24$ | $13.01$ |
| Carbonyl / imino C | 97 | $8.03$ | $16.14$ |
| sp$^3$ quaternary | 48 | $11.33$ | $23.68$ |
| Olefinic C | 98 | $11.79$ | $23.16$ |

The heavy tail of the $^{13}$C error distribution sits on olefinic
and sp$^3$-quaternary carbons. Each of these classes accounts for
under $10\%$ of test atoms, so the network sees correspondingly few
training examples of each. Scaling the training corpus beyond 1,542
molecules is the direct way to bring this tail down.

\subsection*{Split-conformal calibration and the molecule-level coverage question}

Split-conformal calibration on the validation split (*Experimental details: Split-conformal calibration*)
gives quantiles of $q_{C} \approx 14.8$ ppm and $q_{H} \approx 1.06$
ppm at the nominal per-atom level $\alpha = 0.05$, with empirical
test-set coverage of $95.2\%$ on $^{13}$C and $96.7\%$ on $^{1}$H.
The Bonferroni table below was computed on a parallel calibration
run on the same validation split with marginally tighter quantiles
($q_{C} = 13.4$ ppm, $q_{H} = 1.03$ ppm); the small numerical
difference reflects validation-split sampling variance and is
unimportant for the per-atom-versus-Bonferroni comparison.

A molecule-level "all peaks of this molecule lie within their
conformal intervals" decision is not the same statistical object as
the per-atom guarantee. Under an independence assumption the joint
coverage for a molecule with $k$ HSQC peaks would be roughly
$(1-\alpha)^{2k}$, which is about $44\%$ at $k=8$. The empirical
uncorrected joint pass rate on the test set is $66.5\%$, well above
the independence bound but well below the per-atom $95\%$ level.
A Bonferroni correction recovers a formal molecule-level guarantee:
we set $\alpha_{\text{atom}} = \alpha_{\text{mol}} / (2k)$ for each
molecule and use the corresponding atom-level quantile.

**Table 8.** Conformal coverage at $\alpha = 0.05$ (validation
calibration, test evaluation).

| Level | $q_{C}$ (ppm) | $q_{H}$ (ppm) | Joint pass |
|---|---|---|---|
| Per-atom $\alpha = 0.05$ | $13.4$ | $1.03$ | 66.5 $\%$ (uncorrected) |
| Bonferroni at $k = 3$ ($\alpha_{\text{atom}} = 0.0083$) | $24.1$ | $1.62$ | — |
| Bonferroni at median $k = 6$ ($\alpha_{\text{atom}} = 0.0042$) | $28.6$ | $1.98$ | — |
| Bonferroni at max $k = 38$ ($\alpha_{\text{atom}} = 0.0007$) | $37.0$ | $3.31$ | — |
| **Bonferroni molecule-adaptive** | — | — | $\mathbf{94.8\%}$ (147 / 155) |

The Bonferroni-corrected joint pass rate lands at $94.8\%$, within
$0.2\%$ of the theoretical $95\%$ target. This is a rigorous
molecule-level guarantee, but the corrected intervals are substantially
wider than the per-atom version (28.6 ppm $^{13}$C half-width at the
median HSQC peak count). We therefore separate the two deliverables:
the per-atom $\alpha = 0.05$ intervals are tight and
chemistry-actionable for outlier flagging, while the
Bonferroni-corrected intervals are mathematically sound but too wide
for day-to-day structure verification. A tighter inference technique,
such as locally adaptive conformal prediction or conformalized
quantile regression, is the obvious next step.

# Discussion

\subsection*{What the results actually show}

An unassigned HSQC peak set carries enough information to train a
$^{1}$H shift predictor to sub-0.5 ppm MAE with no atom-assigned
$^{1}$H labels, and the causal audit (*Causal audit*) confirms the
mechanism. The same loss, layered on top of full $^{13}$C supervision,
improves both nuclei. Peak dropout, merging, solvent offsets, and
Gaussian noise in the realistic range cost about half a ppm on
$^{13}$C and a few hundredths of a ppm on $^{1}$H. Split-conformal
calibration gives the predictor an honest per-atom uncertainty
quantification; a Bonferroni correction recovers a rigorous
molecule-level guarantee, at the cost of considerably wider intervals.

\subsection*{Honest reframing of the 2-D structure}

The axis-aligned $K = 2$ variant matches sliced-random $K = 16$ within
noise at eight times lower cost (see *Hyperparameter sensitivity*).
The $^{1}$H and
$^{13}$C coordinates of the target distribution contribute almost
independently; applying the 1-D sort-match loss to each coordinate
separately captures essentially all of the signal that the sliced
construction picks up. The sliced-Wasserstein framing is the cleaner
theoretical scaffolding and preserves the inequality
$SW_{2}^{2} \leq W_{2}^{2}$, but for deployment the axis-aligned
variant is what practitioners should use. Reading the paper charitably,
the method is two parallel 1-D sort-match problems coupled through a
shared encoder, and the 2-D machinery is an upper bound rather than
the working engine.

\subsection*{Limitations}

We state these explicitly; each is a concrete direction for future work.

1. **Synthetic HSQC supervision, not scraped literature peak lists.** We
   verified that NMRShiftDB2's native HSQC records are empty field
   stubs: the 2-D spectra are referenced in the metadata but the peak
   lists are not written into the SDF dump. Our training targets are
   therefore derived by averaging the per-H shifts on each H-bearing
   carbon from the atom-assigned $^{1}$H spectrum. This preserves the
   combinatorial structure of a real HSQC peak list but not its
   experimental noise characteristics. *Robustness to realistic HSQC
   degradation* establishes
   robustness to a model of realistic degradation but does not
   substitute for an actual scraped-literature evaluation. **This is
   the single largest outstanding gap.**

2. **Absolute accuracy below published baselines on both
   supervision regimes.** NMRNet [1], the fully supervised
   state of the art, achieves $1.10$ ppm $^{13}$C and $0.18$ ppm
   $^{1}$H on the $\sim\!15{,}000$-molecule atom-assigned NMRShiftDB2
   corpus. Jin et al. [11], the concurrent unassigned-HSQC
   semi-supervised baseline, reach approximately $0.8$ ppm $^{13}$C
   and $0.1$ ppm $^{1}$H using millions of literature-extracted
   spectra. Our combined variant reaches $3.23$ ppm $^{13}$C and
   $0.30$ ppm $^{1}$H on a 1,542-molecule filtered subset — roughly
   $10^{3}$ times less training data than either baseline, with a
   correspondingly larger MAE. We do not claim to beat either
   baseline on headline accuracy, and the loss family and
   calibration advances in this paper are orthogonal to the
   scale-driven gap.

3. **Diastereotopic $^{1}$H averaging excludes stereochemistry
   assignment.** We average the attached-H shifts on each heavy atom
   into a single value, which matches natural-product dereplication
   conventions but discards diastereotopic information. The scope of
   the structure-verification claim is therefore restricted to
   scaffold-level dereplication, not to diastereomer assignment. A
   variant that preserves per-hydrogen atom identity is
   straightforward and left to future work.

4. **Structure verification is a ranker, not a classifier.** The
   22$\%$ false-positive rate on constitutional isomers is too high for
   standalone binary accept/reject. The method is explicitly positioned
   as a candidate ranker for dereplication pipelines, not as a
   replacement for expert-interpreted HMBC/COSY/NOESY spectra.

5. **Bonferroni-corrected molecule-level intervals are too wide for
   routine use.** A $28$ ppm $^{13}$C half-width is not a useful
   deployment interval. Locally-adaptive conformal prediction or
   peak-count-conditional quantile regression is the obvious
   remediation; we have not implemented it here.

\subsection*{Where this sits in the landscape}

The method is not a replacement for NMRNet and related single-nucleus
atom-assigned predictors. It is a second, complementary supervision
signal. In practice, the recommended recipe is to use whatever
atom-assigned $^{13}$C data is available to train the supervised head,
add the axis-aligned sort-match loss on the $(^{1}$H, $^{13}$C$)$
multiset for every molecule that has an HSQC peak list, and calibrate
the predictor with split-conformal on a held-out validation split.
Per-atom intervals are useful for outlier flagging, and the
worst-residual score can rank candidate structures in a dereplication
pipeline.

The validation in this paper is on HSQC peak sets derived from
NMRShiftDB2 atom-assigned spectra rather than scraped literature data.
Running the same pipeline on real literature peak lists is the most
important outstanding step; we flag it here instead of burying it.

# Conclusions

HSQC multisets **derived from NMRShiftDB2 atom-assigned spectra**
carry enough supervisory signal to train a joint $^{1}$H / $^{13}$C
chemical-shift predictor without any atom-assigned $^{1}$H labels.
The general feasibility of unassigned-HSQC set supervision for
joint H/C prediction was established in concurrent work by Jin et
al. [11] via exact Hungarian bipartite matching at million-spectrum
scale. The contributions of the present paper are four technical
and statistical advances that sit alongside — not against — that
result, in the order stated in *Background: Related work and
contributions of this paper*. First, a Bonferroni-corrected
split-conformal calibration that delivers a rigorous $94.8\%$
molecule-level coverage guarantee at $\alpha = 0.05$ alongside
$95.2\%$ / $96.7\%$ per-atom coverage; we are not aware of a prior
NMR chemical-shift predictor that ships with a formal
molecule-level uncertainty certificate, and Jin et al. [11] report
point predictions only. Second, a causal $^{1}$H-zeroing audit
that rules out gradient leakage from the $^{13}$C head as the
source of the $^{1}$H signal and localizes the supervision at the
HSQC target itself. Third, a sliced sort-match loss that replaces
the $O(n^{3})$ Hungarian update with $K$ differentiable 1-D sort
problems at $O(K\,n\log n)$ per molecule, keeping training
tractable at small-lab scale and batchable through `torch.sort`.
Fourth, an explicit constitutional-isomer discrimination
experiment showing a 3.5-fold correct-to-wrong ratio, positioned
as a dereplication ranker rather than a standalone accept/reject
classifier. On a 1,542-molecule NMRShiftDB2
filtered-subset benchmark, the four-layer GIN reaches
$4.53 \pm 0.11$ ppm $^{13}$C and $0.35 \pm 0.02$ ppm $^{1}$H test
MAE with only $10\%$ atom-assigned $^{13}$C labels; layered on full
$^{13}$C supervision the combined recipe reaches $3.23 \pm 0.10$
and $0.30 \pm 0.03$ ppm. These headline numbers sit behind both
NMRNet [1] and Jin et al. [11] on absolute accuracy by roughly the
ratio of training-corpus sizes, and we do not claim otherwise; the
contributions of this paper are in loss family, audit, calibration,
and downstream deployment, not in scaling. The single largest
outstanding step — running the same pipeline on HSQC peak lists
scraped from published literature rather than on multisets derived
from atom-assigned spectra — is flagged as future work.

# Declarations

\subsection*{Availability of data and materials}

All experiments use the publicly available NMRShiftDB2 SDF dump
(`nmrshiftdb2withsignals.sd`), release 2026-03-15, distributed under
CC-BY-SA via SourceForge. The exact filtered 1,542-molecule subset
used in this paper is reproducible deterministically from the
dataset construction filters in `src/nmr2d/data_2d.py` applied to
seed-0 random split indices from `experiments/run_2d_experiment.py`.
Trained model checkpoints, conformal quantile tables, per-seed
result JSON files, and figure-generation scripts are archived in
the public repository listed under *Code availability*, at the
submission-frozen git tag referenced below.

\subsection*{Code availability}

Source code, experiment-orchestration scripts, test harness, and
figure-generation code are publicly available at
`https://github.com/dongzhaohe321418-lab/nmr-ssl` under the MIT
license (software) and CC-BY 4.0 (documentation and figures). The
**submission-frozen version** is archived as the git tag
`v2.0-jin-revision` and the corresponding GitHub release. A
mapping from each table and figure in this paper to its exact
generation script and result JSON is provided in the file
`docs/paper_final/ARTIFACT_MAP.md` at that tag, so each headline
number in the paper can be traced to a specific command. Operating
system, Python version, and dependency versions are pinned in the
repository's `pyproject.toml`. The end-to-end training pipeline is
launched by
`python3 experiments/run_option_b_master.py --seeds 0 1 2`;
the causal $^{1}$H-zeroing audit by
`python3 experiments/run_h_zero_ablation.py`;
and the Bonferroni-corrected conformal calibration by
`python3 experiments/compute_bonferroni_conformal.py`. A Zenodo
DOI for the frozen release will be added to the final
camera-ready version on acceptance.

\subsection*{Abbreviations}

NMR, nuclear magnetic resonance; HSQC, heteronuclear single quantum
coherence; SSL, semi-supervised learning; MAE, mean absolute error;
GIN, graph isomorphism network; SW, sliced Wasserstein; OOD,
out-of-distribution; CC-BY, Creative Commons Attribution; ppm, parts
per million.

\subsection*{Competing interests}

The author declares that they have no competing interests.

\subsection*{Funding}

This research received no external funding.

\subsection*{Authors' contributions}

Z.D. conceived the study, designed the methodology, implemented the
software, performed all experiments and data analyses, validated the
numerical results, curated the dataset, produced the figures and
tables, drafted and revised the manuscript, and is responsible for all
final claims.

\subsection*{Acknowledgements}

The author thanks the maintainers of NMRShiftDB2 (Kuhn and Schlörer
[9]) for making the underlying spectral data freely available under
CC-BY-SA, and the Yusuf Hamied Department of Chemistry at the
University of Cambridge for supporting independent open-source work
in chemoinformatics. Code, figures, and manuscript drafts were
developed with AI assistance (Anthropic Claude Opus 4.6) under the
author's direction; the author verified every methodological
choice and numerical result against the source JSON files in the
repository, and takes sole responsibility for all final claims.
Each headline number in this paper is traceable to a specific
script, seed, and result file at the submission-frozen git tag
`v2.0-jin-revision` via the artefact map (see *Code availability*).

\subsection*{Ethics approval and consent to participate}

Not applicable. This work uses only publicly available NMR spectral
data of small organic molecules and involves no human or animal
subjects.

\subsection*{Consent for publication}

Not applicable.

# References

\begingroup
\raggedright
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.45em}
\sloppy

[1] Xu F, Guo W, Wang F, Yao L, Wang H, Tang F, et al. (2025) Toward
a unified benchmark and framework for deep learning-based prediction
of nuclear magnetic resonance chemical shifts. \textit{Nat Comput Sci}
5:292--300. \url{https://doi.org/10.1038/s43588-025-00783-z}

[2] Dong Z (2026) A sort-match theorem for unassigned-set regression
with application to one-dimensional NMR chemical shift prediction.
Companion working note, archived at git tag \texttt{v2.0-jin-revision}
of the project repository.
\url{https://github.com/dongzhaohe321418-lab/nmr-ssl/releases/tag/v2.0-jin-revision}
The theorem and its full proof are reproduced self-contained in
Supplementary Information \textbf{S8} of the present paper.

[3] Bonneel N, Rabin J, Peyré G, Pfister H (2015) Sliced and Radon
Wasserstein barycenters of measures. \textit{J Math Imaging Vis}
51:22--45. \url{https://doi.org/10.1007/s10851-014-0506-3}

[4] Vovk V, Gammerman A, Shafer G (2005) \textit{Algorithmic Learning
in a Random World}. Springer, New York.
\url{https://doi.org/10.1007/b106715}

[5] Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L (2018)
Distribution-free predictive inference for regression.
\textit{J Am Stat Assoc} 113:1094--1111.
\url{https://doi.org/10.1080/01621459.2017.1307116}

[6] Hardy GH, Littlewood JE, Pólya G (1934) \textit{Inequalities}.
Cambridge University Press, Cambridge.

[7] Xu K, Hu W, Leskovec J, Jegelka S (2019) How powerful are graph
neural networks? In: \textit{Proceedings of the International
Conference on Learning Representations (ICLR)}.
\url{https://arxiv.org/abs/1810.00826}

[8] Landrum G (2024) RDKit: open-source cheminformatics software.
\url{https://www.rdkit.org}. Accessed 15 Mar 2026.

[9] Kuhn S, Schlörer NE (2015) Facilitating quality control for
spectra assignments of small organic molecules: nmrshiftdb2 --- a
free in-house NMR database with integrated LIMS for academic service
laboratories. \textit{Magn Reson Chem} 53:582--589.
\url{https://doi.org/10.1002/mrc.4263}

[10] Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G,
et al. (2019) PyTorch: an imperative style, high-performance deep
learning library. In: \textit{Advances in Neural Information
Processing Systems (NeurIPS)} 32:8024--8035.
\url{https://arxiv.org/abs/1912.01703}

[11] Jin Y, Wang Y, Wang JJ, Zhu R, Ke G, E W (2026) From human labels
to literature: semi-supervised learning of NMR chemical shifts at
scale. \textit{arXiv preprint} arXiv:2601.18524, posted 26 January
2026. Concurrent independent work: establishes unassigned-HSQC set
supervision for joint $^{1}$H / $^{13}$C prediction via exact
Hungarian bipartite matching at million-spectrum scale.
\url{https://arxiv.org/abs/2601.18524}

\endgroup

# Supplementary Information

\subsection*{S1. Per-seed results for the headline 2-D SSL variant}

**Table S1.** Per-seed test MAE for the low-label 2-D SSL headline
($K=16$, $\lambda=2$, 30 epochs, $10\%$ labelled fraction).

| Seed | $^{13}$C (ppm) | $^{1}$H (ppm) |
|---|---|---|
| 0 | $4.689$ | $0.332$ |
| 1 | $4.451$ | $0.351$ |
| 2 | $4.464$ | $0.376$ |
| **Mean $\pm$ std** | **$4.535 \pm 0.110$** | **$0.353 \pm 0.018$** |

\subsection*{S2. Per-seed results for the combined full-supervision variant}

**Table S2.** Per-seed test MAE for the combined recipe (full
$^{13}$C supervision on all 1,542 molecules $+$ 2-D SSL, $K=16$,
$\lambda=0.5$, 30 epochs).

| Seed | $^{13}$C (ppm) | $^{1}$H (ppm) |
|---|---|---|
| 0 | $3.338$ | $0.288$ |
| 1 | $3.104$ | $0.350$ |
| 2 | $3.252$ | $0.275$ |
| **Mean $\pm$ std** | **$3.231 \pm 0.097$** | **$0.304 \pm 0.033$** |

\subsection*{S3. Novel (and neutral) multiplicity-augmented loss}

A chemistry reviewer observed that real HSQC peak lists often carry
multiplicity edit-mode tags (CH / CH$_2$ / CH$_3$) that the raw
sort-match loss discards. We added a small multiplicity-classification
head and a histogram-$\ell^1$ loss that compares the softmax-count
histogram of predicted classes against the observed histogram of true
classes, permutation-invariant across atoms within a molecule. The
histogram loss is **neutral** at the default weight
$\lambda_{\text{mul}} = 1$:

**Table S3.** Multiplicity-augmented loss.

| Variant | $^{13}$C (ppm) | $^{1}$H (ppm) |
|---|---|---|
| 2-D SSL headline ($\lambda=2$, $K=16$) | $4.53 \pm 0.11$ | $0.35 \pm 0.02$ |
| 2-D SSL + multiplicity-hist loss ($\lambda_{\text{mul}}=1$) | $4.66 \pm 0.05$ | $0.39 \pm 0.04$ |

We include this as an honest negative result. The histogram loss only
constrains class counts and does not add useful gradient signal at the
default weight; a stronger per-peak classification loss, conditional on
a peak-to-atom alignment, is the obvious next step. Code:
`experiments/run_multiplicity_loss.py`.

\subsection*{S4. Cross-task gradient isolation (stop-gradient control)}

An earlier version of this paper speculated that the
1-D-SSL-vs-2-D-SSL $^{13}$C gap might be caused by SSL gradient noise
leaking into the $^{13}$C head through the shared encoder. We test this
directly by detaching the predicted $^{13}$C shifts before they enter
the sliced sort-match loss, so the SSL gradient flows only into the
$^{1}$H head and the shared encoder.

**Table S4.** Stop-gradient control (3 seeds, 30 epochs).

| Variant | $^{13}$C (ppm) | $^{1}$H (ppm) |
|---|---|---|
| 2-D SSL headline ($K=16$, $\lambda=2$) | $4.53 \pm 0.11$ | $0.35 \pm 0.02$ |
| 2-D SSL with stop-grad on $^{13}$C through SSL | $5.62 \pm 0.20$ | $0.76 \pm 0.38$ |

The stop-gradient variant is decisively worse on both nuclei, which
**disproves** the leakage hypothesis: if leakage were the mechanism,
stop-grad would improve $^{13}$C. Instead, the coupled gradient
through the $^{13}$C prediction is essential. Detaching it breaks the
2-D sort-match quality. The correct interpretation is that the tuned
$\lambda = 2$ weight makes the sliced loss act as a shared-encoder
regularizer that benefits both heads.

\subsection*{S5. Pretrain-then-finetune transfer baseline}

A two-phase alternative trains a dual-head model on the full
$^{13}$C corpus in phase 1 and then fine-tunes with the 2-D SSL loss on
a $10\%$-labelled split in phase 2. Three seeds, 30 epochs each.

**Table S5.** Pretrain-then-finetune.

| Variant | Phase-1 val $^{13}$C | Phase-2 test $^{13}$C | Phase-2 test $^{1}$H |
|---|---|---|---|
| Full $^{13}$C pretrain $\to$ 2-D SSL finetune | $3.45 \pm 0.08$ | $3.53 \pm 0.13$ | $0.40 \pm 0.10$ |
| **Combined (no phase switch)** | — | **$3.23 \pm 0.10$** | **$0.30 \pm 0.03$** |

The combined recipe is strictly better on both nuclei. The
interpretation is that finetuning on only $10\%$ labels in phase 2
discards most of the $^{13}$C supervision signal that phase 1 paid for.
The recommended deployment recipe is therefore to use the combined loss
from the start, not a phase-switch finetune.

\subsection*{S6. Stereo-rich natural-product stress test}

We ran the structure-verification check on five of the most
stereochemically complex test molecules, each with at least two
chiral or E/Z stereo elements and at least three HSQC peaks. One of
five passes the 95$\%$ joint check: a 33-atom diketopiperazine with
4 chiral centers and 12 HSQC peaks. The other four (a 44-atom
carotenoid fragment, a 39-atom diterpene, glucose, and a 31-atom
triterpene) fail because individual olefinic or quaternary carbons
exceed the conformal band; the worst $\Delta_{C}$ values fall
between 4 and 20 ppm. This failure pattern is consistent with Table
7: the heavy tail is concentrated on olefinic and sp$^{3}$-quaternary
carbons, which are precisely the atom types natural-product
scaffolds are rich in and precisely the atom types our
1,542-molecule training corpus under-represents. Scaling the training
pool is the direct remediation. Raw per-molecule results are in
`experiments/results_2d/stereo_demo.json`.

\subsection*{S7. Loss-code audit}

An independent code review verified that the sliced sort-match
implementation in `src/nmr2d/losses_2d.py` faithfully matches the
the *Methods* construction, that the axis-aligned $K=2$ variant corresponds
to the two canonical coordinate projections, that the ratio check in
*Sliced approximation* reports mean $0.414$ with the expected range for
dimension-2 sliced Wasserstein, and that the gradient is finite and
non-NaN end-to-end. The verification test is
`tests/test_theorem_2d.py`; it reproduces the mean ratio and
gradient-norm numbers to within the stated tolerances.

\subsection*{S8. Self-contained statement and proof of the 1-D sort-match theorem}

This section states and proves the 1-D sort-match theorem used in
*Methods: Set-valued NMR supervision* and cited in the second half
of *Background: Related work and contributions* (third contribution).
The proof is a direct application of the Hardy–Littlewood–Pólya
rearrangement inequality [6] and is included here so that the
methodological backbone of the present paper is self-contained
within a single archival document, independently of any companion
preprint.

**Setup.** Fix an integer $n \geq 1$. Let
$\mathbf{x} = (x_{1}, \dots, x_{n})$ and
$\mathbf{y} = (y_{1}, \dots, y_{n})$ be two real-valued tuples of
the same length. Write
$x_{(1)} \leq x_{(2)} \leq \dots \leq x_{(n)}$ for the order
statistics of $\mathbf{x}$ (and similarly $y_{(i)}$). For a
permutation $\pi \in S_{n}$ define the matched squared error
$$
\Phi(\pi; \mathbf{x}, \mathbf{y})
\;\;=\;\;
\sum_{i=1}^{n} \bigl(x_{i} - y_{\pi(i)}\bigr)^{2},
$$
and the sort-matched squared error
$$
\Phi_{\mathrm{sort}}(\mathbf{x}, \mathbf{y})
\;\;=\;\;
\sum_{i=1}^{n} \bigl(x_{(i)} - y_{(i)}\bigr)^{2}.
$$

**Theorem S8.1 (1-D sort-match is optimal bipartite matching).** For
all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^{n}$,
$$
\min_{\pi \in S_{n}}
   \Phi(\pi; \mathbf{x}, \mathbf{y})
\;\;=\;\;
\Phi_{\mathrm{sort}}(\mathbf{x}, \mathbf{y}).
$$

*Proof.* Expand the squared error:
$$
\sum_{i=1}^{n}(x_{i} - y_{\pi(i)})^{2}
\;\;=\;\;
\sum_{i=1}^{n} x_{i}^{2}
\;+\;
\sum_{i=1}^{n} y_{\pi(i)}^{2}
\;-\;
2\sum_{i=1}^{n} x_{i}\,y_{\pi(i)}.
$$
The first sum does not depend on $\pi$. Because $\pi$ is a bijection
of $\{1, \dots, n\}$, the second sum equals
$\sum_{j} y_{j}^{2}$ and is also independent of $\pi$. The third
term is the inner product of $\mathbf{x}$ with a permutation of
$\mathbf{y}$.

By the Hardy–Littlewood–Pólya rearrangement inequality [6, §10.2],
for any real sequences $a_{1} \leq \dots \leq a_{n}$ and
$b_{1} \leq \dots \leq b_{n}$, and any permutation $\sigma \in S_{n}$,
$$
\sum_{i=1}^{n} a_{i}\,b_{\sigma(i)}
\;\;\leq\;\;
\sum_{i=1}^{n} a_{i}\,b_{i},
$$
with the maximum on the right attained when $\sigma$ is the
identity (i.e.\ both sequences sorted in the same order). Applying
this with $a_{i} = x_{(i)}$ and $b_{i} = y_{(i)}$, and absorbing
the permutation $\pi$ of the unsorted tuples into the permutation
$\sigma$ of the sorted tuples, the inner product
$\sum_{i} x_{i}\,y_{\pi(i)}$ is **maximized** over
$\pi \in S_{n}$ precisely when the two tuples are co-sorted —
that is, when the permutation $\pi$ is the one that sorts $\mathbf{y}$
into the same order as $\mathbf{x}$. Equivalently, after first
sorting $\mathbf{x}$, the optimal matched assignment is
$(x_{(1)}, y_{(1)}), (x_{(2)}, y_{(2)}), \dots, (x_{(n)}, y_{(n)})$.

Therefore
$$
\max_{\pi}\, \sum_{i=1}^{n} x_{i}\,y_{\pi(i)}
\;\;=\;\;
\sum_{i=1}^{n} x_{(i)}\,y_{(i)},
$$
which in turn gives
$$
\min_{\pi}\, \Phi(\pi; \mathbf{x}, \mathbf{y})
\;\;=\;\;
\sum_{i=1}^{n} x_{i}^{2}
\,+\,
\sum_{i=1}^{n} y_{i}^{2}
\,-\,
2\sum_{i=1}^{n} x_{(i)}\,y_{(i)}
\;\;=\;\;
\sum_{i=1}^{n} \bigl(x_{(i)} - y_{(i)}\bigr)^{2}
\;\;=\;\;
\Phi_{\mathrm{sort}}(\mathbf{x}, \mathbf{y}),
$$
establishing the theorem. $\square$

**Remark S8.2 (ties and multisets).** The theorem as stated holds
for real tuples; an identical argument gives the multiset version,
because the order statistics are well-defined under any consistent
tie-breaking rule and the rearrangement inequality is insensitive
to tie breaking.

**Remark S8.3 (from $n^{3}$ Hungarian to $n\log n$ sort).** The
optimal bipartite matching between two $n$-point sets in $\mathbb{R}$
is achievable by any algorithm that solves the assignment problem
with cost $\|x_{i} - y_{j}\|^{2}$; in general this requires the
Hungarian algorithm at $O(n^{3})$. Theorem S8.1 reduces the 1-D
case to a sort, at $O(n \log n)$, with no loss of optimality. It
is this reduction — extended to $d$ dimensions by the
sliced-Wasserstein construction [3] — that underlies the
$O(K\,n \log n)$ complexity of the sliced sort-match loss in the
present paper, and that motivates our use of a cheap approximation
in place of the exact Hungarian loss used by Jin et al. [11].

**Remark S8.4 (why the 2-D problem does not reduce to a single sort).**
In $\mathbb{R}^{d}$ with $d \geq 2$, the optimal-transport problem
between two empirical point distributions does not in general admit
a one-dimensional sort-based closed form, and Theorem S8.1 does not
extend to $d \geq 2$ directly. The sliced-Wasserstein construction
of Bonneel et al. [3] uses this observation constructively: it
expresses the $d$-dimensional problem as an expectation, over
random one-dimensional projections, of one-dimensional problems,
each of which is solved exactly by Theorem S8.1 applied to the
projected tuples. The sliced Monte-Carlo estimator of $SW_{2}^{2}$
is therefore an unbiased average of $K$ exact 1-D solutions, and
our training loss $\mathcal{L}_{\mathrm{SSW}}$ in *Methods:
Sliced approximation* is a direct Monte-Carlo estimator of $SW_{2}^{2}$
built on $K$ independent applications of Theorem S8.1.
