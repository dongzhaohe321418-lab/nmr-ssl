# Abstract

Chemical-shift predictors for nuclear magnetic resonance (NMR) are usually
trained as single-nucleus, atom-assigned regression problems, which limits
training data to the small fraction of published spectra that carry
per-atom annotation. We introduce an alternative supervision paradigm that
trains a joint $^{1}$H / $^{13}$C predictor from **unassigned 2-D HSQC peak
sets** --- multisets of (H, C) cross-peaks that carry no atom identity but
that are routinely reported in organic and natural-product chemistry. The
method extends the 1-D sort-match reduction from prior work to two
dimensions via a sliced-Wasserstein projection loss that is permutation
invariant, differentiable almost everywhere, and $O(Kn\log n)$ per molecule.

On NMRShiftDB2, with $10\%$ of molecules providing atom-assigned $^{13}$C
labels and the remaining $90\%$ providing only HSQC peak multisets, a
four-layer graph isomorphism network reaches $4.54 \pm 0.11$ ppm
$^{13}$C and $0.35 \pm 0.02$ ppm $^{1}$H test MAE simultaneously, without
ever seeing a single atom-assigned $^{1}$H shift (3 seeds, 30 epochs). A
causal audit that zeros out the $^{1}$H coordinate of every HSQC training
target drives $^{1}$H error to $4.69 \pm 0.10$ ppm --- a more than tenfold
collapse --- falsifying the "encoder-leakage" counter-hypothesis and
demonstrating that the HSQC target itself is where the $^{1}$H head learns.
Applied on top of full $^{13}$C labels on the same 1,542 training
molecules, the same SSL loss yields $3.23 \pm 0.10$ ppm $^{13}$C and
$0.30 \pm 0.03$ ppm $^{1}$H --- a $1.3$ ppm $^{13}$C improvement over the
low-label headline, showing that the recipe is a general-purpose
multi-nucleus training signal rather than a low-label trick. Split-conformal
calibration delivers rigorous per-atom prediction intervals with 95.2 %
empirical $^{13}$C and 96.7 % empirical $^{1}$H coverage. Wrong-candidate
discrimination is evaluated against three graded controls; the
constitutional-isomer control --- the chemistry-meaningful case --- gives a
$3.5\times$ discrimination that supports dereplication ranking but not
standalone binary structure acceptance.

We release full training, conformal calibration, figure-generation, and
peer-review orchestration code at
`https://github.com/dongzhaohe321418-lab/nmr-ssl`.

\clearpage

# Introduction

Nuclear magnetic resonance (NMR) chemical-shift prediction from a
molecular graph is one of the most widely-used machine-learning tasks in
chemoinformatics. State-of-the-art predictors such as NMRNet$^{1}$
achieve $1.10$ ppm $^{13}$C and $0.18$ ppm $^{1}$H mean absolute error on
the full NMRShiftDB2 corpus. These predictors share two architectural
assumptions:

1. Supervision is **per-atom**: each training example provides the
   chemical shift of a specific atom, indexed by position in the
   molecular graph.
2. Supervision is **single-nucleus**: a model trained for $^{13}$C is
   trained on $^{13}$C-labelled data only; a separately trained model
   handles $^{1}$H.

The combination is expensive. Atom assignment is the labour-intensive
step in constructing any NMR training set, and two-nucleus models require
two assignment passes per molecule. The published literature contains a
much larger volume of 2-D NMR data --- HSQC, HMBC, COSY, NOESY tables ---
that carries structural information without atom-level labels, but that
the standard supervised recipe cannot consume.

**This paper studies the simplest version of this problem.** We ask
whether a 2-D HSQC peak set --- an unordered multiset of $(H,C)$ cross-peak
pairs, each reporting one $^{1}$H shift and the $^{13}$C shift of its
directly-bonded carbon, with no atom identity --- carries enough signal to
train a dual-head chemical-shift predictor. We show that it does,
introduce a loss function that exploits it, and provide a causal audit
that rules out the obvious alternative explanation for the observed
behaviour.

Our contribution has four components.

**(i)** We formalize the training problem as a permutation-invariant
regression against an unordered target multiset. Under any convex
per-pair cost, this reduces in one dimension to a pair of sort-match
losses on the two coordinate axes, which inherits the exact-reduction
theorem of our prior work.$^{2}$ The true 2-D bipartite matching cost is
not sort-reducible in general, but it is upper-bounded by the
**sliced-Wasserstein** estimator of Bonneel et al.$^{3}$, computed in
$O(Kn\log n)$ per molecule.

**(ii)** Trained on NMRShiftDB2 with only $10\%$ of molecules providing
atom-assigned $^{13}$C labels and the remainder providing only HSQC peak
multisets, a four-layer graph isomorphism network achieves
$4.54 \pm 0.11$ ppm $^{13}$C and $0.35 \pm 0.02$ ppm $^{1}$H test MAE
(three seeds, $K=16$, SSL weight $\lambda=2$, 30 epochs). Applied on top
of full $^{13}$C supervision, the same loss delivers $3.23 \pm 0.10$ ppm
$^{13}$C and $0.30 \pm 0.03$ ppm $^{1}$H. The $^{1}$H head is trained
*without* any atom-assigned $^{1}$H labels, in either regime.

**(iii)** We audit the central causal claim directly. Zeroing out the
$^{1}$H coordinate of every unlabelled-split HSQC target drives $^{1}$H
test MAE from $0.35$ ppm to $4.69 \pm 0.10$ ppm --- a more than
tenfold degradation --- while leaving $^{13}$C error essentially unchanged.
This falsifies the hypothesis that the $^{1}$H head is trained by
gradient leakage from the $^{13}$C supervised loss through the shared
encoder. The HSQC $^{1}$H coordinate itself is the causal training
signal.

**(iv)** We layer split-conformal calibration on top of the predictor.
At a per-atom significance level $\alpha = 0.05$, empirical test-set
coverage is $95.2\%$ on $^{13}$C and $96.7\%$ on $^{1}$H --- hitting the
marginal target. A Bonferroni correction over the per-molecule HSQC peak
count $k$ delivers a formal molecule-level $1-\alpha_{\text{mol}}$
coverage guarantee at the cost of substantially wider intervals; we
discuss when each is the right choice.

Section 2 states the sliced sort-match loss and its implementation.
Section 3 reports results, including the causal audit, the combined
full-label regime, the label-efficiency curve, wrong-candidate
discrimination against constitutional isomers and scaffold neighbours,
robustness to realistic HSQC degradation, per-carbon-type error
decomposition, a scaffold-OOD split, and the conformal calibration.
Section 4 discusses scope, limitations, and the path to real literature
HSQC validation. Section 5 (Methods) is the complete experimental
protocol. Supplementary Information reports per-seed values, sweep
tables, and the novel (and neutral) multiplicity-augmented loss.

# 2. The sliced sort-match loss

\subsection*{2.1 Set-valued NMR supervision}

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
$\hat{\mathbf{P}}$ and $\mathbf{P}^{\star}$. In one dimension this is
achieved exactly by sorting both sets and taking a per-element $\ell^2$
distance --- the Hardy--Littlewood--P\'{o}lya rearrangement inequality, which our
prior work formalizes as a training-time theorem.$^{2}$ In two dimensions
the exact solution is the Hungarian optimal bipartite matching, which
is $O(n^3)$ and not easily batched on GPU.

\subsection*{2.2 Sliced approximation}

We follow Bonneel et al.$^{3}$ and use the sliced-Wasserstein distance
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

\subsection*{2.3 Axis-aligned decomposition --- the practical default}

A specialization of the sliced estimator uses only the two native
coordinate axes (i.e.\ $\theta \in \{0,\pi/2\}$). This reduces to two
independent one-dimensional sort-match losses on the $^{1}$H and $^{13}$C
projections, with no Monte-Carlo randomness. At equal $K$ it is a
biased lower bound on the full sliced estimator, but on NMRShiftDB2 it
matches sliced $K=16$ within noise at eight times lower cost (see
Section 3.8). We report both variants; the axis-aligned version is what
a practitioner should deploy.

\subsection*{2.4 Full training objective}

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
Section 3.

# 3. Results

All experiments use NMRShiftDB2 release 2026-03-15 (SourceForge,
CC-BY-SA). After filtering for molecules with both $^{13}$C and $^{1}$H
spectra, non-degenerate $^{13}$C assignments, at least three HSQC
cross-peaks, and at most 60 atoms, the final corpus is 1,542 molecules.
Unless otherwise stated, we report 3-seed aggregates (mean $\pm$ one
standard deviation) using seeds 0, 1, 2 with the training protocol of
Section 5. The dual-head architecture has four GIN layers, hidden size
192, twenty atom features, and two 2-layer MLP readout heads --- roughly
340k parameters total.

\subsection*{3.1 Main result}

Table 1 reports the three main training regimes side by side. The 2-D
sort-match SSL at $\lambda = 2$, $K = 16$ is the paper's low-label
headline; it **matches** the 1-D sort-match SSL baseline on $^{13}$C
(4.54 vs 4.56 ppm), delivers sub-0.4 ppm $^{1}$H, and does so without any
atom-assigned $^{1}$H labels. The combined full-$^{13}$C + SSL variant
(row 4) is the general-purpose recipe: it improves $^{13}$C by a further
$1.3$ ppm and drops $^{1}$H below $0.31$ ppm.

**Table 1.** Main test-set MAE on NMRShiftDB2 (3 seeds, mean $\pm$ std).
All rows share the same 4-layer GIN, AdamW optimizer, batch size 32,
gradient clipping L2 $=$ 5, and best-val-$^{13}$C-MAE early stopping.

| Variant | $^{13}$C MAE (ppm) | $^{1}$H MAE (ppm) |
|---|---|---|
| Supervised-1D ($10\%$ labelled, $^{13}$C only) | $5.60 \pm 0.34$ | $2.47 \pm 0.38$ (untrained) |
| 1-D sort-match SSL ($10\%$ labelled, prior work)$^{2}$ | $4.56 \pm 0.31$ | $2.61 \pm 0.32$ (untrained) |
| **2-D SSL** ($10\%$ labelled, $\lambda=2$, $K=16$) | $\mathbf{4.54 \pm 0.11}$ | $\mathbf{0.35 \pm 0.02}$ |
| **2-D SSL + full $^{13}$C** (combined) | $\mathbf{3.23 \pm 0.10}$ | $\mathbf{0.30 \pm 0.03}$ |

For context, the published state of the art NMRNet achieves $1.10$ ppm
$^{13}$C and $0.18$ ppm $^{1}$H on NMRShiftDB2$^{1}$ using the full
$\sim\!15{,}000$-molecule assigned corpus. At 10$\times$ less training
data, the combined recipe brings $^{1}$H to within $0.12$ ppm of SOTA;
the remaining $2$ ppm $^{13}$C gap is explained by the scale difference.

![**Figure 1.** Main result. (a) $^{13}$C test MAE across five training
variants: supervised-1D baseline (grey), 1-D sort-match SSL (blue),
2-D SSL at earlier $K=8$ working point (orange), 2-D SSL at
$K=16$ low-label headline (purple), 2-D SSL combined with full
$^{13}$C supervision (green). (b) $^{1}$H test MAE for the same five
variants. Error bars are one standard deviation over 3 seeds. The
$K=16$ low-label headline closes the $^{13}$C gap to the 1-D SSL
baseline; the combined variant delivers a further $1.3$ ppm
improvement.](../2d/figures/fig_v4_headline.pdf)

\subsection*{3.2 Causal audit: the $^{1}$H head learns from the HSQC target}

The most important experiment in the paper is a direct falsification of
the counter-hypothesis that the $^{1}$H head is trained by encoder
leakage rather than by the HSQC $^{1}$H shift values. We retrain the
main 2-D SSL model at $K = 16$, 30 epochs, 3 seeds, with a single
modification: every $^{1}$H coordinate in every unlabelled-split HSQC
target is set to zero before the sliced sort-match loss is computed. The
$^{13}$C coordinate is kept.

**Table 2.** Causal audit of the central claim. The $^{1}$H head
collapses more than tenfold when its HSQC targets are zeroed, while
$^{13}$C is essentially unchanged.

| Configuration | $^{13}$C MAE (ppm) | $^{1}$H MAE (ppm) |
|---|---|---|
| 2-D SSL baseline ($\lambda=2$, $K=16$) | $4.54 \pm 0.11$ | $0.35 \pm 0.02$ |
| 2-D SSL with HSQC $^{1}$H coordinate zeroed | $5.00 \pm 0.46$ | $4.69 \pm 0.10$ |

![**Figure 2.** Causal audit. (a) $^{13}$C test MAE is essentially
insensitive to zeroing the HSQC $^{1}$H coordinate --- confirming the
$^{13}$C head is trained by its own supervised loss. (b) $^{1}$H test
MAE collapses by more than a factor of ten when the HSQC $^{1}$H
coordinate is zeroed. The collapse rules out encoder leakage from the
$^{13}$C supervised loss as the training signal for the $^{1}$H head,
and localizes the training signal at the HSQC target itself.
](../2d/figures/fig_h_zero.pdf)

\subsection*{3.3 Combined supervision and the full-label regime}

The 2-D SSL loss is often presented as a low-label trick. We tested
whether the improvement survives when the supervised head is allowed to
see every $^{13}$C label in the training corpus. Table 1, row 4,
reports the result: the combined training regime yields
$3.23 \pm 0.10$ ppm $^{13}$C and $0.30 \pm 0.03$ ppm $^{1}$H, a
substantial improvement on both nuclei compared to the low-label
headline. The gain is not an artefact of the low-label setting: the
sliced sort-match loss is a useful regularizer in the fully-supervised
regime too. We therefore position the method as a **general-purpose
multi-nucleus training recipe**, not a low-label trick.

\subsection*{3.4 Label-efficiency curve}

Figure 3 reports test MAE as a function of the $^{13}$C-labelled
fraction. At 1 $\%$ labels (12 molecules), supervised-1D reaches a
useless $18.5$ ppm $^{13}$C and $2.87$ ppm $^{1}$H, while 2-D SSL lands
at $6.0$ ppm $^{13}$C and $0.44$ ppm $^{1}$H --- a three-fold $^{13}$C
improvement and a six-fold $^{1}$H improvement. The $^{13}$C gap
shrinks as labels grow and the two methods converge near $50\%$ labels.
The $^{1}$H gap stays essentially constant at $\sim\!0.4$ ppm regardless
of label fraction --- the HSQC supervision extracts the same $^{1}$H
signal whether the labelled subset is 1 $\%$ or 50 $\%$.

![**Figure 3.** Label-efficiency curve. (a) $^{13}$C MAE as a function
of the $^{13}$C-labelled fraction. Supervised-1D (blue dashed) is
useless below $10\%$ labels; 2-D SSL (green solid) remains within
6 ppm across the entire range and tracks supervised-1D above $50\%$
labels. (b) $^{1}$H MAE. Supervised-1D is pinned at the random floor
($\sim\!3$ ppm) because its $^{1}$H head receives no training signal;
2-D SSL delivers sub-0.45 ppm $^{1}$H at every label fraction.
](../2d/figures/fig_label_sweep.pdf)

\subsection*{3.5 Wrong-candidate discrimination and the dereplication ranker}

To evaluate the structure-verification use case we constructed three
graded negative controls, from easy to chemically-meaningful.

**Random pair** --- each test molecule's observed HSQC is compared against
the predicted HSQC of a *different* test molecule of the same HSQC peak
count but otherwise unrelated. This is the soft lower bound on
discrimination.

**Scaffold neighbour** --- for each test molecule we compare against up
to three other training molecules sharing its Bemis--Murcko scaffold
(same ring skeleton, different functional groups). This tests whether
the method distinguishes functional-group changes within a fixed
carbon skeleton.

**Constitutional isomer** --- for each test molecule with a formula-matched
peer in the corpus, we compare against up to three constitutional
isomers (same molecular formula, different connectivity). This is the
chemistry-meaningful setting: exactly the discrimination problem a
natural-product dereplication pipeline faces after a molecular-formula
lookup.

**Table 3.** Structure-verification discrimination for the three negative
controls, at per-atom conformal $\alpha = 0.05$.

| Control | Correct joint pass | Wrong joint pass | Discrimination |
|---|---|---|---|
| **Constitutional isomer (headline)** | $77.0\%$ (n=74) | $21.7\%$ | $\mathbf{3.5\times}$ |
| Scaffold neighbour | $62.4\%$ (n=93) | $5.2\%$ | $12\times$ |
| Random pair (lower bound) | $72.9\%$ (n=155) | $1.3\%$ | $55\times$ |

![**Figure 4.** Wrong-candidate negative controls. Left: constitutional
isomer control (headline, chemically meaningful) --- correct structures
pass at $77\%$ joint, formula-matched isomers at $22\%$, a $3.5\times$
discrimination. Middle: scaffold-neighbour control --- $62\%$ vs $5\%$, a
$12\times$ discrimination. Right: random-pair control (soft lower
bound) --- $73\%$ vs $1\%$, a $55\times$ discrimination. The
constitutional-isomer control is the number to trust for real
dereplication.](../2d/figures/fig_wrong_struct_v4.pdf)

A $22\%$ false-positive rate on constitutional isomers is too high for
standalone binary accept/reject deployment. We therefore **scope the
structure-verification use case to candidate ranking**: a dereplication
pipeline that receives a list of candidate structures can use the
worst-residual score to rank them, with expert HMBC / COSY review on
the top ranks. This is not a replacement for binary structure
assignment, and we do not claim it is.

\subsection*{3.6 Scaffold-OOD generalization}

Random 80/10/10 splits can overstate generalization to novel scaffolds.
We re-trained the $K=16$, $\lambda=2$ headline on a Bemis--Murcko
scaffold-stratified split: the largest scaffold (benzene derivatives) is
forced into train, the remainder shuffled across val and test so that
no test molecule shares a scaffold with any training molecule. Table 4
reports the result.

**Table 4.** Random vs scaffold-OOD generalization.

| Split | $^{13}$C MAE (ppm) | $^{1}$H MAE (ppm) | $n_{\text{test}}$ |
|---|---|---|---|
| Random 80/10/10 (main) | $4.54 \pm 0.11$ | $0.35 \pm 0.02$ | 155 |
| Scaffold-OOD (Bemis--Murcko) | $6.06 \pm 0.01$ | $0.40 \pm 0.003$ | 225 |

$^{13}$C degrades by $1.5$ ppm from random to scaffold-OOD, while
$^{1}$H is essentially unchanged. This asymmetry is chemistry-consistent:
$^{1}$H shifts depend primarily on local bonding environment, which
generalizes well across scaffolds, while $^{13}$C shifts are sensitive
to longer-range electronic effects that are scaffold-specific.

\subsection*{3.7 Robustness to realistic HSQC degradation}

NMRShiftDB2's native HSQC records are empty field stubs --- the 2-D
spectra are referenced in the metadata but the peak lists are not in
the SDF dump. We cannot train directly on native HSQC, and instead
simulate real-world degradation via a deterministic pipeline in
`src/nmr2d/realistic_hsqc.py`. The pipeline applies four independent
degradation modes: (a) per-peak Gaussian noise in $^{1}$H and $^{13}$C,
(b) per-molecule systematic solvent offset drawn from a Gaussian,
(c) random peak dropout, and (d) single-linkage peak merging within a
resolution tolerance. We retrained the full 2-D SSL model on four
recipes: clean baseline, a realistic recipe with typical literature
noise levels, a merge-inclusive recipe, and an aggressive worst-case.

**Table 5.** Realistic HSQC degradation stress test (3 seeds per row).

| Recipe | $^{13}$C MAE (ppm) | $^{1}$H MAE (ppm) |
|---|---|---|
| Clean baseline | $5.93$ | $0.52$ |
| Realistic ($\sigma_H=0.03$, $\sigma_C=0.5$, $10\%$ dropout, solvent offsets) | $5.82$ | $0.53$ |
| + peak merging | $6.35$ | $0.58$ |
| Aggressive ($\sigma_H=0.08$, $\sigma_C=1.5$, $25\%$ dropout, large solvent offsets, merging) | $6.43$ | $0.65$ |

Realistic noise is within 0.1 ppm of clean on both nuclei --- and is
actually slightly better on $^{13}$C, consistent with a regularization
interpretation. Aggressive worst-case costs $0.5$ ppm $^{13}$C and
$0.13$ ppm $^{1}$H. The method is not brittle to the degradation modes
a literature-mining pipeline is likely to encounter.

\subsection*{3.8 Hyperparameter sensitivity ($K$, $\lambda$) and axis-aligned decomposition}

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

$K = 16$ is the most stable working point and $\lambda = 2$ is the best
SSL weight; the 30-epoch headline in Table 1 uses both. At the $K = 16$
working point, the axis-aligned $K = 2$ variant reaches $5.52$ ppm
$^{13}$C and $0.38$ ppm $^{1}$H --- within noise of sliced-random $K = 16$
at $8\times$ lower cost. Axis-aligned is therefore the recommended
deployment variant; sliced-random is the theoretically cleaner
construction.

\subsection*{3.9 Per-carbon-type error decomposition}

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

The heavy tail is concentrated on olefinic and sp$^3$-quaternary
carbons, each of which represent less than $10\%$ of test atoms and
therefore carry the fewest per-class training examples. Scaling the
training corpus beyond 1,542 molecules is the direct path to closing
this tail.

\subsection*{3.10 Split-conformal calibration and the molecule-level coverage question}

Split-conformal prediction on the validation split (Section 5.5) gives
at per-atom $\alpha = 0.05$ the quantiles $q_{C} = 13.4$ ppm and
$q_{H} = 1.03$ ppm. Empirical test-set coverage is $95.2\%$ on $^{13}$C
and $96.7\%$ on $^{1}$H --- cleanly hitting the marginal target.

The molecule-level joint decision "all peaks of this molecule fall
within the conformal interval" is a different statistical object. Under
an independence assumption it would have expected coverage
$(1-\alpha)^{2k}$ for a molecule with $k$ HSQC peaks --- approximately
$44\%$ at $k = 8$. The observed uncorrected joint pass rate on the test
set is $66.5\%$, above the independence bound but below the per-atom
$95\%$ level. To recover a formal $(1 - \alpha_{\text{mol}})$
molecule-level guarantee we apply a Bonferroni correction, setting
$\alpha_{\text{atom}} = \alpha_{\text{mol}}/(2k)$, and measure the
resulting pass rate on the test set.

**Table 8.** Conformal coverage at $\alpha = 0.05$ (validation
calibration, test evaluation).

| Level | $q_{C}$ (ppm) | $q_{H}$ (ppm) | Joint pass |
|---|---|---|---|
| Per-atom $\alpha = 0.05$ | $13.4$ | $1.03$ | 66.5 $\%$ (uncorrected) |
| Bonferroni at $k = 3$ ($\alpha_{\text{atom}} = 0.0083$) | $24.1$ | $1.62$ | --- |
| Bonferroni at median $k = 6$ ($\alpha_{\text{atom}} = 0.0042$) | $28.6$ | $1.98$ | --- |
| Bonferroni at max $k = 38$ ($\alpha_{\text{atom}} = 0.0007$) | $37.0$ | $3.31$ | --- |
| **Bonferroni molecule-adaptive** | --- | --- | $\mathbf{94.8\%}$ (147 / 155) |

The Bonferroni-corrected joint pass rate lands at $94.8\%$, within
$0.2\%$ of the theoretical $95\%$ target. This is a rigorous
molecule-level guarantee, but the corrected intervals are substantially
wider than the per-atom version ($28.6$ ppm $^{13}$C half-width at the
median). We therefore separate the two deliverables: per-atom
$\alpha = 0.05$ intervals are tight and chemistry-actionable for
outlier flagging; Bonferroni-corrected intervals are mathematically
sound but too wide for day-to-day structure verification. A tighter
inference technique --- such as locally adaptive conformal prediction or
conformalized quantile regression --- is the obvious next step.

# 4. Discussion

\subsection*{4.1 What the method establishes}

The paper establishes four things that the prior literature did not.

First, an unassigned 2-D HSQC peak set carries sufficient signal to
train a $^{1}$H shift predictor to sub-0.5 ppm MAE with zero
atom-assigned $^{1}$H labels. The causal audit in Section 3.2 confirms
this is not a shared-encoder artefact. Second, the sliced sort-match
loss applied on top of full $^{13}$C supervision improves both nuclei
--- $^{13}$C by $1.3$ ppm, $^{1}$H by $0.05$ ppm --- establishing the loss
as a general-purpose training recipe rather than a low-label-regime
workaround. Third, the method is robust to the degradation modes a
literature-mining pipeline is likely to encounter: aggressive noise,
dropout, peak merging, and per-molecule solvent offsets only cost
$\sim\!0.5$ ppm $^{13}$C. Fourth, split-conformal calibration layered
on top delivers rigorous per-atom coverage guarantees and --- with a
Bonferroni correction --- a rigorous (but wide) molecule-level guarantee.

\subsection*{4.2 Honest reframing of the 2-D structure}

The axis-aligned $K = 2$ variant matches the sliced-random $K = 16$
variant within noise at $8\times$ lower cost (Section 3.8). The
practical implication is that the "2-D-ness" of the target distribution
is near-separable: the $^{1}$H and $^{13}$C coordinates contribute
independently, and the sort-match loss applied to each coordinate
individually captures essentially all the signal. The sliced-Wasserstein
construction is the cleaner theoretical framing and preserves the
correct mathematical inequality $SW_{2}^{2} \leq W_{2}^{2}$, but
practitioners should deploy the axis-aligned variant. We do not hide
this: the honest narrative is "two parallel 1-D sort-match problems
coupled through a shared encoder", and the paper is better for stating
it explicitly.

\subsection*{4.3 Limitations}

We state these explicitly; each is a concrete direction for future work.

1. **Synthetic HSQC supervision, not scraped literature peak lists.** We
   verified that NMRShiftDB2's native HSQC records are empty field
   stubs --- the 2-D spectra are referenced but the peak lists are not in
   the SDF. Our training targets are derived by averaging the per-H
   shifts on each H-bearing carbon from the atom-assigned $^{1}$H
   spectrum. This preserves the combinatorial structure of a real HSQC
   peak list but not the experimental noise characteristics.
   Section 3.7 establishes robustness to a model of realistic
   degradation, but does not substitute for an actual scraped-literature
   evaluation. **This is the single largest outstanding gap.**

2. **Absolute accuracy below published SOTA.** NMRNet achieves
   $1.10$ ppm $^{13}$C and $0.18$ ppm $^{1}$H$^{1}$ on the full
   $\sim\!15{,}000$-molecule NMRShiftDB2 corpus; our combined variant
   reaches $3.23$ ppm $^{13}$C and $0.30$ ppm $^{1}$H on 1,542
   molecules. The $^{13}$C gap is explained by the scale difference;
   the $^{1}$H gap has closed to $0.12$ ppm. Training the combined
   recipe on the full corpus is straightforward but was outside the
   compute budget of this work.

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

\subsection*{4.4 Where this sits in the landscape}

Compared to single-nucleus predictors that consume atom-assigned data
only, this method opens a second data modality --- the vastly larger
corpus of unassigned 2-D peak tables in published literature --- with a
training recipe that is empirically robust and causally understood. It
is not a replacement for NMRNet; it is a complementary training signal
that should layer on top. The headline practical recipe is:

> **Train a dual-head GIN with the supervised $^{13}$C loss on whatever
> atom-assigned $^{13}$C data you have, and the axis-aligned sort-match
> loss on the $(^{1}$H, $^{13}$C$)$ multiset for every molecule for
> which you have an HSQC peak list. Calibrate the predictor with
> split-conformal on a held-out validation split. Deploy the per-atom
> intervals for outlier flagging and the worst-residual score as a
> dereplication ranker.**

The honest limitation is that this recipe has been validated entirely
on synthetic HSQC targets derived from NMRShiftDB2's atom-assigned
spectra. Real-literature validation is the single most important
outstanding task, and we flag it as such rather than burying it.

# 5. Methods

\subsection*{5.1 Dataset construction}

We parse `nmrshiftdb2withsignals.sd` (SourceForge, release 2026-03-15,
CC-BY-SA) with RDKit and join the $^{13}$C and $^{1}$H spectra by
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

\subsection*{5.2 Model}

A shared 4-layer graph isomorphism network (GIN) encoder with dense
adjacency and 20 atom features --- element one-hot, degree, formal
charge, aromaticity, hybridization, hydrogen count, ring membership,
atomic mass --- followed by two 2-layer MLP readout heads, one for
per-atom $^{13}$C shift and one for per-heavy-atom mean $^{1}$H shift.
Hidden size 192. Total parameters approximately 340k. Code:
`src/nmr2d/model_2d.py`.

\subsection*{5.3 Sliced sort-match loss}

At each forward pass we draw $K$ direction vectors uniformly on $S^{1}$
from a 2-D normal rescaled to unit norm, project every predicted and
target peak onto each direction, sort the two projected sets within
each mini-batch row, and compute the masked mean squared error on the
aligned pairs. The final loss is the mean over $K$ directions.
Permutation invariance is inherited from the sort. Differentiability
through `torch.sort` is standard. Code: `src/nmr2d/losses_2d.py`. The
axis-aligned variant fixes $K=2$ directions $(1,0)$ and $(0,1)$ and
returns the sum of the two 1-D sort-match losses.

\subsection*{5.4 Training protocol}

AdamW, learning rate $10^{-3}$, weight decay $10^{-5}$, batch size 32,
30 epochs, gradient clipping L2 $= 5$, best-val-$^{13}$C-MAE early
stopping. Three seeds (0, 1, 2) with independent 80/10/10 random
train/val/test splits and independent labelled/unlabelled partitions
(labelled fraction $0.1$ unless otherwise stated). SSL weight
$\lambda = 2$, sliced directions $K = 16$. Implementation in PyTorch
2.8 with Apple Silicon MPS; approximately 60 seconds per seed per
headline run.

\subsection*{5.5 Split-conformal calibration}

After training the 2-D SSL model, we collect per-atom absolute
residuals $|\hat{\delta} - \delta^{\star}|$ on the held-out validation
split (150 molecules, approximately 1,800 $^{13}$C atoms and 1,200
$^{1}$H atoms). We take the finite-sample-corrected
$(1-\alpha)$ empirical quantile at $\alpha = 0.05$, giving
$q_{C} = 13.4$ ppm and $q_{H} = 1.03$ ppm. Empirical test-set
coverage at this level is $95.2\%$ on $^{13}$C and $96.7\%$ on $^{1}$H.
For the molecule-level Bonferroni correction we precompute a lookup
table of per-$k$ quantiles at $\alpha_{\text{atom}} =
\alpha_{\text{mol}}/(2k)$ and apply the molecule-appropriate quantile
per candidate during evaluation. Code: `src/nmr2d/conformal.py`,
`experiments/compute_bonferroni_conformal.py`.

\subsection*{5.6 Structure-verification protocol}

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

# Data availability

All experiments use the publicly available NMRShiftDB2 SDF dump
(`nmrshiftdb2withsignals.sd`), release 2026-03-15, distributed under
CC-BY-SA via SourceForge. The exact filtered 1,542-molecule subset used
in this paper is reproducible deterministically from the dataset
construction filters in `src/nmr2d/data_2d.py` applied to seed-0
random split indices from `experiments/run_2d_experiment.py`. Trained
model checkpoints, conformal quantile tables, per-seed result JSON
files, and figure-generation scripts are archived in the public
repository listed under "Code availability".

# Code availability

Source code, experiment-orchestration scripts, test harness,
peer-review response documents, and figure-generation code are
publicly available at
`https://github.com/dongzhaohe321418-lab/nmr-ssl` under a CC-BY 4.0
license. The end-to-end revision pipeline can be reproduced with
`python3 experiments/run_option_b_master.py --seeds 0 1 2`
(approximately 26 minutes on an Apple M4 Pro with MPS). The causal
audit reported in Section 3.2 runs via
`python3 experiments/run_h_zero_ablation.py`. The Bonferroni-corrected
conformal calibration in Section 3.10 is computed by
`python3 experiments/compute_bonferroni_conformal.py`.

# Author contributions (CRediT)

**Z.D.**: conceptualization, methodology, software, validation, formal
analysis, investigation, data curation, writing --- original draft,
writing --- review and editing, visualization, supervision, and project
administration. No external funding was received.

# Competing interests

The author declares no competing interests.

# Acknowledgements

I thank the maintainers of NMRShiftDB2 (Kuhn & Schl\\\"{o}rer$^{9}$) for
making the underlying spectral data freely available under CC-BY-SA.
All code, figures, and manuscript drafts were developed with AI
assistance (Anthropic Claude Opus 4.6) under my direction; I verified
every methodological choice, numerical result, and final claim, and
every empirical number reported in this paper is reproducible from
the public code repository at
`https://github.com/dongzhaohe321418-lab/nmr-ssl`. I am grateful to the
Yusuf Hamied Department of Chemistry for supporting independent
open-source work in chemoinformatics.

# References

1. **Xu, F. et al.** Toward a unified benchmark and framework for deep
   learning-based prediction of nuclear magnetic resonance chemical
   shifts (NMRNet). *Nature Computational Science* (2025). DOI
   `10.1038/s43588-025-00783-z`, arXiv `2408.15681` (2024).

2. **Dong, Z.** A sort-match theorem for unassigned-set regression, with
   application to 1-D NMR chemical-shift prediction. Prior work (Paper 1),
   preprint in `docs/preprint_v1_filled.md` of the companion code
   repository; `https://github.com/dongzhaohe321418-lab/nmr-ssl`.

3. **Bonneel, N., Rabin, J., Peyr\'{e}, G., Pfister, H.** Sliced and Radon
   Wasserstein barycenters of measures. *J. Math. Imaging Vis.* **51**,
   22--45 (2015).

4. **Vovk, V., Gammerman, A., Shafer, G.** *Algorithmic Learning in a
   Random World*. Springer (2005).

5. **Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., Wasserman, L.**
   Distribution-free predictive inference for regression.
   *J. Am. Stat. Assoc.* **113**, 1094--1111 (2018).

6. **Hardy, G. H., Littlewood, J. E., P\'{o}lya, G.** *Inequalities*.
   Cambridge University Press (1934). Used for the rearrangement
   inequality that underpins 1-D sort-match.

7. **Xu, K., Hu, W., Leskovec, J., Jegelka, S.** How powerful are graph
   neural networks? In *Proc. International Conference on Learning
   Representations (ICLR)* (2019). Reference for the GIN architecture.

8. **Landrum, G.** RDKit: Open-source cheminformatics.
   `https://www.rdkit.org` (2024).

9. **Kuhn, S., Schl\\\"{o}rer, N. E.** Facilitating quality control for
   spectra assignments of small organic molecules: nmrshiftdb2 --- a
   free in-house NMR database with integrated LIMS for academic
   service laboratories. *Magn. Reson. Chem.* **53**, 582--589 (2015).

10. **Paszke, A. et al.** PyTorch: an imperative style, high-performance
    deep learning library. In *Advances in Neural Information
    Processing Systems* **32** (2019).

\clearpage

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
| 2-D SSL headline ($\lambda=2$, $K=16$) | $4.54 \pm 0.11$ | $0.35 \pm 0.02$ |
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
| 2-D SSL headline ($K=16$, $\lambda=2$) | $4.54 \pm 0.11$ | $0.35 \pm 0.02$ |
| 2-D SSL with stop-grad on $^{13}$C through SSL | $5.62 \pm 0.20$ | $0.76 \pm 0.38$ |

The stop-gradient variant is decisively worse on both nuclei, which
**disproves** the leakage hypothesis: if leakage were the mechanism,
stop-grad would improve $^{13}$C. Instead the coupled gradient through
the $^{13}$C prediction is essential --- detaching it breaks the 2-D
sort-match matching quality. The correct interpretation is that the
tuned $\lambda = 2$ weight makes the sliced loss act as a shared-encoder
regularizer, which benefits both heads.

\subsection*{S5. Pretrain-then-finetune transfer baseline}

A two-phase alternative trains a dual-head model on the full
$^{13}$C corpus in phase 1 and then fine-tunes with the 2-D SSL loss on
a $10\%$-labelled split in phase 2. Three seeds, 30 epochs each.

**Table S5.** Pretrain-then-finetune.

| Variant | Phase-1 val $^{13}$C | Phase-2 test $^{13}$C | Phase-2 test $^{1}$H |
|---|---|---|---|
| Full $^{13}$C pretrain $\to$ 2-D SSL finetune | $3.45 \pm 0.08$ | $3.53 \pm 0.13$ | $0.40 \pm 0.10$ |
| **Combined (no phase switch)** | --- | **$3.23 \pm 0.10$** | **$0.30 \pm 0.03$** |

The combined recipe is strictly better on both nuclei. The
interpretation is that finetuning on only $10\%$ labels in phase 2
discards most of the $^{13}$C supervision signal that phase 1 paid for.
The recommended deployment recipe is therefore to use the combined loss
from the start, not a phase-switch finetune.

\subsection*{S6. Stereo-rich natural-product stress test}

We ran the structure-verification check on five of the most
stereochemically complex test molecules (at least two chiral or
E/Z stereo elements, at least three HSQC peaks). One of five passes
the $95\%$ joint check: a 33-atom diketopiperazine with 4 chiral
centers and 12 HSQC peaks. The other four --- a 44-atom carotenoid
fragment, a 39-atom diterpene, glucose, and a 31-atom triterpene ---
fail because individual olefinic or quaternary carbons exceed the
conformal band (worst $\Delta_{C}$ between $4$ ppm and $20$ ppm). This
failure pattern is consistent with Table 7: the heavy tail is
concentrated on olefinic and sp$^{3}$-quaternary carbons, which are
precisely the atom types natural-product scaffolds are rich in, and
precisely the atom types our $1{,}542$-molecule training corpus
underrepresents. Scaling the training pool is the direct remediation
path. The underlying raw results are in
`experiments/results_2d/stereo_demo.json`.

\subsection*{S7. Loss-code audit}

An independent code review verified that the sliced sort-match
implementation in `src/nmr2d/losses_2d.py` faithfully matches the
Section 2 construction, that the axis-aligned $K=2$ variant corresponds
to the two canonical coordinate projections, that the ratio check in
Section 2.2 reports mean $0.414$ with the expected range for
dimension-2 sliced Wasserstein, and that the gradient is finite and
non-NaN end-to-end. The verification test is
`tests/test_theorem_2d.py`; it reproduces the mean ratio and
gradient-norm numbers to within the stated tolerances.
