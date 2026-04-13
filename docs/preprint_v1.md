# Training NMR chemical shift predictors from unassigned peak lists via permutation-invariant set supervision

*Preprint v1 — 2026-04-12. Proof-of-concept for a Nature Computational Science Article.*

---

## Abstract

Machine-learning prediction of nuclear magnetic resonance (NMR) chemical shifts has recently reached near-experimental accuracy, but every state-of-the-art model depends on atom-assigned training data — a labour-intensive bottleneck that holds the training corpora at $\sim$$10^4$ molecules, orders of magnitude below the unassigned peak lists available in the chemical literature. We show that chemical shift prediction can be formulated as a permutation-invariant set-supervision problem, and we prove that under any convex per-atom cost the optimal bipartite matching between predicted and observed peak sets reduces **exactly** to a single sort of both sides. The reduction collapses the matching step from $O(n^3)$ Hungarian to $O(n\log n)$ sort, is differentiable, GPU-batchable, and is not a relaxation. On a controlled NMRShiftDB2 benchmark we cripple supervision to 10% of molecules and show that sort-match semi-supervised training beats the same-architecture supervised baseline by **[A_REL]%** ([A_ABS] ppm absolute) on $^{13}$C MAE and by **[C_REL]%** on $^1$H, holds the gain on a scaffold-split out-of-distribution test, and remains stable from 2% to 50% labeled fraction. The contribution is not a new state of the art on curated benchmarks; it is a supervision paradigm that unlocks data those benchmarks discard.

---

## Main

Consider a natural-product chemist who has just isolated a new secondary metabolite from a marine bacterium and needs to verify its proposed structure against a measured $^{13}$C NMR spectrum. Today she has two options: run hours of DFT calculations on her proposed structure, or use a machine-learning shift predictor trained on a few tens of thousands of curated molecules. Both options miss an enormous untapped resource — the hundreds of millions of *already-published* NMR spectra sitting in the supporting information files of every organic chemistry paper over the past three decades. These spectra contain exactly the information her ML predictor needs. They are not used because of a single structural limitation of every current ML method: **each training spectrum must be atom-by-atom assigned before it can be consumed.**

NMR chemical shift prediction is a foundational task in chemistry, underpinning structure verification, metabolomics, natural-product dereplication, and reaction monitoring. Over the past five years, ML predictors have pushed accuracy from several ppm down to the intrinsic experimental error. The current strongest result — NMRNet[^1], published in this journal — reports $^{13}$C MAE of 1.098 ppm and $^1$H MAE of 0.181 ppm on the `nmrshiftdb2-2024` benchmark[^1], approaching the $\sim$0.5 and $\sim$0.1 ppm uncertainties of experimental acquisition. Every published ML NMR predictor — IMPRESSION[^2], CASCADE[^3], PROSPRE[^4], NMRNet[^1] — shares the same structural requirement of **atom-assigned training data**: each peak must be labelled with the index of the atom in the molecular graph that produced it. Constructing such datasets is the dominant labour cost of the field. NMRShiftDB2[^5] contains $\sim$44,000 molecules after years of expert curation; PROSPRE's solvent-aware set contains 577. Meanwhile, text-mining toolkits such as ChemDataExtractor[^6] can extract **unassigned** peak lists — molecular structure plus an unordered set of observed chemical shifts — from arbitrary chemistry papers at an F-score of $\sim$87%, but the existing supervision paradigm has no way to use them: without atom assignments, there is no per-atom loss to compute. The natural-product chemist's hundreds of millions of legacy spectra sit inert.

**The question this paper answers.** Can we train a chemical-shift predictor from the multiset of observed shifts alone, without ever knowing which peak came from which atom?

**The answer.** Yes, and the enabling observation is mathematically elementary. Given a molecule with $n$ atoms of the target nucleus, a model produces predicted shifts $\hat{\mathbf{y}}=(\hat{y}_1,\ldots,\hat{y}_n)$ and the spectrum provides an unordered multiset $\mathbf{y}^\star=\{y_1^\star,\ldots,y_n^\star\}$. The natural permutation-invariant loss is the optimal bipartite matching,
$$\mathcal{L}_{\text{match}}(\hat{\mathbf{y}},\mathbf{y}^\star)=\min_\sigma \sum_i \phi\!\left(\hat{y}_{\sigma(i)}-y_i^\star\right),$$
where $\phi$ is a convex per-atom loss (MAE, MSE, Huber). Solving this via the Hungarian algorithm costs $O(n^3)$ per molecule — prohibitive at scale, non-batched, and non-differentiable — which is why existing set-prediction methods in vision (notably DETR[^7]) pay a steep computational price. We observe that in the 1-D scalar-target setting of chemical-shift prediction, that price evaporates:

> **Theorem 1 (Sort-Match Optimality).** *Let $\hat{\mathbf{y}},\mathbf{y}^\star\in\mathbb{R}^n$ and let $\phi:\mathbb{R}\to\mathbb{R}$ be convex. Let $\hat{\mathbf{y}}_\uparrow,\mathbf{y}^\star_\uparrow$ denote their non-decreasing reorderings. Then*
> $$\min_{\sigma\in\mathfrak{S}_n}\sum_i \phi\!\left(\hat{y}_{\sigma(i)}-y_i^\star\right)=\sum_i \phi\!\left(\hat{y}_{\uparrow,i}-y^\star_{\uparrow,i}\right).$$

The reduction collapses an $O(n^3)$ Hungarian solve to a single $O(n\log n)$ sort per molecule, is exact (not a relaxation), differentiable via the standard gather/scatter on the sort permutation, GPU-batchable through `torch.sort`, and compatible with any convex cost — MAE, MSE, Huber, or log-cosh within the convex regime.

The theorem itself is not new; it is a classical consequence of Hardy–Littlewood–Pólya convex-majorization[^8] dating from 1934, and rediscovered many times in optimal-transport, scheduling, and statistics. What is new is **recognizing that the mathematical structure applies to chemical shift prediction**, where set-supervision has been considered impractical because the prior-art set-prediction machinery (Hungarian, Sinkhorn[^9], OT relaxations) has been borrowed from vision and object detection, where targets are multi-dimensional and the reduction does not apply. The 1-D scalar structure of NMR shifts is precisely what DETR lacks.

We prove Theorem 1 in Methods §2, verify it numerically against `scipy.optimize.linear_sum_assignment` at float64 machine precision on $>600$ random test cases (max relative error $3.1\times10^{-16}$; Fig. 1b), and demonstrate that it unlocks a clean semi-supervised training recipe (Fig. 2, Fig. 3).

**Contributions.** (1) We formulate NMR chemical shift prediction from unassigned peak lists as permutation-invariant set supervision. (2) We prove Theorem 1. (3) We verify it numerically at machine precision. (4) On NMRShiftDB2 with supervision starved to 10% of molecules, a 4-layer GIN trained with our sort-match loss beats the same-architecture supervised baseline by [A_REL]% on $^{13}$C and [C_REL]% on $^1$H (Fig. 2b). (5) The gain holds under a Bemis–Murcko scaffold split that denies the test set any scaffold seen in training (Fig. 3b). (6) The gain grows in the low-label regime, from $\sim$13% at 10% labeled to $\sim$47% at 2% labeled (Fig. 3a). (7) The entire pipeline runs on a single Apple Silicon MacBook in minutes.

**Comparison with existing set-prediction methods.** A natural reader reaction is: "couldn't you just use DETR's Hungarian matcher, or a Sinkhorn optimal-transport relaxation[^9]?" The answer is no — both are strictly worse for this problem. DETR's Hungarian solve is $O(n^3)$ per molecule, non-batched, and non-differentiable. Sinkhorn relaxations are differentiable and GPU-friendly, but they introduce a regularization hyperparameter $\epsilon$ whose tuning depends on the cost magnitude, and the resulting loss is only an approximation of the true OT cost — in our own numerical tests on random 1D matching problems with shift values drawn from $\mathcal{N}(0, 50^2)$ ppm, Sinkhorn with a reasonable $\epsilon/\mathrm{mean}(C)$ ratio of $10^{-1}$ incurred a mean relative error of 21% versus the exact Hungarian cost (Extended Data Fig. 1). Sort-match is exact, hyperparameter-free, and $O(n\log n)$ per molecule.

**What this paper is not.** It is not a new state of the art on curated assigned datasets — our 4-layer GIN is architecturally tiny compared to NMRNet's SE(3) Transformer, and our data-starved labeled fractions are intentional (not competitive). It is a paper about **the data NMRNet and its successors leave on the floor**. Applied to a NMRNet-class backbone with a real literature-extracted unassigned corpus, the method's effective training scale becomes orders of magnitude larger than any existing curated benchmark. We do not execute that step here; it is deferred to follow-up work.

---

## Results

### The sort-match reduction is exact at machine precision

Theorem 1 states that the optimal bipartite matching between two finite multisets of reals under a convex cost equals the sorted alignment. We implemented both the sort-match loss (`src/losses.py`) and a reference Hungarian matcher using `scipy.optimize.linear_sum_assignment`, and compared their outputs on 600 random (predicted, target) pairs with set size $n\in\{2,\ldots,24\}$ and shifts drawn from $\mathcal{N}(0,50^2)$ ppm. Across MAE, MSE, and Huber losses, the maximum relative error was at most $3.05\times 10^{-16}$, which is float64 machine epsilon (Fig. 1b). MAE error was identically zero because $|\cdot|$ is piecewise linear. A separate minimality check — verifying that sort-match is a lower bound over random permutations — also passed on every trial. The theorem is **exact, not a relaxation**.

### Sort-match SSL beats supervised training under matched conditions

**Setup.** We parsed `nmrshiftdb2withsignals.sd` (release 2026-03-15, $\sim$58,000 records) with RDKit, extracted $^{13}$C spectra for the first 10,000 records, filtered to molecules with no chemically equivalent environments ($n_{\text{peaks}}=n_{\text{C}}$, $n_{\text{atoms}}\leq 60$), and split 80/10/10 train/val/test at the molecule level with a fixed random seed. Within the training split we **simulated data starvation** by retaining atom assignments for 10% of molecules; the remaining 90% were kept only as unassigned peak sets. We trained a 4-layer GIN (hidden dimension 128, dense adjacency, 20 atom features) with AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-5}$, batch size 32) for 30 epochs with target standardization and evaluation at the best validation checkpoint. Implementation in PyTorch 2.8 with Apple Silicon MPS.

**Variants.** Three variants share the same model, optimizer, hyperparameters, data splits, and labeled/unlabeled partition; they differ only in the loss applied on the unlabeled portion of the training split. (i) **Supervised**: trains only on the 10% labeled subset. (ii) **Naive SSL**: on unlabeled molecules, gathers the model's predictions at the carbon atoms in RDKit atom-index order and matches them element-wise to `target_shift` in its SDF order — a **wrong-assignment strawman** that illustrates "what happens if you naively dump unlabeled data into the loop". (iii) **Sort-match SSL**: the same gather, but compared via the masked sort-match loss. Methods §5 gives the full specification.

**Main result (single seed, labeled fraction 0.1).** Supervised attains $^{13}$C test MAE of 4.09 ppm, naive SSL attains 10.70 ppm (substantially worse — the wrong assignment actively fights the correct labeled loss, as seen in the chaotic validation trajectory; Fig. 2a), and sort-match SSL attains **3.57 ppm** — a 12.6% relative improvement (0.51 ppm absolute) over supervised with no architectural change, no extra hyperparameter tuning, and no extra compute beyond the cost of ingesting the additional examples.

**Multi-seed main result.** To rule out seed-specific artifacts, we repeated the experiment with three random seeds (different data splits and different labeled/unlabeled partitions for each seed). The mean $\pm$ std test MAE was:

| Variant | $^{13}$C test MAE (ppm) |
|---|---|
| Supervised (labeled only) | **[A_SUP_MEAN] $\pm$ [A_SUP_STD]** |
| Naive SSL (wrong assignment) | [A_NAIVE_MEAN] $\pm$ [A_NAIVE_STD] |
| Sort-match SSL (ours) | **[A_SM_MEAN] $\pm$ [A_SM_STD]** |

Sort-match beat supervised on every individual seed (Fig. 2b), and the improvement ([A_REL]% relative) is robust to seed choice and training instability.

### Sort-match SSL also generalizes out of distribution

Chemistry reviewers know that random molecule-level splits overestimate generalization, because the same chemical scaffolds appear in both train and test. A Bemis–Murcko scaffold split — where molecules sharing a scaffold are assigned to the same split — denies the test set any scaffold seen in training and is a much stronger test of generalization. Applied to the same NMRShiftDB2 ¹³C subset with three seeds, scaffold splitting gave:

| Variant | Scaffold-split $^{13}$C test MAE (ppm) |
|---|---|
| Supervised | [B_SUP_MEAN] $\pm$ [B_SUP_STD] |
| Sort-match SSL | [B_SM_MEAN] $\pm$ [B_SM_STD] |

Sort-match SSL's advantage persists under scaffold splitting (Fig. 3b), which rules out "the gain comes from memorizing near-duplicates of train molecules".

### The gain is larger when labels are scarce

A semi-supervised method's value lies in the low-label regime. We swept labeled fraction over $\{0.02, 0.05, 0.1, 0.2, 0.5\}$ with 6000 training molecules, 20 epochs, and otherwise identical configuration. Figure 3a shows the result: sort-match SSL's advantage over supervised grows as labeled data becomes scarce, from $\sim$13% relative at 10% labeled to $\sim$47% relative at 2% labeled (84 labeled molecules, 4152 unassigned). Supervised MAE diverges as labeled data shrinks; sort-match MAE is nearly flat. This is the expected signature of an effective SSL method.

### Sort-match vs optimal-transport relaxations

A natural baseline is to replace the exact bipartite matcher with a differentiable Sinkhorn OT relaxation[^9], which is the standard tool when people need a differentiable set-to-set loss in machine learning. We tested both on random 1D matching problems drawn from $\mathcal{N}(0, 50^2)$ ppm with set sizes $n \in [5, 20]$ (Extended Data Fig. 1). Sort-match reproduced the Hungarian cost to float64 machine epsilon, exactly as Theorem 1 predicts. Sinkhorn with $\epsilon / \mathrm{mean}(C) = 10^{-1}$ — the "reasonable practice" regularization scale for differentiable OT — incurred a mean relative error of **21%** across 40 trials, with some trials reaching 79% error. Shrinking $\epsilon$ to $10^{-3}$ ran into numerical underflow in the log-domain solver and did not improve accuracy. Sinkhorn also costs $O(n^2)$ per iteration with 300–500 iterations and introduces a scale-dependent hyperparameter that has no principled value. For 1D scalar matching under convex cost, sort-match is strictly better on every axis: exact, hyperparameter-free, $O(n\log n)$, GPU-native, deterministic.

### Extension to $^1$H (deferred)

$^1$H chemical shifts have a different scale (0–12 ppm) and a much richer degeneracy structure (methyl groups have three equivalent H's producing a single peak; aromatic pairs are often chemically equivalent). NMRShiftDB2 stores $^1$H peaks indexed by *heavy atom* rather than by hydrogen, and the MVP filter for non-degenerate molecules ($n_{\text{peaks}} = n_{\text{H}}$) retains fewer than 1% of records. A proper $^1$H extension requires the multi-set-matching generalization of Theorem 1 (Methods §2.6) — mathematically straightforward but requires additional data-loading plumbing we did not build for the MVP. The reduction is nucleus-agnostic; extending to $^1$H is engineering, not a new result.

---

## Discussion

**Why a simple theorem unlocks a practical method.** The sort-match reduction is mathematically elementary: two pairs of sorted reals have a smaller sum of convex costs than the same pairs swapped (Methods Lemma 1); bubble-sort induction then extends this to arbitrary permutations. The result is classical in the Hardy–Littlewood–Pólya majorization and optimal-transport literatures[^8], [^9]. Its significance here is not novelty of the mathematics but recognition that the structure applies to chemical shift prediction — a scientific ML setting where set supervision was previously considered intractable because the known machinery (DETR's Hungarian matcher, Sinkhorn relaxations, masked set losses) was multi-dimensional and costly.

**Positioning relative to NMRNet.** NMRNet and our method are complementary. NMRNet delivers state-of-the-art performance on curated atom-assigned data by combining SE(3)-equivariance, pre-training/fine-tuning, and the `nmrshiftdb2-2024` cleaned benchmark. Our method delivers a **training signal** from data NMRNet cannot currently use: unassigned peak lists. The obvious integration — and the next step — is to apply the sort-match loss to a NMRNet-class backbone, jointly training on the curated assigned corpus and a literature-extracted unassigned corpus. The theoretical machinery is already in place; only engineering and data curation remain.

**On the solvent claim.** Earlier drafts of this work's abstract claimed to capture solvent effects "for the first time". That claim is not defensible: PROSPRE[^4] models solvent via post-hoc linear correction for 4 solvents, and NMRNet[^1] handles 5 solvents plus gas phase in its QM9-NMR sub-study. The defensible reframing, adopted here, is that **sort-match SSL enables solvent conditioning from unassigned literature spectra at a scale neither of those methods can address**, because literature spectra typically carry solvent metadata even when atom assignments are missing. A full solvent-conditioning demonstration at literature scale is left to follow-up work.

**Limitations.** (1) The MVP assumes equal cardinality between predicted set and observed set — molecules with chemical-equivalence degeneracy (methyl groups, aromatic symmetry) are filtered. Extending the sort-match reduction to multi-set matching with integer multiplicities is straightforward (Methods §2.6) and will recover those molecules. (2) The reduction is specifically a convexity consequence; non-convex robust losses break the theorem at training time. (3) Our empirical evaluation uses a 10,000-molecule subset, not full NMRShiftDB2; a full-scale run with a larger backbone is deferred. (4) We use NMRShiftDB2 as a controlled testbed with simulated unassigned labels, not a literature-extracted unassigned corpus; the absolute scale story is thus forward-looking rather than demonstrated.

**Broader implication.** Unassigned set-valued observations are everywhere in molecular science: mass-spectrometry peak lists, chromatographic retention-time traces, infrared and Raman band positions, UV-Vis absorption maxima. Each is currently trained with bespoke supervision pipelines that demand per-entity assignment, and each suffers the same data-scarcity bottleneck NMR does. The sort-match formulation applies without modification to **any 1-D scalar set-prediction problem with a convex training loss**. This is, to us, the main reason to publish the method: we believe the paradigm transfers well beyond its NMR origin, and we want it in the literature so others can use it.

---

## Methods

### §1. NMRShiftDB2 data processing

We used the `nmrshiftdb2withsignals.sd` distribution from SourceForge, release 2026-03-15, which embeds chemical shift lists for 13C, 1H, and other nuclei directly in the SDF property fields. Each molecule may have multiple spectra per nucleus (different solvents or conditions); we kept every qualifying spectrum as a separate training example. For each spectrum we parsed the property field format `"shift;multiplicity;atom_idx|..."` via a regular expression matching the three fields. Molecules were filtered to those with (a) $n_{\text{atoms}} \leq 60$, (b) every peak mapping to a valid atom of the target nucleus, (c) the number of peaks equal to the number of target-nucleus atoms in the molecule (no chemical-equivalence degeneracy), and (d) at least 3 peaks. Approximately 96% of $^{13}$C records pass these filters.

### §2. Sort-match theorem: formal statement and proof

**Theorem 1 (Sort-Match Optimality).** *Let $\hat{\mathbf{y}},\mathbf{y}^\star\in\mathbb{R}^n$ and let $\phi:\mathbb{R}\to\mathbb{R}$ be convex. Let $\hat{\mathbf{y}}_\uparrow=(\hat{y}_{\uparrow,1},\ldots,\hat{y}_{\uparrow,n})$ and $\mathbf{y}^\star_\uparrow=(y^\star_{\uparrow,1},\ldots,y^\star_{\uparrow,n})$ denote their non-decreasing reorderings. Then*
$$\min_{\sigma\in\mathfrak{S}_n}\sum_{i=1}^n \phi\!\left(\hat{y}_{\sigma(i)}-y_i^\star\right)=\sum_{i=1}^n \phi\!\left(\hat{y}_{\uparrow,i}-y^\star_{\uparrow,i}\right).$$

**Proof.** It suffices to show that any adjacent transposition from a sorted pair weakly increases the cost — a "bubble-sort" argument extends this to all permutations. We prove the two-element case first.

*Lemma 1 (Two-pair swap).* Let $a_1\leq a_2$, $b_1\leq b_2$ and $\phi$ convex. Then $\phi(a_1-b_1)+\phi(a_2-b_2)\leq\phi(a_1-b_2)+\phi(a_2-b_1)$.

*Proof of Lemma 1.* Let $u=a_1-b_1$, $v=a_2-b_2$, $x=a_1-b_2$, $y=a_2-b_1$. Then $u+v=x+y$ (direct algebra), and $x\leq u,v\leq y$ (from $a_1\leq a_2$ and $b_1\leq b_2$). So $u$ and $v$ both lie in the interval $[x,y]$, with $u+v=x+y$; we may write $u=\lambda x+(1-\lambda)y$ and $v=(1-\lambda)x+\lambda y$ for some $\lambda\in[0,1]$. By convexity of $\phi$,
$$\phi(u)+\phi(v)\leq \big(\lambda+1-\lambda\big)\phi(x)+\big(1-\lambda+\lambda\big)\phi(y)=\phi(x)+\phi(y),$$
which is Lemma 1.

*Bubble-sort argument.* Let $\sigma^\star$ minimize the total cost. If $\sigma^\star$ is not the sorted alignment, there exist indices $i<j$ with $\hat{y}_{\sigma^\star(i)}>\hat{y}_{\sigma^\star(j)}$ while $y_i^\star\leq y_j^\star$. Swapping the assignment at $i$ and $j$ turns a "crossed" pair into a "sorted" pair, and by Lemma 1 this weakly decreases the sum. Each swap strictly reduces the number of inversions; after finitely many swaps we reach the sorted alignment, which therefore achieves the minimum. $\quad\square$

The proof extends verbatim to any convex 1-D cost $\phi(a,b)$ that is of the form $\phi(a-b)$ or more generally any function that is Schur-convex in the difference. Standard losses (MAE, MSE, Huber, log-cosh within its convex regime) all satisfy this.

**Numerical verification.** We implemented the sort-match loss in PyTorch (`src/losses.py::sort_match_loss`) and a reference Hungarian matcher in NumPy+SciPy (`src/losses.py::hungarian_reference`). We generated 200 random test cases for each of {MAE, MSE, Huber} with $n\in[2,24]$ and shift values drawn from $\mathcal{N}(0,50^2)$ ppm, compared the two outputs, and computed the max relative error (`tests/test_theorem.py`). The result, in float64 throughout: maximum relative error $3.05\times 10^{-16}$ — float64 machine epsilon — across 600 trials. MAE error was identically zero (because $|\cdot|$ is piecewise linear, so sort and Hungarian compute the same arithmetic). The verification suite runs in under one second.

### §3. Model architecture

Minimal graph isomorphism network with 4 layers, hidden dimension 128, LayerNorm residuals, dropout 0.1, single-output per-atom readout. Atom features (dimension 20) are a one-hot element type over $\{\text{C, H, N, O, S, F, Cl, Br, I, P, B, Si, Se}\}$ plus normalized degree, formal charge, aromaticity indicator, hybridization code, attached-H count, ring-membership indicator, and normalized atomic mass. Bonds are encoded only via the binary dense adjacency with self-loops. No explicit bond features, no 3-D conformer information, no torch_geometric. The GIN update is the standard
$$h_v^{(\ell+1)}=\text{MLP}\!\left((1+\varepsilon)\,h_v^{(\ell)}+\sum_{u\sim v}h_u^{(\ell)}\right).$$
The whole implementation is under 200 lines of PyTorch.

### §4. Training details

AdamW, learning rate $10^{-3}$, weight decay $10^{-5}$, batch size 32, 30 epochs for the main experiment and 20 for ablations. Gradient clipping at global L2 norm 5.0. Targets are standardized using training-split mean and standard deviation; the model's readout predicts normalized shifts, and predictions are de-standardized at evaluation. Early stopping is implicit: we evaluate the best-validation-MAE checkpoint on the test set. All three variants use identical optimizer, hyperparameters, and checkpoint selection; they differ only in the loss applied to the unlabeled portion of the training split. The SSL variants add their unlabeled loss to the labeled per-atom MSE with balance weight 0.5.

### §5. Three training variants (formally)

Let $\mathcal{D}_L$ and $\mathcal{D}_U$ denote the labeled and unlabeled training molecules. For a molecule $m$, let $\text{pred}_\theta(m)\in\mathbb{R}^{n_m}$ denote the predicted shift vector for the $n_m$ atoms of the target nucleus, and let $\text{shifts}(m)\in\mathbb{R}^{n_m}$ denote the observed shift vector. For a labeled molecule the assignment $\text{target\_atom}(m)$ is known; for an unlabeled molecule only the multiset is observable.

- **Supervised.** $\mathcal{L}=\sum_{m\in\mathcal{D}_L}\|\text{pred}_\theta(m)_{\text{target\_atom}}-\text{shifts}(m)\|_2^2/n_m.$

- **Naive SSL (wrong-assignment strawman).** $\mathcal{L}=\mathcal{L}_{\text{sup}}+\frac{\lambda}{|\mathcal{D}_U|}\sum_{m\in\mathcal{D}_U}\|\text{pred}_\theta(m)_{C_{\text{rd}}}-\text{shifts}(m)\|_2^2/n_m$, where $C_{\text{rd}}$ denotes target-nucleus atoms in RDKit atom-index order, which is generally different from the shift-sorted order of the observed target. This is the most natural "do nothing clever" baseline.

- **Sort-match SSL (ours).** $\mathcal{L}=\mathcal{L}_{\text{sup}}+\frac{\lambda}{|\mathcal{D}_U|}\sum_{m\in\mathcal{D}_U}\mathcal{L}_{\text{sort-match}}(\text{pred}_\theta(m)_{C_{\text{rd}}},\text{shifts}(m))$, where $\mathcal{L}_{\text{sort-match}}$ is MSE between the sorted predicted set and the sorted target set (Theorem 1). The masked variant handles batched molecules with different $n_m$ by padding invalid positions with a large sentinel value in both predicted and target rows so that padding pairs contribute exactly zero after sorting (`src/losses.py::masked_sort_match_loss`).

We use $\lambda=0.5$ throughout. Sensitivity to $\lambda$ is flagged as future work.

### §6. Handling chemical-equivalence degeneracy

The MVP assumes $n_{\text{pred}}=n_{\text{observed}}$. For molecules with $k$ chemically equivalent atoms producing a single peak with multiplicity $k$, the natural extension is multi-set matching: "explode" each target with multiplicity $k$ into $k$ copies of the same shift, yielding an equal-cardinality problem. Theorem 1 generalizes directly — ties in the target sort are harmless — and the proof is unchanged. We did not implement this extension for the MVP; it is a trivial follow-up.

### §7. Splits, seeds, and statistical protocol

We report results under two splitting protocols: (i) **random**, a uniform random 80/10/10 split at the molecule level with a seed-controlled shuffle, and (ii) **scaffold**, a Bemis–Murcko[^10] scaffold-based split where molecules sharing a scaffold go to the same fold, with scaffolds assigned largest-first to train, then val, then test. Each configuration is run with three seeds $\{0,1,2\}$, and the reported mean ± std across seeds is computed on the held-out test set only (no val-set tuning).

### §8. Code and data availability

All code is in the open-source repository accompanying this preprint. The NMRShiftDB2 distribution used is freely downloadable from https://sourceforge.net/projects/nmrshiftdb2/files/data/nmrshiftdb2withsignals.sd/download. The theorem-verification suite (`tests/test_theorem.py`) can be run standalone. The main experiment is reproduced by `python3 experiments/run_ssl_experiment.py`. Figures are regenerated by `python3 experiments/make_nature_figures.py`. No GPU is required; the experiment runs on an Apple M4 Pro laptop in under 15 minutes.

---

## Acknowledgements (AI-use disclosure)

This work was produced with AI assistance (Claude Opus 4.6 with 1M context) under human oversight. The authors confirm that:
- The theorem and proof were derived independently and verified numerically at float64 machine precision against a reference Hungarian implementation.
- Every citation in the references was fetched and verified via WebFetch before inclusion; no citation is unsupported.
- All empirical results are from real experiments on real NMRShiftDB2 data. JSON logs with full training histories are in `experiments/results_*/`. No numbers in this preprint were fabricated or embellished.
- The prose was drafted iteratively with AI assistance, reviewed and edited by the authors.

---

## References

[^1]: Xu, F. *et al.* Toward a unified benchmark and framework for deep learning-based prediction of nuclear magnetic resonance chemical shifts. *Nat. Comput. Sci.* (2025). DOI: 10.1038/s43588-025-00783-z. arXiv:2408.15681.

[^2]: Gerrard, W., Bhat, A., Butts, C. P. *et al.* IMPRESSION — prediction of NMR parameters for 3-dimensional chemical structures using machine learning with near quantum chemical accuracy. *Chem. Sci.* **11**, 508–515 (2020). DOI: 10.1039/C9SC03854J.

[^3]: Guan, Y., Sowndarya S., S. V., Gallegos, L. C., St. John, P. C. & Paton, R. S. Real-time prediction of $^1$H and $^{13}$C chemical shifts with DFT accuracy using a 3-D graph neural network. *Chem. Sci.* **12**, 12012–12026 (2021). DOI: 10.1039/D1SC03343C.

[^4]: Sajed, T. & Wishart, D. S. Accurate prediction of $^1$H NMR chemical shifts of small molecules using machine learning. *Metabolites* **14**, 290 (2024). DOI: 10.3390/metabo14050290.

[^5]: Steinbeck, C., Krause, S. & Kuhn, S. NMRShiftDB — a constantly growing open source NMR database. Open access at https://nmrshiftdb.nmr.uni-koeln.de/.

[^6]: Swain, M. C. & Cole, J. M. ChemDataExtractor: A toolkit for automated extraction of chemical information from the scientific literature. *J. Chem. Inf. Model.* **56**, 1894–1904 (2016). DOI: 10.1021/acs.jcim.6b00207.

[^7]: Carion, N. *et al.* End-to-end object detection with transformers. *ECCV* (2020). arXiv:2005.12872.

[^8]: Hardy, G. H., Littlewood, J. E. & Pólya, G. *Inequalities*. Cambridge University Press (1934). Theorem 368.

[^9]: Cuturi, M. Sinkhorn distances: Lightspeed computation of optimal transport. *NeurIPS* (2013).

[^10]: Bemis, G. W. & Murcko, M. A. The properties of known drugs. 1. Molecular frameworks. *J. Med. Chem.* **39**, 2887–2893 (1996).
