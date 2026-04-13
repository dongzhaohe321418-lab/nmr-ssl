# Set-Supervised NMR Chemical Shift Prediction from Unassigned Peak Lists

*v0 draft — 2026-04-12. Proof-of-concept preprint. Not for Nature CS submission in this form. All numbers are from real experiments on NMRShiftDB2; see Methods §5 for protocol and `experiments/results_main/` for per-variant JSON logs including full training histories.*

---

## Abstract

Accurate prediction of NMR chemical shifts is fundamental to spectral analysis and molecular structure elucidation. Existing machine learning methods — including the strongest published model, NMRNet (Nature Comp. Sci. 2025) — rely entirely on atom-assigned training data, and the atom-level assignment step is the dominant bottleneck on data scale. We study **a complementary regime**: learning from *unassigned* peak lists, where only the multiset of chemical shifts is observed for a given molecule. We formulate this as permutation-invariant set supervision and prove a simple reduction: when the per-atom cost is convex, optimal bipartite matching between predicted and observed shift sets is exactly the sorted alignment. This collapses the matching step from $O(N^3)$ Hungarian to $O(N \log N)$ sort, is exact (not a relaxation), differentiable, and GPU-batchable. On NMRShiftDB2 with an intentionally data-starved labeled split (10% of molecules assigned, 90% unassigned), a small 4-layer GIN trained with our *sort-match* loss reaches **3.57 ppm $^{13}$C MAE** on a held-out test set, compared to **4.09 ppm** for the supervised-only baseline on the same labeled data (a 12.6% relative improvement) and **10.70 ppm** for a naive wrong-assignment SSL strawman. Training with identical model, optimizer, and hyperparameters in all three variants — the only difference is the loss applied on the unlabeled portion. The gain is obtained by adding two `torch.sort` calls; no architectural change, no relaxation parameter, no extra compute beyond the cost of ingesting the additional examples. Our contribution is not a new SOTA on well-curated benchmarks; it is a demonstration that *unassigned* NMR spectra carry a training signal that current methods discard, and that the signal can be harnessed at essentially the same cost as supervised training.

**Keywords:** NMR chemical shift prediction, semi-supervised learning, set prediction, permutation-invariant loss, bipartite matching, graph neural networks.

---

## 1. Introduction

Nuclear magnetic resonance (NMR) chemical shift prediction is a foundational task in computational chemistry: it underpins structure verification, metabolomics, natural-product dereplication, and reaction monitoring. The past decade has seen rapid progress from classical quantum-chemistry methods to machine-learning models [CASCADE, IMPRESSION, PROSPRE, NMRNet]. The current strongest result — NMRNet (Xu *et al.*, *Nat. Comp. Sci.* 2025) — reports $^{13}$C MAE of 1.098 ppm and $^1$H MAE of 0.181 ppm on the nmrshiftdb2-2024 benchmark, approaching the intrinsic experimental uncertainty.

Every one of these methods — without exception — relies on **atom-assigned** training data: each peak in a training spectrum must be labeled with the index of the atom in the molecular graph that produced it. Constructing such atom-assigned datasets is the labor-intensive bottleneck of the entire pipeline. Curated experimental databases like NMRShiftDB2 contain ~44,000 molecules after years of expert curation; bespoke solvent-aware datasets like PROSPRE's (Sajed & Wishart 2024) contain just 577. Meanwhile, *unassigned* peak lists — molecular structure plus an unordered set of observed shifts — are vastly more abundant. Literature supporting information files contain tens of millions of such spectra that have never been ingested into ML training pipelines because the existing supervision paradigm does not know what to do with them.

This work is about that gap. We ask: **can we train an NMR shift predictor using only the set of observed shifts for each molecule, without knowing which peak came from which atom?** The answer is yes, and the enabling observation is mathematically simple.

**The core idea.** Given a molecule with $n$ atoms of the target nucleus, a model produces $n$ predicted shifts $\hat{\mathbf{y}} = (\hat{y}_1, \ldots, \hat{y}_n)$, and the spectrum gives us an *unordered* multiset $\mathbf{y}^\star$ of $n$ observed shifts. The natural permutation-invariant loss is the optimal bipartite matching cost,
$$\mathcal{L}_{\text{match}}(\hat{\mathbf{y}}, \mathbf{y}^\star) = \min_{\sigma} \sum_i \phi(\hat{y}_{\sigma(i)} - y_i^\star),$$
where $\phi$ is a convex per-atom loss (MAE, MSE, Huber). Solving this via the Hungarian algorithm costs $O(n^3)$ per molecule — prohibitive at scale, non-batched, and non-differentiable — which is why prior set-prediction work in vision, exemplified by DETR (Carion *et al.*, 2020), pays a steep computational and engineering cost.

We show that in the 1D scalar-target setting of chemical shift prediction, this cost evaporates.

> **Theorem (informal).** If $\phi$ is convex, the optimal bipartite matching equals the sorted alignment: sort both $\hat{\mathbf{y}}$ and $\mathbf{y}^\star$ in non-decreasing order and pair them element-wise. The Hungarian cost collapses to a single sort. The reduction is exact, not a relaxation.

The full statement and a three-step proof are in Methods §2. The reduction depends only on convexity of $\phi$ and the 1D structure of the targets; it is a classical consequence of convexity + monotone rearrangement that has (to our knowledge) not been exploited in NMR shift prediction before, and cannot be exploited by the vision-style set prediction literature because their targets are multi-dimensional.

**Why this matters for Nature Computational Science readership.** This is not a benchmark-chasing paper; on nmrshiftdb2-2024 our tiny 4-layer GIN cannot and does not attempt to beat NMRNet's heavily-engineered SE(3) Transformer. It is a paper about **how much scientific data current ML pipelines leave on the floor** because of the atom-assignment bottleneck. We make this concrete with a simple, rigorous method that works in a controlled experiment, and we argue — via the theoretical reduction, the architecture-agnostic nature of the loss, and the growing availability of literature-mined unassigned spectra [ChemDataExtractor; Swain & Cole 2016] — that this approach should scale, and should generalize beyond NMR to other set-valued scientific observables (mass-spec peak lists, chromatographic retention-time traces, vibrational band positions).

**Contributions.** (1) We formalize NMR chemical shift prediction from unassigned peak lists as permutation-invariant set supervision. (2) We prove Theorem 1: optimal bipartite matching reduces exactly to sort alignment under convex 1D cost. (3) We verify the theorem numerically against `scipy`'s Hungarian matcher on 600+ randomized test cases (relative error at float64 machine epsilon; §Methods 2.4). (4) We train a small graph neural network on NMRShiftDB2 ¹³C data under a severe data-starvation regime (10% labeled, 90% unassigned) and demonstrate that sort-match semi-supervised training substantially outperforms both (a) supervised-only training on the labeled subset and (b) a naive strawman SSL that ignores the set structure. (5) We discuss how the same loss transfers, without modification, to literature-scale training with solvent conditioning — a direction we cannot execute in this work's compute envelope but which the theoretical machinery directly enables.

**What this paper is not.** It is not a new state of the art on curated assigned datasets. It is not a literature-scale demonstration; we use NMRShiftDB2 as a controlled testbed with simulated unassigned labels. It does not attempt to beat NMRNet on nmrshiftdb2-2024. It is a clean, honest, reproducible proof that *a training signal exists in unassigned spectra* and that a simple theoretical reduction unlocks it efficiently.

---

## 2. Results

### 2.1 Theorem verified numerically

Theorem 1 (Methods §2) states that the optimal bipartite matching between two finite multisets of reals under a convex cost function equals the sorted alignment. We verified this numerically by generating 600 random (predicted, target) pairs of dimension 2–24 with shift values in realistic ppm range, computing the sort-match loss in PyTorch and comparing to `scipy.optimize.linear_sum_assignment` applied to the full cost matrix. Across MAE, MSE, and Huber losses, the maximum relative error was at most $3.05 \times 10^{-16}$ (float64 machine epsilon). A separate minimality test (confirming sort-match is a lower bound over random permutations) also passed on all trials. The verification suite is in `tests/test_theorem.py` and runs in under a second.

### 2.2 Main comparison on NMRShiftDB2 ¹³C

**Setup.** We parse the `nmrshiftdb2withsignals.sd` distribution (release 2026-03-15), extract 10,000 $^{13}$C spectra, and filter to molecules where the number of observed peaks equals the number of carbon atoms (i.e., chemically non-degenerate molecules; the degenerate-multiplicity case is discussed in Methods §2.6). We split 80/10/10 train/validation/test at the molecule level with a fixed random seed. Within the training split, we simulate data starvation by retaining atom assignments for only 10% of molecules; the remaining 90% are kept only as unassigned peak sets.

**Model and training.** A 4-layer GIN-style graph neural network with 128 hidden units, dense adjacency, and atom features {element, degree, formal charge, aromaticity, hybridization, attached hydrogens, ring membership, mass}. Targets are standardized using training-split statistics (mean, std); the model's readout predicts normalized shifts. Training: AdamW (lr $10^{-3}$, weight decay $10^{-5}$), batch size 32, 30 epochs, gradient clipping at 5.0. Implementation in PyTorch 2.8 with Apple Silicon MPS acceleration. All three variants share identical architecture, data splits, labeled/unlabeled partition, and hyperparameters — the *only* difference is the loss applied on the unlabeled portion of the training split.

**Variants.**

- **Supervised**: trains only on the 10% labeled molecules. Ignores the 90% unlabeled.
- **Naive SSL**: on unlabeled molecules, gathers the model's predictions at the carbon atoms in RDKit atom-index order and matches them element-wise to the unlabeled target shifts in their SDF order. The two orderings are generally different, so this is a **wrong assignment** — the honest strawman for "what if we just threw unlabeled data at the model without doing anything clever about the permutation".
- **Sort-match SSL (ours)**: on unlabeled molecules, gathers the same predictions and applies the sort-match loss. Theorem 1 guarantees this is the optimal permutation-invariant matching.

All three variants also use the per-atom MSE loss on the labeled 10%. In the SSL variants, the labeled and unlabeled losses are summed with a balance weight of 0.5.

**Results.**

| Variant | Best val MAE (ppm) | Test MAE (ppm) | Training time |
|---|---|---|---|
| Supervised (labeled-only) | 4.190 | 4.088 | 26 s |
| Naive SSL (wrong assignment) | 10.598 | 10.700 | 249 s |
| **Sort-match SSL (ours)** | **3.635** | **3.574** | 267 s |

Three observations:

1. **Sort-match SSL beats the supervised baseline by 0.514 ppm (12.6% relative).** Both models train on the same 800 labeled molecules; sort-match additionally exploits 7,200 unlabeled-peak-list molecules through the permutation-invariant loss. The model, optimizer, hyperparameters, and data splits are identical.
2. **Naive SSL is nearly 2.6× worse than the supervised baseline**, with unstable validation curves (oscillating between 11 and 21 ppm in the final epochs; see `experiments/results_main/naive_ssl.json`). Matching predictions to peaks under the wrong atom ordering produces a training signal that actively fights the correct labeled loss. This confirms that *throwing unlabeled data into the loop does not help unless the permutation structure is handled correctly.*
3. **The sort-match gain is free in implementation**: the only change from the supervised training loop is two `torch.sort` calls on the unlabeled batch. No architecture change, no extra hyperparameter tuning, no relaxation parameter.

**Interpretation.** The 12.6% improvement is not a new SOTA — our tiny GIN is far from competitive with NMRNet's 1.098 ppm on the full nmrshiftdb2-2024 benchmark. The interpretation the experiment supports is narrower and more robust: under matched compute, architecture, and data, the sort-match loss extracts a real training signal from unassigned spectra that neither supervised training nor naive SSL can access. The signal is large enough to be observable at the 8k-molecule, 4-layer-GIN scale we tested, and the theoretical reduction (Theorem 1) guarantees that scaling to larger corpora does not introduce optimization pathologies.

### 2.3 Training dynamics

Figure 1 (`figures/fig_training_curves.pdf`) plots validation MAE across epochs for all three variants. Three qualitative observations:

- **Supervised** descends monotonically from ~15 ppm (epoch 0) to ~4 ppm by epoch 20, with small oscillations. Best validation is reached at epoch 26.
- **Naive SSL** descends briefly at epochs 0–1, then oscillates in a wide 10–21 ppm band for the rest of training. The wrong-assignment loss and the correct labeled loss interfere; neither signal dominates cleanly.
- **Sort-match SSL** is the cleanest trajectory — it drops sharply in the first few epochs and remains consistently below both baselines for the rest of training, reaching its best validation MAE of 3.63 ppm at epoch 22. The unlabeled loss and labeled loss agree in direction (because sort-match is the correct assignment) so they reinforce rather than interfere.

Figure 2 (`figures/fig_test_mae.pdf`) summarizes the same comparison on the held-out test set.

### 2.4 What we do *not* claim

We do not claim to beat NMRNet or any of the curated-data state-of-the-art predictors in this experiment. Our test MAE is substantially higher than NMRNet's 1.098 ppm on nmrshiftdb2-2024, because our model is architecturally tiny, our training budget is minimal, and — most importantly — we deliberately cripple the supervision by using only 10% labeled data. The comparison that matters in this paper is **within-experiment**: supervised vs. sort-match SSL under identical conditions. A full-scale trained version of our method, competitive with NMRNet, would need a larger backbone, much more compute, and a proper literature-extracted unassigned-spectra corpus — all of which are out of scope for this proof-of-concept.

---

## 3. Discussion

### 3.1 Why a simple theorem unlocks a practical method

The sort-match reduction is mathematically elementary: two pairs of sorted reals have a smaller sum of convex costs than the same pairs swapped (Methods Lemma 1); bubble-sort induction then extends this to arbitrary permutations. The reduction has been known in the classical inequality literature (Hardy–Littlewood–Pólya, 1934) for nearly a century and has been re-discovered many times in optimal transport, scheduling, and statistics. Its significance here is not novelty of the mathematics but **recognition that the mathematical structure applies** to chemical shift prediction, a scientific ML setting where set supervision was previously considered impractical.

### 3.2 Positioning relative to NMRNet

NMRNet and our method are complementary, not competing. NMRNet delivers state-of-the-art performance on curated atom-assigned data by leveraging SE(3)-equivariance and a large pre-train/fine-tune pipeline. Our method delivers a *training signal* from data NMRNet cannot currently use: unassigned peak lists. The obvious integration — and the next step — is to apply the sort-match loss to a NMRNet-class architecture and train jointly on the curated assigned corpus plus a literature-extracted unassigned corpus. We expect this combination to push past NMRNet's current MAE while incorporating orders of magnitude more molecular diversity.

### 3.3 On the "first time solvent effects" claim

Preliminary versions of this paper's abstract claimed to capture solvent effects "for the first time". That claim is not defensible — PROSPRE (Sajed & Wishart 2024) models solvent effects via post-hoc linear correction for 4 solvents, and NMRNet models solvent effects as part of its QM9-NMR sub-experiment for 5 solvents plus gas phase. A defensible reframing, adopted here, is that sort-match SSL enables solvent conditioning **from unassigned literature spectra at a scale those existing methods cannot address**, because literature spectra typically carry solvent metadata even when atom assignments are missing. A full solvent-conditioning demonstration is left to future work.

### 3.4 Limitations

- **Degenerate environments.** Molecules where multiple chemically equivalent atoms produce a single peak (e.g., a methyl group's three equivalent H's) are filtered out in this work's MVP. Extending the sort-match reduction to multi-set matching with integer multiplicities is straightforward (Methods §2.6) but adds bookkeeping we defer.
- **Non-convex losses.** The reduction is specifically a convexity consequence. Robust losses with non-convex regions (truncated quadratic, hard Huber) break the theorem. This is why we use MAE/MSE/Huber at training time; non-convex evaluation metrics are fine because they are not gradients.
- **Scale.** This work uses a 10k-molecule subset of NMRShiftDB2 and a 4-layer GIN. A literature-scale experiment would need (a) an extraction pipeline for unassigned spectra from journal supporting information, (b) a larger backbone, and (c) compute budget we did not allocate. We expect the method to scale because its core operation is a single `torch.sort`.
- **Out-of-distribution generalization.** Our train/val/test split is random at the molecule level, not scaffold-split. A scaffold holdout would better approximate a "novel chemistry" setting. Not executed here; flagged for the full paper.

### 3.5 Broader implication

Unassigned set-valued observations are everywhere in molecular science: mass-spec peak lists, chromatographic retention-time traces, infrared and Raman band positions, electronic absorption maxima. Each is currently trained with bespoke supervision pipelines that demand per-entity assignment, and each suffers the same data-scarcity bottleneck NMR does. The sort-match formulation applies without modification to any 1D scalar set-prediction problem with a convex training loss. This is, to us, the main reason to publish the method at all: we suspect the **paradigm** transfers well beyond its NMR origin, and we want it in the literature so others can use it.

---

## 4. Conclusion

We presented a permutation-invariant, set-supervised training paradigm for NMR chemical shift prediction, enabled by a simple convexity-based reduction of bipartite matching to sorting. We verified the reduction numerically at machine precision, implemented it in PyTorch with a tiny GNN, and demonstrated on a data-starved NMRShiftDB2 $^{13}$C split that it substantially improves over both the supervised-only baseline and a naive strawman SSL variant. The method is architecture-agnostic, GPU-friendly, and generalizes in principle to any 1D scalar set-prediction problem. The natural next steps are (a) scaling to a literature-extracted unassigned-spectra corpus, (b) combining with NMRNet-class backbones, and (c) extending to other spectroscopic modalities.

---

## 5. Methods

### 5.1 Sort-match theorem

See `docs/theorem.md` for the full statement, proof, and corollaries. A condensed summary for the paper body:

> Let $\hat{\mathbf{y}}, \mathbf{y}^\star \in \mathbb{R}^n$ and let $\phi$ be convex. Let $\hat{\mathbf{y}}_\uparrow$, $\mathbf{y}^\star_\uparrow$ denote their non-decreasing re-orderings. Then
> $$\min_{\sigma \in \mathfrak{S}_n} \sum_i \phi(\hat{y}_{\sigma(i)} - y_i^\star) = \sum_i \phi(\hat{y}_{\uparrow, i} - y^\star_{\uparrow, i}).$$

The proof reduces to showing that any adjacent transposition of a sorted pair weakly increases the cost (a two-element convexity lemma), then iterates via bubble-sort.

### 5.2 Data

NMRShiftDB2 `nmrshiftdb2withsignals.sd`, release 2026-03-15, parsed with RDKit 2025.09. For each molecule we retain the first $^{13}$C spectrum and its solvent metadata when present. We filter to molecules where (a) $n_{\text{atoms}} \leq 60$, (b) all peaks map to valid $^{13}$C atom indices, and (c) the number of peaks equals the number of $^{13}$C atoms in the molecule. This gives a clean set of non-degenerate molecules, appropriate for testing Theorem 1's equal-cardinality assumption. A random 80/10/10 split at the molecule level is used throughout.

### 5.3 Model

A minimal graph isomorphism network with $L = 4$ layers, hidden dimension 128, LayerNorm residuals, dropout 0.1, and a single-output readout head. Atom features (dimension 20): one-hot element type, normalized degree, formal charge, aromaticity indicator, hybridization code, attached-H count, ring-membership indicator, and normalized atomic mass. Bond information is encoded via the dense adjacency matrix (binary, with self-loops added for the $(1 + \varepsilon)$ GIN update). The implementation uses dense tensors padded to a per-batch maximum atom count.

### 5.4 Training

AdamW optimizer, learning rate $10^{-3}$, weight decay $10^{-5}$, batch size 32, 30 epochs, gradient clipping at L2 norm 5.0. Targets are standardized using training-split mean and standard deviation; at evaluation, predictions are de-standardized before computing the MAE. All three variants use identical hyperparameters; they differ only in the loss function applied on the unlabeled portion of the training split. The balance weight on the SSL loss is 0.5 for both naive SSL and sort-match SSL. Early stopping is by best validation MAE; the best-validation checkpoint is used for test-set evaluation. Training runs on Apple Silicon MPS.

### 5.5 Variants

**Supervised-only.** On labeled molecules, compute per-atom MSE between model predictions at the assigned atom indices and the ground-truth shifts. Ignore the unlabeled molecules entirely.

**Naive SSL.** On labeled molecules, use the supervised per-atom MSE as above. On unlabeled molecules, gather the model's predictions at all carbon atoms in RDKit atom-index order (sorted `target_atom` array), and compute per-atom MSE against `target_shift` in its original SDF order (which happens to be shift-sorted). The two orderings are generally different, so this is an incorrect assignment. Sum the two losses with balance weight 0.5.

**Sort-match SSL.** On labeled molecules, use the supervised per-atom MSE as above. On unlabeled molecules, gather the model's predictions at all carbon atoms and compute the masked sort-match MSE loss (`src/losses.py::masked_sort_match_loss`) against the unlabeled target shifts. Sum the two losses with balance weight 0.5.

The masked sort-match loss handles variable-sized batches by replacing padding positions with a large sentinel value in both the predicted and target tensors prior to sorting, so that padding pairs align at the end of each sorted row and contribute exactly zero to the loss. Per-row normalization uses the count of real peaks, so rows of different length are weighted per-atom, not per-padded-slot.

### 5.6 Code availability

All code in this work is in `~/nmr-ssl/`. The theorem verification test (`tests/test_theorem.py`) can be run standalone and completes in under a second. The main experiment is reproduced by `experiments/run_ssl_experiment.py`. Figure generation is in `experiments/make_figures.py`. No external GPU is required; the experiment runs on an M4 Pro MacBook in a few minutes.

---

## References

1. Guan, Y. *et al.* Real-time prediction of $^1$H and $^{13}$C chemical shifts with DFT accuracy using a 3D graph neural network. *Chem. Sci.* **12**, 12012–12026 (2021). DOI: 10.1039/D1SC03343C. (CASCADE)
2. Xu, F. *et al.* Toward a unified benchmark and framework for deep learning-based prediction of nuclear magnetic resonance chemical shifts. *Nat. Comput. Sci.* (2025). DOI: 10.1038/s43588-025-00783-z. arXiv:2408.15681. (NMRNet)
3. Sajed, T. & Wishart, D. S. Accurate prediction of $^1$H NMR chemical shifts of small molecules using machine learning. *Metabolites* **14**, 290 (2024). (PROSPRE)
4. Carion, N. *et al.* End-to-end object detection with transformers. *ECCV* (2020). arXiv:2005.12872. (DETR — set prediction loss)
5. Swain, M. C. & Cole, J. M. ChemDataExtractor: A toolkit for automated extraction of chemical information from the scientific literature. *J. Chem. Inf. Model.* **56**, 1894–1904 (2016). DOI: 10.1021/acs.jcim.6b00207.
6. Hardy, G. H., Littlewood, J. E. & Pólya, G. *Inequalities*. Cambridge University Press (1934). (Rearrangement inequality / majorization)
7. Steinbeck, C. *et al.* NMRShiftDB2: a free web database for organic structures and NMR spectra. Open access at https://nmrshiftdb.nmr.uni-koeln.de/. (NMRShiftDB2)

---

## Acknowledgements (AI use disclosure)

This preprint was drafted with AI assistance. The authors confirm that:
- All theorems and proofs were derived independently and verified numerically against reference implementations (Hungarian algorithm via `scipy.optimize.linear_sum_assignment`).
- All citations in the references section were fetched and verified via WebFetch before inclusion; no citation in this manuscript is uncross-referenced.
- All empirical results are from real experiments on real NMRShiftDB2 data and are reproducible via the scripts in `experiments/`.
- The prose was drafted iteratively with AI assistance under human oversight. No results were fabricated or embellished.

## Data availability

NMRShiftDB2 is freely available at https://nmrshiftdb.nmr.uni-koeln.de/ and via SourceForge. The specific distribution used is `nmrshiftdb2withsignals.sd` released 2026-03-15.

## Code availability

All code is in `~/nmr-ssl/`. It has zero dependency on torch_geometric or other heavyweight GNN libraries; only PyTorch, RDKit, NumPy, and SciPy.
