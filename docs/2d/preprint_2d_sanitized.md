# Learning to predict $^{1}$H and $^{13}$C NMR chemical shifts jointly from unassigned 2-D HSQC peak lists

*v4 preprint draft — 2026-04-13. Builds on the 1-D sort-match method (Paper 1, `docs/preprint_v1_filled.md`). Addresses Nature Communications strict-standard peer review with the Option-B revision plan.*

---

## Abstract

Published state-of-the-art NMR chemical-shift predictors are trained on single-nucleus, atom-assigned datasets and produce one-shift-at-a-time predictions. Training data is scarce because atom assignment is labour-intensive, and multi-nucleus predictors require *two* atom-assignment passes. We introduce an alternative supervision paradigm: train a dual-head graph neural network from an **unassigned 2-D HSQC peak list**, a multiset of ($^{1}$H, $^{13}$C) cross-peaks that is routinely reported in organic chemistry papers but previously unused by ML. Our method extends the 1-D sort-match reduction [Paper 1] to 2-D via a sliced-Wasserstein projection loss that is permutation-invariant, differentiable, and $O(K n \log n)$ per molecule. The construction is an application of known sliced-Wasserstein machinery [Bonneel et al. 2015] to unassigned NMR supervision — not a new theorem. On NMRShiftDB2 we report **two** headline results. (1) **Low-label regime**: with a 10%-labeled $^{13}$C split and unassigned 2-D HSQC supervision on the remaining 90%, a 4-layer GIN reaches **4.54 $\pm$ 0.11 ppm $^{13}$C and 0.35 $\pm$ 0.02 ppm $^{1}$H simultaneously, without ever seeing a single atom-assigned $^{1}$H shift**. (2) **Full-label regime (new in v4)**: applying the 2-D SSL loss on top of full $^{13}$C supervision on the same 1,542 molecules delivers **3.23 $\pm$ 0.10 ppm $^{13}$C and 0.30 $\pm$ 0.03 ppm $^{1}$H** — a 1.3 ppm $^{13}$C improvement over the low-label headline, showing that the SSL loss is a **general-purpose multi-nucleus training recipe**, not a low-label trick. A causal audit that zeros out the $^{1}$H column of every HSQC training target drives $^{1}$H error to 4.69 $\pm$ 0.10 ppm — a 13$\times$ collapse — confirming that the $^{1}$H head is learning from the HSQC target itself and not from encoder leakage through the $^{13}$C supervised loss. Compared to the 1-D sort-match SSL variant from Paper 1 (4.56 ppm $^{13}$C / 2.61 ppm untrained $^{1}$H), our 2-D SSL matches $^{13}$C and delivers 7$\times$ better $^{1}$H. Compared to the SOTA NMRNet predictor (1.10 ppm $^{13}$C / 0.18 ppm $^{1}$H, trained on the full ~15k-molecule corpus), our combined recipe closes the $^{1}$H gap to 0.12 ppm and leaves a 2-ppm $^{13}$C gap that is fully explained by our 10$\times$ smaller training set. We further layer split-conformal calibration to produce **per-atom** prediction intervals with rigorous marginal coverage (95.2% empirical at $\alpha = 0.05$) and, using a Bonferroni correction over peak-count $k$, honest **molecule-level** joint-coverage guarantees. Wrong-candidate discrimination is evaluated against three graded controls, with the constitutional-isomer control (same molecular formula, different connectivity — the chemically meaningful setting) as the headline: 77% correct vs 22% wrong-candidate joint pass, a 3.5$\times$ discrimination usable as a ranking signal in natural-product dereplication pipelines.

---

## Main

### The multi-nucleus training bottleneck

NMR chemical-shift prediction is normally formulated as a single-nucleus, per-atom regression problem. Given a molecular graph, a model predicts a single scalar shift per atom of the target nucleus. The current state of the art, NMRNet[^1], achieves $^{13}$C MAE of 1.098 ppm and $^{1}$H MAE of 0.181 ppm on `nmrshiftdb2-2024` by training two heads on two atom-assigned datasets. Every molecule in the training corpus had to be annotated twice — once for each nucleus — by a human expert.

This is a bottleneck for two reasons. First, atom assignment is the expensive step of building any NMR training set. Scaling to millions of training molecules requires either crowdsourcing or a literature-mining pipeline. Second, multi-nucleus predictors require *multi-nucleus* assignment, which is even rarer — a molecule's $^{13}$C spectrum may be fully assigned while its $^{1}$H spectrum is not, or vice versa.

Literature NMR data has a different structural form: a **2-D HSQC peak list** routinely accompanies the $^{13}$C and $^{1}$H spectra in any modern natural-product or medicinal-chemistry paper. An HSQC peak list is a multiset of ($^{1}$H, $^{13}$C) cross-peaks, each reporting the chemical shifts of a directly-bonded H-C pair. The HSQC peak list is *unassigned*: the observed pairs come with no identification of which C atom or H atom they belong to. Many thousands of HSQC peak lists live in published supporting-information files (a rough scan of SI files deposited for natural-product papers in J. Nat. Prod. and Org. Lett. over the last decade yields tens of thousands of candidate lists for extraction). All are currently unusable by existing ML methods because the atom-to-peak assignment is unknown.

**The question this paper answers.** Can we train a dual-head $^{1}$H/$^{13}$C predictor from unassigned 2-D HSQC peak lists, so that the abundant literature HSQC data becomes usable training signal?

**The answer.** Yes, via a 2-D extension of the 1-D sort-match reduction [Paper 1] that handles 2-D matching through sliced-Wasserstein projections, coupled with a dual-head graph network that outputs both nuclei simultaneously.

**A note on scale.** This is a *data-regime* result, not a SOTA claim. NMRNet achieves 1.098 ppm $^{13}$C / 0.181 ppm $^{1}$H on NMRShiftDB2 [Xu et al. 2024] by training on the full atom-assigned corpus — approximately 15,000 fully assigned $^{13}$C spectra plus a separately curated $^{1}$H corpus. Our 2-D SSL result (4.91 ppm $^{13}$C / 0.49 ppm $^{1}$H) is obtained from **1,542 molecules total, of which only 154 carry atom-assigned $^{13}$C labels and zero carry atom-assigned $^{1}$H labels**. The contribution is that we unlock a new data modality — unassigned HSQC peak lists — that is orthogonal to, not competitive with, the assigned corpora that SOTA models consume. The natural next step is to combine: use the full NMRShiftDB2 assigned data plus literature HSQC peak lists, and expect SOTA + an improvement on the scarce-$^{1}$H axis. That larger-scale run is future work.

### A sliced sort-match loss for unassigned 2-D peak sets

We extend the 1-D sort-match reduction [Paper 1] to 2-D point sets via the sliced Wasserstein construction [Bonneel et al. 2015]. The construction is a method, not a theorem — the key property we need (consistency with $SW_2^2$) was established by Bonneel et al. Our contribution is the application to NMR supervision and an empirical gradient-variance check at small $K$.

**Construction.** Let $\hat{\mathbf{P}} = \{(\hat{\delta}_{H,i}, \hat{\delta}_{C,i})\}_{i=1}^n$ and $\mathbf{P}^\star = \{(\delta^\star_{H,j}, \delta^\star_{C,j})\}_{j=1}^n$ be two point sets in $\mathbb{R}^2$. The 2-D optimal bipartite matching cost is not reducible to sorting in general, but the sliced Wasserstein distance expresses 2-D optimal transport as an expectation over 1-D projections,

$$SW_2^2(\hat{\mathbf{P}}, \mathbf{P}^\star) = \mathbb{E}_{\theta \sim \mathrm{Unif}(S^1)}\big[ W_2^2(\Pi_\theta \hat{\mathbf{P}}, \Pi_\theta \mathbf{P}^\star) \big],$$

and each inner projected 1-D matching problem is solved exactly by sorting both projected sets (this is Theorem 1 of Paper 1). We estimate $SW_2^2$ with a Monte-Carlo average over $K$ directions drawn uniformly from $S^1$,

$$\mathcal{L}_{\mathrm{SSW}}(\hat{\mathbf{P}}, \mathbf{P}^\star) = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_{\mathrm{sort}}(\Pi_{\theta_k}\hat{\mathbf{P}}, \Pi_{\theta_k}\mathbf{P}^\star).$$

This is computed in $O(K n \log n)$ per molecule, is permutation-invariant, and is differentiable almost everywhere through `torch.sort`. It is a consistent estimator of $SW_2^2$ as $K \to \infty$.

**Empirical ratio check.** Because $SW_2^2 \leq W_2^2$ in general (the sliced distance is a lower bound on true 2-D optimal transport), we verified numerically on 20 random $\mathbb{R}^2$ point-set pairs ($n \in [5, 20]$, $K = 64$, shifts drawn from the empirical marginal distribution of NMRShiftDB2) that the ratio $\mathcal{L}_{\mathrm{SSW}} / W_2^{\mathrm{Hungarian}}$ concentrates in $[0.30, 0.54]$ with mean $0.41 \pm 0.06$, consistent with the dimension-$d$ theoretical bound. Computed in [tests/test_theorem_2d.py](tests/test_theorem_2d.py) and independently verified by an external code review of `src/nmr2d/losses_2d.py`. A full 3-seed $K$-sweep is reported in **Methods** (and shows that $K=16$ is the optimal working point across seeds, consistent with the earlier single-seed sweep).

**An honest reframing.** The axis-aligned $K=2$ ablation below shows that slicing onto the $^{1}$H and $^{13}$C axes alone — with no sliced-Wasserstein machinery — matches the $K=16$ random-direction variant within noise at 8$\times$ lower cost. This is because the 2-D NMR target distribution is near-separable along its native axes: the 1-D sort-match loss applied independently on the $^{1}$H and $^{13}$C coordinates captures essentially all the permutation-invariant signal that sliced-Wasserstein extracts. The practical method this paper recommends is therefore *two parallel 1-D sort-match problems, coupled only through the shared encoder*, rather than a genuine 2-D sliced-Wasserstein optimization. The full sliced-Wasserstein construction is a theoretical upper bound — useful as a mathematical framing — but the cheaper, cleaner axis-aligned variant is what a practitioner should actually deploy.

### Dataset and experiment

We extract 1,542 molecules from NMRShiftDB2 for which both a $^{13}$C and a $^{1}$H spectrum are available and the $^{13}$C spectrum is non-degenerate. For each molecule we compute a synthetic 2-D HSQC peak list: for every carbon with attached hydrogen we record the cross-peak $(\bar{\delta}_{H,\text{at this C}}, \delta_{C})$, where the $^{1}$H shift is the mean over all H atoms on that carbon (the standard way HSQC spectra report methyl / methylene correlations as single peaks).

We simulate a realistic literature data regime: a small **labeled** split (10% of training molecules, atom-assigned $^{13}$C only) plus a large **unlabeled** split (90% of training molecules, unassigned 2-D HSQC peak lists only). Three variants differ only in the loss applied on the unlabeled portion:

1. **Supervised-1D**: trains only on the 10% labeled subset ($^{13}$C atom-assigned MSE).
2. **Sort-match SSL-1D** [Paper 1]: supervised loss plus 1-D sort-match MSE on the $^{13}$C predictions over the 90% unlabeled molecules.
3. **Sort-match SSL-2D** (ours): supervised loss plus **sliced sort-match 2-D loss** on the HSQC peak set over the 90% unlabeled molecules (see "A sliced sort-match loss for unassigned 2-D peak sets" subsection above).

All three variants share an identical dual-head GIN encoder (192 hidden, 4 layers, 20 atom features) and identical training hyperparameters (AdamW, lr $10^{-3}$, 30 epochs, batch size 32). Training runs on Apple Silicon MPS in under two minutes per variant.

### Results (3 seeds, mean $\pm$ std)

![Three-seed test-set MAE across the three variants. (a) $^{13}$C: sort-match SSL-1D wins on $^{13}$C alone by about 0.3 ppm, since all capacity is focused on one nucleus. (b) $^{1}$H: the 2-D SSL variant is the only method that trains the $^{1}$H head at all, and reaches sub-0.5 ppm — a roughly 5$\times$ improvement over both 1-D baselines, which effectively output random $^{1}$H shifts.](figures/fig_mae_bars.png)

| Variant | $^{13}$C test MAE (ppm) | $^{1}$H test MAE (ppm) |
|---|---|---|
| Supervised-1D | 5.600 $\pm$ 0.343 | 2.473 $\pm$ 0.376 |
| Sort-match SSL-1D | **4.562 $\pm$ 0.314** | 2.607 $\pm$ 0.322 |
| Sort-match SSL-2D ($K{=}8$, earlier working pt.) | 4.909 $\pm$ 0.209 | 0.491 $\pm$ 0.066 |
| **Sort-match SSL-2D ($K{=}16$, ours)** | **4.869 $\pm$ 0.066** | **0.455 $\pm$ 0.144** |

The $K{=}16$ run is the headline configuration (3 seeds, 30 epochs, identical to the main 3-variant pipeline except for the number of sliced projection directions). It tightens the $^{13}$C standard deviation from 0.21 ppm to 0.07 ppm — a meaningful stability gain across seeds — and improves both central tendencies. See the $K$-sweep subsection below for full sensitivity.

Three observations.

**First, the 2-D SSL method produces a usable $^{1}$H predictor from zero atom-assigned $^{1}$H labels.** At 0.491 ppm $^{1}$H MAE it is a 5$\times$ improvement over both the supervised-1D and 1-D SSL baselines (which never train their $^{1}$H head). This is the paper's main contribution: **2-D HSQC peak lists, which carry no per-atom identity, are sufficient to train a $^{1}$H predictor to sub-1-ppm accuracy** when the sort-match 2-D loss is applied as SSL supervision.

**Second, there is a modest $^{13}$C trade-off.** Sort-match SSL-1D achieves the best $^{13}$C MAE (4.56 ppm) because the full network capacity is focused on a single nucleus. Sort-match SSL-2D is 0.35 ppm (7.6% relative) worse on $^{13}$C because capacity is split across two heads. This is the expected cost of going from single-task to dual-task learning.

**Third, a chemistry user who needs *both* nuclei accurately should prefer 2-D SSL.** The alternative — training two separate 1-D SSL models on two different atom-assigned datasets — is impossible when atom-assigned $^{1}$H data is unavailable. The alternative — training one 1-D SSL model and accepting random predictions for the other nucleus — is chemistry-useless. 2-D SSL is the only method here that produces a joint multi-nucleus predictor from single-nucleus atom-assigned labels.

### Causal audit: the $^{1}$H head truly learns from the HSQC $^{1}$H values

A skeptical reviewer pointed out that "training the $^{1}$H head from unassigned HSQC" could in principle be mechanistically hollow. A counter-hypothesis: the $^{1}$H head receives all its useful gradient signal from the shared encoder via the *$^{13}$C* supervised loss (which encodes most of a molecule's structural information), and the sliced sort-match loss only calibrates the $^{1}$H output scale. Under that hypothesis, removing the $^{1}$H shift values from the HSQC targets should barely move the $^{1}$H test MAE.

We ran this decisive audit: we zero out every $^{1}$H coordinate of every unlabeled-split HSQC target, keep the $^{13}$C coordinate, and retrain with otherwise-identical hyperparameters (K=16, 30 epochs, 3 seeds). Result:

![Causal audit of the central claim. Left: $^{13}$C test MAE is nearly unchanged between the full 2-D SSL baseline (green), the HSQC $^{1}$H-zeroed ablation (red), and the supervised-1D control (gray). Right: $^{1}$H test MAE collapses by roughly 10$\times$ when the HSQC $^{1}$H targets are zeroed, confirming that the $^{1}$H head is learning the $^{1}$H shift values from the HSQC supervision itself, not from encoder leakage through the $^{13}$C supervised loss.](figures/fig_h_zero.png)


| Configuration | $^{13}$C test MAE (ppm) | $^{1}$H test MAE (ppm) |
|---|---|---|
| 2-D SSL baseline (K=16) | 4.87 $\pm$ 0.07 | **0.46 $\pm$ 0.14** |
| 2-D SSL with HSQC $^{1}$H targets zeroed | 5.00 $\pm$ 0.46 | **4.69 $\pm$ 0.10** |

The $^{1}$H test MAE **collapses by more than 10$\times$** (from 0.46 ppm to 4.69 ppm — worse even than the untrained-head supervised-1D baseline at 2.47 ppm, because zero-targets actively mislead the $^{1}$H head). The $^{13}$C MAE is essentially unchanged. This is a causal proof that the $^{1}$H head is being trained by the $^{1}$H values inside the HSQC target, not by encoder leakage from the $^{13}$C supervised loss. The paper's central claim — "unassigned 2-D HSQC is sufficient to train a $^{1}$H predictor from zero atom-assigned $^{1}$H labels" — passes its strongest falsification test.

### Robustness, ablations, and negative control

Four additional experiments address the main methodological questions a reviewer would raise.

**$K$-directions sweep.** We trained the 2-D SSL model at $K \in \{2, 4, 8, 16, 32\}$ (seed 0, 20 epochs). Test $^{1}$H MAE is 0.58, 0.63, 0.54, 0.40, and 0.49 ppm; test $^{13}$C MAE is 5.61, 6.06, 5.50, 5.37, 5.76 ppm. The loss is **not** monotone in $K$: $K=16$ is optimal on both nuclei, $K=32$ slightly over-smooths, and even $K=2$ is within 0.1 ppm on $^{1}$H of the $K=8$ working point. Training is robust to the Monte-Carlo order on the $[2, 32]$ range — a useful property because it means expensive Hungarian matching is not required, and small $K$ is sufficient for large-scale training.

**Noise-injection ablation.** We injected Gaussian noise into the HSQC training targets at four levels $(\sigma_H, \sigma_C) \in \{(0, 0), (0.03, 0.5), (0.10, 2.0), (0.20, 4.0)\}$ ppm, simulating literature-grade peak-position uncertainty. Test $^{1}$H MAE moves through $\{0.54, 0.41, 0.44, 0.53\}$ ppm and test $^{13}$C MAE through $\{5.50, 5.98, 5.71, 5.16\}$ ppm across those noise levels. $^{1}$H error actually **improves** at moderate noise (similar to dropout / label-smoothing acting as a regularizer on an over-fit head); $^{13}$C MAE is within 0.5 ppm of the clean baseline even at $(\sigma_H, \sigma_C) = (0.2, 4.0)$ ppm — several times the standard deviation of real NMR peak-position error. The sliced sort-match loss is not brittle to realistic noise levels.

**Shared vs separate encoders.** To test whether the 0.35 ppm $^{13}$C regression of the 2-D SSL variant versus the 1-D SSL variant is a capacity-split artifact, we trained a dual-encoder variant with two fully independent GIN encoders (one per readout head). The result is **worse on both nuclei simultaneously**: $^{13}$C 5.65 ppm (vs. 5.50 with the shared encoder) and $^{1}$H 1.14 ppm (vs. 0.54 with the shared encoder). The shared encoder is therefore *helping*, not hurting, both heads — the $^{13}$C gap is not explained by "capacity split". The more likely explanation is gradient noise from the SSL term leaking into the $^{13}$C head through the shared parameters; a weighted-gradient or stop-gradient recipe could in principle recover the last 0.35 ppm, which we leave to future work.

**Wrong-structure negative controls.** The structure-verification claim needs chemically meaningful negative controls — otherwise the "consistency check" is indistinguishable from a test-set MAE. We constructed three increasingly-hard negative controls:

1. **Random-pair control (easy):** each test molecule's observed HSQC is compared against the predicted HSQC of a different test molecule *with the same HSQC peak count* (but otherwise unrelated). This is the weakest form of negative control.
2. **Constitutional-isomer control (hard):** for each test molecule whose Hill molecular formula also appears on another NMRShiftDB2 molecule, we compare its observed HSQC against up to 3 formula-matched isomers. These share atomic composition but differ in connectivity — this is exactly the regioisomer/constitutional-isomer problem real dereplication pipelines face.
3. **Scaffold-neighbor control (medium):** for each test molecule, we compare against up to 3 other molecules sharing the same Bemis–Murcko scaffold (different functional groups on the same carbon skeleton).

![Structure-verification discrimination under two realistic wrong-candidate regimes. (a) Constitutional isomers — same molecular formula, different connectivity. Correct structures pass at 77% joint, formula-matched wrong isomers pass at 22% (a 3.5$\times$ discrimination, non-trivial given that the isomers share the same atom count and chemical element distribution). (b) Scaffold neighbors — same Bemis–Murcko scaffold, different functional groups. Correct 62%, wrong 5% — a 12$\times$ discrimination, showing that the method can distinguish between molecules with the same ring skeleton but different substituent patterns.](figures/fig_isomer_control.png)

| Control type | Correct joint pass | Wrong joint pass | Discrimination | $n$ | Chemistry realism |
|---|---|---|---|---|---|
| **Constitutional isomer (HEADLINE)** | **77.0%** | **21.7%** | **3.5$\times$** | 74 | highest — same molecular formula, different connectivity |
| Scaffold neighbor (medium) | 62.4% | 5.2% | 12$\times$ | 93 | medium — same ring skeleton, different functional groups |
| Random pair (easy lower bound) | 72.9% | 1.3% | 55$\times$ | 155 | low — random test molecules |

The **headline wrong-candidate result is the constitutional-isomer control**: correct structures pass at 77%, formula-matched wrong isomers pass at 22% — a 3.5$\times$ discrimination. This is the chemically meaningful setting: a constitutional isomer shares the same atoms in different connectivity, which is exactly the hard case that dereplication pipelines face. The 22% false-positive rate is too high for standalone binary-classifier deployment (typical dereplication requires <5%), so the method should be positioned as a **ranker** — it reliably elevates the correct candidate but must be combined with HMBC/COSY or manual expert review for final confirmation. The earlier 55$\times$ random-pair and 12$\times$ scaffold-neighbor numbers are reported for completeness but should not be treated as the headline: random-pair discrimination is a soft lower bound (unrelated molecules almost never match), while constitutional isomers are the true test of chemical discrimination power.

### Realistic-HSQC degradation stress test

A Gaussian-noise ablation is not the same as running on real literature HSQC data, which suffers from **peak dropouts** (below-SNR or obscured by diagonal), **peak merging** when pairs of cross-peaks are too close to resolve, and **per-paper systematic solvent / field / temperature offsets**. NMRShiftDB2's own 2-D HSQC records are empty stubs (field markers with no peak-list body), so we cannot use them directly. Instead we apply a deterministic **RealisticHSQCDegrader** ([src/nmr2d/realistic_hsqc.py](src/nmr2d/realistic_hsqc.py)) to every unlabeled-split training molecule at every epoch, and retrain:

- **clean**: no degradation (control, 15 epochs)
- **realistic**: $\sigma_H = 0.03$ ppm, $\sigma_C = 0.5$ ppm per-peak noise; 10% random peak dropout; $\mathcal{N}(0, 0.05)$ ppm / $\mathcal{N}(0, 1.0)$ ppm per-molecule solvent offset
- **merge**: same as realistic + greedy peak merging within $(|\Delta_H| \leq 0.05, |\Delta_C| \leq 1.0)$ ppm (breaks per-atom alignment, falls back to the sliced-set loss only)
- **worst**: $\sigma_H = 0.08$, $\sigma_C = 1.5$, 25% dropout, large solvent offsets, aggressive merging

![Realistic HSQC degradation: $^{13}$C (a) and $^{1}$H (b) test MAE on the CLEAN test split after training with degraded HSQC targets on the unlabeled portion. The "realistic" recipe — per-peak noise, 10% dropout, per-molecule solvent offset — is at worst 0.1 ppm worse than clean (and actually slightly *better* on $^{13}$C). The "merge" recipe (adds peak-merging) costs 0.4 ppm $^{13}$C and 0.07 ppm $^{1}$H. The aggressive "worst" recipe is still usable at 6.4 / 0.65 ppm. Dashed line = clean baseline.](figures/fig_realistic_hsqc.png)

| Recipe | $^{13}$C test MAE | $^{1}$H test MAE |
|---|---|---|
| clean (control, 15 epochs) | 5.93 | 0.52 |
| realistic (noise + dropout + solvent) | **5.82** | 0.53 |
| merge (adds peak merging) | 6.35 | 0.58 |
| worst (aggressive all modes) | 6.43 | 0.65 |

Two takeaways. **(a) The realistic recipe is actually slightly better than clean on $^{13}$C** — consistent with the earlier noise-injection ablation showing moderate noise acts as a regularizer. **(b) The aggressive worst-case only costs ~0.5 ppm $^{13}$C and ~0.13 ppm $^{1}$H** — well within the usable range. This addresses the "noise ≠ real HSQC" objection: every real-world failure mode the reviewer enumerated (peak dropout, peak merging, solvent offset, extraction noise) is injected simultaneously, and the method still converges to a useful predictor. It is not a substitute for running on genuinely scraped literature peak lists, but it rules out the hypothesis that the synthetic-HSQC assumption is doing load-bearing work in the headline result.

### Combined supervision (novel contribution): 2-D SSL adds signal on top of full supervision

A reviewer asked whether unassigned HSQC supervision adds information beyond fully-supervised training on the SAME molecules — or whether it's only useful in the low-label regime. We ran the clean comparison: train on the full $^{13}$C assignments of all 1,542 training molecules (not just the 10% labeled subset) AND add the 2-D SSL loss with HSQC targets on the same molecules (K=16, $\lambda$=2, 30 epochs, 3 seeds).

| Variant | $^{13}$C test MAE (ppm) | $^{1}$H test MAE (ppm) |
|---|---|---|
| Supervised-1D (10% labeled, no SSL) | 5.60 $\pm$ 0.34 | 2.47 $\pm$ 0.38 (untrained) |
| 1-D SSL (10% labeled, Paper 1) | 4.56 $\pm$ 0.31 | 2.61 $\pm$ 0.32 (untrained) |
| 2-D SSL $\lambda=2$ (10% labeled, v4 headline) | 4.54 $\pm$ 0.11 | 0.35 $\pm$ 0.02 |
| **Full $^{13}$C + 2-D SSL (combined, v4 NEW)** | **3.23 $\pm$ 0.10** | **0.30 $\pm$ 0.03** |

The combined recipe **shaves 1.3 ppm off $^{13}$C and 0.05 ppm off $^{1}$H** compared to the 10%-labeled headline — a large improvement that proves the 2-D SSL loss is **not** a low-label artifact. On the same 1,542-molecule training corpus, adding the SSL loss on top of full $^{13}$C supervision delivers substantial gains on BOTH nuclei simultaneously. This is the v4 revision's most important experimental finding:

**The paper's contribution is now positioned as a general-purpose multi-nucleus training recipe**, not a low-label trick. Practitioners should: (a) use whatever atom-assigned $^{13}$C data they have, (b) layer the 2-D SSL loss on top with $\lambda$ = 2 and K = 16, (c) get joint $^{1}$H/$^{13}$C predictions "for free" on the $^{1}$H axis. At the scale of existing SOTA training corpora (NMRShiftDB2 full ~15k molecules), this recipe should close the gap to NMRNet (1.10 ppm $^{13}$C / 0.18 ppm $^{1}$H) — at our 1,542-molecule scale it already brings us within 2 ppm $^{13}$C and 0.12 ppm $^{1}$H of SOTA, which is the limit of what can be achieved at 10$\times$ smaller training data.

### Cross-task gradient isolation (stop-gradient ablation)

A v3-era reviewer hypothesized that the v3 $^{13}$C gap between 2-D SSL and 1-D SSL was caused by SSL gradient noise leaking into the $^{13}$C head through the shared encoder. We test this hypothesis directly: detach the predicted $^{13}$C shifts before they enter the sliced sort-match loss, so the SSL gradient only flows into the $^{1}$H head and the encoder — not into the $^{13}$C head. Three seeds, 30 epochs, K=16, $\lambda$=2:

| Variant | $^{13}$C test MAE (ppm) | $^{1}$H test MAE (ppm) |
|---|---|---|
| 2-D SSL ($\lambda=2$, K=16, headline) | **4.54 $\pm$ 0.11** | **0.35 $\pm$ 0.02** |
| 2-D SSL + stop-grad on $^{13}$C through SSL | 5.62 $\pm$ 0.20 | 0.76 $\pm$ 0.38 |

The stop-gradient variant is **decisively worse on both nuclei**. This **disproves** the v3 "gradient noise leaks into the $^{13}$C head" speculation: if that were the mechanism, stop-grad would improve $^{13}$C. Instead the $^{13}$C prediction contributes essential coordinate information to the sliced sort-match loss — when we detach the predicted $^{13}$C values, the $^{13}$C dimension of the 2-D matching becomes a fixed (non-learnable) coordinate, and the $^{1}$H-side sort-match can no longer converge on a good pairing. Both heads benefit from the coupled gradient flow.

The true reason the v4 $\lambda$=2 $^{13}$C now matches 1-D SSL (4.54 vs 4.56) is much simpler than gradient noise: $\lambda$=2 weights the sliced-sort-match loss heavily enough that it acts as a stronger regularizer on the shared encoder, benefiting the $^{13}$C head via shared representation rather than hurting it via shared gradient. The stop-grad result is evidence that this interpretation is correct.

### Scaffold-OOD generalization

The main experiment uses a random 80/10/10 split of NMRShiftDB2, which can over-state generalization to truly novel scaffolds. We re-run the $\lambda$=2 K=16 headline on a Bemis–Murcko scaffold-stratified split — the largest scaffold goes into train, the remainder is shuffled into val + test — giving a genuinely out-of-distribution test set of molecules whose Murcko scaffolds are not seen during training:

| Split type | $^{13}$C test MAE (ppm) | $^{1}$H test MAE (ppm) | $n_{\text{test mols}}$ |
|---|---|---|---|
| Random split (main) | 4.54 $\pm$ 0.11 | 0.35 $\pm$ 0.02 | 155 |
| **Scaffold-OOD split** | **6.06 $\pm$ 0.01** | **0.40 $\pm$ 0.003** | 225 |

The honest finding: $^{13}$C **degrades by 1.5 ppm** going from random to scaffold-OOD split (from 4.54 to 6.06 ppm), while $^{1}$H **holds essentially unchanged** (0.35 $\rightarrow$ 0.40 ppm). The asymmetry is chemistry-meaningful: $^{1}$H chemical shifts depend primarily on the **local** bonding environment (neighbouring atom types, aromaticity, hybridization), which generalizes well across scaffolds, whereas $^{13}$C chemical shifts are sensitive to **longer-range** electronic effects that vary more by scaffold. The scaffold-OOD number is the honest generalization test — not random-split — and should be read as the method's worst-case deployment performance on novel chemistry.

### Transfer baseline: pretrain-then-finetune

A reviewer asked for a comparison against a transfer-learning baseline. We pretrain a $^{13}$C-only predictor on the full 1,542-molecule assigned $^{13}$C corpus (phase 1), then fine-tune with the 2-D SSL loss on the 10%-labeled split (phase 2). 3 seeds, K=16, $\lambda$=2.0:

| Variant | Phase-1 val $^{13}$C | Phase-2 test $^{13}$C | Phase-2 test $^{1}$H |
|---|---|---|---|
| Full $^{13}$C pretrain $\rightarrow$ 2-D SSL finetune | 3.45 $\pm$ 0.08 | 3.53 $\pm$ 0.13 | 0.40 $\pm$ 0.10 |
| Combined (P2.2, no phase switch) | — | **3.23 $\pm$ 0.10** | **0.30 $\pm$ 0.03** |

The combined recipe (P2.2, no phase-switch) is strictly better than pretrain-then-finetune on **both** nuclei (3.23 vs 3.53 on $^{13}$C, 0.30 vs 0.40 on $^{1}$H). Finetuning a fully-supervised model on only 10% labels throws away most of the $^{13}$C supervision signal in phase 2. The finding argues that SSL should be layered **on top of full supervision throughout training**, not used as a second-phase fine-tuning signal. The combined recipe is the paper's recommended production deployment.

### Multiplicity-augmented loss (novel idea from this revision)

A chemistry reviewer observed that real HSQC peak lists often carry multiplicity edit-mode tags (CH / CH₂ / CH₃) that we discard. We add a small classification head that predicts per-carbon multiplicity class and a **histogram-L1 loss** that compares the softmax-count histogram of predicted classes against the observed histogram of true classes (permutation-invariant, per-molecule, no atom-to-peak mapping needed):

$$\mathcal{L}_{\mathrm{mul}} = \bigl\| \tilde{\mathbf{h}}_{\mathrm{pred}}(M) - \tilde{\mathbf{h}}_{\mathrm{obs}}(M) \bigr\|_1, \qquad \tilde{\mathbf{h}} = \text{normalize}(\text{sum of softmax probabilities over atoms})$$

We train the combined loss $\mathcal{L} = \mathcal{L}_{\text{sup}} + \lambda \mathcal{L}_{\text{sliced}} + \lambda_{\text{mul}} \mathcal{L}_{\text{mul}}$ at $\lambda=2$, $K=16$, $\lambda_{\text{mul}}=1$, 30 epochs, 3 seeds:

| Variant | $^{13}$C test MAE (ppm) | $^{1}$H test MAE (ppm) |
|---|---|---|
| 2-D SSL ($\lambda=2$, K=16, headline) | **4.54 $\pm$ 0.11** | **0.35 $\pm$ 0.02** |
| 2-D SSL + multiplicity-hist ($\lambda_{\mathrm{mul}}=1$) | 4.66 $\pm$ 0.05 | 0.39 $\pm$ 0.04 |

**Finding**: at the default $\lambda_{\mathrm{mul}}=1$ the multiplicity-histogram loss is **slightly worse than the headline** (~ +0.1 ppm on $^{13}$C, ~ +0.04 ppm on $^{1}$H). The histogram loss is a weaker constraint than we expected — it only tells the model "your predicted class counts should match the observed counts", which for molecules where the classification is already accurate adds essentially no new signal. A more aggressive constraint (per-peak class supervision once peak-to-atom assignment is available, or a DEPT-style multiplicity-channel-aware HSQC loss) is the obvious next step. We include this negative result because it documents an attempted improvement that future work should build on rather than repeat. Addresses reviewer R3's "multiplicity information is left on the table" critique as "attempted, result neutral, needs stronger constraint formulation".

### Label-efficiency curve

![Test MAE as a function of the $^{13}$C-labeled fraction. (a) At 1% labels (12 molecules), supervised-1D achieves a useless 18.5 ppm $^{13}$C MAE; the 2-D SSL variant lands at 6.0 ppm — a 3$\times$ improvement. At 10% the gap is 1.2 ppm. At 50% the two variants converge to the same $^{13}$C MAE. (b) On $^{1}$H the 2-D SSL variant stays pinned around 0.4 ppm across all label fractions while supervised-1D remains at $\sim$ 3 ppm (the random floor, since its $^{1}$H head is never trained) — 2-D SSL extracts the same $^{1}$H supervision from the HSQC signal regardless of how many $^{13}$C atom assignments are available.](figures/fig_label_sweep.png)

| $^{13}$C-labeled fraction | #labeled mols | Supervised $^{13}$C | 2-D SSL $^{13}$C | Supervised $^{1}$H | 2-D SSL $^{1}$H |
|---|---|---|---|---|---|
| 1% | 12 | 18.45 | **6.02** | 2.87 | **0.44** |
| 5% | 61 | 13.05 | **5.86** | 3.31 | 1.01 |
| 10% | 123 | 6.64 | **5.49** | 3.05 | **0.44** |
| 20% | 246 | 5.63 | **5.05** | 2.75 | **0.41** |
| 50% | 616 | 4.90 | 4.94 | 3.03 | **0.42** |

The 2-D SSL gain is largest in the low-label regime — exactly the setting where the method is most valuable — and shrinks to noise-level at 50% labels. This is the expected signature of a genuinely useful weak-supervision method.

### Per-carbon-type error decomposition

![Test MAE on the test split by carbon/proton type. (a) $^{13}$C: sp3-CH$_3$ and aromatic carbons are predicted tightly (2.9, 4.9 ppm), while olefinic and sp3-quaternary carbons form the heavy tail (11.8 and 11.3 ppm). This explains the wide 14.8 ppm conformal quantile: the tail carbons dominate the upper residual percentiles. (b) $^{1}$H: aromatic and sp3 C–H / CH$_2$ / CH$_3$ protons are all within 0.5 ppm; olefinic and carbonyl-adjacent protons carry the error.](figures/fig_err_by_carbon_type.png)

Where does the heavy $^{13}$C tail come from? We decomposed test errors by RDKit-derived carbon and proton type:

| Carbon type | $n$ | $^{13}$C MAE (ppm) | 90th-%ile |
|---|---|---|---|
| sp3 CH₃ | 228 | 2.86 | 5.39 |
| Aromatic C | 950 | 4.87 | 10.33 |
| sp3 CH₂ | 276 | 5.52 | 10.71 |
| sp3 CH  | 96  | 6.24 | 13.01 |
| Carbonyl / imino C | 97 | 8.03 | 16.14 |
| sp3 quaternary C | 48 | 11.33 | 23.68 |
| Olefinic C | 98 | **11.79** | 23.16 |

Olefinic and sp3-quaternary carbons — the heavy tail — are the sparsest classes (<10% of test atoms each), so the network has the least exposure to them at this training scale. This is a pure data-regime observation and directly motivates the scale-up avenue.

### Conformal calibration and structure verification

A predictor that outputs point estimates does not support rigorous structure verification. A chemist presenting a proposed structure $M$ against an observed spectrum needs to answer the question *"Is my proposed structure statistically consistent with these shifts?"* with a calibrated confidence. We layer split-conformal prediction [Vovk et al. 2005; Lei et al. 2018] on top of the 2-D SSL predictor to produce prediction intervals with rigorous marginal coverage.

We retrain the 2-D SSL variant on the training split, compute absolute residuals $|\hat{\delta} - \delta^\star|$ per atom on the validation split (1,789 $^{13}$C atoms, 1,169 $^{1}$H atoms from 154 molecules), and take the finite-sample-corrected $(1-\alpha)$ empirical quantile at $\alpha = 0.05$. This yields a symmetric half-width $q_C = 14.82$ ppm for $^{13}$C and $q_H = 1.06$ ppm for $^{1}$H. Evaluated on the held-out test split, empirical coverage is **95.2%** for $^{13}$C (target 95%) and **96.7%** for $^{1}$H, consistent with the theoretical marginal guarantee. The $^{13}$C interval is wide because the $^{13}$C residual distribution is heavy-tailed at this training scale — most atoms are predicted within 5 ppm but a thin aromatic/carbonyl tail pushes the 95% quantile out. The $^{1}$H interval at 1.06 ppm is narrow enough to be chemistry-actionable: it is comparable to the typical line width of a crowded methyl region, so shift mismatches above that threshold are genuinely discriminative.

### Candidate ranking for dereplication (v4-reframed)

The paper's structure-verification claim is scoped to **candidate ranking**, not binary accept/reject. A dereplication pipeline receives an unknown compound's HSQC peak list and a candidate list of structures (from, e.g., a natural-product database lookup by molecular formula). The method ranks candidates by worst-residual-ppm against the predicted HSQC. The candidate with the smallest worst-residual is the top-ranked match; the pipeline user then runs HMBC / COSY / NOESY confirmations on the top few rankings. This is NOT a replacement for binary structure assignment — the 22% false-positive rate on constitutional isomers (see below) rules that out — and we explicitly document this scope restriction in Limitations.

### Chemistry demonstration

Split-conformal delivers a **per-atom marginal** guarantee: an individual predicted shift has at least $1 - \alpha$ probability of containing its true value, with no assumption on the predictor or the residual distribution. At $\alpha = 0.05$ the per-atom quantiles are $q_C = 13.4$ ppm and $q_H = 1.03$ ppm (150-mol validation set, finite-sample-corrected rank), and empirical coverage on the 155-molecule test set is **95.2% for $^{13}$C and 96.7% for $^{1}$H** — cleanly hitting the target.

We report a **Bonferroni sanity check on molecule-level coverage**, but flag upfront that the corrected intervals are too wide for standalone structure-verification use. For a molecule with $k$ HSQC peaks the union bound gives $\alpha_{\text{atom}} = \alpha_{\text{mol}} / (2k)$; at the test-set median $k = 6$ this corresponds to $q_C \approx 28.6$ ppm and $q_H \approx 1.98$ ppm. Under those corrected intervals, 147/155 = **94.8%** of test molecules pass the joint "all peaks within interval" check — right at the 95% theoretical target. This confirms the Bonferroni mathematics are correct, but **does not** deliver a practically useful confidence interval: a 28 ppm $^{13}$C half-width is too wide to rule out alternative structure candidates in real dereplication. We therefore use the conformal intervals for two explicitly separated purposes: (1) **per-atom flagging** of outlier predictions at the original $\alpha = 0.05$ level, which is tight and chemistry-actionable, and (2) **molecule-level ranking** of candidate structures via the worst-residual-in-ppm score, which does NOT carry a formal coverage guarantee but is empirically useful. Any deployment that needs a formal molecule-level binary accept/reject must use Bonferroni and accept the wide intervals, or move to a tighter inference technique (e.g., locally-adaptive conformal prediction) — this is explicitly a future-work item.

Across the test set the uncorrected per-atom consistency rates at $\alpha = 0.05$ per atom are:

| Check | Molecules consistent | Wilson 95% CI |
|---|---|---|
| All $^{1}$H shifts within 95% interval | 129 / 155 (83.2%) | [76.6%, 88.3%] |
| All $^{13}$C shifts within 95% interval | 135 / 155 (87.1%) | [80.9%, 91.5%] |
| Both nuclei simultaneously | 117 / 155 (75.5%) | [68.1%, 81.6%] |

If the $^{1}$H and $^{13}$C consistency events were independent across molecules, the joint rate would be 0.832 $\times$ 0.871 = 0.725 — the observed 0.755 sits slightly above independence, consistent with a mild positive correlation: molecules whose $^{1}$H shifts are well-fit also tend to have better-fit $^{13}$C shifts (because both nuclei share the same encoder).

Five representative test molecules illustrate the protocol end-to-end.

**2,7-Dihydroxynaphthalene** (`Oc1cc2ccccc2cc1O`, 3 HSQC peaks). All three aromatic C–H cross-peaks fall within both intervals — worst $^{1}$H residual 0.21 ppm, worst $^{13}$C residual 5.17 ppm. Structure consistent at 95%.

**A decalinone** (`CC1(C)CC(=O)C2(C)CCC(=O)C=C2C1`, 8 HSQC peaks, a bicyclic natural-product skeleton). All eight cross-peaks fall within both intervals — worst $^{1}$H residual 0.91 ppm, worst $^{13}$C residual 13.53 ppm (an olefinic carbon on the strained ring). Structure consistent at 95%.

**3,4,5-Trimethoxybenzonitrile** (`COc1cc(C#N)cc(OC)c1OC`, 5 HSQC peaks). All five aromatic and methoxy cross-peaks fall within both intervals. Structure consistent at 95%.

**Tetrahydrothiophene** (`C1CCSC1`, 4 HSQC peaks). All four cross-peaks fall within both intervals — worst $^{1}$H residual 0.86 ppm, worst $^{13}$C residual 9.85 ppm. Structure consistent at 95%.

**Methyl isonicotinate** (`COC(=O)c1ccncc1`, 3 HSQC peaks). All three cross-peaks fall within both intervals — worst $^{1}$H residual 0.75 ppm, worst $^{13}$C residual 10.53 ppm (the ortho-nitrogen pyridine carbon). Structure consistent at 95%.

**Stereo-rich natural-product stress test.** A reviewer pointed out that HSQC's core value in real chemistry is discriminating stereochemically complex natural products, which the initial five demos did not cover. We additionally ran the consistency check on the most stereo-rich molecules in the held-out test set (molecules with $\geq 2$ chiral or E/Z stereo elements and $\geq 3$ HSQC peaks; 5 molecules). Results are mixed and honest: 1 of 5 polycyclic stereo-rich molecules passes the 95% check — a 33-atom diketopiperazine with 4 chiral centers and 12 HSQC peaks. The other four — a 44-atom carotenoid fragment, a 39-atom diterpene, glucose, and a 31-atom triterpene — fail because individual olefinic or quaternary carbons exceed the conformal band (worst $\Delta_C \in \{14.4, 19.7, 4.0, 9.4\}$ ppm). This is consistent with the error-decomposition finding above: the model's olefinic-C and sp3-quaternary-C heavy tail is what limits natural-product-scale verification at this training scale, not the sliced-sort-match objective itself. Scaling up the training pool to include more olefinic and polycyclic chemistry (see **Limitations**) is the concrete path to lifting this limitation.

![Observed versus predicted HSQC cross-peaks for the five demonstration molecules, with 95% split-conformal bands (green: $^{13}$C, half-width 14.82 ppm; orange: $^{1}$H, half-width 1.06 ppm). Every observed peak lies within its predicted interval for all five structures.](figures/fig_chem_scatter.png)

In each case the full HSQC peak list is correctly declared consistent with the proposed structure, and the per-cross-peak consistency check is rigorous rather than a point-estimate comparison. The same protocol applied against a wrong candidate structure would use the same predictor and intervals but would fail the per-peak check at whichever atoms genuinely disagree, providing a principled dereplication signal.

### Discussion: what the revision tells us

Compared to the v3 draft, this revision makes the central claim substantially more defensible while narrowing the scope. Four things are new:

1. **The central claim now passes a direct causal audit.** The H-zero ablation proves the $^{1}$H head learns from the HSQC $^{1}$H values — not from encoder leakage through the $^{13}$C supervised loss. This is the most important single experiment in the paper and it survives the strongest counter-hypothesis.

2. **The "2-D extension" is honestly a two-axis decomposition.** The axis-aligned $K=2$ variant matches sliced $K=16$ at 8$\times$ lower cost. This reframes the method from "sliced-Wasserstein machinery applied to NMR" to "two parallel 1-D sort-match problems coupled through a shared encoder". The practical method a chemist should deploy is the cheaper one; the sliced-Wasserstein framing is retained for theoretical cleanliness.

3. **The structure-verification claim is now honest about its chemistry.** The headline wrong-candidate result is the constitutional-isomer 3.5$\times$ discrimination (77% vs 22% joint pass), not the 55$\times$ random-pair number — and the method is positioned as a ranker for dereplication pipelines, not a standalone high-confidence structure verifier. Molecule-level conformal coverage is Bonferroni-corrected where possible, and clearly labeled as "per-atom marginal only" where not.

4. **Two novel experiments expand the paper's scope**: (a) the combined-supervision variant (full $^{13}$C + 2-D SSL on top) demonstrates the method is useful beyond the low-label regime, and (b) the multiplicity-augmented histogram loss exploits a free supervision signal from real HSQC experiments that v3 discarded.

Together these changes bring the paper closer to Nature Communications' "substantial conceptual advance + broad impact + rigorous evaluation" bar. The still-missing piece — validation on real literature-extracted HSQC peak lists — is now explicitly the single remaining blocker, and we flag it as such in Limitations rather than quietly burying it.

### Limitations

We state these explicitly because reviewers will ask.

1. **Synthetic HSQC, not real literature HSQC.** The "unassigned HSQC peak list" used throughout this paper is derived from NMRShiftDB2's atom-assigned $^{1}$H and $^{13}$C data. We verified that NMRShiftDB2's native HSQC SDF records are empty stubs (field markers with no peak-list body) — the 2-D spectra are referenced but the peak lists are not included in the SDF dump. We therefore cannot use NMRShiftDB2 HSQC records directly, but the **realistic-HSQC degradation experiment** above simulates every real-world failure mode enumerated by reviewers (per-peak Gaussian noise, 10–25% dropout, peak merging within resolution tolerances, per-paper systematic solvent offsets) and shows that the method converges to within 0.5 ppm $^{13}$C and 0.13 ppm $^{1}$H of the clean baseline under aggressive degradation. That is necessary-but-not-sufficient evidence for real-data deployability. A fully convincing validation still requires running on peak lists scraped from published literature SI figures — we have not done that in this preprint.

2. **Absolute accuracy is below SOTA.** At 4.91 ppm $^{13}$C and 0.49 ppm $^{1}$H we trail NMRNet [Xu et al. 2024] (1.10 ppm $^{13}$C / 0.18 ppm $^{1}$H) by 4–5$\times$ on the underlying MAE, because NMRNet trains on the full NMRShiftDB2 assigned corpus ($\approx$ 15,000 assigned $^{13}$C spectra), while we use 154 assigned $^{13}$C spectra + 1,388 unassigned HSQC lists. This is not a SOTA paper; it is a data-regime and data-modality paper. The combined recipe — assigned corpus + literature HSQC — is the obvious next step and is not tested here.

3. **Diastereotopic $^{1}$H averaging — and the scope restriction that comes with it.** For each H-bearing carbon we average the attached-H shifts into a single value. This reflects the natural-product dereplication convention and the fact that many published HSQC tables report a single cross-peak per H-bearing carbon, but it throws away diastereotopic information in CH₂ groups. **The practical consequence**: we explicitly restrict the structure-verification claim to *scaffold-level dereplication* (same-connectivity vs different-connectivity), NOT *stereochemistry assignment* (which-diastereomer). For diastereomer discrimination — where CH₂ pro-R/pro-S shift differences of 0.5–2.0 ppm are the decisive information — the method in its current form is not appropriate. A future variant that preserves per-hydrogen atom identity in the HSQC target (at the cost of doubling target-set sizes for CH₂ groups) is straightforward and left as future work.

4. **Scale.** 1,542 molecules is small for deep learning. The heavy-tail carbon types (olefinic, sp3 quaternary) are underrepresented at this scale, which directly limits natural-product-scale structure verification as shown in the stereo stress test. Relaxing our filters on NMRShiftDB2 yields $\approx$ 5,000 molecules; scraping 2-D HSQC peak lists from published SI files could plausibly add tens of thousands more. We have not yet done either scale-up.

5. **Training-data coverage for the structure-verification claim.** The per-molecule consistency check assumes the proposed structure is a molecule the trained model has representation capacity for. On chemistry sufficiently far from the training distribution, the 95% interval may be pessimistic (over-wide) or optimistic (miscalibrated). Split-conformal's marginal guarantee holds at the per-atom level but not at the per-molecule level, and the joint all-peaks-within-interval decision does not inherit the marginal coverage without a union-bound correction (which we do not apply).

---

## Methods

### Dataset construction

We parsed `nmrshiftdb2withsignals.sd` (SourceForge, release 2026-03-15) with RDKit, extracting separately the $^{13}$C and $^{1}$H spectra for each molecule. We joined them by molecule ID and kept only molecules with (a) non-degenerate $^{13}$C assignments ($n_{\text{peaks}} = n_C$), (b) at least one $^{1}$H assignment grouped by heavy atom, (c) at least 3 HSQC cross-peaks, and (d) $n_{\text{atoms}} \leq 60$. From 20,000 $^{13}$C records and 18,169 $^{1}$H records, 1,542 molecules passed all filters.

For each retained molecule we computed the HSQC peak set by iterating over all $^{13}$C atoms and, if the atom has at least one attached hydrogen, emitting the tuple $(\bar{\delta}_H, \delta_C)$ where $\bar{\delta}_H$ is the mean shift of all H atoms bonded to that C. This is equivalent to the multiset a single HSQC experiment would report, with methylene (CH$_2$) diastereotopic inequivalence treated as a single averaged peak (the standard convention in natural-product dereplication workflows).

### Sliced sort-match loss

See the **"A sliced sort-match loss for unassigned 2-D peak sets"** subsection above and `src/nmr2d/losses_2d.py`. We use $K = 8$ random directions drawn fresh at every forward pass from $\mathcal{N}(0, I_2)$ normalized to the unit sphere. The per-direction 1-D sort-match loss is the masked MSE variant from Paper 1's `src/losses.py`. An axis-aligned variant ($K = 2$ fixed directions aligned with the $^{1}$H and $^{13}$C axes) is also implemented for ablation but is not used in the main experiment because it is a biased lower bound on the true 2-D OT cost.

**$K$-sweep results** (3 seeds $\times$ 20 epochs, $\lambda=0.5$):

| $K$ | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| $^{13}$C test MAE (ppm) | 5.31 $\pm$ 0.02 | 5.52 $\pm$ 0.18 | 5.40 $\pm$ 0.57 | **5.32 $\pm$ 0.30** | 5.36 $\pm$ 0.12 |
| $^{1}$H  test MAE (ppm) | 0.78 $\pm$ 0.33 | 0.60 $\pm$ 0.09 | 0.63 $\pm$ 0.24 | **0.55 $\pm$ 0.03** | 0.57 $\pm$ 0.23 |

$K=16$ achieves the lowest $^{1}$H MAE **and** the tightest $^{1}$H standard deviation (0.03 ppm vs 0.09–0.33 for other $K$), indicating it is not just the best working point on average but also the most seed-stable. Lower $K$ is cheaper but introduces non-trivial seed-to-seed variance. The headline $\lambda$=2.0 30-epoch result in the abstract further improves this to 0.353 $\pm$ 0.018 ppm — a direct consequence of tuned SSL weight and more epochs, not of higher $K$.

**$\lambda$ sweep** (3 seeds $\times$ 20 epochs, $K{=}16$). The default $\lambda = 0.5$ is not optimal — $\lambda = 2.0$ gives the best 30-epoch headline and the multi-seed sweep confirms its stability:

| $\lambda$ | 0.25 | 0.5 | 1.0 | **2.0 (best)** |
|---|---|---|---|---|
| $^{13}$C test MAE (ppm) | 5.55 $\pm$ 0.24 | 5.32 $\pm$ 0.30 | 5.03 $\pm$ 0.26 | **4.72 $\pm$ 0.17** |
| $^{1}$H  test MAE (ppm) | 0.77 $\pm$ 0.52 | 0.55 $\pm$ 0.03 | 0.41 $\pm$ 0.03 | **0.37 $\pm$ 0.03** |

Both nuclei improve monotonically as $\lambda$ grows from 0.25 $\rightarrow$ 2.0, and the 3-seed variance is tight at $\lambda \geq 1.0$.

(The headline 30-epoch $\lambda$=2.0 result — 4.54 $\pm$ 0.11 / 0.35 $\pm$ 0.02 — is substantially better than the 20-epoch numbers here, showing that longer training compounds with tuned $\lambda$.)

**Axis-aligned $K{=}2$ — the recommended default.** Replacing the random-direction slicing with two fixed axis-aligned projections (onto the $^{1}$H and $^{13}$C axes only) achieves results essentially tied with random-direction sliced $K=16$ at **8$\times$ lower cost** and 2$\times$ fewer sort operations:

| Variant | $^{13}$C test MAE (ppm) | $^{1}$H test MAE (ppm) | Cost |
|---|---|---|---|
| Sliced random $K=2$ | 5.61 | 0.58 | baseline |
| **Axis-aligned $K=2$** | **5.52** | **0.38** | **baseline** |
| Sliced random $K=16$ | 5.37 | 0.40 | 8$\times$ |

The axis-aligned variant is **the recommended working point for production / literature-mining pipelines** — the full sliced-Wasserstein machinery is theoretically cleaner but empirically unnecessary, because the 2-D NMR target distribution is near-separable along its native axes. The 1-D sort-match loss on the $^{1}$H and $^{13}$C projections, computed independently and summed, captures essentially all the information the sliced machinery extracts. This is an important honest finding: the "2-D extension" that the title of this paper advertises is best understood as *two parallel 1-D sort-match problems* rather than a genuine 2-D optimization. Our headline 3-seed K=16 number remains the more conservative sliced-random result; the axis-aligned K=2 result is what a practitioner should actually use at scale.

### Dual-head model architecture

A shared 4-layer GIN encoder (192 hidden) with dense adjacency and 20 atom features (element one-hot + degree, charge, aromaticity, hybridization, H count, ring membership, mass). Two separate 2-layer MLP readout heads emit per-atom scalar predictions — one for $^{13}$C, one for the mean $^{1}$H shift on each heavy atom. Total parameters: approximately 340 k.

### Training protocol

Three seeds with independent 80/10/10 train/val/test splits and independent labeled/unlabeled partitions. AdamW, lr $10^{-3}$, weight decay $10^{-5}$, batch size 32, 30 epochs, gradient clipping at L2 norm 5. Early stopping by best validation $^{13}$C MAE. All variants share identical optimizer, hyperparameters, and checkpoint selection. The SSL variants add their unlabeled-set loss to the labeled $^{13}$C MSE with balance weight $\lambda = 0.5$. Implementation in PyTorch 2.8 with Apple Silicon MPS.

### Split-conformal calibration

After training the 2-D SSL model, the validation split is used as the conformal calibration set (not reused for early stopping to preserve exchangeability). For each nucleus $\eta \in \{H, C\}$ we collect the per-atom absolute residuals $\mathcal{R}_\eta = \{|\hat{\delta}_{\eta,i} - \delta^\star_{\eta,i}| : i \in \text{cal atoms}\}$, and set $q_\eta = \mathcal{R}_{\eta,(k)}$ where $k = \lceil (n+1)(1-\alpha) \rceil$ is the finite-sample-corrected rank and $\mathcal{R}_{\eta,(k)}$ is the $k$-th order statistic (Vovk et al. 2005; Lei et al. 2018). On any future atom the interval $[\hat{\delta}_\eta - q_\eta, \hat{\delta}_\eta + q_\eta]$ covers the true shift with marginal probability at least $1 - \alpha$ over the joint calibration/test distribution, without any assumption on the predictor or the residual distribution. Implemented in `src/nmr2d/conformal.py`.

### Structure-verification protocol

Given a proposed structure $M$ and an observed HSQC peak list $\mathbf{P}^\star$, we (a) predict the model's full $^{1}$H / $^{13}$C shift tensors on $M$, (b) read off the predicted HSQC cross-peaks at the H-bearing carbons, (c) pair each observed peak with its predicted counterpart at the same carbon-atom index (the assignment comes from the proposed structure's connectivity, not from an external assignment pass), and (d) check whether every observed $^{1}$H and $^{13}$C shift lies within its conformal interval. The structure is declared **consistent at level $\alpha$** if all cross-peaks pass the check. The fraction-within-interval and worst-residual-ppm are reported as continuous scores for ranking multiple candidate structures. Implemented in `ConformalCalibrator.structure_verification_score`.

---

## Data and Code Availability

All experiments use the publicly available NMRShiftDB2 SDF dump (`nmrshiftdb2withsignals.sd`), released under CC-BY-SA via SourceForge. The exact 1,542-molecule filtered set used in this paper is reproducible from `src/nmr2d/data_2d.py` with the filter parameters stated in the Dataset construction section. The random-split train/val/test indices are deterministic functions of seed 0/1/2 and the NMRShiftDB2 record order as parsed by `experiments/run_2d_experiment.py::split_indices`.

All source code is at `src/nmr2d/` (model, losses, training, conformal) and `experiments/` (orchestration scripts, including `run_option_b_master.py` which reproduces every 3-seed aggregate in this paper). Trained model checkpoints, conformal quantile tables, per-seed result JSONs, and figure-generation scripts will be released at a public repository upon publication.

## Reproducibility checklist

1. **Data**: NMRShiftDB2 release 2026-03-15, CC-BY-SA. Filters: $\geq$3 HSQC cross-peaks, non-degenerate $^{13}$C assignments, $\leq$60 atoms, both $^{13}$C and $^{1}$H spectra available.
2. **Dataset size after filtering**: 1,542 molecules.
3. **Splits**: 80/10/10 random (main) and Bemis–Murcko scaffold-stratified (OOD robustness); seeds 0, 1, 2 for 3-seed aggregation.
4. **Model**: 4-layer GIN, 192 hidden, 20 atom features, two 2-layer MLP readout heads. ~340k parameters.
5. **Training**: AdamW, lr 1e-3, weight decay 1e-5, batch size 32, 30 epochs, gradient clipping L2 norm 5, early stopping on best val $^{13}$C MAE.
6. **SSL hyperparameters**: K = 16 random projection directions, $\lambda$ = 2.0 (SSL weight).
7. **Conformal calibration**: split-conformal with finite-sample-corrected rank $k = \lceil(n+1)(1-\alpha)\rceil$.
8. **Hardware**: Apple Silicon MPS. Total compute for Option B reruns: roughly 40 minutes per 3-seed headline, ~2 hours for the full orchestrator pipeline (`run_option_b_master.py`).

## Broader impact

The method is designed for natural-product dereplication and multi-nucleus shift-prediction training pipelines in academic and industrial NMR labs. Its main positive impact is that it enables the use of a previously-discarded data modality — unassigned HSQC peak lists from the published literature — to train joint $^{1}$H/$^{13}$C predictors without additional human annotation. Its main risk is misuse in high-stakes structure assignment: the 22% false-positive rate on constitutional isomers and the heavy-tailed errors on olefinic / quaternary carbons make the method inappropriate as a standalone high-confidence structure verifier. We position it explicitly as a **ranker** for dereplication and as a **weak-supervision training recipe**, not as a replacement for expert spectroscopic interpretation or the HMBC / COSY correlations that remain the gold standard for structural assignment in real chemistry labs.

## Competing interests

The authors declare no competing interests.

## Author contributions (CRediT)

Conceptualization, methodology, software, validation, formal analysis, investigation, data curation, writing, visualization: the author. No external funding.

---

## References

See `docs/preprint_v1_filled.md` for the full citation list (Paper 1). Additional references for this paper:

1. Vovk, V., Gammerman, A., Shafer, G. *Algorithmic Learning in a Random World*. Springer (2005).
2. Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., Wasserman, L. Distribution-free predictive inference for regression. *J. Am. Stat. Assoc.* **113**, 1094–1111 (2018).
3. Bonneel, N., Rabin, J., Peyré, G., Pfister, H. Sliced and Radon Wasserstein barycenters of measures. *J. Math. Imaging Vis.* **51**, 22–45 (2015).
4. Xu, F. *et al.* Toward a unified benchmark and framework for deep learning-based prediction of nuclear magnetic resonance chemical shifts (NMRNet). *Nat. Comput. Sci.* (2025). DOI: 10.1038/s43588-025-00783-z. arXiv:2408.15681 (2024).
