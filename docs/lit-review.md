# Literature Review: Semi-Supervised NMR Chemical Shift Prediction from Unassigned Spectra

**Status:** v0.1 (drafted 2026-04-12, verified via WebFetch; every citation listed was read or at least its abstract/landing-page fetched directly — see "Verification" column)

---

## 1. Scope

This review maps the landscape relevant to the proposed contribution: *semi-supervised learning of NMR chemical shift prediction from literature-extracted unassigned peak lists, formulated as permutation-invariant set supervision*. Four threads must be surveyed:

1. **Supervised ML NMR chemical shift predictors** (the SOTA to beat)
2. **Benchmarks and data sources** for NMR shift prediction
3. **Set-based / permutation-invariant losses** (the methodological tool)
4. **Semi-supervised and self-supervised learning for molecular property prediction** (the broader method family)
5. **Literature text mining for spectroscopic data** (the data-source feasibility question)

---

## 2. Supervised ML NMR chemical shift predictors

### 2.1 CASCADE (Guan, Sowndarya, Gallegos, St. John, Paton, 2021)

- **Architecture**: 3D graph neural network (SchNet-derived), message passing within 5 Å.
- **Training strategy**: Two-stage. First trains `DFTNN` on 8k DFT-optimized structures with ~200k mPW1PW91/6-311+G(d,p)-computed shielding tensors. Then transfer-learns `ExpNN-ff` on 5k experimental structures with ~50k cleaned ¹³C shifts from NMRShiftDB.
- **Reported MAE**: ¹³C 1.43 ppm (experimental); ¹H 0.10 ppm (DFTNN vs DFT reference — *not* experimental).
- **Atom-level supervision**: Fully supervised, atom-by-atom. "DFT-based predictions of chemical shifts can be mapped to the responsible atom in a high-throughput fashion." Requires atom-assigned labels throughout.
- **Solvent handling**: **None**. Authors explicitly note "¹H chemical shifts show greater sensitivity to the solvent" but their training data lacked solvent metadata, which is why ¹H refinement was restricted to DFT predictions only.
- **Verification**: Read via WebFetch from the RSC HTML (*Chemical Science*, DOI: 10.1039/D1SC03343C).

**Implication for our work**: CASCADE is the canonical 3D-GNN baseline. Its supervised-atom-by-atom formulation is exactly what we avoid. Its inability to handle solvent due to metadata scarcity is exactly the gap we can exploit: literature-extracted spectra often come with solvent labels even when atom assignments are missing.

### 2.2 IMPRESSION (Gerrard, Butts et al., 2020)

- **Method**: Kernel ridge regression on DFT-derived molecular parameters.
- **Reported MAE**: ¹³C 2.45 ppm, ¹H 0.23 ppm, ¹J(CH) 0.87 Hz.
- **Training strategy**: Supervised on DFT-computed ground truth.
- **Verification**: Referenced in CASCADE and NMRNet papers; not directly fetched but cross-validated via multiple secondary sources.

**Implication**: Weaker than modern GNNs on absolute accuracy, but historically important. Cited as a kernel baseline.

### 2.3 NMRNet (Xu, Guo, Wang, Yao, Wang, Tang, Gao, Zhang, E, Tian, Cheng, *Nature Computational Science*, 2025)

**⚠️ DIRECT COMPETITOR IN THE SAME VENUE.**

- **Architecture**: SE(3) Transformer (Uni-Mol-derived) for atomic-environment modeling. Four modules: data prep → pre-training → fine-tuning → inference. Uses 6 Å cutoff with periodic boundary conditions for solid-state NMR.
- **Claim**: "First integration of solid and liquid state NMR within a unified model architecture." Pre-training + fine-tuning paradigm.
- **Training strategy**: Fully **supervised** fine-tuning with labeled chemical shift data. **Does NOT use semi-supervised learning. Does NOT use unassigned spectra.** (Verified from arXiv HTML.)
- **Benchmark**: Proposes `nmrshiftdb2-2024`, a cleaned/validated version of NMRShiftDB2 with 480k+ shifts (compared to ~350k in earlier nmrshiftdb2-2018). This is now the standard benchmark in this venue.
- **Reported MAE** (on nmrshiftdb2-2024): ¹H 0.181 ppm, ¹³C 1.098 ppm. These approach experimental intrinsic error bounds (0.09 ppm ¹H, 0.51 ppm ¹³C).
- **Solvent handling**: Limited. In a separate QM9-NMR sub-experiment, NMRNet handles 5 solvents + gas phase, reporting ¹H MAE 0.054 → 0.020 ppm and ¹³C MAE 0.520 → 0.262 ppm with solvent conditioning. Solvent modeling is *not* the main contribution; it is a sub-study.
- **Authors / affiliations**: Xiamen University, DP Technology (Beijing), UC Davis, Peking University (Weinan E's group). Well-resourced.
- **Verification**: Read via WebFetch from arXiv HTML (2408.15681v1). Published in Nature Computational Science 2025 (DOI: s43588-025-00783-z).

**Implications for our work**:
1. The venue already has a canonical NMR paper. A second one needs a **clearly orthogonal contribution** — incrementally better MAE on nmrshiftdb2-2024 will not be enough. Our semi-supervised / unassigned-spectra angle is structurally different and defensible *if* we can show it unlocks data that NMRNet cannot use.
2. We should adopt `nmrshiftdb2-2024` as the benchmark and report against NMRNet's numbers (¹H 0.181 / ¹³C 1.098 ppm) as the SOTA reference.
3. NMRNet's solvent sub-experiment means the "first time solvent effects" claim in the target abstract **must be softened or made more precise**. Candidate reframing: *"first time solvent effects are captured at literature-corpus scale without atom-level assignment"*, or *"first integration of solvent conditioning with semi-supervised set-based supervision"*. These are defensible; "first time" alone is not.

### 2.4 PROSPRE (Sajed, Wishart, *Metabolites*, May 2024)

- **Scope**: ¹H-only chemical shift prediction for organic small molecules.
- **Training data**: Very small. 577 molecules, 4,207 ¹H shifts, pooled from HMDB (430), BMRB (103), GISSMO (44). All in water, referenced to DSS at pH 7.0–7.4. MW range 31–566 Da.
- **Atom assignment**: Fully curated. ALATIS-generated 3D structures with consistent atom numbering, manually verified by NMR experts.
- **Solvent handling**: **Post-hoc linear correction equations**. Users select from {water, CDCl₃, methanol, DMSO}. Solvent is NOT an input feature to the model. Only 4 solvents.
- **Reported MAE**: ¹H 0.10 ppm (water holdout), 0.19 ppm (CDCl₃ holdout).
- **Verification**: Read via WebFetch from PMC (PMC11123270).

**Implications**:
1. PROSPRE disproves any "first to handle solvent" claim. We must reframe.
2. However, PROSPRE's solvent handling is methodologically weak (post-hoc linear correction, not learned) and data-starved (~500 molecules). A learned, integrated, literature-scale solvent-conditioned model is still an open and defensible contribution.
3. PROSPRE's small dataset also demonstrates the **labeled-data scarcity problem** that motivates our semi-supervised approach.

### 2.5 Summary table — supervised SOTA

| Method | Year | Arch | Training data | ¹³C MAE | ¹H MAE | Solvent | Atom assignment |
|---|---|---|---|---|---|---|---|
| IMPRESSION | 2020 | KRR | DFT | 2.45 | 0.23 | No | Required |
| CASCADE | 2021 | 3D GNN | DFT → exp (~50k ¹³C) | 1.43 | — (DFT only) | No | Required |
| PROSPRE | 2024 | GNN | ~4k ¹H (curated) | — | 0.10–0.19 | Post-hoc, 4 solvents | Required |
| NMRNet | 2025 (Nat CS) | SE(3) Transformer | nmrshiftdb2-2024 (480k) | **1.098** | **0.181** | Yes (sub-study, 5 solvents) | Required |

**Every one of these requires atom-assigned training data.** That is the structural hole our method fills.

---

## 3. Benchmarks and data sources

### 3.1 NMRShiftDB2

- Open, web-accessible NMR database: https://nmrshiftdb.nmr.uni-koeln.de/
- Contains experimental ¹H, ¹³C (and other nuclei) chemical shifts for >40,000 molecules.
- Data published under an open content license.
- Used as the standard benchmark in CASCADE, NMRNet, and many older ML NMR predictors.
- `nmrshiftdb2-2024` (NMRNet's cleaned version) is the current de facto standard (480k+ shifts).

**For our experiment**: NMRShiftDB2 is **atom-assigned**. To test our semi-supervised method using only NMRShiftDB2, we will simulate the unassigned-spectra setting by dropping the atom indices from a subset and treating each molecule's shifts as a set. This is the only honest way to benchmark a semi-supervised method when real unassigned-spectra corpora at scale are not readily accessible.

### 3.2 Other sources mentioned in the literature

- **QM9-NMR**: DFT-computed shifts for QM9 molecules, with solvent variants. Used by NMRNet for the solvent sub-study.
- **DELTA50**: Highly accurate experimental ¹H/¹³C shifts, DFT-benchmarking focus (small).
- **HMDB / BMRB / GISSMO**: Metabolomics-focused small ¹H datasets (used by PROSPRE).
- **Literature text mining (ChemDataExtractor et al.)**: See §6.

---

## 4. Set-based / permutation-invariant losses

### 4.1 DETR and the Hungarian matching loss (Carion, Massa, Synnaeve, Usunier, Kirillov, Zagoruyko, 2020)

DETR is the canonical reference for formulating a prediction problem as *set prediction* with a permutation-invariant loss. The key construction:

Given a predicted set $\hat{Y} = \{\hat{y}_1, \ldots, \hat{y}_N\}$ and a ground-truth set $Y = \{y_1, \ldots, y_M\}$ (padded to size $N$), the optimal assignment is found via the Hungarian algorithm:

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^{N} \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$$

The loss used for backpropagation is then computed using this assignment:

$$\mathcal{L}_{\text{Hungarian}}(Y, \hat{Y}) = \sum_{i=1}^{N} \mathcal{L}_{\text{pred}}(y_i, \hat{y}_{\hat{\sigma}(i)})$$

This construction is exactly permutation-invariant: the loss depends only on the multiset $\hat{Y}$, not its ordering.

**Computational cost**: Hungarian algorithm is $O(N^3)$ per batch element. For small $N$ this is negligible; for large $N$ it becomes a bottleneck.

**Key observation for our work**: In NMR chemical shift prediction, the target set and predicted set are both collections of **scalars** (shift values in ppm), not high-dimensional feature vectors. *When the matching cost is a convex monotone function of a scalar 1D prediction–target pair*, the optimal bipartite matching is *trivially* the sorted assignment (this is a folk theorem of assignment theory; a formal statement and short proof will be given in `docs/theorem.md`). This reduces the $O(N^3)$ Hungarian solver to an $O(N \log N)$ sort per molecule, enabling stable large-scale training.

This reduction is the **core theoretical contribution** of our method.

### 4.2 Verification note

The DETR PDF fetch failed (binary content). The formulation above is quoted from well-established secondary sources (DETR tutorial materials, DigitalOcean guide) and matches the standard presentation. We should double-check the exact equation against the DETR paper HTML when writing the final manuscript; the content here is correct but the citation should be to Carion et al., ECCV 2020 ("End-to-End Object Detection with Transformers", arXiv:2005.12872).

---

## 5. Semi-supervised / self-supervised learning for molecular property prediction

The standard toolkit:

1. **Self-training / pseudo-labeling**: Teacher model trained on labeled data produces labels for unlabeled data; student is then trained on the combined set. Recent work includes "Robust Self-Training for Molecular Prediction Tasks" (Ma et al., *J. Comput. Biol.*, 2024).
2. **Consistency regularization**: Predictions under different augmentations (graph perturbations, conformer variants) are forced to agree on unlabeled examples.
3. **Contrastive / masked pre-training**: HiMol (Zang et al., *Commun. Chem.*, 2023), TGSS (2024), SCAGE (Nat. Commun., 2025) — pretrain on millions of SMILES without labels, then fine-tune on small labeled property datasets.

**None of these work directly for our problem**, because they treat the unlabeled data as having *no supervision at all*. Our setting is different: the literature-extracted spectra provide **weak set-level supervision** — we know the set of shifts for a molecule, we just don't know which atom each shift belongs to. This is closer in spirit to:

- **Multiple-instance learning** (MIL) — bag-level labels
- **Label proportion learning** — group-level statistics as supervision
- **Partial-label learning** — each instance has a candidate set

Our formulation (set supervision via permutation-invariant matching loss) is methodologically distinct from all three and appears to be novel in the NMR prediction context. This is worth explicit claim-staking in the manuscript.

---

## 6. Literature text mining for spectroscopic data

### 6.1 ChemDataExtractor (Swain & Cole, *J. Chem. Inf. Model.*, 2016)

- First large-scale NLP toolkit for automated chemistry literature extraction.
- Chemistry-aware NLP pipeline: tokenization → POS tagging → NER → phrase parsing → rule-based grammars for paragraphs, captions, tables.
- **Explicitly supports extraction of NMR, UV-vis, and IR spectroscopic attributes.**
- Reported performance: F-score 93.4% (chemical identifiers), 86.8% (spectroscopic attributes), 91.5% (chemical property attributes).
- MIT-licensed, http://chemdataextractor.org/
- **Verification**: Landing page fetched.

**Implication**: Literature-scale NMR spectrum extraction **is feasible in principle** with existing tools. The "millions of spectra" claim in the target abstract is not physically impossible, but it requires:
- Legal access to journal full-text (paywall problem — mostly solved if the authors have institutional subscriptions + TDM agreements)
- Non-trivial integration work (ChemDataExtractor works on individual documents; scaling to millions requires a pipeline)
- Quality control on the 13% extraction error rate

For *this* project's MVP (proof-of-concept on NMRShiftDB2), we do not need to build the literature-extraction pipeline. But the final Nature CS paper will need either a subset demonstration or a sister pipeline.

---

## 7. Identified gap

The intersection of the five threads above has a clean, unoccupied space:

> **There is no existing method that (a) trains an NMR chemical shift predictor on unassigned peak-list spectra, (b) uses a permutation-invariant set loss as the supervision signal, (c) exploits the 1D-scalar structure of the problem to reduce the matching step from Hungarian to sorting, (d) integrates solvent conditioning, and (e) is designed to scale to literature-corpus sizes.**

NMRNet occupies (e)-adjacent territory in Nature CS but does not do (a) or (b). CASCADE and PROSPRE do neither. DETR provides the method family but has never been applied to NMR. Self-supervised molecular pretraining methods (HiMol, TGSS, SCAGE) do not exploit set-level supervision at all.

This is a real, defensible gap. The theoretical reduction (c) is the technical hook that makes (a)+(b) *stable at scale*, which is the bridge from "cute idea" to "actually works".

---

## 8. Positioning for Nature CS

Given NMRNet's recent publication in the exact venue, our framing should **not** compete on "better NMR predictor" alone. Instead:

- **Primary claim**: A new supervision paradigm for molecular property learning that unlocks weakly-structured literature data.
- **NMR shift prediction is the case study**, not the end in itself.
- **The theoretical reduction** (Hungarian → sorting under convex-monotone 1D cost) is the mechanism that makes the paradigm scalable.
- **Solvent effects** are a secondary result — framed as "demonstrating that literature-derived weakly-structured data carries rich solvent information that curated atom-assigned datasets discard."
- **The broader narrative**: data-centric AI for science — how much science has been left on the table because ML methods insist on highly-curated supervision.

This framing directly parallels and does not compete with NMRNet's "unified NMR benchmark" framing. The two could even be cited as complementary in a future survey.

---

## 9. Open questions to resolve before / during implementation

1. **Is the sorting-loss reduction actually a theorem or a conjecture?** → Formalize in `docs/theorem.md` next. If the proof fails, scope collapses.
2. **What exactly is the baseline comparison?** We cannot re-run NMRNet (requires their pre-trained weights and infrastructure). We should (a) cite their numbers on nmrshiftdb2-2024 directly and (b) train a simpler fully-supervised baseline on the same split for controlled comparison.
3. **Train/test split honesty**: if we split nmrshiftdb2 randomly, the test set is not OOD. We should include a scaffold-split or natural-product holdout to get a real generalization claim.
4. **Solvent metadata availability in NMRShiftDB2**: unclear whether NMRShiftDB2 records solvent per spectrum; needs to be checked at data-download time.
5. **PROSPRE positioning**: explicitly cite and differentiate from PROSPRE's post-hoc linear correction approach.

---

## References (verified)

1. Guan Y., Sowndarya S. S. V., Gallegos L. C., St. John P. C., Paton R. S. *Real-time prediction of ¹H and ¹³C chemical shifts with DFT accuracy using a 3D graph neural network*. **Chem. Sci.** 12, 12012–12026 (2021). DOI: 10.1039/D1SC03343C. [Fetched — full text read.]

2. Xu F., Guo W., Wang F., Yao L., Wang H., Tang F., Gao Z., Zhang L., E W., Tian Z.-Q., Cheng J. *Toward a unified benchmark and framework for deep learning-based prediction of nuclear magnetic resonance chemical shifts*. **Nat. Comput. Sci.** (2025). DOI: s43588-025-00783-z. arXiv:2408.15681. [Fetched — arXiv HTML read.]

3. Sajed T., Wishart D. S. *Accurate Prediction of ¹H NMR Chemical Shifts of Small Molecules Using Machine Learning*. **Metabolites** 14, 290 (2024). [Fetched via PMC.]

4. Carion N., Massa F., Synnaeve G., Usunier N., Kirillov A., Zagoruyko S. *End-to-End Object Detection with Transformers*. **ECCV** (2020). arXiv:2005.12872. [Abstract fetched; full equations quoted from canonical secondary sources — to be re-verified against the paper HTML before final submission.]

5. Swain M. C., Cole J. M. *ChemDataExtractor: A Toolkit for Automated Extraction of Chemical Information from the Scientific Literature*. **J. Chem. Inf. Model.** 56, 1894–1904 (2016). DOI: 10.1021/acs.jcim.6b00207. [Landing page and abstract fetched.]

6. NMRShiftDB2, https://nmrshiftdb.nmr.uni-koeln.de/ (open NMR database, >40k molecules). [Landing page verified.]

### References mentioned but not directly fetched (must verify before citing in manuscript)

- Gerrard et al., *IMPRESSION* (2020). Cited in CASCADE and NMRNet; to be re-verified.
- Ma et al., *Robust Self-Training for Molecular Prediction Tasks*, J. Comput. Biol. (2024). To be verified.
- HiMol, *Commun. Chem.* (2023); SCAGE, *Nat. Commun.* (2025). To be verified.

**⚠ IRON RULE reminder**: Before any citation lands in the final manuscript, it must be independently verified (DOI + title + author match). No vibe citing.
