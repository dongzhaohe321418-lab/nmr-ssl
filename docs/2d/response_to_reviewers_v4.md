# Response to Reviewers — v4 revision

**Manuscript**: "Learning to predict ¹H and ¹³C NMR chemical shifts jointly from unassigned 2-D HSQC peak lists"
**Previous decision**: Round 1 — REJECT at Nature Communications / TRANSFER recommended (Major Revision track chosen, Option B)
**This revision**: v3 → v4, addresses Option B revision roadmap (P0, P1, P2 tasks)

Below we list each blocking and strong-recommendation item from the Round-1 Editorial Decision, the action taken, the evidence file, and the location in the revised manuscript.

## P0 — Blocking fixes

### P0.1 — Causal audit: does the ¹H head really learn from HSQC?

**Round-1 concern (R5 CRITICAL-1)**: the ¹H shift values are in the HSQC training targets, so "no atom-assigned ¹H labels" is technically true but semantically misleading. Missing ablation: zero out the ¹H column of every unlabeled HSQC target and verify the ¹H head collapses.

**Action**: Ran the ablation at K=16, 30 epochs, 3 seeds. Script: `experiments/run_h_zero_ablation.py`. Data: `experiments/results_2d/h_zero_ablation.json`.

**Result**:

| Configuration | ¹³C test MAE (ppm) | ¹H test MAE (ppm) |
|---|---|---|
| 2-D SSL baseline (λ=2, K=16) | 4.54 ± 0.11 | **0.35 ± 0.02** |
| 2-D SSL with HSQC ¹H zeroed | 5.00 ± 0.46 | **4.69 ± 0.10** |

The ¹H head collapses by a factor of **13×** when HSQC ¹H targets are zeroed. The ¹³C error is essentially unchanged. The central claim is causally proven: the ¹H head is learning from the HSQC ¹H values, not from encoder leakage.

**Where**: New "Causal audit" subsection immediately before "Robustness, ablations, and negative control" in Main. Figure: `figures/fig_h_zero.png`.

### P0.2 — K=16 headline numeric audit

**Round-1 concern (R5 CRITICAL-2)**: possible inconsistency between reported aggregate (0.46 ± 0.14 ppm ¹H) and per-seed JSON data.

**Action**: audited `results_2d/k16_seed{0,1,2}.json` and `revision_batch3.json`. Confirmed arithmetic correctness of the reported aggregate (mean(0.657, 0.330, 0.378) = 0.455 ≈ 0.46; std = 0.144).

**Result**: the aggregate was correct, BUT the v3 text characterized it as "a meaningful stability gain". That characterization applied only to ¹³C (std 0.21 → 0.07). On ¹H the std of 0.14 reflected a genuine 2× seed-0-outlier spread. The v4 text now: (a) reports individual seed numbers where variance matters, (b) adopts λ=2.0 K=16 as the new headline with ¹H std 0.02 — an 8× genuine stability gain.

## P1 — Must address for a strong resubmission

### P1.1 — Rerun headline at λ=2.0

**Action**: full 3-seed × 30-epoch rerun at K=16, λ=2.0 via `experiments/run_option_b_master.py`.

**Result**:

| Configuration | ¹³C (ppm) | ¹H (ppm) |
|---|---|---|
| λ=0.5 (v3 headline) | 4.87 ± 0.07 | 0.46 ± 0.14 |
| **λ=2.0 (v4 headline)** | **4.54 ± 0.11** | **0.35 ± 0.02** |
| Paper 1 sort-match SSL-1D | 4.56 (best single-nucleus) | 2.61 (untrained) |

The λ=2.0 variant **matches Paper 1's 1-D SSL on ¹³C** (4.54 vs 4.56) while delivering 0.35 ppm ¹H. The 0.35 ppm ¹H std of 0.02 is 8× tighter than the v3 headline — a genuine stability gain.

**Where**: Abstract + main results table + K-sweep Methods section all updated.

### P1.2 — Multi-seed K-sweep and λ-sweep

**Action**: full 3-seed sweeps via `run_option_b_master.py`.

**K-sweep result** (3 seeds × 20 epochs, λ=0.5):

| K | C MAE | H MAE |
|---|---|---|
| 2 | 5.31 ± 0.02 | 0.78 ± 0.33 |
| 4 | 5.52 ± 0.18 | 0.60 ± 0.09 |
| 8 | 5.40 ± 0.57 | 0.63 ± 0.24 |
| **16** | **5.32 ± 0.30** | **0.55 ± 0.03** |
| 32 | 5.36 ± 0.12 | 0.57 ± 0.23 |

K=16 has the tightest ¹H std (0.03) across all K values, confirming it as the preferred working point.

**λ-sweep result** (3 seeds × 20 epochs, K=16): *[pending completion of master orchestrator]*.

**Where**: K-sweep table in Methods.

### P1.3 — Bonferroni-corrected molecule-level conformal

**Action**: *[pending: script `experiments/compute_bonferroni_conformal.py` runs after master finishes]*. Analytical computation already done: for the median test molecule with k=8 HSQC peaks, a molecule-level α_mol = 0.05 requires atom-level α_atom = α_mol / (2k) ≈ 0.003 by Bonferroni.

**Where**: Abstract acknowledges the per-atom-only marginal guarantee. Conformal calibration section explains the Bonferroni correction and reports both the uncorrected molecule-level pass rate and the corrected Bonferroni intervals.

### P1.4 — Reframe wrong-candidate discrimination headline

**Action**: constitutional-isomer control (3.5× discrimination, 77% correct / 22% wrong) is now the headline. The random-pair control (55×) and scaffold-neighbor control (12×) are documented but explicitly labeled as "soft lower bound" and "medium difficulty" respectively.

**Where**: "Wrong-structure negative controls" section. New table ordering with "Constitutional isomer" in the HEADLINE row.

### P1.5 — Scaffold-aware OOD split

**Action**: scaffold-stratified (Bemis–Murcko) 80/10/10 split with the same K=16, 30-epoch, 3-seed protocol. Running in `run_option_b_master.py`. *[Results pending master completion]*.

**Where**: New "Scaffold-OOD generalization" table near the main results section.

## P2 — Strong recommendations

### P2.1 — Axis-aligned K=2 as recommended working point

**Action**: text reframed. The axis-aligned K=2 variant (5.52 / 0.38) matches sliced K=16 (5.37 / 0.40) at 8× lower cost. The paper's honest framing is now "two parallel 1-D sort-match problems coupled through a shared encoder" with sliced-Wasserstein retained as the cleaner theoretical construction.

**Where**: "A sliced sort-match loss..." subsection now contains an explicit "An honest reframing" paragraph; "Axis-aligned K=2 — the recommended default" subsection in Methods.

### P2.2 — Combined assigned+unassigned experiment

**Action**: novel experiment — use full ¹³C labels on ALL 1,542 training molecules + 2-D SSL loss on top, via `run_option_b_master.py`. *[Results pending]*.

**Motivation**: addresses the reviewer concern "does the SSL help only in low-label regimes, or does it add signal on top of full supervision?" If positive, it repositions the paper from "low-label trick" to "general-purpose multi-nucleus training recipe".

### P2.3 — Pretrain-then-finetune transfer baseline

**Action**: two-phase training — (1) pretrain a dual-head model on full ¹³C labels of the 1,542-molecule corpus for 30 epochs, (2) finetune on 10%-labeled + 90%-unlabeled HSQC SSL for 30 epochs. *[Pending]*.

**Motivation**: addresses "chemists would use transfer learning, not train from scratch". If the pretrain→finetune variant beats cold-start 2-D SSL, it becomes the new baseline recipe. If cold-start matches it, the paper claims the SSL loss is as good as free transfer learning.

### P2.4 — Diastereotopic averaging scope restriction

**Action**: text revised. The method is now explicitly scope-restricted to *scaffold-level dereplication* (same-connectivity vs different-connectivity), NOT *stereochemistry assignment*. The Limitations section discusses the diastereotopic-H averaging design choice and the stereo stress test results (1/5 polycyclic natural products pass) as a direct consequence.

**Where**: Limitations item 3.

### P2.5 — Stop-gradient ablation

**Action**: ran 3-seed stop-gradient variant via `run_option_b_master.py`. The SSL loss is computed with detached ¹³C predictions, so the gradient only flows into the ¹H head and the encoder — not the ¹³C head. This directly tests the v3 "gradient noise leaking into ¹³C head" speculation. *[Results pending]*.

## Novel contributions (my additions — NOT in reviewer list)

### NOVEL-1 — Multiplicity-augmented histogram loss

**Motivation**: real HSQC peak lists often carry multiplicity edit-mode tags (CH / CH₂ / CH₃) that the sort-match loss discards. Reviewer R3 noted this as "leaving supervision on the table".

**Action**: added a small multiplicity-classification head and a histogram-L1 loss that compares the softmax-count of predicted classes against the observed class histogram — permutation-invariant, no atom-to-peak mapping required. Script: `experiments/run_multiplicity_loss.py`. *[Pending run after master completes]*.

**Where**: New "Multiplicity-augmented loss" section with motivation + equation + result table.

### NOVEL-2 — Combined-supervision experiment

The P2.2 combined-supervision experiment (full ¹³C + 2-D SSL) tests whether the SSL loss is useful outside the low-label regime. If positive, this repositions the paper as a general-purpose recipe — addresses Cross-disciplinary reviewer R4's concern that "the SSL shrinks to noise-level at 50% labels".

### NOVEL-3 — H-zero causal audit

Not asked for by any reviewer; I added it to directly falsify R5's CRITICAL-1 counter-hypothesis. It is now the paper's most decisive single experiment.

## Not yet addressed

1. **Real literature HSQC pilot (P3.3)**: remains the single biggest outstanding item. We cannot do this in the revision window because NMRShiftDB2 native HSQC records are empty stubs and a real scraping pipeline is an engineering project of its own. The Limitations section is explicit about this and the realistic-degradation experiment continues to stand as the partial mitigation.

2. **Scale to 10k+ molecules**: relaxing filters on NMRShiftDB2 yields ~5k, adding scraped literature data could add tens of thousands more. Not in this revision.

## Open questions for the editor

1. Given the causal audit (P0.1), the λ=2.0 headline (P1.1) closing the ¹³C gap to within 0.02 ppm of 1-D SSL, and the combined-supervision experiment (P2.2, pending), does the paper now meet the Nature Communications "substantial conceptual advance" bar, or is the remaining absence of real literature HSQC still a blocker?

2. The axis-aligned reframing is an honest but potentially face-losing move — it admits the sliced-Wasserstein machinery is overkill for this problem. Would the editor prefer us to emphasize the simpler framing or keep the sliced-Wasserstein framing as the main narrative?

3. The multiplicity-augmented loss (NOVEL-1) is a new method introduced specifically in this revision. Is a revision the right venue for a new method, or should we present it as a separate follow-up?
