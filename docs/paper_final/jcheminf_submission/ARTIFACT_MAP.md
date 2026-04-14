# ARTIFACT MAP — `nmr-ssl`, submission frozen at tag `v2.0-jin-revision`

This file maps every table, figure, and headline number in the paper
`docs/paper_final/paper_final.pdf` to the exact script that generated
it and the exact result JSON it was read from. It exists so that a
reviewer or an independent reader can trace every claim in the paper
to a specific command and a specific file in the repository.

All paths are relative to the repository root.

---

## Primary pipeline entry points

| Command | Purpose | Runtime (order of magnitude, consumer GPU) |
|---|---|---|
| `python3 experiments/run_option_b_master.py --seeds 0 1 2` | End-to-end main pipeline: dataset build, 3-seed training for the five main training variants, label-efficiency sweep, K-sweep, lambda-sweep, scaffold-OOD, realistic-HSQC stress test, conformal calibration | tens of minutes |
| `python3 experiments/run_h_zero_ablation.py --seeds 0 1 2` | Causal H-zeroing audit (zero the H coordinate of every HSQC target and retrain) | minutes |
| `python3 experiments/run_realistic_isomer_control.py --seeds 0 1 2` | Constitutional isomer + scaffold-neighbour discrimination | minutes |
| `python3 experiments/run_reviewer_experiments.py --seeds 0 1 2` | Random-pair wrong-structure control + reviewer ablations | minutes |
| `python3 experiments/compute_bonferroni_conformal.py` | Bonferroni-corrected split-conformal calibration and Table 8 | seconds |
| `python3 experiments/run_error_decomposition.py` | Per-carbon-type error decomposition for Table 7 | seconds |
| `python3 experiments/run_multiplicity_loss.py` | SI S3 multiplicity-augmented loss experiment | minutes |

The exact Python and dependency versions are pinned in
`pyproject.toml` at the submission tag.

---

## Figure → script → JSON

| Figure | Script that draws it | Result JSON it reads | Generator command |
|---|---|---|---|
| Figure 1 — main result bar chart | `experiments/make_option_b_figures.py` | `experiments/results_2d/option_b_master.json` | `python3 experiments/make_option_b_figures.py` |
| Figure 2 — H-zeroing causal audit | `experiments/make_h_zero_figure.py` | `experiments/results_2d/h_zero_ablation.json` + `option_b_master.json` (baselines) | `python3 experiments/make_h_zero_figure.py` |
| Figure 3 — label-efficiency sweep | `experiments/make_label_sweep_figure.py` | `experiments/results_2d/label_sweep.json` | `python3 experiments/make_label_sweep_figure.py` |
| Figure 4 — wrong-candidate discrimination | `experiments/remake_wrong_struct_figure.py` | `experiments/results_2d/realistic_isomer_control.json` + `reviewer_experiments.json` (random-pair block) | `python3 experiments/remake_wrong_struct_figure.py` |

---

## Table → source JSON → headline numbers

| Table | Source JSON | Headline numbers exported |
|---|---|---|
| Table 1 — main 4 training variants | `option_b_master.json` (blocks `p10_supervised_1d`, `p11_lambda2_headline`, `p22_combined`); 1-D SSL row from `revision_batch3.json` (block `p09`) | 5.60 / 2.47 (sup-1D); 4.56 / 2.61 (1-D SSL); 4.53 / 0.35 (2-D SSL headline); 3.23 / 0.30 (combined) |
| Table 2 — causal audit | `h_zero_ablation.json` + `option_b_master.json` baseline | 4.53 / 0.35 (baseline); 5.00 / 4.69 (H-zeroed) |
| Table 3 — structure-verification discrimination | `realistic_isomer_control.json` (constitutional isomer + scaffold neighbour blocks) + `reviewer_experiments.json` (wrong_structure block) | 77.0% vs 21.7% (n=74, iso); 62.4% vs 5.2% (n=93, scaffold); 72.9% vs 1.3% (n=155, random) |
| Table 4 — random vs scaffold-OOD | `option_b_master.json` (`p15_scaffold_ood`) + headline baseline | 4.53 / 0.35 (random); 6.06 / 0.40 (scaffold-OOD) |
| Table 5 — realistic HSQC stress test | `realistic_hsqc.json` | 4 rows: clean, realistic, +merging, aggressive |
| Table 6 — K-sweep and λ-sweep | `option_b_master.json` (`p12a_k_sweep`, `p12b_lambda_sweep`) | 2–32 sweep; 0.25–2 sweep |
| Table 7 — per-carbon-type error decomposition | `error_decomposition.json` | 7 carbon-type rows |
| Table 8 — Bonferroni conformal | `bonferroni_conformal.json` + `chemistry_demo.json` (per-atom coverage) | per-atom 13.4 / 1.03; Bonferroni rows up to k=38; molecule-adaptive 94.8% (147/155) |
| Table S1 — per-seed headline 2-D SSL | `option_b_master.json` (`p11_lambda2_headline.seeds`) | 4.689 / 0.332; 4.451 / 0.351; 4.464 / 0.376 |
| Table S2 — per-seed combined | `option_b_master.json` (`p22_combined.seeds`) | 3.104 / 0.350; 3.252 / 0.275; 3.337 / 0.287 |
| Table S3 — multiplicity-loss ablation | `multiplicity_loss.json` | c=4.66±0.05; h=0.39±0.04 |
| Table S4 — stop-gradient control | `option_b_master.json` (`p25_stopgrad`) | |
| Table S5 — pretrain→finetune | `option_b_master.json` (`p23_pretrain_finetune`) | |

---

## Abstract and body numerical claims → file

| Claim in text | Source |
|---|---|
| "4.53 ± 0.11 ppm 13C and 0.35 ± 0.02 ppm 1H" | `option_b_master.json::p11_lambda2_headline.aggregate` |
| "3.23 ± 0.10 ppm 13C and 0.30 ± 0.03 ppm 1H" (combined) | `option_b_master.json::p22_combined.aggregate` |
| "4.69 ± 0.10 ppm 1H" (H-zeroed) | `h_zero_ablation.json::aggregate` |
| "95.2% 13C and 96.7% 1H per-atom coverage" | `chemistry_demo.json::c_empirical_coverage`, `h_empirical_coverage` |
| "94.8% molecule-level coverage (147/155)" | `bonferroni_conformal.json::molecule_adaptive` |
| "77.0% vs 21.7% on constitutional isomers" | `realistic_isomer_control.json::constitutional_isomer.{own_both_rate,wrong_both_rate}` |
| "1,542-molecule filtered subset" | `option_b_master.json::dataset.n_molecules`, `reviewer_experiments.json::n_molecules`, `revision_batch3.json::n_molecules` |
| "1,959 13C val atoms, 1,307 1H val atoms" | `bonferroni_conformal.json::val_atoms` |
| "SW/Hungarian ratio mean 0.414 on dim-2 test" | `tests/test_theorem_2d.py` + SI S7 audit |

---

## SI tables

| SI Table | Source |
|---|---|
| Table S1 | `option_b_master.json::p11_lambda2_headline.seeds` |
| Table S2 | `option_b_master.json::p22_combined.seeds` |
| Table S3 | `multiplicity_loss.json` |
| Table S4 | `option_b_master.json::p25_stopgrad` |
| Table S5 | `option_b_master.json::p23_pretrain_finetune` |

---

## Reproducibility statement

Every empirical number in the main text, the supplementary tables,
and the figures of `paper_final.pdf` is traceable to one of the
result JSON files listed above via one of the generator scripts
listed above. The submission-frozen state is the git tag
`v2.0-jin-revision`. This ARTIFACT_MAP.md is distributed inside the
LaTeX submission zip (`jcheminf_submission.zip`) alongside the
manuscript source.

No result JSONs were regenerated between the integrity audit and
the final submission. All JSONs are committed at the submission tag
and can be spot-checked against the paper with, for example:

```bash
python3 - <<'PY'
import json
d = json.load(open("experiments/results_2d/option_b_master.json"))
print("p11 13C aggregate:", d["p11_lambda2_headline"]["aggregate"]["c_mean"])
print("p11  1H aggregate:", d["p11_lambda2_headline"]["aggregate"]["h_mean"])
PY
```

This will print numbers that match Table 1 row 3 of the paper to
within the rounding convention (2 decimal places, 2 s.d.).
