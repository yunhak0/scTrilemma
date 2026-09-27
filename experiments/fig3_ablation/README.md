# Figure 3 / Table 24 — component ablation

Each component is removed from the released recipe and the model retrained with the same
data, steps and seed; the full recipe is also retrained unchanged as a variability reference.

| Arm | Config | Change |
|---|---|---|
| Full (retrained) | `configs/experiment/ablation_full_retrained.yaml` | none |
| w/o E-Gate | `ablation_no_egate.yaml` | `model.expr_proj_mode: linear` |
| w/o C-Route | `ablation_no_croute.yaml` | `model.decoder_query_mode: "off"` (mean-pooled readout kept) |
| w/o PB-Cond | `ablation_no_pbcond.yaml` | `model.tissue_prior_mode: ""` |
| PB-Cond, shuffled codes | `ablation_pbcond_shuffled_codes.yaml` | group codes permuted (seed 42) |
| PB-Cond, constant code | `ablation_pbcond_constant_code.yaml` | every group gets the mean code |

The two control code files come from `scripts/preprocessing/6_build_control_codes.sh`.

## Pipeline

```bash
bash experiments/fig3_ablation/run_ablation_arms.sh     # train the six arms (4 GPUs each) and export their embeddings
bash experiments/fig3_ablation/score_arms.sh            # Table 1 scorers + HVG fidelity + reconstruction MSE per arm
pixi run python -m experiments.fig3_ablation.build_long # -> ablation_per_dataset_long.csv
pixi run python -m experiments.fig3_ablation.wilcoxon   # paired Wilcoxon tests, BH-FDR over all tests
pixi run python -m experiments.fig3_ablation.plot       # Figure 3
```

Metrics per dataset: NMI/ARI (FAISS k-means, 20 seeds), ASW/BRAS (scib-metrics, 20 samples),
PCR (20 samples, 75 multi-donor datasets), training-style HVG Pearson and gene-mean Pearson
(`experiments.reconstruction.hvg_fidelity`), cell-wise MSE on the shared gene list
(`experiments.reconstruction.score_agreement`, detection ≥ 2%).

## Results

- `results/ablation_per_dataset_long.csv` — per (condition, dataset, metric) values for the
  released checkpoint (`Full`) and the six arms.
- `results/ablation_wilcoxon_tests.csv` — paired Wilcoxon tests of each removal against `Full`
  with BH-FDR q-values (`wilcoxon.py`).
- `results/ablation_seven_metric_table.csv` — Table 24 (all arms vs. the retrained reference).
- `results/fig3_ablation.pdf/png` — Figure 3. The plotted value is mean(Δ) / SD(Δ) across
  datasets with MSE sign-flipped; `plot.py` regenerates it from the two CSVs above and prints
  the same labels as the paper figure.
