# Figure 4 / Table 20 — demand-targeted latent interventions

Intervene on the posterior-mean latent tokens of the released checkpoint at inference time
and measure all three demands on the same intervened latent: identity and context
invariance on the pooled embedding, expression fidelity by decoding the tokens with the
unchanged decoder over the shared gene list of each dataset.

| Intervention | Pushes | Definition (alpha in [0, 1]) |
|---|---|---|
| `donor_centering` | context invariance | T' = T − alpha · (mu_donor − mu_all) |
| `centroid_shrinkage` | identity | T' = (1 − alpha) · T + alpha · c_k, k-means centroids on the pooled embedding (K = #cell types, seed 0) |
| `label_shrinkage` | identity (oracle) | same with annotated cell-type centroids |
| `latent_refinement` | fidelity | T' = T + alpha · (T* − T), T* = 30 Adam steps on the per-cell reconstruction Pearson |

## Inputs

- `checkpoints/sctrilemma/final.ckpt` (see the top-level README for the download)
- sampled datasets: `pixi run python -m experiments.prepare_samples` (≤ 2,500 cells per held-out dataset, stratified by cell type × disease, seed 42)
- gene lists: `experiments/data/genelists/`

## Run

```bash
bash experiments/fig4_interventions/run_all.sh          # five alpha grids, then summary + figure
```

Each run scores all 89 held-out datasets on one GPU; `run_interventions.py` is resumable
(datasets already present in `interventions_long.csv` are skipped). Outputs go to
`outputs/experiments/fig4_interventions/`.

## Results

`results/` holds the summary the paper was built from:

- `paired_deltas.csv` — per (intervention, alpha, metric): number of datasets `n`, mean of the
  unmodified model `base_mean`, mean paired delta vs alpha = 0 with its SEM, counts of datasets
  moving down/up, and the two-sided Wilcoxon p-value. Rows of `alpha_fine`, `alpha_mid`,
  `alpha_over` and `refine_label` runs are pooled; deltas are always taken against the
  alpha = 0 reference of the same run and dataset.
- `coupling_matrix.csv` — slope of each responding demand per unit relative change of the
  pushed demand (least squares through the origin), with the fraction of datasets moving in
  the mean direction at the middle alpha.
- `trilemma_response_curves.pdf/png` — Figure 4.

Embedding metrics use scib-metrics (k-means with seeds 0–4 for NMI/ARI, pynndescent k-NN for
the LISI scores), so re-runs reproduce the reported values up to the tolerance of the
approximate k-NN graph.
