# Table 1 — zero-shot benchmark on 89 held-out datasets

Every model is scored on the same cells: for each held-out dataset of the 2025-11-08 Census
release, at most 100,000 cells (seed 0), from which 20 stratified metric samples
(≤ 500 cells per cell type, ≤ 10,000 cells, dataset-specific seeds) are drawn.

| Metric | Repeats | Scorer |
|---|---|---|
| NMI, ARI | 20 k-means seeds on metric sample 0 (FAISS k-means, 20 iterations, 1 restart) | `experiments.scoring.kmeans_repeat` |
| ASW, cLISI, Iso. Label, BRAS, iLISI | 20 metric samples (scib-metrics; k-NN graphs from pynndescent, k = 90) | `experiments.scoring.scib_repeat` |
| PCR comparison | 20 metric samples; PCA of log1p(CP10K) counts vs. embedding, categorical donor covariate; datasets with ≥ 2 donors (75) | `experiments.scoring.pcr_repeat` |

Per dataset the repeats are averaged first; the table reports mean and SD of these
per-dataset values across datasets (n = 89, or 75 for BRAS/iLISI/PCR). Stars mark
scTrilemma gains over the strongest baseline of each metric by a dataset-paired two-sided
Wilcoxon test (* p < 0.05, *** p < 0.001).

## Reproduce the scTrilemma row

```bash
bash scripts/evaluation/export_embeddings.sh                         # embedding caches (GPU)
bash experiments/table1_benchmark/run_all.sh                        # metric samples, scorers, table
```

## Results

- `results/table1_long.csv` — one row per (dataset, model, metric, repeat) for all six models,
  80,400 rows. `repeat_kind` says whether `repeat` indexes a k-means seed or a metric sample.
  Baseline rows were produced with the official implementations at the versions listed in
  Appendix A (scVI census model, Geneformer, scGPT, CellPLM 20230926_85M, scPRINT large-v1);
  their code is not part of this repository.
- `build_table.py` turns the long file into `table1_summary.csv`, `table1_wilcoxon.csv` and the
  LaTeX rows of Table 1. Run on the shipped file it reproduces every cell of the paper table
  (means, SDs, bold/italic ranks and stars).
