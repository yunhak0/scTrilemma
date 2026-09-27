# Reconstruction fidelity — Table 2, Table 19 and the expression-fidelity panel of the ablation figure

Expression fidelity of the released checkpoint on the 89 held-out datasets
(`configs/zsb/full_89_ids.txt`), in two protocols:

| Module | Computes | Needs |
|---|---|---|
| `generate.py` | decodes the staged sample of every held-out dataset over the dataset's shared gene list and writes one reconstruction cache per dataset | checkpoint, GPU recommended |
| `score_agreement.py` | Table 2 / Table 19: per-cell Pearson and Spearman, MAE and MSE of a cache against the raw sampled counts (log1p(CP10K), detection filter 2%) | caches only, CPU is fine |
| `hvg_fidelity.py` | ablation figure: training-style HVG Pearson on raw counts and gene-mean Pearson/Spearman | checkpoint and the Census data, GPU recommended |
| `_util.py` | shared-gene decoding with the public inference helpers, default paths, CLI options of the checkpoint | — |
| `data/table2_excluded_genes.tsv` | per-dataset genes of the shared lists that lie outside the paper's scored gene universe (see below) | — |

Run every module from the repository root, e.g. `pixi run python -m experiments.reconstruction.generate`;
`--help` lists all options. Defaults: checkpoint `checkpoints/sctrilemma/final.ckpt`, gene
vocabulary `$SCTRILEMMA_GENE_VOCAB` or `$SCTRILEMMA_DATA_ROOT/20250130/gene_vocab_homo_sapiens_20250130.json`,
samples `outputs/experiments/samples`, gene lists `experiments/data/genelists`, outputs under
`outputs/experiments/reconstruction/`. The released checkpoint needs neither the pseudo-bulk
nor the tissue-code dictionary for inference (the tissue code only enters the KL prior and the
decoder ignores it), so the `--pseudo-bulk` / `--tissue-code` options can stay unset.

## Table 2 / Table 19 — direct reconstruction agreement

```bash
pixi run python -m experiments.prepare_samples                       # once: <= 2,500 cells per dataset
pixi run python -m experiments.reconstruction.generate               # one GPU; resumable
pixi run python -m experiments.reconstruction.score_agreement        # CPU, about a minute
```

`generate.py` encodes each sampled cell exactly as `sctrilemma.inference.reconstruct_adata`
does (log1p(CP10K) of the vocabulary genes, per-cell top-4,096 crop, full-cell library size)
and decodes the ZINB mean over the dataset's shared gene list in list order (genes missing
from the sample or the vocabulary are dropped; in practice every listed gene is decoded).
Batch size 8 as in the paper. Output per dataset:
`outputs/experiments/reconstruction/cache/sctrilemma/<dataset_id>.npz` with `recon`
(N, G) float32 means on the raw-count scale, `gene_names` (G), `soma_joinid` (N, staged
sample order), `model` and a JSON `metadata` string, plus a `run_config.json`. Existing
valid caches (same cell order) are skipped unless `--force` is given.

`score_agreement.py` scores one cache directory (`--cache-dir`, `--model-name`,
`--model-display`); any cache with `recon`, `gene_names` and `soma_joinid` can be scored.
Per dataset:

1. cells: the staged sample, matched to the cache by exact ordered `soma_joinid`;
2. genes: the shared gene list, minus the dataset's rows of `data/table2_excluded_genes.tsv`,
   intersected with the genes present in the raw matrix and in the cache, then filtered to genes
   detected (raw count > 0) in at least `--min-detection-rate` = 0.02 of the sampled cells
   ("det02"); datasets with fewer than `--min-common-genes` = 100 genes fail;
3. raw counts and reconstruction are independently normalised to log1p(CP10K) on that gene set
   (`experiments.common.log1p_cp10k`) and scored with `experiments.common.score_matrix`:
   per-cell Pearson and Spearman averaged over cells, MAE and MSE over all entries.

Outputs (`outputs/experiments/reconstruction/results_det02/`): `per_dataset.csv`
(`dataset_id, model, model_display, n_cells, n_common_genes, recon_pearson, recon_spearman,
recon_mae, recon_mse`), `aggregate_equal_dataset.csv` (`model, model_display, n_datasets` and
`<metric>_mean`, `<metric>_sd` with equal dataset weight and sample SD), `evaluation_support.csv`,
`failures.csv`, `RESULTS.md` and `run_config.json`.

**Gene universe.** In the paper every method was scored on the genes that all compared methods
had decoded for a dataset. The shared gene lists in `experiments/data/genelists` are that
universe up to 4–35 genes per dataset (1,738 gene–dataset pairs in total) which one of the
compared methods (scPRINT) could not decode; `data/table2_excluded_genes.tsv` lists them so
that the released scorer reproduces the paper's gene sets exactly. `--ignore-excluded-genes`
scores the full shared lists instead (metrics move by less than 1e-3), `--no-genelist` scores
every gene shared by the raw matrix and the cache.

## Ablation figure — training-style HVG Pearson and gene-mean correlations

```bash
pixi run python -m experiments.reconstruction.hvg_fidelity           # one GPU
```

Unlike Table 2 this protocol works on raw counts and on the model's own decode set, and it
decodes the cells itself (it does not read the `generate.py` caches). Per held-out dataset:

1. cells: all cells of the dataset (every shard under
   `$SCTRILEMMA_DATA_ROOT/20251108/by_dataset/<dataset_id>/`), or a uniform random subset of
   `--max-cells-per-dataset` = 20,000 cells drawn with `numpy.random.default_rng` seeded by
   `stable_seed(--sample-seed, dataset_id)` (`--sample-seed` = 0); rows keep the global order
   of the sorted shards. Only the selected rows are read from disk, which is identical to
   sampling the in-memory concatenation of the shards (checked on a two-shard dataset);
2. reconstruction: `ScTrilemmaInference.reconstruct` → `sctrilemma.inference.reconstruct_adata`
   with batch size 128: encoder input as above, decode set = dataset-level top-4,096 vocabulary
   genes by summed raw expression over the sampled cells, ZINB mean with the full-cell library;
3. `training_style_recon_pearson_hvg`: raw counts vs ZINB means (no normalisation), restricted
   to the `--top-k-hvg` = 2,000 decoded genes with the largest raw-count variance (float64,
   zero-variance genes excluded); per-cell Pearson averaged over the cells whose centred norms
   are both above 1e-8 (`n_cells` in the output; `n_hvg` = 2,000);
4. `gene_mean_pearson`, `gene_mean_spearman`: raw counts and ZINB means independently
   log1p(CP10K)-normalised (float64) over the decoded genes, per-gene mean over cells, then
   Pearson / Spearman across genes (`n_hvg` = number of decoded genes, 4,096).

Output: `outputs/experiments/reconstruction/hvg_fidelity/<run>.csv` with columns
`run, dataset, metric, value, n_hvg, n_cells, sampled_cells, sample_seed` (three rows per
dataset and seed). The ablation figure uses `training_style_recon_pearson_hvg` and
`gene_mean_pearson`. `--sample-seeds 0 1 2` repeats the cell sample (file suffix `_repeatN`);
`--limit N` processes only the first N dataset IDs.

## Results shipped

- Table 19 (direct reconstruction agreement) ships with the matched-cohort analyses in
  `experiments/table2_joint/results/reconstruction/`; re-scoring the paper's scTrilemma caches
  with `score_agreement.py` (CPU) reproduces those values to 5e-7 per dataset and 2e-8 in the aggregate.
- `results/hvg_fidelity_per_dataset.csv` — the paper's values of the released checkpoint
  (89 datasets; produced with the script defaults, before the `sample_seed` column existed).

## Verification of this port

- `score_agreement.py`: the paper's scTrilemma caches for all 89 datasets, scored on CPU,
  give identical `n_common_genes` and metrics within 4.8e-7 (per dataset) / 2e-8 (aggregate)
  of the paper's result files.
- `generate.py`: one 2,500-cell dataset decoded end to end on CPU (`--device cpu`) matches the
  paper's cache for that dataset (same genes and cell order; max relative difference 1.4e-5,
  99.9th percentile 3.4e-6, float32 CPU-vs-GPU noise), and scoring it with
  `score_agreement.py` returns the paper's row (Pearson identical, other metrics within
  2.4e-7). A 24-cell spot check on a second cell subset gave the same picture.
- `hvg_fidelity.py`: the smallest held-out dataset (796 cells, so all cells are used) run on
  CPU matches the paper's three values within 3.5e-7 (`training_style_recon_pearson_hvg`
  0.494916, `gene_mean_pearson` 0.578395, `gene_mean_spearman` 0.378640). The 20,000-cell
  subsets of the other datasets need a GPU and were not re-run here (command below).
- Sampling: the selected rows of a two-shard held-out dataset (166,085 cells, 500 sampled) are
  identical (matrix, gene names, `soma_joinid` order, donors) to an in-memory concatenation of
  the shards sampled with the same seed.

## Compute

- `generate.py`: about 0.2 s per cell on 8–12 CPU threads (8 min per 2,500-cell dataset, about
  12 h for the 89 datasets on CPU); a single GPU is recommended. The paper ran it on one GPU
  with batch size 8; the GPU runtime was not recorded.
- `score_agreement.py`: under 1 s per dataset on CPU.
- `hvg_fidelity.py`: about 1.54 million cells in total (67 datasets hit the 20,000-cell cap);
  on CPU this is days, so run it on one GPU (batch size 128). GPU runtime was not recorded.
  Full verification against `results/hvg_fidelity_per_dataset.csv`:

  ```bash
  pixi run python -m experiments.reconstruction.hvg_fidelity --output-dir outputs/experiments/reconstruction/hvg_fidelity
  # then compare outputs/experiments/reconstruction/hvg_fidelity/sctrilemma.csv with
  # experiments/reconstruction/results/hvg_fidelity_per_dataset.csv (join on dataset, metric)
  ```
