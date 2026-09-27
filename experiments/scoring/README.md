# Table 1 — embedding scoring

Scores cached scTrilemma embeddings of the 89 held-out datasets under the paper's repeat
protocol. Every module takes a generic cache directory (`--cache-dir`, one `<dataset>.npz`
per dataset with aligned arrays `embeddings` (N, D) float32, `labels` (N,) str and
`batches` (N,) str) and tags its rows with `--model-name` (default `sctrilemma`), so the
same code scores any compatible cache.

Run modules from the repository root with `pixi run python -m experiments.scoring.<module>`.

## Pipeline

```bash
# 0. embeddings (GPU): outputs/experiments/embeddings/sctrilemma/embeddings/<dataset>.npz
bash scripts/evaluation/export_embeddings.sh

# 1. metric samples (CPU, seconds)
pixi run python -m experiments.scoring.metric_samples

# 2. FAISS K-means seed sensitivity + cLISI/BRAS (CPU, ~minutes)
pixi run python -m experiments.scoring.kmeans_repeat

# 3. scib-metrics protocol (GPU for jax; ~1-3 min per dataset)
pixi run python -m experiments.scoring.scib_repeat

# 4. PCR comparison (CPU jax; needs the held-out Census release, ~1 min per dataset)
SCTRILEMMA_DATA_ROOT=/path/to/cellxgene pixi run python -m experiments.scoring.pcr_repeat
```

Defaults: cache dir `outputs/experiments/embeddings/sctrilemma/embeddings`, metric samples
`outputs/experiments/scoring/metric_samples`, results under `outputs/experiments/scoring/`.
Dataset list: `configs/zsb/full_89_ids.txt` (`--dataset-ids` selects a subset).

## Input protocol (fixed by the paper)

- Input cap: at most 100,000 cells per dataset, drawn with `numpy.random.default_rng` seeded by
  `(0 + int(sha1(dataset_id)[:8], 16)) mod 2**32` (`sctrilemma.benchmark.run._sample_cells_for_analysis`).
  Labels are `cell_type_ontology_term_id`, batches `donor_id` (or `dataset_id` when absent).
- Embeddings: posterior mean of the latent encoder, 512-d, batch size 256
  (`sctrilemma.benchmark.export_embeddings`, console script `sctrilemma-export-embeddings`).
- Metric samples: for sample seed `s` in 0..19 the dataset-specific NumPy seed
  `(s + int(sha1(dataset_id)[:8], 16)) mod 2**32` seeds the **legacy** NumPy RNG
  (`np.random.seed`), under which the benchmark's stratified sampler draws at most 500 cells per
  cell type (types with fewer than 50 cells are kept whole) and at most 10,000 cells in total, with
  proportional redistribution when the total is exceeded. `experiments/scoring/_common.py`
  re-implements this sampler with `numpy.random.RandomState`, which yields the same stream.

## Modules

### `metric_samples.py`

Inputs: cache dir. Outputs: `<dataset>_sample_seed_<s:02d>.npz` with `cache_row_indices`
(int64, ordered by cell type), `dataset_numpy_seed` (uint32) and `sample_seed` (int64);
`metric_sample_manifest.csv`, `metadata.json`. Existing files are reused unless `--force`.

### `kmeans_repeat.py`

Seed-0 metric sample only. NMI/ARI from `faiss.Kmeans` (CPU, `k` = number of cell types,
`niter` 20, `nredo` 1) at K-means seeds 0..19; cLISI (`sctrilemma.utils.metrics.compute_clisi`,
perplexity-weighted Simpson index on the k = 50 Euclidean k-NN graph) and BRAS
(`compute_bras`, cosine; only datasets whose cache has more than one batch) once on the same
sample. Outputs: `kmeans_repeat_long__<model>.csv` with columns
`dataset, model, metric, kmeans_seed, sample_seed, nredo, value`,
`kmeans_repeat_summary__<model>.csv` (per-dataset seed means, per-seed global means,
fixed-sample cLISI/BRAS), `kmeans_repeat_sample_manifest__<model>.csv` and a metadata JSON.

### `scib_repeat.py`

scib-metrics library metrics. NMI/ARI: `scib_metrics.utils.KMeans` (k = number of cell types)
at seeds 0..19 on the seed-0 sample, scored with the same scikit-learn calls as
`nmi_ari_cluster_labels_kmeans` (`--verify` asserts seed-0 equivalence with the wrapper).
`silhouette_label`, `isolated_labels`, `clisi_knn`, and for multi-batch datasets `ilisi_knn`
and `bras`, on each of the 20 samples; the LISI scores use the `pynndescent` k = 90 graph.
Output: `scib_repeat20_long__<model>.csv` with columns
`dataset, model, metric, value, sample_seed, kmeans_seed, n_cells, n_labels, n_batches`
(`kmeans_seed` is -1 for sample-level metrics). Resumable (datasets already in the file are
skipped); `--num-shards`/`--shard-index` split the datasets into `__shard<k>` files.

### `pcr_repeat.py`

scIB principal-component-regression comparison for every dataset whose cache has more than
one batch (75 of 89) and each of the 20 samples: the sampled cells are re-read from the
held-out Census release under the same input cap (row alignment with the cache is asserted),
normalised (`normalize_total` 1e4, `log1p`), reduced with `sc.tl.pca(svd_solver="arpack")` on
all genes; `pcr_pre` = PCR of the PCA coordinates on the categorical batch covariate,
`pcr_post` = PCR of the embedding, `value = max(0, (pcr_pre - pcr_post) / pcr_pre)`.
One shard per dataset under `shards/<model>/` (resumable; `--worker-index/--worker-count`
partition datasets; failures are recorded under `failures/<model>/`). The aggregate step
(also `--aggregate-only`) writes `pcr_repeat20_long.csv` (columns `dataset, model, sample_seed,
value, pcr_pre, pcr_post, n_sample_requested, n_sample_common, n_batches, seconds, status`),
`pcr_dataset_repeat_summary.csv`, `pcr_repeat20_summary.csv` and `progress.json`.

## Runtime environment

- `CPU_THREADS` (default 4) seeds `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `NUMEXPR_NUM_THREADS` in every module (set before NumPy is imported; existing values win);
  `kmeans_repeat` also applies it to FAISS (`--threads`).
- `pcr_repeat` additionally sets `NUMBA_NUM_THREADS`, `JAX_NUM_THREADS`, `JAX_PLATFORMS=cpu`
  and `XLA_FLAGS=--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4` by
  default (the reference values were produced on the CPU jax backend); export any of them to
  override.
- `scib_repeat` runs jax on the GPU when one is visible (`CUDA_VISIBLE_DEVICES`) and sets
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` so the process does not reserve the whole card.
- Data: `SCTRILEMMA_DATA_ROOT` (default `/scratch/$USER/datasets/cellxgene`), or
  `SCTRILEMMA_TARGET_PATH` / `--target-path` for the held-out release
  `<root>/20251108/by_dataset` and `SCTRILEMMA_GENE_VOCAB` for the gene vocabulary
  (`export_embeddings` only).

## Verification against the reported results

The port was checked against the working copies of the paper's result files (embedding
cache of the reported checkpoint, identical package versions except scib-metrics 0.5.9 vs
0.5.7, pynndescent 0.6.0 vs 0.5.13, jax 0.8.3 vs 0.8.2, scipy 1.17.1 vs 1.16.3,
scikit-learn 1.7.2 vs 1.8.0, anndata 0.12.11 vs 0.12.7):

- `export_embeddings`: one dataset re-exported through `scripts/evaluation/export_embeddings.sh`
  with the released checkpoint is bit-identical to the reported cache (7,750 x 512, max abs
  diff 0.0; labels and batches identical).
- `metric_samples`: 4 datasets x 20 seeds (80 files, incl. a 100,000-cell dataset hitting the
  10,000-cell cap) reproduce the reported `cache_row_indices`, `dataset_numpy_seed` and
  `sample_seed` exactly.
- `kmeans_repeat`: 3 datasets, 126 rows; max abs difference 0.0 for NMI, ARI, cLISI and BRAS.
- `scib_repeat`: 2 datasets x (40 K-means rows + 20 samples x 5-7 metrics) = 280 rows, jax on GPU;
  NMI/ARI max abs diff 0.0, `silhouette_label` and `isolated_labels` <= 6e-8, `bras` <= 1.8e-7,
  `clisi_knn` <= 1.6e-5 and `ilisi_knn` <= 8.3e-4 (approximate `pynndescent` graph, 0.6.0 vs 0.5.13);
  `--verify` reproduces `nmi_ari_cluster_labels_kmeans` at seed 0.
- `pcr_repeat`: 2 multi-batch datasets x 20 samples; `pcr_post` identical, `pcr_pre` max abs
  diff 2.1e-7 (PCA numerics), `value` max abs diff 1.6e-6.
