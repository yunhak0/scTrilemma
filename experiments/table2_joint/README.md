# Table 2 and appendix Tables 3, 4, 7, 18, 19 — matched-cohort analyses

Matched evaluation of the three representation demands (biological identity and state,
donor-context invariance, expression fidelity) on the same held-out disease cohorts, plus the
supplementary donor-supported expression-fidelity panel and the direct reconstruction agreement.
Every module scores caches generically (`--embedding-cache NAME=DIR`, `--recon-cache NAME=DIR`),
so the same code scores any compatible cache; the release computes scTrilemma only and ships
the baseline rows of the paper as CSV under `results/`.

| Module | Computes | Paper |
|---|---|---|
| `cohort_audit.py` | metadata-only audit of fixed-context two-state contrasts in all 89 held-out datasets (`candidate_pairs.csv`, `dataset_best_candidates.csv`) | Table 4; input of the cohort selection audit in `joint.py` |
| `joint.py` | five common cohorts, 20 balanced repeats: identity / state retrieval and donor invariance on embeddings, DEG fidelity on reconstructions; cohort audit; paired comparisons; Pareto audit | Table 2 (all rows but the pathway rows), Table 3 |
| `pathway.py` | pathway-level fidelity (Enrichr) of the repeat-averaged disease logFC on the same cohorts; metric-family Pareto sensitivity | Table 2 (Pathway Jaccard, Pathway-score rho) |
| `rq4_panel.py` | outcome-blind donor-supported screen of the 89 samples, seven disease cohorts, up to six cell-type contrasts each; DEG and pathway concordance | Tables 7, 18 |
| `deg_concordance.py`, `pathway_concordance.py` | contrast-level DEG and Enrichr scorers used by `rq4_panel.py` and `pathway.py` (also usable standalone on staged samples) | — |
| `context_metrics.py` | residual donor variance (adjusted multivariate omega-squared) and local donor mixing | — |
| `experiments.reconstruction.score_agreement` (sibling folder) | direct reconstruction agreement of a cache with the raw counts | Table 19 |

Run modules from the repository root, e.g. `pixi run python -m experiments.table2_joint.joint`;
`--help` lists all options. `run_all.sh` chains the whole pipeline.

## Cohorts and protocol

**Common cohorts (Table 2 / Table 3).** Five datasets were fixed from metadata and cache coverage
only (`joint.COHORTS`): Kidney `867757c1` (10x multiome, kidney; obstructive nephropathy vs
normal), Liver `e3ed2ba4` (BD Rhapsody WTA, liver; metastatic colorectal carcinoma), Brain
`203025fe` (10x multiome, dorsolateral prefrontal cortex; cognitive disorder), Tendon `acd544d0`
(10x 3' v3, quadriceps femoris tendon; injury), Colon `829a3cd1` (10x 3' v3, sigmoid colon;
colorectal cancer). The selection rule is re-derived on every run (`candidate_audit_89.csv`):
one fixed-context normal contrast per dataset from `candidate_pairs.csv` (the normal contrast
with most cells, then most eligible cell types), then *strict* support on the common pool
(sampled cells that are also in the embedded cell universe): a cell type is eligible when both
states have at least 40 cells and at least two donors with at least 10 cells each; a dataset is
selected when at least three cell types are eligible. The run aborts if this rule disagrees with
the frozen panel.

**Balanced repeats.** For repeat `r` in 0..19 the seed is `42 + r`, and
`numpy.random.default_rng(stable_seed(seed, dataset_id))` draws, for every eligible cell type and
each of the two states, two donors (from those with at least ten cells, sorted) and ten cells per
donor: `n_cells = 40 x n_eligible_cell_types` per repeat (Table 3 column "cells per repeat").
The same cells are scored on all three demands.

**Biological identity and state** (`joint.biological_scores`, embeddings L2-normalised): for each
query cell the reference pool is the other cells of the same state (identity) or the same cell
type (state) from other donors, subsampled to equal size per target label
(`default_rng(seed)`, seed = repeat seed), and the label is predicted by cosine k-NN majority
(k = 10). Reported: balanced accuracy minus chance `1/n_cell_types` (identity, averaged over
states), macro neighbour purity, balanced accuracy of the disease state (averaged over cell
types).

**Context invariance** (`joint.context_scores`, `context_metrics.py`): within each cell type x
state group, `1 - residual_donor_variance` (adjusted omega-squared of donor) and local donor
mixing (chance-corrected same-donor enrichment among the 10 cosine nearest neighbours); donor
BRAS conditional on cell type x state (`scib_metrics.bras`, cosine, `mean_other`).

**Expression fidelity** (`joint.prepare_expression`, `joint.expression_scores`): per cohort the
common genes are the raw features present in every reconstruction cache, restricted to the paper's
gene universe (below) and to genes detected in at least 2% of the pooled cells. Raw counts are
log1p(CP10K) with the full-cell library size; each reconstruction is log1p(CP10K) with the sum over
all genes of its cache. Per repeat and cell type the disease-vs-normal logFC is computed on genes
whose summed detection rate over the two 20-cell groups exceeds 0.02; metrics are the Spearman
correlation of raw vs reconstructed logFC, the Jaccard overlap of the top-100 genes by |logFC|
and the sign concordance on the raw top-100 genes, averaged over cell types.

**Pathway fidelity** (`pathway.py`): logFC vectors are averaged over the 20 repeats (genes kept
when detected in at least 80% of the repeats), the top-100 up and down gene symbols of the raw and
of each reconstructed profile are submitted to Enrichr (`MSigDB_Hallmark_2020`,
`Reactome_Pathways_2024`, `GO_Biological_Process_2025`), and per direction and library the
reconstruction is scored by the Jaccard of the top-10 terms (adjusted p-value) and the Spearman
correlation of the combined scores over the union of those terms; then averaged over cell types
and cohorts.

**Donor-supported panel (Tables 7, 18)** (`rq4_panel.py`): on the 89 staged samples, every
normal-vs-condition pair is screened with the same support rule (40 cells per state, two donors
with 10 cells); the best-supported pair per dataset wins (more eligible cell types, larger
minimum-state support, total support, donors, cells, lexical label), `injury` is not a named
disease, and the seven winners are frozen (the run aborts otherwise). Up to six cell-type contrasts
per cohort (largest minimum state size first) are scored on all sampled cells with the DEG scorer
(detection 0.02, top-100) and the pathway scorer (top-100 genes, top-10 terms); cohort values
average the contrasts, the final block averages cohorts (Table 18 SDs are sample SDs across the
seven cohort values in `cohort_metric_summary.csv`).

**Gene universe.** The paper scored expression fidelity on the genes that all compared methods had
decoded for a dataset. With several `--recon-cache` entries the intersection is taken on the fly;
in addition the shared gene lists (`experiments/data/genelists`) minus the shipped
`experiments/reconstruction/data/table2_excluded_genes.tsv` (1,738 dataset-gene pairs one compared
method could not decode) are applied before the detection filter, so that the release reproduces
the paper's gene sets exactly with only the scTrilemma cache. With the paper's four caches the
subtraction is a no-op (verified below). `--ignore-excluded-genes` / `--no-genelist` switch the
restrictions off.

## Inputs

- Sampled cells: `outputs/experiments/samples/<dataset>.h5ad` (`pixi run python -m experiments.prepare_samples`).
- Embedding caches: `outputs/experiments/embeddings/sctrilemma/embeddings/<dataset>.npz`
  (`bash scripts/evaluation/export_embeddings.sh`; arrays `embeddings`, `labels`, `batches`,
  `soma_joinid`). The `soma_joinid` array defines the embedded cell universe of the dataset; caches
  without one (`-1` or absent) are matched by row order to `--manifest-dir/<dataset>.npz`
  (`soma_joinid`), which is only a fallback for external caches. `--embedding-cache` also accepts a
  `{dataset_id}` path pattern and caches whose array is named `embedding`.
- Reconstruction caches: `outputs/experiments/reconstruction/cache/sctrilemma/<dataset>.npz`
  (`pixi run python -m experiments.reconstruction.generate`; arrays `recon`, `gene_names`,
  `soma_joinid`).
- Gene lists `experiments/data/genelists/<dataset>.txt` and the excluded-gene table above.
- Candidate contrasts: `results/cohort_audit/candidate_pairs.csv` (default of `joint.py
  --candidate-pairs`); regenerate with `cohort_audit.py` from the Census release
  (`$SCTRILEMMA_DATA_ROOT/20251108/by_dataset`, obs tables only).
- Enrichr (network) for `pathway.py` and `rq4_panel.py pathway`.

## Run

```bash
bash experiments/table2_joint/run_all.sh        # samples -> embeddings -> reconstructions -> tables
```

or step by step (outputs under `outputs/experiments/table2_joint/`):

```bash
pixi run python -m experiments.prepare_samples
bash scripts/evaluation/export_embeddings.sh                         # GPU
pixi run python -m experiments.reconstruction.generate               # GPU
pixi run python -m experiments.table2_joint.cohort_audit             # optional, Census metadata
pixi run python -m experiments.table2_joint.joint                    # Table 2 rows, Table 3
pixi run python -m experiments.table2_joint.pathway                  # Table 2 pathway rows (Enrichr)
pixi run python -m experiments.table2_joint.rq4_panel all            # Tables 7, 18 (Enrichr)
pixi run python -m experiments.reconstruction.score_agreement        # Table 19
```

`joint.py` and `pathway.py` take `--cohorts` for a subset and `--no-selection-audit` to skip the
89-dataset re-derivation (which needs every sample and the embedding cache of the 18 candidate
datasets). `rq4_panel.py` has the sub-commands `screen`, `deg`, `pathway`, `summarize`, `all`;
`summarize` works on the DEG outputs alone when the pathway step was not run. Enrichr responses
are cached in `enrichr_cache.jsonl` (`--enrichr-cache` points to an existing cache), so interrupted
runs resume without re-querying.

## Outputs

- `joint/`: `repeat_metrics.csv` (cohort x repeat x model, three-axis models), `embedding_repeat_metrics.csv`
  (all embedding models), `cohort_means.csv`, `equal_cohort_summary.csv` (Table 2 identity / state /
  context / DEG rows, `pareto_frontier` flag), `embedding_equal_cohort_summary.csv` (Table 2 embedding
  rows of every embedding model), `cohort_audit.csv` (Table 3: eligible cell types, common-pool cells,
  common genes), `candidate_audit_89.csv`, `paired_model_comparisons.csv` (win/tie/loss of
  `--reference-model` per repeat), `repeat_pareto.csv`, `pareto_frequency.csv`,
  `embedding_rank_relationships*.csv` (three or more embedding models), `run_config.json`.
- `pathway/`: `pathway_metrics_long.csv`, `pathway_cohort_summary.csv`, `pathway_equal_cohort_summary.csv`
  (Table 2 pathway rows), `selected_contrasts.csv`, `metric_combination_sensitivity.csv`, `enrichr_cache.jsonl`.
- `rq4_panel/`: `screen_*.csv`, `selected_cohorts.csv`, `selected_cell_type_contrasts.csv`, `evaluation_support.csv`
  (Table 7: supported cell types / contrasts / common genes), `deg_metrics_long.csv`, `deg_cohort_summary.csv`,
  `deg_equal_cohort_summary.csv`, `pathway/…`, `cohort_metric_summary.csv` and `equal_cohort_summary.csv`
  (Table 18), rank counts and pairwise wins.
- `cohort_audit/`: `candidate_pairs.csv`, `dataset_best_candidates.csv`, `failures.csv`.

Model names: `joint.py` / `pathway.py` use the cache names as given (`sctrilemma`, `scvi`, …);
`rq4_panel.py` uses table labels (`scTrilemma`, `scVI`, `CellPLM`, `scPRINT`) as in the paper files.

## Results shipped (`results/`)

Files the paper tables were built from, with the scTrilemma cache name normalised.

| Table | Files | Cells of the table |
|---|---|---|
| Table 2 | `joint/embedding_equal_cohort_summary.csv` (identity, purity, state, residual invariance, BRAS, local mixing of six models), `joint/equal_cohort_summary.csv` (logFC Spearman, DEG Jaccard, DEG sign of the four reconstruction models), `pathway/pathway_equal_cohort_summary.csv` (pathway rows) | equal-cohort means |
| Table 2 support | `joint/repeat_metrics.csv`, `joint/embedding_repeat_metrics.csv`, `joint/paired_model_comparisons.csv`, `pathway/pathway_cohort_summary.csv` | per-repeat values, paired wins |
| Table 3 | `joint/cohort_audit.csv` | `n_eligible_cell_types / n_common_pool_cells / 40 x n_eligible_cell_types / n_common_genes` |
| Table 4 | `cohort_audit/dataset_best_candidates.csv`, `cohort_audit/candidate_pairs.csv` | donors (`n_donors_pair`) and eligible cell types (`n_eligible_cell_types`) of the four RQ2 cohorts; the "sampled cells" column (12,000) comes from the RQ2 retrieval protocol, which is not part of this folder |
| Table 7 | `rq4_panel/selected_cohorts.csv` (`n_eligible_cell_types`), `rq4_panel/evaluation_support.csv` (`selected_contrasts`, `common_genes`) | |
| Table 18 | `rq4_panel/cohort_metric_summary.csv` (cohort block), `rq4_panel/deg_cohort_summary.csv`, `rq4_panel/deg_metrics_long.csv`, `rq4_panel/pathway/pathway_cohort_summary.csv` | means and sample SDs across the seven cohort values give the final block |
| Table 19 | `reconstruction/aggregate_equal_dataset.csv`, `reconstruction/per_dataset.csv` (all shared genes, 2% detection), `reconstruction_det20/…` (genes detected in at least 20% of cells) | equal-dataset mean and sample SD |

## Verification of this port

All checks were run on CPU (`JAX_PLATFORMS=cpu`) against the paper's caches.

- `joint.py`, four reconstruction caches and six embedding caches: `candidate_audit_89.csv`,
  `cohort_audit.csv`, `pareto_frequency.csv`, the win/tie/loss counts and every metric of
  `repeat_metrics.csv` / `embedding_repeat_metrics.csv` (400 and 600 rows) are identical to the
  paper's files except donor BRAS, which is computed by `scib_metrics` in JAX: at most 6.0e-4 per
  repeat (largest for the two highest-dimensional baselines), 9.1e-5 in the equal-cohort means
  (2.3e-5 for the four three-axis models), attributable to GPU-vs-CPU float arithmetic.
- `joint.py`, release path (scTrilemma caches only, shared gene list minus excluded genes): the
  100 scTrilemma repeat rows have identical `expression_fidelity`, `deg_jaccard`, `deg_sign` and
  identical `n_common_genes` (1,524 / 2,135 / 1,821 / 1,555 / 1,639) as the four-cache run, i.e.
  both gene-universe paths give the same numbers.
- `rq4_panel.py deg`, four caches: `screen_*.csv`, `selected_cohorts.csv`,
  `selected_cell_type_contrasts.csv`, `evaluation_support.csv`, `deg_metrics_long.csv`,
  `deg_cohort_summary.csv`, `deg_equal_cohort_summary.csv` are byte-identical to the paper's files;
  with the scTrilemma cache alone the scTrilemma rows are identical.
- `cohort_audit.py` on the Census release: `candidate_pairs.csv`, `dataset_best_candidates.csv`
  and `failures.csv` are byte-identical to the paper's files (371 candidate pairs, 18 datasets).
- `pathway.py` and `rq4_panel.py pathway`, replaying the paper's cached Enrichr responses
  (`--enrichr-cache`): `pathway_metrics_long.csv`, `pathway_cohort_summary.csv`,
  `pathway_equal_cohort_summary.csv`, `selected_contrasts.csv` (Table 2) and
  `pathway/*.csv`, `cohort_metric_summary.csv`, `equal_cohort_summary.csv`, the rank counts
  (Table 18) are byte-identical to the paper's files, so the gene-list construction hits the same
  cache keys and the metric code is unchanged. Two of the 410 gene lists of the panel were not in
  that cache and were answered live by Enrichr with identical results. Two files differ by design:
  `contrast_metric_pairwise_wins.csv` now compares the reference model with every other model
  (the paper's file has scVI and CellPLM only; those rows are identical), and
  `metric_combination_sensitivity.csv` is computed from the joint results of the same run (the
  paper's file combined the pathway summary with an earlier three-model joint run and is
  reproduced exactly when that run's `repeat_metrics.csv` is passed as `--joint-dir`).

## Enrichr caveat

`pathway.py` and `rq4_panel.py pathway` query Enrichr over the network. Term statistics depend on
the library versions served at query time; the three libraries used in the paper were still served
when this port was verified (`MSigDB_Hallmark_2020`: 50 terms, `Reactome_Pathways_2024`: 2,105
terms, `GO_Biological_Process_2025`: 5,343 terms), but Enrichr updates libraries in place, so
re-runs can differ from the shipped pathway values even with identical gene lists. Renamed
libraries fall back to the closest available version (`pathway_concordance.FALLBACK_LIBRARIES`).
Replaying the paper's own cached Enrichr responses through the ported code reproduces the paper's
pathway files (see the verification note), which checks the gene-list construction and the metric
code independently of the service.

## Not ported

In-process baseline inference and the baseline cache defaults of the private scripts; the
mean-imputation fallback for cells missing from an external cache (all caches must cover the sampled
cells exactly); the diagnostic three-axis scatter figure and the pathway tornado plot, which do not
appear in the paper.
