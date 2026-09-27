# Paper analyses

Each folder reproduces one table or figure of the paper and ships the result files the
paper was built from under `results/`. The folders assume the released checkpoint at
`checkpoints/sctrilemma/final.ckpt` and the CELLxGENE Census data layout described in the
top-level README (`SCTRILEMMA_DATA_ROOT`).

| Folder | Reproduces |
|---|---|
| `table1_benchmark/` | Table 1 — zero-shot benchmark on 89 held-out datasets (20-repeat protocol) |
| `table2_joint/` | Table 2 — matched-cohort joint evaluation; appendix Tables 3, 4, 7, 18 and 19 |
| `fig3_ablation/` | Figure 3, Table 24 — component ablation |
| `fig4_interventions/` | Figure 4, Table 20 — demand-targeted latent interventions |

Shared pieces: `scoring/` (Table 1 embedding scorers: metric samples, FAISS k-means repeats, scib-metrics repeats, PCR repeats), `reconstruction/` (reconstruction generation and scoring), `prepare_samples.py` (stratified ≤ 2,500-cell sample per held-out dataset),
`common.py` (sampling, gene alignment and reconstruction scoring helpers),
`figure_style.py` (typography), `data/genelists/` (per-dataset shared gene lists).

Run modules from the repository root, e.g. `pixi run python -m experiments.prepare_samples`.

Not included: the appendix analyses that require retraining many additional arms or the baseline
environments (hyperparameter sensitivity, PB-Cond controls beyond the two ablation arms, runtime
profiling of the baselines) and the mechanistic figures; their numbers are in the paper and can be
provided on request.
