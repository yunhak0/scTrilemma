# Shared reconstruction gene lists

One file per held-out dataset (Ensembl gene IDs, one per line, ordered by seurat_v3
variance rank on the sampled cells). These lists define the genes on which expression
fidelity is scored in the matched-cohort analyses (Table 2, Tables 18–19) and in the
latent interventions (Figure 4, Table 20).

Definition: genes expressed in the staged 2,500-cell sample of the dataset
(`experiments/prepare_samples.py`) that are present in the vocabularies of all four
compared reconstruction models (scTrilemma, scVI, CellPLM, scPRINT), capped at 4,096
genes by seurat_v3 variance rank. `genelist_summary.csv` records the size of each
intersection step per dataset. The lists are shipped as data because rebuilding them
requires the baseline models' environments.
