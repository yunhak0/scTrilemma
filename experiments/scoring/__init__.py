"""Embedding-scoring stack of Table 1 (see experiments/scoring/README.md).

Modules, in execution order:

* ``metric_samples``  - 20 stratified metric samples per dataset (cache row indices)
* ``kmeans_repeat``   - FAISS K-means NMI/ARI over 20 seeds + cLISI/BRAS on the seed-0 sample
* ``scib_repeat``     - scib-metrics protocol (NMI/ARI, silhouette, isolated labels, LISI, BRAS)
* ``pcr_repeat``      - scIB principal-component-regression comparison on the 20 samples

All modules read a generic embedding cache directory (``<dataset>.npz`` with ``embeddings``,
``labels``, ``batches``) written by ``sctrilemma.benchmark.export_embeddings`` and tag their
rows with ``--model-name``.
"""
