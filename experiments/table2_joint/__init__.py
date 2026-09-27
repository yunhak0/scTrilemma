"""Matched-cohort analyses of the paper (Table 2 and appendix Tables 3, 4, 7, 18, 19).

Modules, in execution order (see README.md):

* ``cohort_audit``        - metadata-only candidate audit of the held-out datasets (Table 4)
* ``joint``               - five common cohorts, 20 balanced repeats, identity / state /
                            context-invariance metrics on embeddings and DEG fidelity on
                            reconstructions (Table 2, Table 3)
* ``pathway``             - pathway-level fidelity on the same repeats via Enrichr (Table 2)
* ``rq4_panel``           - donor-supported seven-cohort DEG and pathway panel (Tables 7, 18)
* ``deg_concordance``, ``pathway_concordance`` - contrast-level scorers used by ``rq4_panel``
* ``context_metrics``     - residual donor variance and local donor mixing
"""
