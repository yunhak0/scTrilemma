"""Shared gene-mapping utility.

The inference path uses this helper to map AnnData variable names into the
model's fixed gene vocabulary.
"""

import re

import numpy as np

# Matches trailing .<digits> or -<digits> suffixes (e.g. "GENE1.2" → "GENE1")
_GENE_SUFFIX_RE = re.compile(r"[\.-]\d+$")


def build_gene_mapping(
    var_names: list[str],
    gene_vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Map file-local gene names to global vocab indices.

    For each gene in *var_names* that exists in *gene_vocab* (after
    stripping trailing ``.<digits>`` / ``-<digits>`` suffixes), record:

    * the global vocab index, and
    * the file-local column index.

    Parameters
    ----------
    var_names:
        Gene names from ``adata.var_names`` (file-local ordering).
    gene_vocab:
        ``{gene_name: global_index}`` mapping loaded from
        ``gene_vocab_*.json``.

    Returns
    -------
    vocab_indices_map:
        1-D int array of global vocab indices (length = # matched genes).
    valid_file_indices:
        1-D int array of file-local column positions (same length).
    """
    mapping: list[int] = []
    valid_file_indices: list[int] = []
    for i, g in enumerate(var_names):
        g_clean = _GENE_SUFFIX_RE.sub("", g)
        if g_clean in gene_vocab:
            mapping.append(gene_vocab[g_clean])
            valid_file_indices.append(i)
    return np.array(mapping, dtype=np.intp), np.array(valid_file_indices, dtype=np.intp)
