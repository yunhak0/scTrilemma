"""Defaults and small helpers shared by the matched-cohort analyses."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import normalize_gene

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

MODEL_NAME = "sctrilemma"
MODEL_LABELS = {
    "scvi": "scVI",
    "geneformer": "Geneformer",
    "scgpt": "scGPT",
    "cellplm": "CellPLM",
    "scprint": "scPRINT",
    "sctrilemma": "scTrilemma",
}

DEFAULT_IDS_FILE = ROOT / "configs/zsb/full_89_ids.txt"
DEFAULT_SAMPLES_DIR = ROOT / "outputs/experiments/samples"
DEFAULT_EMBEDDING_DIR = ROOT / "outputs/experiments/embeddings" / MODEL_NAME / "embeddings"
DEFAULT_RECON_DIR = ROOT / "outputs/experiments/reconstruction/cache" / MODEL_NAME
DEFAULT_GENELIST_DIR = ROOT / "experiments/data/genelists"
DEFAULT_EXCLUDED_GENES = ROOT / "experiments/reconstruction/data/table2_excluded_genes.tsv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/experiments/table2_joint"
DEFAULT_CANDIDATE_PAIRS = HERE / "results/cohort_audit/candidate_pairs.csv"

SOMA_KEY = "soma_joinid"
ASSAY_KEY = "assay"
TISSUE_KEY = "tissue"
STATE_KEY = "disease"
CELL_TYPE_KEY = "cell_type"
DONOR_KEY = "donor_id"


def display_name(model: str) -> str:
    """Table label of a model name (unknown names pass through)."""
    return MODEL_LABELS.get(model, model)


def describe_path(path: Path | None) -> str | None:
    """Repository-relative string for run metadata (absolute only outside the repository)."""
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def parse_model_dirs(values: list[str] | None, default: dict[str, Path]) -> dict[str, Path]:
    """Parse repeated ``NAME=DIR`` options into an ordered model -> directory mapping.

    ``DIR`` is a directory of ``<dataset_id>.npz`` files or a path pattern containing
    ``{dataset_id}`` (see ``cache_path``).
    """
    if not values:
        return dict(default)
    mapping: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"expected NAME=DIR, got {value!r}")
        name, directory = value.split("=", 1)
        name = name.strip()
        if not name or name in mapping:
            raise argparse.ArgumentTypeError(f"invalid or repeated model name in {value!r}")
        mapping[name] = Path(directory).expanduser()
    return mapping


def cache_path(location: Path, dataset_id: str) -> Path:
    """Per-dataset cache file: ``<location>/<dataset_id>.npz`` or a ``{dataset_id}`` pattern."""
    text = str(location)
    if "{dataset_id}" in text:
        return Path(text.format(dataset_id=dataset_id))
    return location / f"{dataset_id}.npz"


def add_gene_universe_arguments(parser: argparse.ArgumentParser) -> None:
    """CLI options fixing the expression-fidelity gene universe of the paper."""
    group = parser.add_argument_group("gene universe")
    group.add_argument("--genelist-dir", type=Path, default=DEFAULT_GENELIST_DIR)
    group.add_argument(
        "--no-genelist",
        action="store_true",
        help="Do not restrict genes to the shared per-dataset gene list",
    )
    group.add_argument(
        "--excluded-genes",
        type=Path,
        default=DEFAULT_EXCLUDED_GENES,
        help="(dataset_id, gene) TSV of shared genes outside the paper's common gene universe",
    )
    group.add_argument(
        "--ignore-excluded-genes",
        action="store_true",
        help="Keep the genes listed in --excluded-genes",
    )
    group.add_argument(
        "--min-sample-detection-rate",
        type=float,
        default=0.02,
        help="Keep genes detected (raw count > 0) in at least this fraction of the pooled "
        "cells before any contrast-level filtering (paper: 0.02; 0 disables)",
    )


def read_genelist(path: Path) -> list[str]:
    """Read one shared per-dataset gene list (one identifier per line, file order kept)."""
    genes = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not genes:
        raise ValueError(f"Empty gene list: {path}")
    return genes


def load_excluded_genes(path: Path | None) -> dict[str, set[str]]:
    """Normalised excluded genes per dataset from a (dataset_id, gene) TSV."""
    if path is None:
        return {}
    excluded: dict[str, set[str]] = {}
    table = pd.read_csv(path, sep="\t", dtype=str)
    for dataset_id, gene in zip(table["dataset_id"], table["gene"]):
        excluded.setdefault(str(dataset_id), set()).add(normalize_gene(gene))
    return excluded


class GeneUniverse:
    """Per-dataset restriction of the common gene set to the paper's gene universe.

    The paper scored expression fidelity on the genes that every compared method had
    decoded for a dataset. With several reconstruction caches the intersection is taken
    on the fly; with fewer caches the shared gene list minus the shipped excluded genes
    reproduces that intersection exactly.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.genelist_dir: Path | None = None if args.no_genelist else args.genelist_dir
        excluded_path = None if args.ignore_excluded_genes else args.excluded_genes
        self.excluded_path: Path | None = excluded_path
        self.excluded = load_excluded_genes(excluded_path)
        self.min_sample_detection_rate = float(args.min_sample_detection_rate)

    def restrict(self, dataset_id: str, common: set[str]) -> set[str]:
        """Apply the gene-list and excluded-gene restrictions to a common gene set."""
        if self.genelist_dir is not None:
            listed = {
                normalize_gene(gene)
                for gene in read_genelist(self.genelist_dir / f"{dataset_id}.txt")
            }
            common = common & listed
        return common - self.excluded.get(dataset_id, set())

    def config(self) -> dict[str, object]:
        """Run-metadata description."""
        return {
            "genelist_dir": describe_path(self.genelist_dir),
            "excluded_genes": describe_path(self.excluded_path),
            "min_sample_detection_rate": self.min_sample_detection_rate,
        }


def normalized_obs(obs: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Copy selected metadata columns as plain strings with missing values as 'unknown'."""
    missing = [column for column in columns if column not in obs.columns]
    if missing:
        raise KeyError(f"missing obs columns {missing}")
    frame = obs[columns].copy()
    for column in columns:
        frame[column] = frame[column].astype("string").fillna("unknown").astype(str)
    return frame


def cache_soma_ids(payload: np.lib.npyio.NpzFile) -> np.ndarray | None:
    """Return the ``soma_joinid`` array of a cache as strings, or None when absent/unknown."""
    if SOMA_KEY not in payload.files:
        return None
    values = np.asarray(payload[SOMA_KEY])
    if values.dtype.kind in "iu" and np.any(values < 0):
        return None
    return values.astype(str)
