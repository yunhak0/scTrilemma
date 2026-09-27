"""Download CellxGene Census metadata and build vocabularies.

This module creates the metadata files expected by the scTrilemma preprocessing
pipeline:

    {version}/census_info_{version}.json
    {version}/gene_metadata_{organism}_{version}.parquet
    {version}/gene_vocab_{organism}_{version}.json
    {version}/cell_metadata_{organism}_{version}.parquet
    {version}/cell_type_vocab_{organism}_{version}.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cellxgene_census
import pandas as pd
from tqdm import tqdm

CENSUS_VERSION = "2025-01-30"
DEFAULT_ORGANISM = "homo_sapiens"
DEFAULT_OUTPUT_DIR = "/scratch/${USER}/datasets/cellxgene"
DEFAULT_FILTER = "suspension_type != 'na' and is_primary_data == True"


def _version_tag(census_version: str) -> str:
    return census_version.replace("-", "")


def _version_dir(output_dir: str | Path, census_version: str) -> Path:
    return Path(os.path.expandvars(str(output_dir))).expanduser() / _version_tag(census_version)


def download_census_info(
    census_version: str = CENSUS_VERSION,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Download Census summary metadata as JSON."""
    version = _version_tag(census_version)
    out_path = _version_dir(output_dir, census_version) / f"census_info_{version}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with cellxgene_census.open_soma(census_version=census_version) as census:
        summary = census["census_info"]["summary"].read().concat().to_pandas()

    out_path.write_text(json.dumps(summary.to_dict(orient="records"), indent=2))
    print(f"Saved census info to {out_path}", flush=True)
    return out_path


def download_gene_metadata(
    census_version: str = CENSUS_VERSION,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    organism: str = DEFAULT_ORGANISM,
) -> tuple[Path, Path]:
    """Download gene metadata and build an Ensembl-ID vocabulary."""
    version = _version_tag(census_version)
    out_dir = _version_dir(output_dir, census_version)
    metadata_path = out_dir / f"gene_metadata_{organism}_{version}.parquet"
    vocab_path = out_dir / f"gene_vocab_{organism}_{version}.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    with cellxgene_census.open_soma(census_version=census_version) as census:
        gene_metadata = census["census_data"][organism].ms["RNA"].var.read()
        gene_metadata = gene_metadata.concat().to_pandas()

    gene_metadata.to_parquet(metadata_path, index=False)

    vocab: dict[str, int] = {"<pad>": 0, "<mask>": 1}
    for gene_id in gene_metadata["feature_id"].astype(str).tolist():
        if gene_id not in vocab:
            vocab[gene_id] = len(vocab)

    vocab_path.write_text(json.dumps(vocab, indent=2))
    print(f"Saved gene metadata to {metadata_path}", flush=True)
    print(f"Saved gene vocabulary with {len(vocab)} tokens to {vocab_path}", flush=True)
    return metadata_path, vocab_path


def get_ensembl_data(
    release: int,
    gtf_path: str | None = None,
    fasta_path: str | None = None,
):
    """Return a pyensembl reference object for metadata enrichment.

    Ensembl release 114 is not available through the standard pyensembl
    release helper in our environment, so the 20251108 workflow supplies a
    local GTF file.
    """
    import pyensembl
    from pyensembl import Genome

    if release == 114:
        if not gtf_path or not Path(gtf_path).exists():
            raise FileNotFoundError(
                "Ensembl 114 requires a local GTF file. Expected path was "
                f"{gtf_path!r}. Download, for example, "
                "https://ftp.ensembl.org/pub/release-114/gtf/homo_sapiens/"
                "Homo_sapiens.GRCh38.114.gtf.gz"
            )
        transcript_fasta_paths = (
            [fasta_path] if fasta_path and Path(fasta_path).exists() else []
        )
        data = Genome(
            reference_name="GRCh38",
            annotation_name="Ensembl_114",
            gtf_path_or_url=gtf_path,
            transcript_fasta_paths_or_urls=transcript_fasta_paths,
        )
        data.index()
        return data

    data = pyensembl.EnsemblRelease(release)
    try:
        data.genes()
    except Exception:
        print(f"Downloading Ensembl release {release} data...", flush=True)
        data.download()
        data.index()
    return data


def enrich_gene_metadata(
    input_path: str | Path,
    output_path: str | Path,
    *,
    ensembl_release: int,
    id_column: str = "feature_id",
    symbol_column: str = "feature_name",
    gtf_path: str | None = None,
    fasta_path: str | None = None,
) -> Path:
    """Add Ensembl genomic coordinates and gene biotypes to gene metadata."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    print(f"Enriching gene metadata with Ensembl release {ensembl_release}", flush=True)
    gene_metadata = pd.read_parquet(input_path)
    ensembl = get_ensembl_data(ensembl_release, gtf_path=gtf_path, fasta_path=fasta_path)

    enriched_cols: dict[str, list[object]] = {
        "chromosome": [],
        "start": [],
        "end": [],
        "strand": [],
        "gene_type": [],
        "ensembl_gene_name": [],
        "source": [],
        "version": [],
    }

    unknown_count = 0
    for _, row in tqdm(gene_metadata.iterrows(), total=len(gene_metadata)):
        gene = None
        gene_id = row.get(id_column)
        if gene_id is not None and bool(pd.notna(gene_id)):
            try:
                gene = ensembl.gene_by_id(str(gene_id))
            except ValueError:
                pass

        gene_symbol = row.get(symbol_column)
        if gene is None and gene_symbol is not None and bool(pd.notna(gene_symbol)):
            try:
                matches = ensembl.genes_by_name(str(gene_symbol))
                if matches:
                    gene = matches[0]
            except ValueError:
                pass

        if gene is None:
            unknown_count += 1
            enriched_cols["chromosome"].append("unknown")
            enriched_cols["start"].append(-1)
            enriched_cols["end"].append(-1)
            enriched_cols["strand"].append("unknown")
            enriched_cols["gene_type"].append("unknown")
            enriched_cols["ensembl_gene_name"].append("unknown")
            enriched_cols["source"].append("unknown")
            enriched_cols["version"].append(-1)
            continue

        enriched_cols["chromosome"].append(gene.contig)
        enriched_cols["start"].append(gene.start)
        enriched_cols["end"].append(gene.end)
        enriched_cols["strand"].append(gene.strand)
        enriched_cols["gene_type"].append(gene.biotype)
        enriched_cols["ensembl_gene_name"].append(gene.gene_name)
        enriched_cols["source"].append(getattr(gene, "source", "ensembl"))
        enriched_cols["version"].append(getattr(gene, "version", 1))

    for col, values in enriched_cols.items():
        gene_metadata[col] = values

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gene_metadata.to_parquet(output_path, index=False)
    print(
        f"Saved enriched gene metadata to {output_path}; "
        f"{unknown_count}/{len(gene_metadata)} genes unresolved.",
        flush=True,
    )
    return output_path


def download_cell_metadata(
    census_version: str = CENSUS_VERSION,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    organism: str = DEFAULT_ORGANISM,
    value_filter: str = DEFAULT_FILTER,
    min_cells: int = 2,
) -> Path:
    """Download filtered cell metadata used for dataset-level h5ad downloads."""
    version = _version_tag(census_version)
    out_dir = _version_dir(output_dir, census_version)
    metadata_path = out_dir / f"cell_metadata_{organism}_{version}.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)

    with cellxgene_census.open_soma(census_version=census_version) as census:
        cell_metadata = census["census_data"][organism].obs.read(
            value_filter=value_filter
        )
        cell_metadata = cell_metadata.concat().to_pandas()

    if "is_primary_data" in cell_metadata.columns:
        if cell_metadata["is_primary_data"].dtype == "object":
            cell_metadata["is_primary_data"] = cell_metadata["is_primary_data"].apply(
                lambda value: value[0] if isinstance(value, list) and value else value
            )
        cell_metadata = cell_metadata[cell_metadata["is_primary_data"].astype(bool)]

    if "dataset_id" in cell_metadata.columns:
        if isinstance(cell_metadata["dataset_id"].dtype, pd.CategoricalDtype):
            cell_metadata["dataset_id"] = cell_metadata["dataset_id"].astype(str)
        dataset_counts = cell_metadata["dataset_id"].value_counts()
        valid_datasets = dataset_counts[dataset_counts >= min_cells].index
        cell_metadata = cell_metadata[cell_metadata["dataset_id"].isin(valid_datasets)]

    cell_metadata.to_parquet(metadata_path, index=False)
    print(f"Saved cell metadata with {len(cell_metadata):,} cells to {metadata_path}", flush=True)
    return metadata_path


def build_cell_type_vocab(
    cell_metadata_path: str | Path,
    output_dir: str | Path,
    organism: str,
    census_version: str,
) -> Path:
    """Build a cell-type ontology vocabulary for validation labels."""
    version = _version_tag(census_version)
    out_path = _version_dir(output_dir, census_version) / f"cell_type_vocab_{organism}_{version}.json"
    df = pd.read_parquet(cell_metadata_path, columns=["cell_type_ontology_term_id"])
    labels = df["cell_type_ontology_term_id"].astype(str).fillna("unknown")
    counts = labels.value_counts()
    sorted_labels = sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    vocab = {"unknown": 0}
    for label, _count in sorted_labels:
        if label != "unknown" and label.strip():
            vocab[label] = len(vocab)

    out_path.write_text(json.dumps(vocab, indent=2))
    print(f"Saved cell-type vocabulary with {len(vocab)} labels to {out_path}", flush=True)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CellxGene Census metadata")
    parser.add_argument("--census_version", default=CENSUS_VERSION)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--organism", default=DEFAULT_ORGANISM)
    parser.add_argument("--value_filter", default=DEFAULT_FILTER)
    parser.add_argument("--min_cells", type=int, default=2)
    parser.add_argument("--ensembl_release", type=int, default=110)
    parser.add_argument("--id_column", default="feature_id")
    parser.add_argument("--symbol_column", default="feature_name")
    parser.add_argument("--gtf_path", default=None)
    parser.add_argument("--fasta_path", default=None)
    parser.add_argument(
        "--skip_ensembl_enrichment",
        action="store_true",
        help="Only download Census metadata and vocabularies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Starting Census metadata download", flush=True)
    print(f"Version: {args.census_version}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)

    download_census_info(args.census_version, args.output_dir)
    gene_metadata_path, _ = download_gene_metadata(
        args.census_version,
        args.output_dir,
        args.organism,
    )
    cell_metadata_path = download_cell_metadata(
        args.census_version,
        args.output_dir,
        args.organism,
        args.value_filter,
        args.min_cells,
    )
    build_cell_type_vocab(
        cell_metadata_path,
        args.output_dir,
        args.organism,
        args.census_version,
    )
    if not args.skip_ensembl_enrichment:
        enrich_gene_metadata(
            gene_metadata_path,
            gene_metadata_path,
            ensembl_release=args.ensembl_release,
            id_column=args.id_column,
            symbol_column=args.symbol_column,
            gtf_path=args.gtf_path,
            fasta_path=args.fasta_path,
        )
    print("Metadata download finished.", flush=True)


if __name__ == "__main__":
    main()
