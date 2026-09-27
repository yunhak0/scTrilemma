"""Pathway-level concordance of reconstructed differential expression (Enrichr).

For a within-cell-type contrast the raw and each reconstructed logFC ranking over the detected
genes give top-``top_genes`` up- and down-regulated gene symbols, which are submitted to
Enrichr (``MSigDB_Hallmark_2020``, ``Reactome_Pathways_2024``, ``GO_Biological_Process_2025``).
Per direction and library the reconstruction is scored against the raw ranking by the Jaccard
overlap of the top-``top_terms`` terms (by adjusted p-value) and the Spearman correlation of
the combined scores over the union of those terms. ``rq4_panel.py`` and ``pathway.py`` drive
this module; the command line below scores staged datasets with a fixed contrast.

Enrichr is queried over the network (https://maayanlab.cloud/Enrichr); responses are cached in
``enrichr_cache.jsonl`` keyed by the exact gene list and libraries. Enrichr library versions
and term statistics can change over time, so re-runs need not reproduce the paper's values.

    pixi run python -m experiments.table2_joint.pathway_concordance --positive-label "colorectal cancer"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

from experiments.common import load_sampled_adata, normalize_gene, read_ids
from experiments.table2_joint._common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RECON_DIR,
    MODEL_NAME,
    GeneUniverse,
    describe_path,
    display_name,
    parse_model_dirs,
)
from experiments.table2_joint.deg_concordance import (
    Contrast,
    add_cache_arguments,
    align_common_matrices,
    compute_logfc,
    load_recon_cache,
    select_contrasts,
)

ENRICHR_BASE = "https://maayanlab.cloud/Enrichr"
ENRICHR_MAX_ATTEMPTS = 8
ENRICHR_INITIAL_BACKOFF = 5.0
ENRICHR_MAX_BACKOFF = 60.0
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}
DEFAULT_LIBRARIES = [
    "MSigDB_Hallmark_2020",
    "Reactome_Pathways_2024",
    "GO_Biological_Process_2025",
]
FALLBACK_LIBRARIES = {
    "Reactome_Pathways_2024": ["Reactome_2022", "Reactome_2016"],
    "GO_Biological_Process_2025": [
        "GO_Biological_Process_2023",
        "GO_Biological_Process_2021",
    ],
}
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "pathway_concordance"


def retry_delay(retry_after: str | None, attempt: int) -> float:
    """Return a bounded Retry-After or exponential-backoff delay."""
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), ENRICHR_MAX_BACKOFF)
        except ValueError:
            pass
    return min(ENRICHR_INITIAL_BACKOFF * (2**attempt), ENRICHR_MAX_BACKOFF)


def request_json(url: str, timeout: int = 45) -> Any:
    """GET JSON with bounded retries for transient Enrichr failures.

    Uses ``requests`` (certifi CA bundle) so that the call works in environments without a
    system certificate store; non-retryable HTTP errors are raised as ``requests.HTTPError``.
    """
    for attempt in range(ENRICHR_MAX_ATTEMPTS):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status not in RETRYABLE_HTTP_STATUS or attempt + 1 == ENRICHR_MAX_ATTEMPTS:
                raise
            retry_after = error.response.headers.get("Retry-After") if error.response is not None else None
            delay = retry_delay(retry_after, attempt)
        except requests.RequestException:
            if attempt + 1 == ENRICHR_MAX_ATTEMPTS:
                raise
            delay = retry_delay(None, attempt)
        print(
            f"Enrichr request retry {attempt + 1}/{ENRICHR_MAX_ATTEMPTS - 1} "
            f"after {delay:.1f}s: {url}",
            flush=True,
        )
        time.sleep(delay)
    raise RuntimeError("Unreachable Enrichr retry state")


def available_libraries() -> set[str]:
    """Library names currently served by Enrichr."""
    payload = request_json(f"{ENRICHR_BASE}/datasetStatistics")
    libraries: set[str] = set()
    for item in payload.get("statistics", payload if isinstance(payload, list) else []):
        if isinstance(item, dict) and item.get("libraryName"):
            libraries.add(str(item["libraryName"]))
    return libraries


def resolve_libraries(requested: list[str]) -> list[str]:
    """Requested libraries that Enrichr serves, with documented fallbacks for renamed ones."""
    available = available_libraries()
    resolved: list[str] = []
    for library in requested:
        if library in available:
            resolved.append(library)
            continue
        for fallback in FALLBACK_LIBRARIES.get(library, []):
            if fallback in available:
                resolved.append(fallback)
                break
        else:
            print(f"Warning: Enrichr library unavailable and skipped: {library}")
    if not resolved:
        raise RuntimeError("No requested Enrichr libraries are available.")
    return resolved


class EnrichrClient:
    """Minimal Enrichr client with an append-only JSONL response cache."""

    def __init__(self, *, cache_path: Path, sleep: float) -> None:
        self.cache_path = cache_path
        self.sleep = sleep
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, list[dict[str, Any]]] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                self.cache[str(item["key"])] = list(item["rows"])

    def _key(self, genes: list[str], libraries: list[str]) -> str:
        payload = json.dumps(
            {"genes": genes, "libraries": libraries},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _store(self, key: str, rows: list[dict[str, Any]]) -> None:
        self.cache[key] = rows
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "rows": rows}) + "\n")

    def _submit_gene_list(self, genes: list[str], description: str) -> int:
        """Submit one gene list with bounded retries for transient failures."""
        for attempt in range(ENRICHR_MAX_ATTEMPTS):
            time.sleep(self.sleep)
            try:
                response = requests.post(
                    f"{ENRICHR_BASE}/addList",
                    files={
                        "list": (None, "\n".join(genes)),
                        "description": (None, description),
                    },
                    timeout=45,
                )
                response.raise_for_status()
                return int(response.json()["userListId"])
            except requests.RequestException as error:
                status = error.response.status_code if error.response is not None else None
                if (
                    status is not None and status not in RETRYABLE_HTTP_STATUS
                ) or attempt + 1 == ENRICHR_MAX_ATTEMPTS:
                    raise
                retry_after = (
                    error.response.headers.get("Retry-After")
                    if error.response is not None
                    else None
                )
                delay = retry_delay(retry_after, attempt)
                print(
                    f"Enrichr addList retry {attempt + 1}/"
                    f"{ENRICHR_MAX_ATTEMPTS - 1} after {delay:.1f}s "
                    f"(status={status})",
                    flush=True,
                )
                time.sleep(delay)
        raise RuntimeError("Unreachable Enrichr retry state")

    def enrich(
        self,
        genes: list[str],
        *,
        libraries: list[str],
        description: str,
    ) -> list[dict[str, Any]]:
        """Enrichment rows of one gene list over ``libraries`` (cached)."""
        key = self._key(genes, libraries)
        if key in self.cache:
            return self.cache[key]

        user_list_id = self._submit_gene_list(genes, description)

        rows: list[dict[str, Any]] = []
        for library in libraries:
            payload = None
            for resubmit in range(3):
                query = urllib.parse.urlencode(
                    {"userListId": str(user_list_id), "backgroundType": library}
                )
                try:
                    payload = request_json(f"{ENRICHR_BASE}/enrich?{query}")
                    break
                except requests.HTTPError as error:
                    # Enrichr intermittently answers 400 when a freshly added list is not
                    # yet queryable; re-submit the list and try again.
                    status = error.response.status_code if error.response is not None else None
                    if status != 400 or resubmit == 2:
                        raise
                    print(
                        f"Enrichr enrich 400 for list {user_list_id}; re-submitting "
                        f"({resubmit + 1}/2)",
                        flush=True,
                    )
                    time.sleep(max(self.sleep, 2.0) * (resubmit + 1))
                    user_list_id = self._submit_gene_list(genes, description)
            assert payload is not None
            for item in payload.get(library, []):
                rows.append(
                    {
                        "library": library,
                        "rank": int(item[0]),
                        "term": str(item[1]),
                        "p_value": float(item[2]),
                        "z_score": float(item[3]),
                        "combined_score": float(item[4]),
                        "overlap_genes": ";".join(item[5]),
                        "adjusted_p_value": float(item[6]),
                    }
                )
            time.sleep(self.sleep)

        self._store(key, rows)
        return rows


def gene_symbol_lookup(adata) -> dict[str, str]:
    """Map normalised feature identifiers to gene symbols (``feature_name`` when present)."""
    lookup: dict[str, str] = {}
    feature_names = (
        adata.var["feature_name"].astype(str)
        if "feature_name" in adata.var.columns
        else pd.Series(adata.var_names.astype(str), index=adata.var.index)
    )
    for idx, symbol in zip(adata.var_names.astype(str), feature_names, strict=False):
        symbol_str = str(symbol).strip()
        if not symbol_str or symbol_str.upper().startswith("ENSG"):
            symbol_str = str(idx).strip()
        lookup.setdefault(normalize_gene(idx), symbol_str)
    for column in ("feature_id", "feature_name"):
        if column not in adata.var.columns:
            continue
        for value, symbol in zip(adata.var[column].astype(str), feature_names, strict=False):
            symbol_str = str(symbol).strip()
            if symbol_str and not symbol_str.upper().startswith("ENSG"):
                lookup.setdefault(normalize_gene(value), symbol_str)
    return lookup


def clean_gene_symbol(symbol: str) -> str | None:
    """Upper-case symbol, or None for empty or Ensembl-only identifiers."""
    value = str(symbol).strip()
    if not value or value.upper().startswith("ENSG"):
        return None
    return value.upper()


def ranked_gene_set(
    values: np.ndarray,
    symbols: list[str],
    *,
    direction: str,
    top_n: int,
) -> list[str]:
    """Top-``top_n`` unique gene symbols by logFC in one direction (finite values only)."""
    order = np.argsort(values)
    if direction == "up":
        order = order[::-1]
    elif direction != "down":
        raise ValueError(f"Unknown direction: {direction}")

    genes: list[str] = []
    seen: set[str] = set()
    for idx in order:
        if not np.isfinite(values[idx]):
            continue
        gene = clean_gene_symbol(symbols[int(idx)])
        if gene is None or gene in seen:
            continue
        genes.append(gene)
        seen.add(gene)
        if len(genes) >= top_n:
            break
    return genes


def metric_rows_for_terms(
    *,
    model: str,
    contrast: Contrast,
    direction: str,
    library: str,
    raw_terms: pd.DataFrame,
    model_terms: pd.DataFrame,
    top_terms: int,
) -> list[dict[str, object]]:
    """Top-term Jaccard and combined-score Spearman of one model against the raw ranking."""
    raw_top = raw_terms.sort_values(["adjusted_p_value", "rank"]).head(top_terms)
    model_top = model_terms.sort_values(["adjusted_p_value", "rank"]).head(top_terms)
    raw_set = set(raw_top["term"].astype(str))
    model_set = set(model_top["term"].astype(str))
    union_set = raw_set | model_set
    jaccard = len(raw_set & model_set) / len(union_set) if union_set else np.nan

    score_terms = list(union_set)
    raw_score = raw_terms.set_index("term")["combined_score"].to_dict()
    model_score = model_terms.set_index("term")["combined_score"].to_dict()
    raw_values = np.asarray([float(raw_score.get(term, 0.0)) for term in score_terms])
    model_values = np.asarray([float(model_score.get(term, 0.0)) for term in score_terms])
    if len(score_terms) >= 3 and np.any(raw_values) and np.any(model_values):
        rho = spearmanr(raw_values, model_values).statistic
    else:
        rho = np.nan

    base = {
        "model": model,
        "dataset_id": contrast.dataset_id,
        "contrast_id": contrast.contrast_id,
        "cell_type": contrast.cell_type,
        "positive_label": contrast.positive_label,
        "negative_label": contrast.negative_label,
        "direction": direction,
        "library": library,
    }
    return [
        {**base, "metric": f"top{top_terms}_pathway_jaccard", "value": float(jaccard)},
        {
            **base,
            "metric": f"top{top_terms}_pathway_score_spearman",
            "value": float(rho) if np.isfinite(rho) else np.nan,
        },
    ]


def detected_gene_mask(
    raw_counts: np.ndarray,
    *,
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
    min_detection_rate: float,
) -> np.ndarray:
    """Genes whose summed detection rate over the two groups exceeds the threshold."""
    return (
        (raw_counts[pos_mask] > 0).mean(axis=0) + (raw_counts[neg_mask] > 0).mean(axis=0)
    ) > min_detection_rate


def evaluate_pathway_contrast(
    *,
    adata,
    genes: list[str],
    symbols: list[str],
    raw_counts: np.ndarray,
    raw_log: np.ndarray,
    recon_log_by_model: dict[str, np.ndarray],
    contrast: Contrast,
    contrast_key: str,
    cell_type_key: str,
    min_detection_rate: float,
    top_genes: int,
    top_terms: int,
    libraries: list[str],
    client: EnrichrClient,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Enrichment rows and pathway-concordance metric rows for one contrast."""
    obs = adata.obs
    pos_mask = (obs[cell_type_key].astype(str).to_numpy() == contrast.cell_type) & (
        obs[contrast_key].astype(str).to_numpy() == contrast.positive_label
    )
    neg_mask = (obs[cell_type_key].astype(str).to_numpy() == contrast.cell_type) & (
        obs[contrast_key].astype(str).to_numpy() == contrast.negative_label
    )
    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        return [], []

    raw_logfc = compute_logfc(raw_log, pos_mask, neg_mask)
    keep = detected_gene_mask(
        raw_counts,
        pos_mask=pos_mask,
        neg_mask=neg_mask,
        min_detection_rate=min_detection_rate,
    ) & np.isfinite(raw_logfc)
    if int(keep.sum()) < max(30, top_genes):
        return [], []

    kept_symbols = [symbols[idx] for idx in np.flatnonzero(keep)]
    raw_eval = raw_logfc[keep]
    enrichment_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    source_values = {"raw": raw_eval}
    for model_name, recon_log in recon_log_by_model.items():
        source_values[model_name] = compute_logfc(recon_log, pos_mask, neg_mask)[keep]

    for direction in ("up", "down"):
        source_enrichment: dict[str, pd.DataFrame] = {}
        for source, values in source_values.items():
            gene_set = ranked_gene_set(values, kept_symbols, direction=direction, top_n=top_genes)
            if len(gene_set) < 10:
                continue
            rows = client.enrich(
                gene_set,
                libraries=libraries,
                description=f"{contrast.contrast_id}:{source}:{direction}",
            )
            enrichment_rows.extend(
                {
                    **row,
                    "source": source,
                    "model": source,
                    "dataset_id": contrast.dataset_id,
                    "contrast_id": contrast.contrast_id,
                    "cell_type": contrast.cell_type,
                    "positive_label": contrast.positive_label,
                    "negative_label": contrast.negative_label,
                    "direction": direction,
                    "n_input_genes": len(gene_set),
                    "input_genes": ";".join(gene_set),
                }
                for row in rows
            )
            source_enrichment[source] = pd.DataFrame(rows)

        raw_df = source_enrichment.get("raw")
        if raw_df is None or raw_df.empty:
            continue
        for model_name in recon_log_by_model:
            model_df = source_enrichment.get(model_name)
            if model_df is None or model_df.empty:
                continue
            for library in libraries:
                raw_library = raw_df[raw_df["library"] == library]
                model_library = model_df[model_df["library"] == library]
                if raw_library.empty or model_library.empty:
                    continue
                metric_rows.extend(
                    metric_rows_for_terms(
                        model=model_name,
                        contrast=contrast,
                        direction=direction,
                        library=library,
                        raw_terms=raw_library,
                        model_terms=model_library,
                        top_terms=top_terms,
                    )
                )

    return enrichment_rows, metric_rows


def summarize(metrics: list[dict[str, object]]) -> pd.DataFrame:
    """Mean per (model, metric) over (contrast, direction, library) comparisons."""
    df = pd.DataFrame(metrics)
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["model", "metric"], as_index=False)
        .agg(
            mean_value=("value", "mean"),
            n_comparisons=("contrast_id", "count"),
            n_contrasts=("contrast_id", "nunique"),
        )
        .sort_values(["metric", "mean_value"], ascending=[True, False])
    )


def add_enrichr_arguments(parser: argparse.ArgumentParser) -> None:
    """CLI options of the Enrichr step."""
    group = parser.add_argument_group("enrichr")
    group.add_argument("--top-genes", type=int, default=100)
    group.add_argument("--top-terms", type=int, default=10)
    group.add_argument("--libraries", nargs="*", default=DEFAULT_LIBRARIES)
    group.add_argument("--sleep", type=float, default=0.05, help="Pause between requests (s)")
    group.add_argument(
        "--enrichr-cache",
        type=Path,
        default=None,
        help="JSONL response cache (default: <output dir>/enrichr_cache.jsonl)",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_cache_arguments(parser)
    add_enrichr_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    libraries = resolve_libraries(list(args.libraries))
    cache_dirs = parse_model_dirs(args.recon_cache, {MODEL_NAME: DEFAULT_RECON_DIR})
    universe = GeneUniverse(args)
    dataset_ids = read_ids(args.dataset_ids_file) if args.dataset_ids_file else args.dataset_ids
    client = EnrichrClient(
        cache_path=args.enrichr_cache or args.output_dir / "enrichr_cache.jsonl",
        sleep=args.sleep,
    )
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "recon_caches": {k: describe_path(v) for k, v in cache_dirs.items()},
        **universe.config(),
        "resolved_libraries": libraries,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_enrichment: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []
    all_contrasts: list[dict[str, object]] = []
    for dataset_id in dataset_ids:
        print(f"\n=== Dataset {dataset_id} ===", flush=True)
        adata = load_sampled_adata(args.samples_dir / f"{dataset_id}.h5ad", dataset_id)
        contrasts = select_contrasts(
            adata,
            dataset_id=dataset_id,
            contrast_key=args.contrast_key,
            cell_type_key=args.cell_type_key,
            positive_label=args.positive_label,
            negative_label=args.negative_label,
            min_cells_per_group=args.min_cells_per_group,
            max_contrasts=args.max_contrasts,
        )
        print(f"Sampled {adata.n_obs:,} cells; selected {len(contrasts)} contrasts")
        all_contrasts.extend(asdict(contrast) for contrast in contrasts)
        if not contrasts:
            continue

        results = [
            load_recon_cache(display_name(model), location, dataset_id, adata)
            for model, location in cache_dirs.items()
        ]
        genes, raw_counts, raw_log, recon_log_by_model = align_common_matrices(
            adata, results, universe=universe
        )
        symbol_lookup = gene_symbol_lookup(adata)
        symbols = [symbol_lookup.get(normalize_gene(gene), gene) for gene in genes]
        print(f"Common pathway gene universe: {len(genes):,} genes")

        for contrast in contrasts:
            print(f"  Enrichment: {contrast.cell_type}", flush=True)
            enrichment_rows, metric_rows = evaluate_pathway_contrast(
                adata=adata,
                genes=genes,
                symbols=symbols,
                raw_counts=raw_counts,
                raw_log=raw_log,
                recon_log_by_model=recon_log_by_model,
                contrast=contrast,
                contrast_key=args.contrast_key,
                cell_type_key=args.cell_type_key,
                min_detection_rate=args.min_detection_rate,
                top_genes=args.top_genes,
                top_terms=args.top_terms,
                libraries=libraries,
                client=client,
            )
            all_enrichment.extend(enrichment_rows)
            all_metrics.extend(metric_rows)

    pd.DataFrame(all_contrasts).to_csv(args.output_dir / "selected_contrasts.csv", index=False)
    enrichment_df = pd.DataFrame(all_enrichment)
    metrics_df = pd.DataFrame(all_metrics)
    if not enrichment_df.empty:
        enrichment_df.to_csv(args.output_dir / "enrichment_long.csv", index=False)
    if not metrics_df.empty:
        metrics_df.to_csv(args.output_dir / "pathway_metrics_long.csv", index=False)
        summary = summarize(all_metrics)
        summary.to_csv(args.output_dir / "summary.csv", index=False)
        print("\nSummary")
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    else:
        print("\nNo pathway metric rows produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
