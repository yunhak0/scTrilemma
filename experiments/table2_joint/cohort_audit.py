"""Outcome-blind audit of biological-state contrasts in the held-out datasets (Table 4).

Candidates are evaluated from Census metadata only (no model output). A candidate fixes the
assay and the anatomical tissue and compares two disease labels within each cell type; each
state must have at least ``--min-cells`` cells from ``--min-donors`` donors, both overall and
within an eligible cell type. The output ``candidate_pairs.csv`` is the input of the cohort
selection audit in ``joint.py`` (``--candidate-pairs``); ``dataset_best_candidates.csv`` keeps
the best-supported candidate per dataset (normal contrasts first, then more eligible cell
types, more donors, more cells).

Reads only the ``obs`` tables of ``$SCTRILEMMA_DATA_ROOT/20251108/by_dataset/<dataset>/*.h5ad``.

    pixi run python -m experiments.table2_joint.cohort_audit
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
from pathlib import Path
from typing import Any, cast

import anndata as ad
import pandas as pd

from experiments.common import read_ids
from experiments.table2_joint._common import (
    ASSAY_KEY,
    CELL_TYPE_KEY,
    DEFAULT_IDS_FILE,
    DEFAULT_OUTPUT_ROOT,
    DONOR_KEY,
    STATE_KEY,
    TISSUE_KEY,
)

DATA_ROOT = Path(
    os.environ.get(
        "SCTRILEMMA_DATA_ROOT", f"/scratch/{os.environ.get('USER', 'user')}/datasets/cellxgene"
    )
)
DEFAULT_DATA_ROOT = DATA_ROOT / "20251108" / "by_dataset"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "cohort_audit"

REQUIRED_COLUMNS = (ASSAY_KEY, CELL_TYPE_KEY, STATE_KEY, DONOR_KEY, TISSUE_KEY)
UNKNOWN_VALUES = {"", "nan", "none", "unknown"}


def read_metadata(dataset_dir: Path) -> pd.DataFrame:
    """Read required observation metadata across H5AD shards without loading X."""
    frames: list[pd.DataFrame] = []
    for shard in sorted(dataset_dir.glob("*.h5ad")):
        source = ad.read_h5ad(shard, backed="r")
        try:
            missing = [column for column in REQUIRED_COLUMNS if column not in source.obs]
            if missing:
                raise ValueError(f"Missing columns: {missing}")
            frame = cast(pd.DataFrame, pd.DataFrame(source.obs)[list(REQUIRED_COLUMNS)]).copy()
            frames.append(frame)
        finally:
            source.file.close()
    if not frames:
        raise FileNotFoundError(f"No H5AD shard under {dataset_dir}")
    metadata = pd.concat(frames, ignore_index=True)
    for column in REQUIRED_COLUMNS:
        metadata[column] = metadata[column].astype("string").fillna("unknown").astype(str)
    return metadata


def eligible_cell_types(
    frame: pd.DataFrame,
    state_a: str,
    state_b: str,
    min_cells: int,
    min_donors: int,
) -> list[str]:
    """Return cell types supported by both biological states across donors."""
    eligible: list[str] = []
    for cell_type, group in frame.groupby(CELL_TYPE_KEY, observed=True, sort=True):
        state_counts = group.groupby(STATE_KEY, observed=True).agg(
            n_cells=(DONOR_KEY, "size"),
            n_donors=(DONOR_KEY, "nunique"),
        )
        if all(
            state in state_counts.index
            and int(state_counts.loc[state, "n_cells"]) >= min_cells
            and int(state_counts.loc[state, "n_donors"]) >= min_donors
            for state in (state_a, state_b)
        ):
            eligible.append(str(cell_type))
    return eligible


def candidate_rows(
    metadata: pd.DataFrame,
    dataset_id: str,
    min_cells: int,
    min_donors: int,
) -> list[dict[str, Any]]:
    """Enumerate assay/tissue/two-state comparisons supported by metadata."""
    valid = metadata.copy()
    for column in REQUIRED_COLUMNS:
        valid = valid.loc[~valid[column].str.lower().isin(UNKNOWN_VALUES)]

    rows: list[dict[str, Any]] = []
    for group_key, frame in valid.groupby([ASSAY_KEY, TISSUE_KEY], observed=True, sort=True):
        if not isinstance(group_key, tuple) or len(group_key) != 2:
            raise RuntimeError("Expected an assay/tissue group key.")
        assay, tissue = (str(group_key[0]), str(group_key[1]))
        disease_summary = frame.groupby(STATE_KEY, observed=True).agg(
            n_cells=(DONOR_KEY, "size"),
            n_donors=(DONOR_KEY, "nunique"),
        )
        states = [
            str(state)
            for state, summary in disease_summary.iterrows()
            if int(summary["n_cells"]) >= min_cells and int(summary["n_donors"]) >= min_donors
        ]
        for state_a, state_b in itertools.combinations(states, 2):
            cell_types = eligible_cell_types(frame, state_a, state_b, min_cells, min_donors)
            if not cell_types:
                continue
            state_frame = frame.loc[frame[STATE_KEY].isin((state_a, state_b))]
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "assay": str(assay),
                    "tissue": str(tissue),
                    "state_a": state_a,
                    "state_b": state_b,
                    "normal_contrast": "normal" in {state_a.lower(), state_b.lower()},
                    "n_eligible_cell_types": len(cell_types),
                    "n_cells_pair": len(state_frame),
                    "n_donors_pair": int(state_frame[DONOR_KEY].nunique()),
                    "n_donors_state_a": int(
                        state_frame.loc[state_frame[STATE_KEY] == state_a, DONOR_KEY].nunique()
                    ),
                    "n_donors_state_b": int(
                        state_frame.loc[state_frame[STATE_KEY] == state_b, DONOR_KEY].nunique()
                    ),
                    "eligible_cell_types": "; ".join(cell_types),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write records with stable field order."""
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--dataset-ids-file", type=Path, default=DEFAULT_IDS_FILE)
    parser.add_argument("--dataset-ids", nargs="+", help="Subset of dataset IDs to audit")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-cells", type=int, default=20)
    parser.add_argument("--min-donors", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_ids = args.dataset_ids or read_ids(args.dataset_ids_file)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for dataset_id in dataset_ids:
        try:
            rows.extend(
                candidate_rows(
                    read_metadata(args.data_root / dataset_id),
                    dataset_id,
                    args.min_cells,
                    args.min_donors,
                )
            )
            print(f"audited {dataset_id}", flush=True)
        except Exception as error:  # noqa: BLE001
            failures.append({"dataset_id": dataset_id, "reason": str(error)})
            print(f"failed {dataset_id}: {error}", flush=True)

    rows.sort(
        key=lambda row: (
            not bool(row["normal_contrast"]),
            -int(row["n_eligible_cell_types"]),
            -int(row["n_donors_pair"]),
            -int(row["n_cells_pair"]),
        )
    )
    best_by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        best_by_dataset.setdefault(str(row["dataset_id"]), row)
    write_csv(rows, args.output_dir / "candidate_pairs.csv")
    write_csv(list(best_by_dataset.values()), args.output_dir / "dataset_best_candidates.csv")
    write_csv(failures, args.output_dir / "failures.csv")
    print(f"Wrote {len(rows)} candidate pairs across {len(best_by_dataset)} datasets")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
