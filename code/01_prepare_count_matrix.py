#!/usr/bin/env python3
# Author: Ximan Ding (x.ding25@imperial.ac.uk)
# Script: 01_prepare_count_matrix.py
# Description: Filter the target seawater project, validate sequencing counts,
#              and create a biological-sample-by-OTU count matrix.
#
# Arguments: 1 -> Path to the long-format CSV
#                (optional; default:
#                results/intermediate/crosssec_datatax.csv)
# Date: July 2026

"""
Step 1: Prepare the biological-sample-by-OTU count matrix
=========================================================

This script reads the long-format datatax table exported in Step 0.

It then:

1. retains only project MGYS00002437;
2. retains only records classified as seawater;
3. cleans text and numeric columns;
4. checks the relationship between sample_id and run_id;
5. compares summed OTU counts with recorded nreads;
6. combines multiple runs belonging to the same biological sample;
7. creates a sample-by-OTU count matrix.

Input:
    results/intermediate/crosssec_datatax.csv

Outputs:
    results/intermediate/01_run_consistency_check.csv

    results/intermediate/crosssec_sample_metadata.csv

    results/intermediate/crosssec_sample_OTU_count_matrix.pkl.gz

    results/intermediate/
        crosssec_sample_OTU_count_matrix_preview.csv

Usage:
    python3 code/01_prepare_count_matrix.py

or:

    python3 code/01_prepare_count_matrix.py \
        results/intermediate/crosssec_datatax.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import INTERMEDIATE_DIR, banner, out


# Target study used in this project.
TARGET_PROJECT_ID = "MGYS00002437"
TARGET_CLASSIFICATION = "seawater"

# Read the large CSV in chunks to reduce memory usage.
CSV_CHUNK_SIZE = 500_000

REQUIRED_COLUMNS = [
    "otu_id",
    "count",
    "project_id",
    "sample_id",
    "run_id",
    "nreads",
    "classification",
]


def clean_text_column(series: pd.Series) -> pd.Series:
    """Convert values to strings and remove surrounding spaces."""

    return (
        series
        .astype("string")
        .str.strip()
    )


def load_target_project(input_path: str | Path) -> pd.DataFrame:
    """Read the long table in chunks and retain the target project."""

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV file not found: {input_path}"
        )

    selected_parts = []
    total_rows_loaded = 0
    total_rows_retained = 0

    reader = pd.read_csv(
        input_path,
        usecols=REQUIRED_COLUMNS,
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        total_rows_loaded += len(chunk)

        # Clean text fields before filtering.
        for column in [
            "otu_id",
            "project_id",
            "sample_id",
            "run_id",
            "classification",
        ]:
            chunk[column] = clean_text_column(
                chunk[column]
            )

        # Standardise classification to lower case.
        chunk["classification"] = (
            chunk["classification"]
            .str.lower()
        )

        # Retain only the target project and environment.
        selected_chunk = chunk.loc[
            (
                chunk["project_id"]
                == TARGET_PROJECT_ID
            )
            & (
                chunk["classification"]
                == TARGET_CLASSIFICATION
            )
        ].copy()

        if selected_chunk.empty:
            print(
                f"Chunk {chunk_number}: "
                f"loaded {len(chunk):,} rows; retained 0 rows"
            )
            continue

        # Convert numeric columns after target filtering.
        selected_chunk["count"] = pd.to_numeric(
            selected_chunk["count"],
            errors="coerce",
        )

        selected_chunk["nreads"] = pd.to_numeric(
            selected_chunk["nreads"],
            errors="coerce",
        )

        # Remove records missing essential information.
        selected_chunk = selected_chunk.dropna(
            subset=[
                "otu_id",
                "count",
                "sample_id",
                "run_id",
                "nreads",
            ]
        )

        total_rows_retained += len(selected_chunk)
        selected_parts.append(selected_chunk)

        print(
            f"Chunk {chunk_number}: "
            f"loaded {len(chunk):,} rows; "
            f"cumulative retained rows "
            f"{total_rows_retained:,}"
        )

    if not selected_parts:
        raise ValueError(
            "No rows matched the target project and classification:\n"
            f"project_id = {TARGET_PROJECT_ID}\n"
            f"classification = {TARGET_CLASSIFICATION}"
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    print(f"\nTotal rows loaded: {total_rows_loaded:,}")
    print(f"Total rows retained: {len(selected):,}")

    return selected


def validate_selected_data(selected: pd.DataFrame) -> None:
    """Validate identifiers, counts and sequencing-depth values."""

    if selected["count"].isna().any():
        raise ValueError(
            "Missing OTU count values were found."
        )

    if selected["nreads"].isna().any():
        raise ValueError(
            "Missing nreads values were found."
        )

    if (selected["count"] < 0).any():
        raise ValueError(
            "Negative OTU count values were found."
        )

    if (selected["nreads"] <= 0).any():
        raise ValueError(
            "Non-positive nreads values were found "
            "within the target project."
        )

    if selected["sample_id"].isna().any():
        raise ValueError(
            "Missing sample_id values were found."
        )

    if selected["run_id"].isna().any():
        raise ValueError(
            "Missing run_id values were found."
        )

    print("\nBasic validation passed.")


def check_identifier_relationships(
    selected: pd.DataFrame,
) -> None:
    """Check mappings between biological samples and sequencing runs."""

    runs_per_sample = (
        selected
        .groupby("sample_id", observed=True)["run_id"]
        .nunique()
    )

    samples_per_run = (
        selected
        .groupby("run_id", observed=True)["sample_id"]
        .nunique()
    )

    multi_run_samples = runs_per_sample.loc[
        runs_per_sample > 1
    ]

    multi_sample_runs = samples_per_run.loc[
        samples_per_run > 1
    ]

    print(
        f"Unique biological sample IDs: "
        f"{selected['sample_id'].nunique():,}"
    )

    print(
        f"Unique sequencing run IDs: "
        f"{selected['run_id'].nunique():,}"
    )

    print(
        f"Samples associated with multiple runs: "
        f"{len(multi_run_samples):,}"
    )

    if not multi_run_samples.empty:
        print("\nFirst samples with multiple runs:")
        print(
            multi_run_samples
            .head(10)
            .to_string()
        )

    if not multi_sample_runs.empty:
        print("\nRuns associated with multiple samples:")
        print(
            multi_sample_runs
            .head(10)
            .to_string()
        )

        raise ValueError(
            "At least one run_id maps to multiple sample_id values."
        )


def build_run_consistency_table(
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Compare summed OTU counts with recorded nreads for every run."""

    # A run should contain one recorded nreads value.
    nreads_per_run = (
        selected
        .groupby("run_id", observed=True)["nreads"]
        .nunique()
    )

    inconsistent_nreads = nreads_per_run.loc[
        nreads_per_run > 1
    ]

    if not inconsistent_nreads.empty:
        raise ValueError(
            "Some run_id values contain multiple recorded "
            "nreads values:\n"
            f"{inconsistent_nreads.head(10).to_dict()}"
        )

    run_consistency = (
        selected
        .groupby(
            [
                "run_id",
                "sample_id",
                "project_id",
                "classification",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            observed_otu_records=("otu_id", "size"),
            unique_otus=("otu_id", "nunique"),
            calculated_total_reads=("count", "sum"),
            recorded_nreads=("nreads", "first"),
        )
    )

    run_consistency["read_difference"] = (
        run_consistency["calculated_total_reads"]
        - run_consistency["recorded_nreads"]
    )

    run_consistency["calculated_to_recorded_ratio"] = (
        run_consistency["calculated_total_reads"]
        / run_consistency["recorded_nreads"]
    )

    run_consistency["reads_match_recorded"] = np.isclose(
        run_consistency["calculated_total_reads"],
        run_consistency["recorded_nreads"],
        rtol=0,
        atol=0,
    )

    return run_consistency


def build_sample_metadata(
    run_consistency: pd.DataFrame,
) -> pd.DataFrame:
    """Combine run-level information into biological-sample metadata."""

    sample_metadata = (
        run_consistency
        .groupby(
            [
                "sample_id",
                "project_id",
                "classification",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            run_count=("run_id", "nunique"),
            first_run_id=("run_id", "first"),
            calculated_total_reads=(
                "calculated_total_reads",
                "sum",
            ),
            recorded_total_reads=(
                "recorded_nreads",
                "sum",
            ),
            all_runs_match_recorded=(
                "reads_match_recorded",
                "all",
            ),
        )
    )

    sample_metadata["read_difference"] = (
        sample_metadata["calculated_total_reads"]
        - sample_metadata["recorded_total_reads"]
    )

    return sample_metadata


def build_sample_count_matrix(
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Create a biological-sample-by-OTU count matrix."""

    # First combine duplicate records belonging to the same
    # biological sample and OTU.
    sample_otu_counts = (
        selected
        .groupby(
            ["sample_id", "otu_id"],
            as_index=False,
            observed=True,
        )["count"]
        .sum()
    )

    count_matrix = (
        sample_otu_counts
        .pivot(
            index="sample_id",
            columns="otu_id",
            values="count",
        )
        .fillna(0)
    )

    # All counts should be whole numbers.
    if not np.allclose(
        count_matrix.to_numpy(),
        np.round(count_matrix.to_numpy()),
    ):
        raise ValueError(
            "Non-integer OTU count values were found."
        )

    count_matrix = (
        count_matrix
        .round()
        .astype("int64")
    )

    count_matrix.index = (
        count_matrix.index.astype(str)
    )

    count_matrix.columns = (
        count_matrix.columns.astype(str)
    )

    count_matrix.index.name = "sample_id"

    return count_matrix


def main(input_path: str | Path) -> None:
    """Prepare and save the target sample-by-OTU count matrix."""

    banner(
        "Step 1: Prepare the biological-sample-by-OTU count matrix"
    )

    selected = load_target_project(input_path)

    validate_selected_data(selected)
    check_identifier_relationships(selected)

    print(
        f"Unique OTUs in target project: "
        f"{selected['otu_id'].nunique():,}"
    )

    run_consistency = build_run_consistency_table(
        selected
    )

    matching_runs = int(
        run_consistency["reads_match_recorded"].sum()
    )

    print(
        f"\nRuns where summed OTU counts equal recorded nreads: "
        f"{matching_runs:,} / {len(run_consistency):,}"
    )

    if matching_runs != len(run_consistency):
        print(
            "\nNote: calculated OTU counts do not equal recorded "
            "nreads for every run."
        )

        print(
            run_consistency.loc[
                ~run_consistency["reads_match_recorded"],
                [
                    "run_id",
                    "sample_id",
                    "calculated_total_reads",
                    "recorded_nreads",
                    "read_difference",
                    "calculated_to_recorded_ratio",
                ],
            ]
            .head(10)
            .to_string(index=False)
        )

    sample_metadata = build_sample_metadata(
        run_consistency
    )

    count_matrix = build_sample_count_matrix(
        selected
    )

    # Confirm that metadata and count matrix contain identical samples.
    metadata_samples = set(
        sample_metadata["sample_id"].astype(str)
    )

    matrix_samples = set(
        count_matrix.index.astype(str)
    )

    if metadata_samples != matrix_samples:
        only_in_metadata = sorted(
            metadata_samples - matrix_samples
        )

        only_in_matrix = sorted(
            matrix_samples - metadata_samples
        )

        raise ValueError(
            "Sample sets differ between metadata and count matrix.\n"
            f"Only in metadata: {only_in_metadata[:10]}\n"
            f"Only in matrix: {only_in_matrix[:10]}"
        )

    # Confirm that matrix row sums equal calculated sample totals.
    matrix_row_sums = count_matrix.sum(axis=1)

    expected_sample_totals = (
        sample_metadata
        .set_index("sample_id")[
            "calculated_total_reads"
        ]
        .reindex(count_matrix.index)
    )

    if not np.allclose(
        matrix_row_sums.to_numpy(),
        expected_sample_totals.to_numpy(),
        rtol=0,
        atol=0,
    ):
        raise ValueError(
            "Count-matrix row sums do not match calculated "
            "sample totals."
        )

    run_check_output = out(
        "01_run_consistency_check.csv",
        INTERMEDIATE_DIR,
    )

    metadata_output = out(
        "crosssec_sample_metadata.csv",
        INTERMEDIATE_DIR,
    )

    matrix_output = out(
        "crosssec_sample_OTU_count_matrix.pkl.gz",
        INTERMEDIATE_DIR,
    )

    preview_output = out(
        "crosssec_sample_OTU_count_matrix_preview.csv",
        INTERMEDIATE_DIR,
    )

    run_consistency.to_csv(
        run_check_output,
        index=False,
    )

    sample_metadata.to_csv(
        metadata_output,
        index=False,
    )

    count_matrix.to_pickle(
        matrix_output,
        compression="gzip",
    )

    # Save a small human-readable preview.
    count_matrix.iloc[
        :20,
        :30,
    ].reset_index().to_csv(
        preview_output,
        index=False,
    )

    print(
        f"\nFinal count matrix: "
        f"{count_matrix.shape[0]:,} biological samples × "
        f"{count_matrix.shape[1]:,} OTUs"
    )

    print(
        "Calculated sample-depth range: "
        f"{int(matrix_row_sums.min()):,}–"
        f"{int(matrix_row_sums.max()):,}"
    )

    print("\nSaved files:")
    print(f"  {run_check_output}")
    print(f"  {metadata_output}")
    print(f"  {matrix_output}")
    print(f"  {preview_output}")

    print(
        "\nStep 1 completed successfully.\n"
        "Next script:\n"
        "  python3 code/02_filter_reference_samples.py"
    )


if __name__ == "__main__":

    default_input = (
        INTERMEDIATE_DIR
        / "crosssec_datatax.csv"
    )

    input_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else default_input
    )

    main(input_path)