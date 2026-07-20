#!/usr/bin/env python3
# Author: Ximan Ding (x.ding25@imperial.ac.uk)
# Script: 02_filter_reference_samples.py
# Description: Retain biological samples with sufficient sequencing depth
#              to serve as high-depth reference samples.
#
# Arguments: 1 -> Path to the sample metadata CSV
#            2 -> Path to the sample-by-OTU count matrix
#            Both arguments are optional.
# Date: July 2026

"""
Step 2: Filter eligible high-depth reference samples
=====================================================

This script applies the minimum sequencing-depth threshold used to define
eligible high-depth reference samples.

Inputs:
    results/intermediate/crosssec_sample_metadata.csv

    results/intermediate/
        crosssec_sample_OTU_count_matrix.pkl.gz

Outputs:
    results/prepared/01_eligible_reference_samples.csv

    results/intermediate/
        eligible_sample_OTU_count_matrix.pkl.gz

    results/intermediate/
        eligible_sample_OTU_count_matrix_preview.csv

Eligibility rule:
    calculated_total_reads >= 10,000

The high-depth samples are treated as reference observations. They are not
assumed to represent absolute abundance or noise-free biological truth.

Usage:
    python3 code/02_filter_reference_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import INTERMEDIATE_DIR, PREPARED_DIR, banner, out


# Minimum sequencing depth required for a reference sample.
MIN_REFERENCE_DEPTH = 10_000


def load_sample_metadata(
    metadata_path: str | Path,
) -> pd.DataFrame:
    """Load and validate biological-sample metadata."""

    metadata_path = Path(metadata_path)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Sample metadata file not found: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    required_columns = {
        "sample_id",
        "project_id",
        "classification",
        "run_count",
        "first_run_id",
        "calculated_total_reads",
        "recorded_total_reads",
        "all_runs_match_recorded",
        "read_difference",
    }

    missing_columns = required_columns.difference(
        metadata.columns
    )

    if missing_columns:
        raise ValueError(
            "Sample metadata is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    for column in [
        "sample_id",
        "project_id",
        "classification",
        "first_run_id",
    ]:
        metadata[column] = (
            metadata[column]
            .astype("string")
            .str.strip()
        )

    numeric_columns = [
        "run_count",
        "calculated_total_reads",
        "recorded_total_reads",
        "read_difference",
    ]

    for column in numeric_columns:
        metadata[column] = pd.to_numeric(
            metadata[column],
            errors="coerce",
        )

    if metadata[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing or non-numeric values were found "
            "in sample metadata."
        )

    if metadata["sample_id"].isna().any():
        raise ValueError(
            "Missing sample_id values were found."
        )

    if metadata["sample_id"].duplicated().any():
        duplicated_samples = (
            metadata.loc[
                metadata["sample_id"].duplicated(
                    keep=False
                ),
                "sample_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate sample_id values were found: "
            f"{duplicated_samples[:10]}"
        )

    if (
        metadata["calculated_total_reads"] <= 0
    ).any():
        raise ValueError(
            "Non-positive calculated sample depths were found."
        )

    return metadata


def load_count_matrix(
    matrix_path: str | Path,
) -> pd.DataFrame:
    """Load and validate the sample-by-OTU count matrix."""

    matrix_path = Path(matrix_path)

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Count matrix file not found: {matrix_path}"
        )

    count_matrix = pd.read_pickle(
        matrix_path,
        compression="gzip",
    )

    if not isinstance(
        count_matrix,
        pd.DataFrame,
    ):
        raise TypeError(
            "The count matrix file does not contain "
            "a pandas DataFrame."
        )

    count_matrix.index = (
        count_matrix.index
        .astype(str)
        .str.strip()
    )

    count_matrix.columns = (
        count_matrix.columns
        .astype(str)
        .str.strip()
    )

    count_matrix.index.name = "sample_id"

    if count_matrix.index.duplicated().any():
        duplicated_samples = (
            count_matrix.index[
                count_matrix.index.duplicated(
                    keep=False
                )
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate sample IDs were found in the count matrix: "
            f"{duplicated_samples[:10]}"
        )

    if count_matrix.columns.duplicated().any():
        duplicated_otus = (
            count_matrix.columns[
                count_matrix.columns.duplicated(
                    keep=False
                )
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate OTU columns were found: "
            f"{duplicated_otus[:10]}"
        )

    if count_matrix.isna().any().any():
        raise ValueError(
            "Missing values were found in the count matrix."
        )

    if (count_matrix < 0).any().any():
        raise ValueError(
            "Negative values were found in the count matrix."
        )

    return count_matrix


def validate_metadata_and_matrix(
    metadata: pd.DataFrame,
    count_matrix: pd.DataFrame,
) -> None:
    """Check that metadata and count matrix contain identical samples."""

    metadata_samples = set(
        metadata["sample_id"].astype(str)
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
            "Metadata and count matrix contain different samples.\n"
            f"Only in metadata: {only_in_metadata[:10]}\n"
            f"Only in count matrix: {only_in_matrix[:10]}"
        )

    matrix_depth = count_matrix.sum(axis=1)

    expected_depth = (
        metadata
        .set_index("sample_id")[
            "calculated_total_reads"
        ]
        .reindex(count_matrix.index)
    )

    if expected_depth.isna().any():
        missing_samples = (
            expected_depth.index[
                expected_depth.isna()
            ]
            .tolist()
        )

        raise ValueError(
            "Some count-matrix samples are missing "
            "calculated sequencing depths: "
            f"{missing_samples[:10]}"
        )

    if not np.allclose(
        matrix_depth.to_numpy(),
        expected_depth.to_numpy(),
        rtol=0,
        atol=0,
    ):
        differences = pd.DataFrame(
            {
                "matrix_total_reads": matrix_depth,
                "metadata_total_reads": expected_depth,
            }
        )

        differences["difference"] = (
            differences["matrix_total_reads"]
            - differences["metadata_total_reads"]
        )

        mismatches = differences.loc[
            differences["difference"] != 0
        ]

        print("\nFirst depth mismatches:")
        print(
            mismatches
            .head(10)
            .to_string()
        )

        raise ValueError(
            "Count-matrix row sums do not match "
            "metadata sequencing depths."
        )


def main(
    metadata_path: str | Path,
    matrix_path: str | Path,
) -> None:
    """Filter and save eligible high-depth reference samples."""

    banner(
        "Step 2: Filter eligible high-depth reference samples"
    )

    metadata = load_sample_metadata(
        metadata_path
    )

    count_matrix = load_count_matrix(
        matrix_path
    )

    print(
        f"Sample metadata loaded: "
        f"{len(metadata):,} biological samples"
    )

    print(
        f"Count matrix loaded: "
        f"{count_matrix.shape[0]:,} samples × "
        f"{count_matrix.shape[1]:,} OTUs"
    )

    validate_metadata_and_matrix(
        metadata=metadata,
        count_matrix=count_matrix,
    )

    print(
        "\nMetadata and count-matrix validation passed."
    )

    # Apply the reference-depth threshold.
    eligible_samples = metadata.loc[
        metadata["calculated_total_reads"]
        >= MIN_REFERENCE_DEPTH
    ].copy()

    excluded_samples = metadata.loc[
        metadata["calculated_total_reads"]
        < MIN_REFERENCE_DEPTH
    ].copy()

    if eligible_samples.empty:
        raise ValueError(
            "No samples passed the minimum reference-depth threshold."
        )

    eligible_samples = (
        eligible_samples
        .sort_values(
            "calculated_total_reads",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    excluded_samples = (
        excluded_samples
        .sort_values(
            "calculated_total_reads",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    eligible_sample_ids = (
        eligible_samples["sample_id"]
        .astype(str)
        .tolist()
    )

    eligible_count_matrix = (
        count_matrix
        .loc[eligible_sample_ids]
        .copy()
    )

    # Final sample-set check.
    if set(
        eligible_count_matrix.index.astype(str)
    ) != set(
        eligible_sample_ids
    ):
        raise ValueError(
            "Eligible count matrix does not match "
            "the eligible sample list."
        )

    # Check eligible matrix row sums again.
    eligible_matrix_depth = (
        eligible_count_matrix.sum(axis=1)
    )

    eligible_expected_depth = (
        eligible_samples
        .set_index("sample_id")[
            "calculated_total_reads"
        ]
        .reindex(eligible_count_matrix.index)
    )

    if not np.allclose(
        eligible_matrix_depth.to_numpy(),
        eligible_expected_depth.to_numpy(),
        rtol=0,
        atol=0,
    ):
        raise ValueError(
            "Eligible count-matrix row sums do not match "
            "eligible sample metadata."
        )

    # Add explicit eligibility information.
    eligible_samples[
        "minimum_reference_depth"
    ] = MIN_REFERENCE_DEPTH

    eligible_samples[
        "eligible_reference_sample"
    ] = True

    # Define output paths.
    eligible_metadata_output = out(
        "01_eligible_reference_samples.csv",
        PREPARED_DIR,
    )

    eligible_matrix_output = out(
        "eligible_sample_OTU_count_matrix.pkl.gz",
        INTERMEDIATE_DIR,
    )

    eligible_preview_output = out(
        "eligible_sample_OTU_count_matrix_preview.csv",
        INTERMEDIATE_DIR,
    )

    # Save outputs.
    eligible_samples.to_csv(
        eligible_metadata_output,
        index=False,
    )

    eligible_count_matrix.to_pickle(
        eligible_matrix_output,
        compression="gzip",
    )

    eligible_count_matrix.iloc[
        :20,
        :30,
    ].reset_index().to_csv(
        eligible_preview_output,
        index=False,
    )

    print(
        f"\nOriginal biological samples: "
        f"{len(metadata):,}"
    )

    print(
        f"Eligible reference samples "
        f"(≥ {MIN_REFERENCE_DEPTH:,} reads): "
        f"{len(eligible_samples):,}"
    )

    print(
        f"Excluded samples "
        f"(< {MIN_REFERENCE_DEPTH:,} reads): "
        f"{len(excluded_samples):,}"
    )

    print(
        "Eligible sequencing-depth range: "
        f"{int(eligible_samples['calculated_total_reads'].min()):,}"
        "–"
        f"{int(eligible_samples['calculated_total_reads'].max()):,}"
    )

    if not excluded_samples.empty:
        print("\nExcluded samples:")
        print(
            excluded_samples[
                [
                    "sample_id",
                    "first_run_id",
                    "calculated_total_reads",
                ]
            ]
            .to_string(index=False)
        )

    print("\nSaved files:")
    print(f"  {eligible_metadata_output}")
    print(f"  {eligible_matrix_output}")
    print(f"  {eligible_preview_output}")

    print(
        "\nStep 2 completed successfully.\n"
        "Next script:\n"
        "  python3 code/03_split_samples.py"
    )


if __name__ == "__main__":

    default_metadata = (
        INTERMEDIATE_DIR
        / "crosssec_sample_metadata.csv"
    )

    default_matrix = (
        INTERMEDIATE_DIR
        / "crosssec_sample_OTU_count_matrix.pkl.gz"
    )

    metadata_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else default_metadata
    )

    matrix_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else default_matrix
    )

    main(
        metadata_path=metadata_path,
        matrix_path=matrix_path,
    )