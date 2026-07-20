#!/usr/bin/env python3
# Author: Ximan Ding (x.ding25@imperial.ac.uk)
# Script: 05_build_reference_matrix.py
# Description: Build the fixed high-depth reference relative-abundance
#              matrix using selected OTUs and an aggregated OTHER component.
#
# Arguments: 1 -> Path to the eligible sample-by-OTU count matrix
#            2 -> Path to the selected OTU vocabulary CSV
#            3 -> Path to the fixed sample split CSV
#            All arguments are optional.
# Date: July 2026

"""
Step 5: Build the high-depth reference relative-abundance matrix
================================================================

This script constructs the response matrix used in later modelling.

For every eligible biological sample:

1. retain counts for the 254 selected OTUs;
2. sum all remaining OTU counts into one OTHER component;
3. divide every component by the sample's total OTU count;
4. verify that every sample-level relative-abundance vector sums to one.

Inputs:
    results/intermediate/
        eligible_sample_OTU_count_matrix.pkl.gz

    results/prepared/
        03_selected_otu_vocabulary.csv

    results/prepared/
        02_sample_split.csv

Outputs:
    results/prepared/
        04_reference_ra_matrix.pkl.gz

    results/prepared/
        04_reference_ra_matrix_preview.csv

The high-depth relative abundance is a reference observation derived from
finite sequencing counts. It is not absolute abundance and is not assumed
to be a noise-free biological truth.

Usage:
    python3 code/05_build_reference_matrix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import INTERMEDIATE_DIR, PREPARED_DIR, banner, out


def load_count_matrix(
    matrix_path: str | Path,
) -> pd.DataFrame:
    """Load and validate the eligible sample-by-OTU count matrix."""

    matrix_path = Path(matrix_path)

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Eligible count matrix not found: {matrix_path}"
        )

    count_matrix = pd.read_pickle(
        matrix_path,
        compression="gzip",
    )

    if not isinstance(count_matrix, pd.DataFrame):
        raise TypeError(
            "The eligible count-matrix file does not contain "
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
        raise ValueError(
            "Duplicate sample IDs were found in the count matrix."
        )

    if count_matrix.columns.duplicated().any():
        raise ValueError(
            "Duplicate OTU columns were found in the count matrix."
        )

    if count_matrix.isna().any().any():
        raise ValueError(
            "Missing values were found in the count matrix."
        )

    if (count_matrix < 0).any().any():
        raise ValueError(
            "Negative values were found in the count matrix."
        )

    row_sums = count_matrix.sum(axis=1)

    if (row_sums <= 0).any():
        invalid_samples = (
            row_sums.loc[
                row_sums <= 0
            ]
            .index
            .tolist()
        )

        raise ValueError(
            "Some samples have non-positive total counts: "
            f"{invalid_samples[:10]}"
        )

    return count_matrix


def load_selected_vocabulary(
    vocabulary_path: str | Path,
) -> list[str]:
    """Load and validate the selected OTU vocabulary."""

    vocabulary_path = Path(vocabulary_path)

    if not vocabulary_path.exists():
        raise FileNotFoundError(
            f"Selected OTU vocabulary not found: {vocabulary_path}"
        )

    vocabulary = pd.read_csv(
        vocabulary_path
    )

    required_columns = {
        "otu_id",
        "selection_rank",
        "selected_for_model",
    }

    missing_columns = required_columns.difference(
        vocabulary.columns
    )

    if missing_columns:
        raise ValueError(
            "Selected OTU vocabulary is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    vocabulary["otu_id"] = (
        vocabulary["otu_id"]
        .astype("string")
        .str.strip()
    )

    vocabulary["selection_rank"] = pd.to_numeric(
        vocabulary["selection_rank"],
        errors="coerce",
    )

    if vocabulary["otu_id"].isna().any():
        raise ValueError(
            "Missing OTU IDs were found in the vocabulary."
        )

    if vocabulary["otu_id"].duplicated().any():
        duplicated_otus = (
            vocabulary.loc[
                vocabulary["otu_id"].duplicated(
                    keep=False
                ),
                "otu_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate OTU IDs were found in the vocabulary: "
            f"{duplicated_otus[:10]}"
        )

    if vocabulary["selection_rank"].isna().any():
        raise ValueError(
            "Missing or non-numeric selection ranks were found."
        )

    vocabulary = (
        vocabulary
        .sort_values("selection_rank")
        .reset_index(drop=True)
    )

    expected_ranks = np.arange(
        1,
        len(vocabulary) + 1,
    )

    if not np.array_equal(
        vocabulary["selection_rank"].to_numpy(),
        expected_ranks,
    ):
        raise ValueError(
            "OTU selection ranks are not consecutive from 1."
        )

    selected_flags = (
        vocabulary["selected_for_model"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not selected_flags.isin(
        ["true", "1"]
    ).all():
        raise ValueError(
            "The vocabulary contains OTUs not marked as selected."
        )

    selected_otus = (
        vocabulary["otu_id"]
        .astype(str)
        .tolist()
    )

    if "OTHER" in selected_otus:
        raise ValueError(
            "The specific OTU vocabulary must not already contain OTHER."
        )

    return selected_otus


def load_sample_split(
    split_path: str | Path,
) -> pd.DataFrame:
    """Load and validate the fixed biological-sample split."""

    split_path = Path(split_path)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Sample split file not found: {split_path}"
        )

    sample_split = pd.read_csv(
        split_path,
        usecols=[
            "sample_id",
            "split",
        ],
    )

    sample_split["sample_id"] = (
        sample_split["sample_id"]
        .astype("string")
        .str.strip()
    )

    sample_split["split"] = (
        sample_split["split"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    if sample_split["sample_id"].isna().any():
        raise ValueError(
            "Missing sample IDs were found in the split file."
        )

    if sample_split["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample IDs were found in the split file."
        )

    expected_splits = {
        "train",
        "valid",
        "test",
    }

    observed_splits = set(
        sample_split["split"].astype(str)
    )

    if observed_splits != expected_splits:
        raise ValueError(
            "The split file must contain train, valid and test. "
            f"Observed values: {sorted(observed_splits)}"
        )

    return sample_split


def validate_inputs(
    count_matrix: pd.DataFrame,
    selected_otus: list[str],
    sample_split: pd.DataFrame,
) -> None:
    """Check consistency among samples, OTUs and split assignments."""

    matrix_samples = set(
        count_matrix.index.astype(str)
    )

    split_samples = set(
        sample_split["sample_id"].astype(str)
    )

    if matrix_samples != split_samples:
        only_in_matrix = sorted(
            matrix_samples - split_samples
        )

        only_in_split = sorted(
            split_samples - matrix_samples
        )

        raise ValueError(
            "The count matrix and sample split contain "
            "different biological samples.\n"
            f"Only in count matrix: {only_in_matrix[:10]}\n"
            f"Only in sample split: {only_in_split[:10]}"
        )

    matrix_otus = set(
        count_matrix.columns.astype(str)
    )

    missing_selected_otus = sorted(
        set(selected_otus) - matrix_otus
    )

    if missing_selected_otus:
        raise ValueError(
            "Some selected OTUs are absent from the count matrix: "
            f"{missing_selected_otus[:10]}"
        )


def build_reference_matrix(
    count_matrix: pd.DataFrame,
    selected_otus: list[str],
    ordered_sample_ids: list[str],
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build selected-OTU plus OTHER reference relative abundances."""

    # Keep samples in the stable train-valid-test order from the split file.
    ordered_counts = (
        count_matrix
        .reindex(
            index=ordered_sample_ids
        )
        .copy()
    )

    if ordered_counts.isna().any().any():
        missing_samples = (
            ordered_counts.index[
                ordered_counts.isna().any(axis=1)
            ]
            .tolist()
        )

        raise ValueError(
            "Some split samples are missing from the count matrix: "
            f"{missing_samples[:10]}"
        )

    sample_total_reads = (
        ordered_counts
        .sum(axis=1)
        .astype(float)
    )

    selected_count_matrix = (
        ordered_counts
        .reindex(
            columns=selected_otus,
            fill_value=0,
        )
        .copy()
    )

    selected_count_sum = (
        selected_count_matrix.sum(axis=1)
    )

    other_count = (
        sample_total_reads
        - selected_count_sum
    )

    # Small floating-point errors should not occur because these are integer
    # counts, but clip tiny negative values defensively.
    if (other_count < 0).any():
        negative_samples = (
            other_count.loc[
                other_count < 0
            ]
        )

        print("\nNegative OTHER counts:")
        print(
            negative_samples
            .head(10)
            .to_string()
        )

        raise ValueError(
            "Selected OTU counts exceed total sample counts."
        )

    reference_count_matrix = (
        selected_count_matrix
        .copy()
    )

    reference_count_matrix["OTHER"] = (
        other_count
    )

    reference_matrix = (
        reference_count_matrix
        .div(
            sample_total_reads,
            axis=0,
        )
    )

    reference_matrix.index.name = "sample_id"

    return (
        reference_matrix,
        sample_total_reads,
        other_count,
    )


def validate_reference_matrix(
    reference_matrix: pd.DataFrame,
    selected_otus: list[str],
) -> None:
    """Validate dimensions, values and compositional closure."""

    expected_columns = (
        selected_otus
        + ["OTHER"]
    )

    observed_columns = (
        reference_matrix
        .columns
        .astype(str)
        .tolist()
    )

    if observed_columns != expected_columns:
        raise ValueError(
            "Reference-matrix columns do not match "
            "selected OTUs followed by OTHER."
        )

    if reference_matrix.isna().any().any():
        raise ValueError(
            "Missing values were found in the reference matrix."
        )

    if (reference_matrix < 0).any().any():
        raise ValueError(
            "Negative relative-abundance values were found."
        )

    if (reference_matrix > 1).any().any():
        raise ValueError(
            "Relative-abundance values greater than one were found."
        )

    row_sums = (
        reference_matrix.sum(axis=1)
    )

    if not np.allclose(
        row_sums.to_numpy(),
        1.0,
        rtol=0,
        atol=1e-12,
    ):
        invalid_rows = pd.DataFrame(
            {
                "ra_sum": row_sums,
            }
        )

        invalid_rows = invalid_rows.loc[
            ~np.isclose(
                invalid_rows["ra_sum"],
                1.0,
                rtol=0,
                atol=1e-12,
            )
        ]

        print("\nFirst invalid relative-abundance sums:")
        print(
            invalid_rows
            .head(10)
            .to_string()
        )

        raise ValueError(
            "Some reference relative-abundance rows do not sum to one."
        )


def main(
    matrix_path: str | Path,
    vocabulary_path: str | Path,
    split_path: str | Path,
) -> None:
    """Build, validate and save the high-depth reference RA matrix."""

    banner(
        "Step 5: Build the high-depth reference RA matrix"
    )

    count_matrix = load_count_matrix(
        matrix_path
    )

    selected_otus = load_selected_vocabulary(
        vocabulary_path
    )

    sample_split = load_sample_split(
        split_path
    )

    validate_inputs(
        count_matrix=count_matrix,
        selected_otus=selected_otus,
        sample_split=sample_split,
    )

    ordered_sample_ids = (
        sample_split["sample_id"]
        .astype(str)
        .tolist()
    )

    print(
        f"Eligible count matrix loaded: "
        f"{count_matrix.shape[0]:,} samples × "
        f"{count_matrix.shape[1]:,} original OTUs"
    )

    print(
        f"Selected specific OTUs loaded: "
        f"{len(selected_otus):,}"
    )

    (
        reference_matrix,
        sample_total_reads,
        other_count,
    ) = build_reference_matrix(
        count_matrix=count_matrix,
        selected_otus=selected_otus,
        ordered_sample_ids=ordered_sample_ids,
    )

    validate_reference_matrix(
        reference_matrix=reference_matrix,
        selected_otus=selected_otus,
    )

    reference_output = out(
        "04_reference_ra_matrix.pkl.gz",
        PREPARED_DIR,
    )

    preview_output = out(
        "04_reference_ra_matrix_preview.csv",
        PREPARED_DIR,
    )

    # Keep sample_id as a normal column in the saved Pickle so that later
    # scripts can inspect it easily after loading.
    reference_matrix.reset_index().to_pickle(
        reference_output,
        compression="gzip",
    )

    # Save a limited preview for manual inspection.
    reference_matrix.iloc[
        :20,
        :30,
    ].reset_index().to_csv(
        preview_output,
        index=False,
    )

    row_sums = (
        reference_matrix.sum(axis=1)
    )

    other_ra = (
        reference_matrix["OTHER"]
    )

    print("\nReference-matrix summary:")
    print(
        f"  Biological samples: "
        f"{reference_matrix.shape[0]:,}"
    )

    print(
        f"  Specific OTUs: "
        f"{len(selected_otus):,}"
    )

    print(
        f"  Total model components including OTHER: "
        f"{reference_matrix.shape[1]:,}"
    )

    print(
        f"  Original sequencing-depth range: "
        f"{int(sample_total_reads.min()):,}–"
        f"{int(sample_total_reads.max()):,}"
    )

    print(
        f"  Reference RA row-sum range: "
        f"{row_sums.min():.12f}–"
        f"{row_sums.max():.12f}"
    )

    print(
        f"  Mean OTHER relative abundance: "
        f"{other_ra.mean():.6f}"
    )

    print(
        f"  Median OTHER relative abundance: "
        f"{other_ra.median():.6f}"
    )

    print(
        f"  Maximum OTHER relative abundance: "
        f"{other_ra.max():.6f}"
    )

    print(
        f"  Total counts assigned to OTHER across all samples: "
        f"{int(other_count.sum()):,}"
    )

    print("\nValidation passed:")
    print(
        "  all selected OTUs exist in the original count matrix"
    )
    print(
        "  all eligible samples occur exactly once"
    )
    print(
        "  all remaining OTU counts are represented by OTHER"
    )
    print(
        "  every reference relative-abundance vector sums to one"
    )

    print("\nSaved files:")
    print(f"  {reference_output}")
    print(f"  {preview_output}")

    print(
        "\nStep 5 completed successfully.\n"
        "Next script:\n"
        "  python3 code/06_compute_train_statistics.py"
    )


if __name__ == "__main__":

    default_matrix = (
        INTERMEDIATE_DIR
        / "eligible_sample_OTU_count_matrix.pkl.gz"
    )

    default_vocabulary = (
        PREPARED_DIR
        / "03_selected_otu_vocabulary.csv"
    )

    default_split = (
        PREPARED_DIR
        / "02_sample_split.csv"
    )

    matrix_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else default_matrix
    )

    vocabulary_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else default_vocabulary
    )

    split_path = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else default_split
    )

    main(
        matrix_path=matrix_path,
        vocabulary_path=vocabulary_path,
        split_path=split_path,
    )