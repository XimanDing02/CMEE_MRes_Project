#!/usr/bin/env python3
# Author: Ximan Ding (x.ding25@imperial.ac.uk)
# Script: 06_compute_train_statistics.py
# Description: Calculate training-only relative-abundance statistics for
#              all selected OTUs and the aggregated OTHER component.
#
# Arguments: 1 -> Path to the reference RA matrix
#            2 -> Path to the biological-sample split CSV
#            Both arguments are optional.
# Date: July 2026

"""
Step 6: Compute training-only OTU statistics
============================================

This script calculates summary statistics for every model component using
training samples only.

Model components:
    254 selected OTUs
    1 aggregated OTHER component

Statistics:
    otu_mean_ra_train
        Mean reference relative abundance across all training samples,
        including zeros.

    otu_prevalence_train
        Proportion of training samples in which the component has
        positive relative abundance.

    otu_std_ra_train
        Standard deviation of reference relative abundance across
        training samples.

    otu_max_ra_train
        Maximum reference relative abundance observed in the training set.

The validation and test samples are not used when calculating these values.

Inputs:
    results/prepared/04_reference_ra_matrix.pkl.gz
    results/prepared/02_sample_split.csv

Output:
    results/prepared/05_model_otu_train_statistics.csv

Usage:
    python3 code/06_compute_train_statistics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import PREPARED_DIR, banner, out


def load_reference_matrix(
    matrix_path: str | Path,
) -> pd.DataFrame:
    """Load and validate the high-depth reference RA matrix."""

    matrix_path = Path(matrix_path)

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Reference RA matrix not found: {matrix_path}"
        )

    reference_data = pd.read_pickle(
        matrix_path,
        compression="gzip",
    )

    if not isinstance(reference_data, pd.DataFrame):
        raise TypeError(
            "The reference matrix file does not contain "
            "a pandas DataFrame."
        )

    if "sample_id" not in reference_data.columns:
        raise ValueError(
            "The reference matrix does not contain a sample_id column."
        )

    reference_data["sample_id"] = (
        reference_data["sample_id"]
        .astype("string")
        .str.strip()
    )

    if reference_data["sample_id"].isna().any():
        raise ValueError(
            "Missing sample_id values were found."
        )

    if reference_data["sample_id"].duplicated().any():
        duplicated_samples = (
            reference_data.loc[
                reference_data["sample_id"].duplicated(
                    keep=False
                ),
                "sample_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate sample IDs were found in the reference matrix: "
            f"{duplicated_samples[:10]}"
        )

    reference_matrix = (
        reference_data
        .set_index("sample_id")
    )

    reference_matrix.index = (
        reference_matrix.index.astype(str)
    )

    reference_matrix.columns = (
        reference_matrix.columns
        .astype(str)
        .str.strip()
    )

    reference_matrix.index.name = "sample_id"

    if reference_matrix.columns.duplicated().any():
        duplicated_components = (
            reference_matrix.columns[
                reference_matrix.columns.duplicated(
                    keep=False
                )
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Duplicate model components were found: "
            f"{duplicated_components[:10]}"
        )

    reference_matrix = reference_matrix.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if reference_matrix.isna().any().any():
        raise ValueError(
            "Missing or non-numeric values were found "
            "in the reference matrix."
        )

    if (reference_matrix < 0).any().any():
        raise ValueError(
            "Negative relative-abundance values were found."
        )

    if (reference_matrix > 1).any().any():
        raise ValueError(
            "Relative-abundance values greater than one were found."
        )

    row_sums = reference_matrix.sum(axis=1)

    if not np.allclose(
        row_sums.to_numpy(),
        1.0,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "Some reference RA rows do not sum to one."
        )

    if "OTHER" not in reference_matrix.columns:
        raise ValueError(
            "The reference matrix does not contain the OTHER component."
        )

    return reference_matrix


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


def validate_matrix_and_split(
    reference_matrix: pd.DataFrame,
    sample_split: pd.DataFrame,
) -> None:
    """Check that reference matrix and split contain identical samples."""

    matrix_samples = set(
        reference_matrix.index.astype(str)
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
            "The reference matrix and sample split contain "
            "different biological samples.\n"
            f"Only in reference matrix: {only_in_matrix[:10]}\n"
            f"Only in sample split: {only_in_split[:10]}"
        )


def calculate_training_statistics(
    reference_matrix: pd.DataFrame,
    train_ids: list[str],
) -> pd.DataFrame:
    """Calculate training-only statistics for each model component."""

    train_reference_matrix = (
        reference_matrix
        .loc[train_ids]
        .copy()
    )

    if len(train_reference_matrix) != len(train_ids):
        raise ValueError(
            "The number of extracted training samples is incorrect."
        )

    training_statistics = pd.DataFrame(
        {
            "otu_id": (
                train_reference_matrix
                .columns
                .astype(str)
            ),

            "otu_mean_ra_train": (
                train_reference_matrix
                .mean(axis=0)
                .to_numpy()
            ),

            "otu_prevalence_train": (
                (train_reference_matrix > 0)
                .mean(axis=0)
                .to_numpy()
            ),

            "otu_std_ra_train": (
                train_reference_matrix
                .std(axis=0, ddof=1)
                .fillna(0)
                .to_numpy()
            ),

            "otu_max_ra_train": (
                train_reference_matrix
                .max(axis=0)
                .to_numpy()
            ),
        }
    )

    training_statistics["is_other"] = (
        training_statistics["otu_id"]
        == "OTHER"
    ).astype(np.int8)

    return training_statistics


def validate_training_statistics(
    training_statistics: pd.DataFrame,
    reference_matrix: pd.DataFrame,
) -> None:
    """Validate model-component statistics."""

    expected_components = (
        reference_matrix.columns.astype(str).tolist()
    )

    observed_components = (
        training_statistics["otu_id"]
        .astype(str)
        .tolist()
    )

    if observed_components != expected_components:
        raise ValueError(
            "Training-statistics rows do not match "
            "the reference-matrix component order."
        )

    numeric_columns = [
        "otu_mean_ra_train",
        "otu_prevalence_train",
        "otu_std_ra_train",
        "otu_max_ra_train",
    ]

    if (
        training_statistics[numeric_columns]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "Missing values were found in training statistics."
        )

    if (
        training_statistics[numeric_columns]
        < 0
    ).any().any():
        raise ValueError(
            "Negative values were found in training statistics."
        )

    if (
        training_statistics[
            [
                "otu_mean_ra_train",
                "otu_prevalence_train",
                "otu_max_ra_train",
            ]
        ]
        > 1
    ).any().any():
        raise ValueError(
            "A mean, prevalence or maximum value greater than one "
            "was found."
        )

    mean_ra_sum = (
        training_statistics[
            "otu_mean_ra_train"
        ].sum()
    )

    if not np.isclose(
        mean_ra_sum,
        1.0,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "Training mean relative abundances do not sum to one. "
            f"Observed sum: {mean_ra_sum}"
        )

    if (
        training_statistics["otu_max_ra_train"]
        < training_statistics["otu_mean_ra_train"]
    ).any():
        raise ValueError(
            "At least one maximum RA is smaller than its mean RA."
        )

    other_count = int(
        training_statistics["is_other"].sum()
    )

    if other_count != 1:
        raise ValueError(
            "Training statistics must contain exactly one OTHER row."
        )


def main(
    matrix_path: str | Path,
    split_path: str | Path,
) -> None:
    """Calculate, validate and save training-only OTU statistics."""

    banner(
        "Step 6: Compute training-only OTU statistics"
    )

    reference_matrix = load_reference_matrix(
        matrix_path
    )

    sample_split = load_sample_split(
        split_path
    )

    validate_matrix_and_split(
        reference_matrix=reference_matrix,
        sample_split=sample_split,
    )

    train_ids = (
        sample_split.loc[
            sample_split["split"] == "train",
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    valid_ids = (
        sample_split.loc[
            sample_split["split"] == "valid",
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    test_ids = (
        sample_split.loc[
            sample_split["split"] == "test",
            "sample_id",
        ]
        .astype(str)
        .tolist()
    )

    if len(train_ids) == 0:
        raise ValueError(
            "No training samples were found."
        )

    print(
        f"Reference matrix loaded: "
        f"{reference_matrix.shape[0]:,} samples × "
        f"{reference_matrix.shape[1]:,} model components"
    )

    print(
        f"Training samples used: {len(train_ids):,}"
    )

    print(
        f"Validation samples excluded from statistics: "
        f"{len(valid_ids):,}"
    )

    print(
        f"Test samples excluded from statistics: "
        f"{len(test_ids):,}"
    )

    training_statistics = (
        calculate_training_statistics(
            reference_matrix=reference_matrix,
            train_ids=train_ids,
        )
    )

    validate_training_statistics(
        training_statistics=training_statistics,
        reference_matrix=reference_matrix,
    )

    output_path = out(
        "05_model_otu_train_statistics.csv",
        PREPARED_DIR,
    )

    training_statistics.to_csv(
        output_path,
        index=False,
    )

    other_statistics = (
        training_statistics.loc[
            training_statistics["otu_id"]
            == "OTHER"
        ]
        .iloc[0]
    )

    print("\nTraining-statistics summary:")
    print(
        f"  Model components: "
        f"{len(training_statistics):,}"
    )

    print(
        f"  Specific OTUs: "
        f"{len(training_statistics) - 1:,}"
    )

    print(
        "  OTHER components: 1"
    )

    print(
        f"  Sum of training mean RA: "
        f"{training_statistics['otu_mean_ra_train'].sum():.12f}"
    )

    print(
        f"  Mean component prevalence: "
        f"{training_statistics['otu_prevalence_train'].mean():.6f}"
    )

    print(
        f"  Minimum component prevalence: "
        f"{training_statistics['otu_prevalence_train'].min():.6f}"
    )

    print(
        f"  Maximum component prevalence: "
        f"{training_statistics['otu_prevalence_train'].max():.6f}"
    )

    print("\nOTHER training statistics:")
    print(
        f"  mean RA: "
        f"{other_statistics['otu_mean_ra_train']:.6f}"
    )

    print(
        f"  prevalence: "
        f"{other_statistics['otu_prevalence_train']:.6f}"
    )

    print(
        f"  standard deviation: "
        f"{other_statistics['otu_std_ra_train']:.6f}"
    )

    print(
        f"  maximum RA: "
        f"{other_statistics['otu_max_ra_train']:.6f}"
    )

    print("\nLeakage safeguards:")
    print(
        "  OTU statistics were calculated from training samples only"
    )
    print(
        "  validation samples were not used"
    )
    print(
        "  test samples were not used"
    )

    print("\nFirst 10 rows:")
    print(
        training_statistics
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved file:")
    print(f"  {output_path}")

    print(
        "\nStep 6 completed successfully.\n"
        "Next script:\n"
        "  python3 code/07_generate_supervised_data.py"
    )


if __name__ == "__main__":

    default_matrix = (
        PREPARED_DIR
        / "04_reference_ra_matrix.pkl.gz"
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

    split_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else default_split
    )

    main(
        matrix_path=matrix_path,
        split_path=split_path,
    )