#!/usr/bin/env python3
# Author: Ximan Ding (x.ding25@imperial.ac.uk)
# Script: 04_select_otu_vocabulary.py
# Description: Select a fixed OTU vocabulary using training samples only.
#
# Arguments: 1 -> Path to the eligible sample-by-OTU count matrix
#            2 -> Path to the biological-sample split CSV
#            Both arguments are optional.
# Date: July 2026

"""
Step 4: Select the training-derived OTU vocabulary
==================================================

This script selects the specific OTUs that will be represented individually
in the machine-learning dataset.

To prevent data leakage, all OTU-selection statistics are calculated using
training samples only. Validation and test samples do not contribute to:

    OTU prevalence;
    OTU mean relative abundance;
    OTU ranking;
    OTU inclusion decisions.

Inputs:
    results/intermediate/
        eligible_sample_OTU_count_matrix.pkl.gz

    results/prepared/
        02_sample_split.csv

Output:
    results/prepared/
        03_selected_otu_vocabulary.csv

Selection rules:
    minimum training prevalence: 2%
    target cumulative mean relative abundance: 99%
    minimum number of specific OTUs: 100
    maximum number of specific OTUs: 500

OTUs not selected here will be combined into an OTHER component in the
next processing step.

Usage:
    python3 code/04_select_otu_vocabulary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import INTERMEDIATE_DIR, PREPARED_DIR, banner, out


# OTU-selection parameters.
MIN_OTU_PREVALENCE = 0.02
TARGET_CUMULATIVE_MASS = 0.99
MIN_MODEL_OTUS = 100
MAX_MODEL_OTUS = 500


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
        invalid_samples = row_sums.loc[
            row_sums <= 0
        ].index.tolist()

        raise ValueError(
            "Some samples have non-positive total counts: "
            f"{invalid_samples[:10]}"
        )

    return count_matrix


def load_sample_split(
    split_path: str | Path,
) -> pd.DataFrame:
    """Load and validate the fixed biological-sample split."""

    split_path = Path(split_path)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Sample split file not found: {split_path}"
        )

    sample_split = pd.read_csv(split_path)

    required_columns = {
        "sample_id",
        "split",
    }

    missing_columns = required_columns.difference(
        sample_split.columns
    )

    if missing_columns:
        raise ValueError(
            "Sample split file is missing required columns: "
            f"{sorted(missing_columns)}"
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
            "Missing sample_id values were found in the split file."
        )

    if sample_split["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample_id values were found in the split file."
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
    count_matrix: pd.DataFrame,
    sample_split: pd.DataFrame,
) -> None:
    """Check that the count matrix and split contain identical samples."""

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


def calculate_training_otu_statistics(
    count_matrix: pd.DataFrame,
    train_ids: list[str],
) -> pd.DataFrame:
    """Calculate OTU prevalence and relative-abundance statistics."""

    train_counts = count_matrix.loc[
        train_ids
    ].copy()

    train_total_reads = train_counts.sum(axis=1)

    # Convert OTU counts into within-sample relative abundances.
    train_ra = train_counts.div(
        train_total_reads,
        axis=0,
    )

    # Every training sample should sum to one after closure.
    train_ra_sums = train_ra.sum(axis=1)

    if not np.allclose(
        train_ra_sums.to_numpy(),
        1.0,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "Training relative-abundance rows do not sum to one."
        )

    number_train_samples = len(train_ids)

    otu_statistics = pd.DataFrame(
        {
            "otu_id": train_ra.columns.astype(str),

            "detected_train_samples": (
                (train_counts > 0)
                .sum(axis=0)
                .to_numpy()
            ),

            "otu_prevalence_train": (
                (train_counts > 0)
                .mean(axis=0)
                .to_numpy()
            ),

            "otu_mean_ra_train": (
                train_ra
                .mean(axis=0)
                .to_numpy()
            ),

            "otu_std_ra_train": (
                train_ra
                .std(axis=0)
                .fillna(0)
                .to_numpy()
            ),

            "otu_max_ra_train": (
                train_ra
                .max(axis=0)
                .to_numpy()
            ),
        }
    )

    # Internal consistency check for prevalence.
    expected_prevalence = (
        otu_statistics["detected_train_samples"]
        / number_train_samples
    )

    if not np.allclose(
        otu_statistics["otu_prevalence_train"],
        expected_prevalence,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "Calculated OTU prevalence values are inconsistent."
        )

    # Mean relative abundances across all OTUs should sum to one.
    mean_ra_sum = (
        otu_statistics["otu_mean_ra_train"].sum()
    )

    if not np.isclose(
        mean_ra_sum,
        1.0,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "Training OTU mean relative abundances do not sum to one. "
            f"Observed sum: {mean_ra_sum}"
        )

    return otu_statistics


def select_otu_vocabulary(
    otu_statistics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply prevalence, abundance-mass and vocabulary-size rules."""

    # First remove OTUs that occur in fewer than 2% of training samples.
    otu_candidates = (
        otu_statistics.loc[
            otu_statistics["otu_prevalence_train"]
            >= MIN_OTU_PREVALENCE
        ]
        .sort_values(
            [
                "otu_mean_ra_train",
                "otu_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    if otu_candidates.empty:
        raise ValueError(
            "No OTUs passed the minimum training-prevalence threshold."
        )

    otu_candidates["cumulative_mean_ra"] = (
        otu_candidates["otu_mean_ra_train"]
        .cumsum()
    )

    # Identify how many ranked OTUs are needed to reach the
    # requested cumulative mean relative-abundance mass.
    mass_positions = np.flatnonzero(
        otu_candidates[
            "cumulative_mean_ra"
        ].to_numpy()
        >= TARGET_CUMULATIVE_MASS
    )

    if len(mass_positions) > 0:
        number_by_mass = int(
            mass_positions[0] + 1
        )
    else:
        # The prevalence filter may remove rare OTUs whose total
        # contribution prevents the candidates from reaching 99%.
        number_by_mass = len(
            otu_candidates
        )

    number_keep = max(
        MIN_MODEL_OTUS,
        number_by_mass,
    )

    number_keep = min(
        number_keep,
        MAX_MODEL_OTUS,
        len(otu_candidates),
    )

    selected_otus = (
        otu_candidates
        .iloc[:number_keep]
        .copy()
    )

    selected_otus.insert(
        0,
        "selection_rank",
        np.arange(
            1,
            len(selected_otus) + 1,
        ),
    )

    selected_otus[
        "selected_for_model"
    ] = True

    return selected_otus, otu_candidates


def main(
    matrix_path: str | Path,
    split_path: str | Path,
) -> None:
    """Select and save the fixed training-derived OTU vocabulary."""

    banner(
        "Step 4: Select the training-derived OTU vocabulary"
    )

    count_matrix = load_count_matrix(
        matrix_path
    )

    sample_split = load_sample_split(
        split_path
    )

    validate_matrix_and_split(
        count_matrix=count_matrix,
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

    if len(train_ids) == 0:
        raise ValueError(
            "No training samples were found in the split file."
        )

    print(
        f"Eligible count matrix loaded: "
        f"{count_matrix.shape[0]:,} samples × "
        f"{count_matrix.shape[1]:,} OTUs"
    )

    print(
        f"Training samples used for OTU selection: "
        f"{len(train_ids):,}"
    )

    otu_statistics = (
        calculate_training_otu_statistics(
            count_matrix=count_matrix,
            train_ids=train_ids,
        )
    )

    selected_otus, otu_candidates = (
        select_otu_vocabulary(
            otu_statistics
        )
    )

    output_path = out(
        "03_selected_otu_vocabulary.csv",
        PREPARED_DIR,
    )

    selected_otus.to_csv(
        output_path,
        index=False,
    )

    total_otus = len(
        otu_statistics
    )

    prevalence_candidates = len(
        otu_candidates
    )

    selected_count = len(
        selected_otus
    )

    selected_mass = float(
        selected_otus[
            "otu_mean_ra_train"
        ].sum()
    )

    candidate_mass = float(
        otu_candidates[
            "otu_mean_ra_train"
        ].sum()
    )

    print("\nOTU-selection summary:")
    print(
        f"  OTUs observed in training samples: "
        f"{total_otus:,}"
    )

    print(
        f"  OTUs with training prevalence "
        f"≥ {MIN_OTU_PREVALENCE:.2%}: "
        f"{prevalence_candidates:,}"
    )

    print(
        f"  Specific OTUs retained: "
        f"{selected_count:,}"
    )

    print(
        f"  Mean RA mass represented by prevalence candidates: "
        f"{candidate_mass:.6f}"
    )

    print(
        f"  Mean RA mass represented by selected OTUs: "
        f"{selected_mass:.6f}"
    )

    print(
        f"  Remaining mean RA mass assigned to OTHER: "
        f"{1.0 - selected_mass:.6f}"
    )

    print("\nSelection safeguards:")
    print(
        "  validation samples were not used"
    )
    print(
        "  test samples were not used"
    )
    print(
        "  OTU prevalence was calculated from training samples only"
    )
    print(
        "  OTU mean abundance was calculated from training samples only"
    )

    print("\nSaved file:")
    print(f"  {output_path}")

    print(
        "\nStep 4 completed successfully.\n"
        "Next script:\n"
        "  python3 code/05_build_reference_matrix.py"
    )


if __name__ == "__main__":

    default_matrix = (
        INTERMEDIATE_DIR
        / "eligible_sample_OTU_count_matrix.pkl.gz"
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