#!/usr/bin/env python3
# Author: Ximan Ding (x.ding25@imperial.ac.uk)
# Script: 03_split_samples.py
# Description: Split eligible biological samples into fixed training,
#              validation and test subsets before shallow subsampling.
#
# Arguments: 1 -> Path to eligible reference-sample metadata
#                (optional; default:
#                results/prepared/01_eligible_reference_samples.csv)
# Date: July 2026

"""
Step 3: Split eligible biological samples
==========================================

This script assigns each eligible biological sample to one of three subsets:

    training set
    validation set
    test set

The split is performed at the biological-sample level using sample_id.

It must be completed before shallow sequencing repeats are generated.
Therefore, all shallow repeats originating from the same biological sample
will remain in the same subset.

Input:
    results/prepared/01_eligible_reference_samples.csv

Output:
    results/prepared/02_sample_split.csv

Expected split for 474 eligible samples:
    train: 331
    valid: 71
    test:  72

Usage:
    python3 code/03_split_samples.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from _paths import PREPARED_DIR, banner, out


# Biological-sample split proportions.
TRAIN_SIZE = 0.70
VALID_SIZE = 0.15
TEST_SIZE = 0.15

# Fixed random seed for reproducibility.
RANDOM_STATE = 42


def load_eligible_samples(
    input_path: str | Path,
) -> pd.DataFrame:
    """Load and validate eligible reference-sample metadata."""

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Eligible sample file not found: {input_path}"
        )

    eligible_samples = pd.read_csv(input_path)

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
        "minimum_reference_depth",
        "eligible_reference_sample",
    }

    missing_columns = required_columns.difference(
        eligible_samples.columns
    )

    if missing_columns:
        raise ValueError(
            "Eligible sample file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Clean text columns.
    for column in [
        "sample_id",
        "project_id",
        "classification",
        "first_run_id",
    ]:
        eligible_samples[column] = (
            eligible_samples[column]
            .astype("string")
            .str.strip()
        )

    # Convert sequencing-depth columns to numeric values.
    numeric_columns = [
        "run_count",
        "calculated_total_reads",
        "recorded_total_reads",
        "read_difference",
        "minimum_reference_depth",
    ]

    for column in numeric_columns:
        eligible_samples[column] = pd.to_numeric(
            eligible_samples[column],
            errors="coerce",
        )

    if eligible_samples[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing or non-numeric values were found "
            "in eligible sample metadata."
        )

    if eligible_samples["sample_id"].isna().any():
        raise ValueError(
            "Missing sample_id values were found."
        )

    if eligible_samples["sample_id"].duplicated().any():
        duplicated_samples = (
            eligible_samples.loc[
                eligible_samples["sample_id"].duplicated(
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

    # Ensure that all records are eligible reference samples.
    eligibility_values = (
        eligible_samples["eligible_reference_sample"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if not eligibility_values.isin(
        ["true", "1"]
    ).all():
        raise ValueError(
            "The input contains samples not marked as eligible."
        )

    if (
        eligible_samples["calculated_total_reads"]
        < eligible_samples["minimum_reference_depth"]
    ).any():
        raise ValueError(
            "At least one sample is below its recorded "
            "minimum reference-depth threshold."
        )

    return eligible_samples


def validate_split_proportions() -> None:
    """Check that training, validation and test proportions are valid."""

    proportions = np.array(
        [
            TRAIN_SIZE,
            VALID_SIZE,
            TEST_SIZE,
        ],
        dtype=float,
    )

    if (proportions <= 0).any():
        raise ValueError(
            "All split proportions must be greater than zero."
        )

    if not np.isclose(
        proportions.sum(),
        1.0,
    ):
        raise ValueError(
            "TRAIN_SIZE, VALID_SIZE and TEST_SIZE "
            "must sum to 1."
        )


def create_sample_split(
    eligible_samples: pd.DataFrame,
) -> pd.DataFrame:
    """Assign eligible biological samples to train, valid and test."""

    # Sort sample IDs before random splitting so the result is
    # independent of the row order in the input CSV.
    all_sample_ids = (
        eligible_samples["sample_id"]
        .astype(str)
        .sort_values()
        .to_numpy()
    )

    # First separate the training samples from the combined
    # validation-and-test samples.
    train_ids, temporary_ids = train_test_split(
        all_sample_ids,
        test_size=VALID_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    # Split the temporary set into validation and test subsets.
    relative_test_size = (
        TEST_SIZE
        / (VALID_SIZE + TEST_SIZE)
    )

    valid_ids, test_ids = train_test_split(
        temporary_ids,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    split_records = []

    for split_name, sample_ids in [
        ("train", train_ids),
        ("valid", valid_ids),
        ("test", test_ids),
    ]:
        for sample_id in sample_ids:
            split_records.append(
                {
                    "sample_id": str(sample_id),
                    "split": split_name,
                }
            )

    split_assignments = pd.DataFrame(
        split_records
    )

    sample_split = split_assignments.merge(
        eligible_samples,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    return sample_split


def validate_sample_split(
    sample_split: pd.DataFrame,
    eligible_samples: pd.DataFrame,
) -> None:
    """Check completeness, uniqueness and separation of the split."""

    expected_samples = set(
        eligible_samples["sample_id"].astype(str)
    )

    split_samples = set(
        sample_split["sample_id"].astype(str)
    )

    if expected_samples != split_samples:
        only_in_input = sorted(
            expected_samples - split_samples
        )

        only_in_split = sorted(
            split_samples - expected_samples
        )

        raise ValueError(
            "The split does not contain exactly the eligible samples.\n"
            f"Only in eligible input: {only_in_input[:10]}\n"
            f"Only in split output: {only_in_split[:10]}"
        )

    if sample_split["sample_id"].duplicated().any():
        duplicated_samples = (
            sample_split.loc[
                sample_split["sample_id"].duplicated(
                    keep=False
                ),
                "sample_id",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Some biological samples appear more than once: "
            f"{duplicated_samples[:10]}"
        )

    valid_split_names = {
        "train",
        "valid",
        "test",
    }

    observed_split_names = set(
        sample_split["split"].astype(str)
    )

    if observed_split_names != valid_split_names:
        raise ValueError(
            "Unexpected split labels were found: "
            f"{sorted(observed_split_names)}"
        )

    train_set = set(
        sample_split.loc[
            sample_split["split"] == "train",
            "sample_id",
        ]
    )

    valid_set = set(
        sample_split.loc[
            sample_split["split"] == "valid",
            "sample_id",
        ]
    )

    test_set = set(
        sample_split.loc[
            sample_split["split"] == "test",
            "sample_id",
        ]
    )

    if train_set & valid_set:
        raise ValueError(
            "Training and validation samples overlap."
        )

    if train_set & test_set:
        raise ValueError(
            "Training and test samples overlap."
        )

    if valid_set & test_set:
        raise ValueError(
            "Validation and test samples overlap."
        )


def create_split_summary(
    sample_split: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate sample counts and depth summaries for each subset."""

    split_order = [
        "train",
        "valid",
        "test",
    ]

    split_summary = (
        sample_split
        .groupby(
            "split",
            as_index=False,
            observed=True,
        )
        .agg(
            biological_samples=(
                "sample_id",
                "nunique",
            ),
            min_reads=(
                "calculated_total_reads",
                "min",
            ),
            median_reads=(
                "calculated_total_reads",
                "median",
            ),
            mean_reads=(
                "calculated_total_reads",
                "mean",
            ),
            max_reads=(
                "calculated_total_reads",
                "max",
            ),
        )
    )

    split_summary["split"] = pd.Categorical(
        split_summary["split"],
        categories=split_order,
        ordered=True,
    )

    split_summary = (
        split_summary
        .sort_values("split")
        .reset_index(drop=True)
    )

    split_summary["split"] = (
        split_summary["split"]
        .astype(str)
    )

    return split_summary


def main(input_path: str | Path) -> None:
    """Create, validate and save the biological-sample split."""

    banner(
        "Step 3: Split eligible biological samples"
    )

    validate_split_proportions()

    eligible_samples = load_eligible_samples(
        input_path
    )

    print(
        f"Eligible biological samples loaded: "
        f"{len(eligible_samples):,}"
    )

    sample_split = create_sample_split(
        eligible_samples
    )

    validate_sample_split(
        sample_split=sample_split,
        eligible_samples=eligible_samples,
    )

    split_summary = create_split_summary(
        sample_split
    )

    split_counts = (
        sample_split["split"]
        .value_counts()
        .reindex(
            ["train", "valid", "test"]
        )
    )

    print("\nSample split counts:")
    print(split_counts.to_string())

    print("\nSequencing-depth summary by split:")
    print(
        split_summary.to_string(
            index=False,
            formatters={
                "median_reads": "{:.1f}".format,
                "mean_reads": "{:.1f}".format,
            },
        )
    )

    # Keep the output in a stable and readable order.
    split_order_mapping = {
        "train": 0,
        "valid": 1,
        "test": 2,
    }

    sample_split["_split_order"] = (
        sample_split["split"]
        .map(split_order_mapping)
    )

    sample_split = (
        sample_split
        .sort_values(
            [
                "_split_order",
                "sample_id",
            ]
        )
        .drop(columns="_split_order")
        .reset_index(drop=True)
    )

    split_output = out(
        "02_sample_split.csv",
        PREPARED_DIR,
    )

    sample_split.to_csv(
        split_output,
        index=False,
    )

    print(
        "\nSplit validation passed:"
        "\n  each biological sample appears exactly once"
        "\n  train, valid and test contain no overlapping samples"
        "\n  the random split is reproducible with RANDOM_STATE = 42"
    )

    print("\nSaved file:")
    print(f"  {split_output}")

    print(
        "\nStep 3 completed successfully.\n"
        "Next script:\n"
        "  python3 code/04_select_otu_vocabulary.py"
    )


if __name__ == "__main__":

    default_input = (
        PREPARED_DIR
        / "01_eligible_reference_samples.csv"
    )

    input_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else default_input
    )

    main(input_path)