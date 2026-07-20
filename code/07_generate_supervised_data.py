#!/usr/bin/env python3
# Author: Ximan Ding (x.ding25@imperial.ac.uk)
# Script: 07_generate_supervised_data.py
# Description: Generate shallow-sequencing supervised-learning datasets
#              from the high-depth reference relative-abundance matrix.
#
# Arguments: 1 -> Path to the reference RA matrix
#            2 -> Path to the sample split CSV
#            3 -> Path to the training-only OTU statistics CSV
#            All arguments are optional.
# Date: July 2026

"""
Step 7: Generate shallow-sequencing supervised datasets
=======================================================

For every eligible biological sample, this script generates repeated
shallow sequencing observations using multinomial sampling from the
high-depth reference relative-abundance vector.

Each output row represents:

    one biological sample
    × one shallow-subsampling repeat
    × one model component

Model components:
    254 selected OTUs
    1 aggregated OTHER component

Inputs:
    results/prepared/04_reference_ra_matrix.pkl.gz
    results/prepared/02_sample_split.csv
    results/prepared/05_model_otu_train_statistics.csv

Outputs:
    results/prepared/06_model_train.pkl.gz
    results/prepared/06_model_valid.pkl.gz
    results/prepared/06_model_test.pkl.gz

    results/prepared/06_model_train_preview.csv
    results/prepared/06_model_valid_preview.csv
    results/prepared/06_model_test_preview.csv

Important:
    Biological samples were split before shallow repeats were generated.
    Therefore, all repeats from the same biological sample remain in the
    same train, validation or test subset.

Usage:
    python3 code/07_generate_supervised_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import PREPARED_DIR, banner, out


# Shallow sequencing simulation parameters.
SHALLOW_DEPTH = 2_000
N_SUBSAMPLE_REPEATS = 5

# Pseudocount used before log10 transformation.
PSEUDOCOUNT = 1e-8

# Fixed random seed for reproducibility.
RANDOM_STATE = 42


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
            "Missing sample IDs were found in the reference matrix."
        )

    if reference_data["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate sample IDs were found in the reference matrix."
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

    reference_sums = reference_matrix.sum(axis=1)

    if not np.allclose(
        reference_sums.to_numpy(),
        1.0,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "Some reference RA vectors do not sum to one."
        )

    return reference_matrix


def load_sample_split(
    split_path: str | Path,
) -> pd.DataFrame:
    """Load and validate the fixed sample split."""

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
            "The sample split must contain train, valid and test. "
            f"Observed values: {sorted(observed_splits)}"
        )

    return sample_split


def load_otu_statistics(
    statistics_path: str | Path,
) -> pd.DataFrame:
    """Load and validate training-only OTU statistics."""

    statistics_path = Path(statistics_path)

    if not statistics_path.exists():
        raise FileNotFoundError(
            f"Training OTU statistics not found: {statistics_path}"
        )

    statistics = pd.read_csv(
        statistics_path
    )

    required_columns = {
        "otu_id",
        "otu_mean_ra_train",
        "otu_prevalence_train",
        "otu_std_ra_train",
        "otu_max_ra_train",
        "is_other",
    }

    missing_columns = required_columns.difference(
        statistics.columns
    )

    if missing_columns:
        raise ValueError(
            "Training OTU statistics are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    statistics["otu_id"] = (
        statistics["otu_id"]
        .astype("string")
        .str.strip()
    )

    numeric_columns = [
        "otu_mean_ra_train",
        "otu_prevalence_train",
        "otu_std_ra_train",
        "otu_max_ra_train",
        "is_other",
    ]

    for column in numeric_columns:
        statistics[column] = pd.to_numeric(
            statistics[column],
            errors="coerce",
        )

    if statistics["otu_id"].isna().any():
        raise ValueError(
            "Missing OTU IDs were found in training statistics."
        )

    if statistics["otu_id"].duplicated().any():
        raise ValueError(
            "Duplicate OTU IDs were found in training statistics."
        )

    if statistics[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing or non-numeric training statistics were found."
        )

    return statistics


def validate_inputs(
    reference_matrix: pd.DataFrame,
    sample_split: pd.DataFrame,
    otu_statistics: pd.DataFrame,
) -> None:
    """Check consistency among the three Stage 1 input files."""

    matrix_samples = set(
        reference_matrix.index.astype(str)
    )

    split_samples = set(
        sample_split["sample_id"].astype(str)
    )

    if matrix_samples != split_samples:
        raise ValueError(
            "The reference matrix and sample split "
            "contain different samples."
        )

    matrix_components = (
        reference_matrix
        .columns
        .astype(str)
        .tolist()
    )

    statistics_components = (
        otu_statistics["otu_id"]
        .astype(str)
        .tolist()
    )

    if matrix_components != statistics_components:
        raise ValueError(
            "The OTU statistics do not match the reference-matrix "
            "component order."
        )

    if "OTHER" not in matrix_components:
        raise ValueError(
            "The reference matrix does not contain OTHER."
        )

    if SHALLOW_DEPTH <= 0:
        raise ValueError(
            "SHALLOW_DEPTH must be greater than zero."
        )

    if N_SUBSAMPLE_REPEATS <= 0:
        raise ValueError(
            "N_SUBSAMPLE_REPEATS must be greater than zero."
        )

    if PSEUDOCOUNT <= 0:
        raise ValueError(
            "PSEUDOCOUNT must be greater than zero."
        )


def create_supervised_dataset(
    sample_ids: list[str],
    split_name: str,
    reference_matrix: pd.DataFrame,
    otu_statistics: pd.DataFrame,
    shallow_depth: int,
    repeats: int,
    random_seed: int,
) -> pd.DataFrame:
    """Generate one split of shallow-sequencing supervised data."""

    rng = np.random.default_rng(
        random_seed
    )

    otu_ids = (
        reference_matrix
        .columns
        .astype(str)
        .to_numpy()
    )

    number_components = len(
        otu_ids
    )

    statistics = (
        otu_statistics
        .set_index("otu_id")
        .reindex(otu_ids)
    )

    required_stat_columns = [
        "otu_mean_ra_train",
        "otu_prevalence_train",
        "otu_std_ra_train",
        "otu_max_ra_train",
    ]

    if (
        statistics[required_stat_columns]
        .isna()
        .any()
        .any()
    ):
        missing_components = (
            statistics.index[
                statistics[
                    required_stat_columns
                ].isna().any(axis=1)
            ]
            .tolist()
        )

        raise ValueError(
            "Some model components are missing "
            "training statistics: "
            f"{missing_components[:10]}"
        )

    mean_ra_train = (
        statistics[
            "otu_mean_ra_train"
        ]
        .to_numpy(dtype=float)
    )

    prevalence_train = (
        statistics[
            "otu_prevalence_train"
        ]
        .to_numpy(dtype=float)
    )

    std_ra_train = (
        statistics[
            "otu_std_ra_train"
        ]
        .to_numpy(dtype=float)
    )

    max_ra_train = (
        statistics[
            "otu_max_ra_train"
        ]
        .to_numpy(dtype=float)
    )

    is_other = (
        otu_ids == "OTHER"
    ).astype(np.int8)

    output_parts = []

    for sample_number, sample_id in enumerate(
        sample_ids,
        start=1,
    ):

        if sample_id not in reference_matrix.index:
            raise ValueError(
                f"Sample {sample_id} is absent "
                "from the reference matrix."
            )

        reference_probability = (
            reference_matrix
            .loc[sample_id]
            .to_numpy(dtype=float)
        )

        reference_probability = np.clip(
            reference_probability,
            0,
            None,
        )

        probability_sum = (
            reference_probability.sum()
        )

        if probability_sum <= 0:
            raise ValueError(
                f"Sample {sample_id} has a non-positive "
                "reference probability sum."
            )

        # Re-normalise defensively to remove tiny floating-point error.
        reference_probability = (
            reference_probability
            / probability_sum
        )

        for repeat_id in range(
            1,
            repeats + 1,
        ):

            shallow_counts = rng.multinomial(
                shallow_depth,
                reference_probability,
            )

            shallow_ra = (
                shallow_counts
                / shallow_depth
            )

            zero_in_shallow = (
                shallow_counts == 0
            ).astype(np.int8)

            shallow_richness = int(
                (shallow_counts > 0).sum()
            )

            # Detected components are ranked from highest to lowest
            # shallow relative abundance.
            #
            # Undetected components all receive rank richness + 1.
            shallow_ranks = np.full(
                number_components,
                shallow_richness + 1,
                dtype=np.int32,
            )

            positive_indices = np.flatnonzero(
                shallow_counts > 0
            )

            if len(positive_indices) > 0:

                ordered_indices = positive_indices[
                    np.argsort(
                        -shallow_ra[
                            positive_indices
                        ],
                        kind="mergesort",
                    )
                ]

                shallow_ranks[
                    ordered_indices
                ] = np.arange(
                    1,
                    len(ordered_indices) + 1,
                    dtype=np.int32,
                )

            shallow_rank_norm = (
                shallow_ranks
                / (shallow_richness + 1)
            )

            current_data = pd.DataFrame(
                {
                    # Biological-sample identifiers.
                    "sample_id": sample_id,
                    "split": split_name,
                    "subsample_repeat": repeat_id,
                    "otu_id": otu_ids,

                    # Shallow sequencing features.
                    "shallow_depth": shallow_depth,
                    "shallow_count": (
                        shallow_counts.astype(
                            np.int32
                        )
                    ),
                    "shallow_ra": (
                        shallow_ra.astype(
                            np.float64
                        )
                    ),
                    "log10_shallow_ra": np.log10(
                        shallow_ra
                        + PSEUDOCOUNT
                    ),
                    "log1p_shallow_count": np.log1p(
                        shallow_counts
                    ),
                    "zero_in_shallow": (
                        zero_in_shallow
                    ),
                    "shallow_rank": (
                        shallow_ranks
                    ),
                    "shallow_rank_norm": (
                        shallow_rank_norm
                    ),
                    "shallow_richness": (
                        shallow_richness
                    ),

                    # Training-only OTU-level features.
                    "otu_mean_ra_train": (
                        mean_ra_train
                    ),
                    "otu_prevalence_train": (
                        prevalence_train
                    ),
                    "otu_std_ra_train": (
                        std_ra_train
                    ),
                    "otu_max_ra_train": (
                        max_ra_train
                    ),

                    # High-depth reference targets.
                    "target_reference_ra": (
                        reference_probability
                    ),
                    "target_log10_reference_ra": np.log10(
                        reference_probability
                        + PSEUDOCOUNT
                    ),

                    # Indicator for the aggregated component.
                    "is_other": is_other,
                }
            )

            output_parts.append(
                current_data
            )

        if (
            sample_number % 20 == 0
            or sample_number == len(sample_ids)
        ):
            print(
                f"{split_name}: processed "
                f"{sample_number}/{len(sample_ids)} samples"
            )

    if not output_parts:
        raise ValueError(
            f"No data were generated for split {split_name}."
        )

    return pd.concat(
        output_parts,
        ignore_index=True,
    )


def validate_generated_dataset(
    dataset: pd.DataFrame,
    split_name: str,
    expected_sample_ids: list[str],
    expected_components: int,
) -> None:
    """Validate dimensions, identities and compositional closure."""

    expected_rows = (
        len(expected_sample_ids)
        * N_SUBSAMPLE_REPEATS
        * expected_components
    )

    if len(dataset) != expected_rows:
        raise ValueError(
            f"{split_name} contains {len(dataset):,} rows; "
            f"expected {expected_rows:,}."
        )

    if dataset["sample_id"].nunique() != len(
        expected_sample_ids
    ):
        raise ValueError(
            f"{split_name} contains an incorrect number "
            "of biological samples."
        )

    observed_samples = set(
        dataset["sample_id"].astype(str)
    )

    expected_samples = set(
        expected_sample_ids
    )

    if observed_samples != expected_samples:
        raise ValueError(
            f"{split_name} contains unexpected sample IDs."
        )

    observed_splits = set(
        dataset["split"].astype(str)
    )

    if observed_splits != {
        split_name
    }:
        raise ValueError(
            f"{split_name} contains incorrect split labels."
        )

    if (
        dataset["subsample_repeat"].nunique()
        != N_SUBSAMPLE_REPEATS
    ):
        raise ValueError(
            f"{split_name} contains an incorrect number "
            "of subsampling repeats."
        )

    group_columns = [
        "sample_id",
        "subsample_repeat",
    ]

    shallow_count_sums = (
        dataset
        .groupby(
            group_columns,
            observed=True,
        )["shallow_count"]
        .sum()
    )

    if not np.all(
        shallow_count_sums.to_numpy()
        == SHALLOW_DEPTH
    ):
        raise ValueError(
            f"{split_name} shallow counts do not "
            "sum to SHALLOW_DEPTH."
        )

    shallow_ra_sums = (
        dataset
        .groupby(
            group_columns,
            observed=True,
        )["shallow_ra"]
        .sum()
    )

    if not np.allclose(
        shallow_ra_sums.to_numpy(),
        1.0,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{split_name} shallow RA vectors "
            "do not sum to one."
        )

    target_ra_sums = (
        dataset
        .groupby(
            group_columns,
            observed=True,
        )["target_reference_ra"]
        .sum()
    )

    if not np.allclose(
        target_ra_sums.to_numpy(),
        1.0,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{split_name} target reference RA vectors "
            "do not sum to one."
        )

    if dataset.isna().any().any():
        missing_columns = (
            dataset.columns[
                dataset.isna().any(axis=0)
            ]
            .tolist()
        )

        raise ValueError(
            f"{split_name} contains missing values in: "
            f"{missing_columns}"
        )


def save_model_dataset(
    dataset: pd.DataFrame,
    output_name: str,
    preview_rows: int,
) -> tuple[Path, Path]:
    """Save a complete compressed Pickle and a small CSV preview."""

    pickle_output = out(
        f"{output_name}.pkl.gz",
        PREPARED_DIR,
    )

    preview_output = out(
        f"{output_name}_preview.csv",
        PREPARED_DIR,
    )

    dataset.to_pickle(
        pickle_output,
        compression="gzip",
    )

    dataset.head(
        preview_rows
    ).to_csv(
        preview_output,
        index=False,
    )

    return (
        pickle_output,
        preview_output,
    )


def main(
    matrix_path: str | Path,
    split_path: str | Path,
    statistics_path: str | Path,
) -> None:
    """Generate, validate and save train, validation and test data."""

    banner(
        "Step 7: Generate shallow supervised datasets"
    )

    reference_matrix = load_reference_matrix(
        matrix_path
    )

    sample_split = load_sample_split(
        split_path
    )

    otu_statistics = load_otu_statistics(
        statistics_path
    )

    validate_inputs(
        reference_matrix=reference_matrix,
        sample_split=sample_split,
        otu_statistics=otu_statistics,
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

    number_components = (
        reference_matrix.shape[1]
    )

    print(
        f"Reference matrix loaded: "
        f"{reference_matrix.shape[0]:,} samples × "
        f"{number_components:,} components"
    )

    print(
        f"Shallow sequencing depth: "
        f"{SHALLOW_DEPTH:,} reads"
    )

    print(
        f"Subsampling repeats per sample: "
        f"{N_SUBSAMPLE_REPEATS}"
    )

    print("\nExpected output rows:")

    print(
        f"  train: "
        f"{len(train_ids):,} × "
        f"{N_SUBSAMPLE_REPEATS} × "
        f"{number_components} = "
        f"{len(train_ids) * N_SUBSAMPLE_REPEATS * number_components:,}"
    )

    print(
        f"  valid: "
        f"{len(valid_ids):,} × "
        f"{N_SUBSAMPLE_REPEATS} × "
        f"{number_components} = "
        f"{len(valid_ids) * N_SUBSAMPLE_REPEATS * number_components:,}"
    )

    print(
        f"  test: "
        f"{len(test_ids):,} × "
        f"{N_SUBSAMPLE_REPEATS} × "
        f"{number_components} = "
        f"{len(test_ids) * N_SUBSAMPLE_REPEATS * number_components:,}"
    )

    print("\nGenerating training data...")

    train_data = create_supervised_dataset(
        sample_ids=train_ids,
        split_name="train",
        reference_matrix=reference_matrix,
        otu_statistics=otu_statistics,
        shallow_depth=SHALLOW_DEPTH,
        repeats=N_SUBSAMPLE_REPEATS,
        random_seed=RANDOM_STATE + 1,
    )

    print("\nGenerating validation data...")

    valid_data = create_supervised_dataset(
        sample_ids=valid_ids,
        split_name="valid",
        reference_matrix=reference_matrix,
        otu_statistics=otu_statistics,
        shallow_depth=SHALLOW_DEPTH,
        repeats=N_SUBSAMPLE_REPEATS,
        random_seed=RANDOM_STATE + 2,
    )

    print("\nGenerating test data...")

    test_data = create_supervised_dataset(
        sample_ids=test_ids,
        split_name="test",
        reference_matrix=reference_matrix,
        otu_statistics=otu_statistics,
        shallow_depth=SHALLOW_DEPTH,
        repeats=N_SUBSAMPLE_REPEATS,
        random_seed=RANDOM_STATE + 3,
    )

    validate_generated_dataset(
        dataset=train_data,
        split_name="train",
        expected_sample_ids=train_ids,
        expected_components=number_components,
    )

    validate_generated_dataset(
        dataset=valid_data,
        split_name="valid",
        expected_sample_ids=valid_ids,
        expected_components=number_components,
    )

    validate_generated_dataset(
        dataset=test_data,
        split_name="test",
        expected_sample_ids=test_ids,
        expected_components=number_components,
    )

    print("\nAll generated datasets passed validation.")

    train_file, train_preview = save_model_dataset(
        dataset=train_data,
        output_name="06_model_train",
        preview_rows=10_000,
    )

    valid_file, valid_preview = save_model_dataset(
        dataset=valid_data,
        output_name="06_model_valid",
        preview_rows=5_000,
    )

    test_file, test_preview = save_model_dataset(
        dataset=test_data,
        output_name="06_model_test",
        preview_rows=5_000,
    )

    print("\nGenerated dataset summary:")

    for split_name, dataset in [
        ("train", train_data),
        ("valid", valid_data),
        ("test", test_data),
    ]:
        print(
            f"  {split_name}: "
            f"{len(dataset):,} rows, "
            f"{dataset['sample_id'].nunique():,} samples, "
            f"zero rate = "
            f"{dataset['zero_in_shallow'].mean():.6f}"
        )

    print("\nSaved files:")
    print(f"  {train_file}")
    print(f"  {valid_file}")
    print(f"  {test_file}")
    print(f"  {train_preview}")
    print(f"  {valid_preview}")
    print(f"  {test_preview}")

    print(
        "\nStep 7 completed successfully.\n"
        "Next script:\n"
        "  python3 code/08_check_stage1_quality.py"
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

    default_statistics = (
        PREPARED_DIR
        / "05_model_otu_train_statistics.csv"
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

    statistics_path = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else default_statistics
    )

    main(
        matrix_path=matrix_path,
        split_path=split_path,
        statistics_path=statistics_path,
    )