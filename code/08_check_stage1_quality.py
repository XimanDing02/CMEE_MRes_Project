#!/usr/bin/env python3
# Author: Ximan Ding (x.ding25@imperial.ac.uk)
# Script: 08_check_stage1_quality.py
# Description: Perform final Stage 1 quality checks and save the analysis
#              configuration and data-preparation summary.
#
# Arguments: 1 -> Path to the training dataset
#            2 -> Path to the validation dataset
#            3 -> Path to the test dataset
#            4 -> Path to the sample split CSV
#            5 -> Path to the selected OTU vocabulary CSV
#            All arguments are optional.
# Date: July 2026

"""
Step 8: Check Stage 1 data quality
==================================

This script performs final quality-control checks on the prepared
supervised-learning datasets.

Inputs:
    results/prepared/06_model_train.pkl.gz
    results/prepared/06_model_valid.pkl.gz
    results/prepared/06_model_test.pkl.gz
    results/prepared/02_sample_split.csv
    results/prepared/03_selected_otu_vocabulary.csv

Outputs:
    results/prepared/07_stage1_quality_summary.csv
    results/prepared/analysis_config.json
    results/prepared/README_stage1.txt

The script verifies:

1. expected numbers of biological samples;
2. expected numbers of shallow repeats;
3. expected numbers of model components;
4. no overlap between train, validation and test samples;
5. shallow counts sum to the requested shallow depth;
6. shallow relative abundances sum to one;
7. target reference relative abundances sum to one;
8. training-derived OTU statistics remain constant across datasets;
9. exactly one OTHER component is present;
10. no missing values are present.

Usage:
    python3 code/08_check_stage1_quality.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _paths import PREPARED_DIR, banner, out


MIN_REFERENCE_DEPTH = 10_000
SHALLOW_DEPTH = 2_000
N_SUBSAMPLE_REPEATS = 5
TRAIN_SIZE = 0.70
VALID_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42
PSEUDOCOUNT = 1e-8

TARGET_PROJECT_ID = "MGYS00002437"
TARGET_CLASSIFICATION = "seawater"

EXPECTED_COMPONENTS = 255
EXPECTED_SPECIFIC_OTUS = 254

EXPECTED_COLUMNS = [
    "sample_id",
    "split",
    "subsample_repeat",
    "otu_id",
    "shallow_depth",
    "shallow_count",
    "shallow_ra",
    "log10_shallow_ra",
    "log1p_shallow_count",
    "zero_in_shallow",
    "shallow_rank",
    "shallow_rank_norm",
    "shallow_richness",
    "otu_mean_ra_train",
    "otu_prevalence_train",
    "otu_std_ra_train",
    "otu_max_ra_train",
    "target_reference_ra",
    "target_log10_reference_ra",
    "is_other",
]


def load_model_dataset(
    dataset_path: str | Path,
    expected_split: str,
) -> pd.DataFrame:
    """Load and validate one prepared model dataset."""

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"{expected_split} dataset not found: {dataset_path}"
        )

    dataset = pd.read_pickle(
        dataset_path,
        compression="gzip",
    )

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError(
            f"The {expected_split} file does not contain a DataFrame."
        )

    missing_columns = set(
        EXPECTED_COLUMNS
    ).difference(
        dataset.columns
    )

    if missing_columns:
        raise ValueError(
            f"The {expected_split} dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataset = dataset[
        EXPECTED_COLUMNS
    ].copy()

    for column in [
        "sample_id",
        "split",
        "otu_id",
    ]:
        dataset[column] = (
            dataset[column]
            .astype("string")
            .str.strip()
        )

    dataset["split"] = (
        dataset["split"]
        .str.lower()
    )

    observed_splits = set(
        dataset["split"]
        .astype(str)
        .unique()
    )

    if observed_splits != {expected_split}:
        raise ValueError(
            f"The {expected_split} dataset contains "
            f"unexpected split labels: {sorted(observed_splits)}"
        )

    if dataset.isna().any().any():
        missing_value_columns = (
            dataset.columns[
                dataset.isna().any(axis=0)
            ]
            .tolist()
        )

        raise ValueError(
            f"The {expected_split} dataset contains missing values in: "
            f"{missing_value_columns}"
        )

    return dataset


def load_sample_split(
    split_path: str | Path,
) -> pd.DataFrame:
    """Load the fixed biological-sample split."""

    split_path = Path(split_path)

    if not split_path.exists():
        raise FileNotFoundError(
            f"Sample split file not found: {split_path}"
        )

    sample_split = pd.read_csv(
        split_path
    )

    required_columns = {
        "sample_id",
        "split",
        "calculated_total_reads",
    }

    missing_columns = required_columns.difference(
        sample_split.columns
    )

    if missing_columns:
        raise ValueError(
            "The sample split file is missing columns: "
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

    sample_split["calculated_total_reads"] = pd.to_numeric(
        sample_split["calculated_total_reads"],
        errors="coerce",
    )

    if sample_split["calculated_total_reads"].isna().any():
        raise ValueError(
            "Missing or non-numeric sequencing depths "
            "were found in the split file."
        )

    if sample_split["sample_id"].duplicated().any():
        raise ValueError(
            "Duplicate biological samples were found "
            "in the split file."
        )

    return sample_split


def load_vocabulary(
    vocabulary_path: str | Path,
) -> pd.DataFrame:
    """Load and validate the selected OTU vocabulary."""

    vocabulary_path = Path(vocabulary_path)

    if not vocabulary_path.exists():
        raise FileNotFoundError(
            f"OTU vocabulary not found: {vocabulary_path}"
        )

    vocabulary = pd.read_csv(
        vocabulary_path
    )

    required_columns = {
        "otu_id",
        "selection_rank",
    }

    missing_columns = required_columns.difference(
        vocabulary.columns
    )

    if missing_columns:
        raise ValueError(
            "The OTU vocabulary is missing columns: "
            f"{sorted(missing_columns)}"
        )

    vocabulary["otu_id"] = (
        vocabulary["otu_id"]
        .astype("string")
        .str.strip()
    )

    if vocabulary["otu_id"].duplicated().any():
        raise ValueError(
            "Duplicate OTUs were found in the vocabulary."
        )

    return vocabulary


def validate_dataset_structure(
    dataset: pd.DataFrame,
    split_name: str,
    expected_sample_ids: set[str],
) -> dict:
    """Validate one split and return quality-summary values."""

    observed_sample_ids = set(
        dataset["sample_id"]
        .astype(str)
    )

    if observed_sample_ids != expected_sample_ids:
        only_in_dataset = sorted(
            observed_sample_ids - expected_sample_ids
        )

        only_in_split = sorted(
            expected_sample_ids - observed_sample_ids
        )

        raise ValueError(
            f"{split_name} sample IDs do not match the split file.\n"
            f"Only in dataset: {only_in_dataset[:10]}\n"
            f"Only in split file: {only_in_split[:10]}"
        )

    component_counts = (
        dataset
        .groupby(
            [
                "sample_id",
                "subsample_repeat",
            ],
            observed=True,
        )["otu_id"]
        .nunique()
    )

    if not (
        component_counts == EXPECTED_COMPONENTS
    ).all():
        raise ValueError(
            f"{split_name} contains shallow observations "
            "with an incorrect number of model components."
        )

    repeats_per_sample = (
        dataset
        .groupby(
            "sample_id",
            observed=True,
        )["subsample_repeat"]
        .nunique()
    )

    if not (
        repeats_per_sample == N_SUBSAMPLE_REPEATS
    ).all():
        raise ValueError(
            f"{split_name} does not contain exactly "
            f"{N_SUBSAMPLE_REPEATS} repeats per sample."
        )

    repeat_values = set(
        dataset["subsample_repeat"]
        .astype(int)
        .unique()
    )

    expected_repeat_values = set(
        range(
            1,
            N_SUBSAMPLE_REPEATS + 1,
        )
    )

    if repeat_values != expected_repeat_values:
        raise ValueError(
            f"{split_name} contains unexpected repeat IDs: "
            f"{sorted(repeat_values)}"
        )

    shallow_depth_values = set(
        dataset["shallow_depth"]
        .astype(int)
        .unique()
    )

    if shallow_depth_values != {SHALLOW_DEPTH}:
        raise ValueError(
            f"{split_name} contains unexpected shallow depths: "
            f"{sorted(shallow_depth_values)}"
        )

    group_columns = [
        "sample_id",
        "subsample_repeat",
    ]

    shallow_count_sum = (
        dataset
        .groupby(
            group_columns,
            observed=True,
        )["shallow_count"]
        .sum()
    )

    shallow_ra_sum = (
        dataset
        .groupby(
            group_columns,
            observed=True,
        )["shallow_ra"]
        .sum()
    )

    target_ra_sum = (
        dataset
        .groupby(
            group_columns,
            observed=True,
        )["target_reference_ra"]
        .sum()
    )

    if not (
        shallow_count_sum == SHALLOW_DEPTH
    ).all():
        raise ValueError(
            f"{split_name} shallow counts do not sum "
            f"to {SHALLOW_DEPTH}."
        )

    if not np.allclose(
        shallow_ra_sum.to_numpy(),
        1.0,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{split_name} shallow RA vectors do not sum to one."
        )

    if not np.allclose(
        target_ra_sum.to_numpy(),
        1.0,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{split_name} target RA vectors do not sum to one."
        )

    if not dataset["zero_in_shallow"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            f"{split_name} contains invalid zero indicators."
        )

    expected_zero_indicator = (
        dataset["shallow_count"] == 0
    ).astype(np.int8)

    if not np.array_equal(
        dataset["zero_in_shallow"].to_numpy(),
        expected_zero_indicator.to_numpy(),
    ):
        raise ValueError(
            f"{split_name} zero indicators do not match shallow counts."
        )

    expected_shallow_ra = (
        dataset["shallow_count"]
        / dataset["shallow_depth"]
    )

    if not np.allclose(
        dataset["shallow_ra"].to_numpy(),
        expected_shallow_ra.to_numpy(),
        rtol=0,
        atol=1e-15,
    ):
        raise ValueError(
            f"{split_name} shallow RA values do not match "
            "shallow_count / shallow_depth."
        )

    if not np.allclose(
        dataset["log10_shallow_ra"].to_numpy(),
        np.log10(
            dataset["shallow_ra"].to_numpy()
            + PSEUDOCOUNT
        ),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{split_name} log10 shallow RA values are inconsistent."
        )

    if not np.allclose(
        dataset["target_log10_reference_ra"].to_numpy(),
        np.log10(
            dataset["target_reference_ra"].to_numpy()
            + PSEUDOCOUNT
        ),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            f"{split_name} log10 target values are inconsistent."
        )

    other_rows = dataset.loc[
        dataset["otu_id"] == "OTHER"
    ]

    expected_other_rows = (
        len(expected_sample_ids)
        * N_SUBSAMPLE_REPEATS
    )

    if len(other_rows) != expected_other_rows:
        raise ValueError(
            f"{split_name} contains an incorrect number "
            "of OTHER rows."
        )

    if not (
        other_rows["is_other"] == 1
    ).all():
        raise ValueError(
            f"{split_name} OTHER rows are not marked correctly."
        )

    non_other_rows = dataset.loc[
        dataset["otu_id"] != "OTHER"
    ]

    if not (
        non_other_rows["is_other"] == 0
    ).all():
        raise ValueError(
            f"{split_name} specific OTUs are incorrectly "
            "marked as OTHER."
        )

    expected_rows = (
        len(expected_sample_ids)
        * N_SUBSAMPLE_REPEATS
        * EXPECTED_COMPONENTS
    )

    if len(dataset) != expected_rows:
        raise ValueError(
            f"{split_name} contains {len(dataset):,} rows; "
            f"expected {expected_rows:,}."
        )

    return {
        "dataset": split_name,
        "rows": len(dataset),
        "biological_samples": (
            dataset["sample_id"].nunique()
        ),
        "subsample_repeats": (
            dataset["subsample_repeat"].nunique()
        ),
        "model_components": (
            dataset["otu_id"].nunique()
        ),
        "shallow_depth": SHALLOW_DEPTH,
        "zero_rate_in_shallow": (
            dataset["zero_in_shallow"].mean()
        ),
        "mean_shallow_richness": (
            dataset[
                [
                    "sample_id",
                    "subsample_repeat",
                    "shallow_richness",
                ]
            ]
            .drop_duplicates()[
                "shallow_richness"
            ]
            .mean()
        ),
        "min_shallow_ra_sum": (
            shallow_ra_sum.min()
        ),
        "max_shallow_ra_sum": (
            shallow_ra_sum.max()
        ),
        "min_target_ra_sum": (
            target_ra_sum.min()
        ),
        "max_target_ra_sum": (
            target_ra_sum.max()
        ),
        "minimum_shallow_count_sum": (
            shallow_count_sum.min()
        ),
        "maximum_shallow_count_sum": (
            shallow_count_sum.max()
        ),
    }


def validate_no_sample_overlap(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Confirm that biological samples do not overlap across subsets."""

    train_samples = set(
        train_data["sample_id"].astype(str)
    )

    valid_samples = set(
        valid_data["sample_id"].astype(str)
    )

    test_samples = set(
        test_data["sample_id"].astype(str)
    )

    if train_samples & valid_samples:
        raise ValueError(
            "Training and validation samples overlap."
        )

    if train_samples & test_samples:
        raise ValueError(
            "Training and test samples overlap."
        )

    if valid_samples & test_samples:
        raise ValueError(
            "Validation and test samples overlap."
        )


def validate_training_statistics_consistency(
    datasets: list[pd.DataFrame],
) -> None:
    """Check that training-derived OTU statistics are constant everywhere."""

    statistic_columns = [
        "otu_mean_ra_train",
        "otu_prevalence_train",
        "otu_std_ra_train",
        "otu_max_ra_train",
        "is_other",
    ]

    combined = pd.concat(
        [
            dataset[
                ["otu_id"] + statistic_columns
            ]
            for dataset in datasets
        ],
        ignore_index=True,
    )

    unique_counts = (
        combined
        .groupby(
            "otu_id",
            observed=True,
        )[statistic_columns]
        .nunique(
            dropna=False
        )
    )

    inconsistent = unique_counts.loc[
        (unique_counts > 1).any(axis=1)
    ]

    if not inconsistent.empty:
        raise ValueError(
            "Training-derived OTU statistics are not constant "
            "across generated datasets:\n"
            f"{inconsistent.head(10).to_string()}"
        )


def create_depth_summary(
    sample_split: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise reference sequencing depths by subset."""

    split_order = [
        "train",
        "valid",
        "test",
    ]

    depth_summary = (
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
            min_reference_reads=(
                "calculated_total_reads",
                "min",
            ),
            median_reference_reads=(
                "calculated_total_reads",
                "median",
            ),
            mean_reference_reads=(
                "calculated_total_reads",
                "mean",
            ),
            max_reference_reads=(
                "calculated_total_reads",
                "max",
            ),
        )
    )

    depth_summary["split"] = pd.Categorical(
        depth_summary["split"],
        categories=split_order,
        ordered=True,
    )

    depth_summary = (
        depth_summary
        .sort_values("split")
        .reset_index(drop=True)
    )

    depth_summary["split"] = (
        depth_summary["split"]
        .astype(str)
    )

    return depth_summary


def main(
    train_path: str | Path,
    valid_path: str | Path,
    test_path: str | Path,
    split_path: str | Path,
    vocabulary_path: str | Path,
) -> None:
    """Run all final Stage 1 quality checks."""

    banner(
        "Step 8: Check Stage 1 data quality"
    )

    train_data = load_model_dataset(
        train_path,
        expected_split="train",
    )

    valid_data = load_model_dataset(
        valid_path,
        expected_split="valid",
    )

    test_data = load_model_dataset(
        test_path,
        expected_split="test",
    )

    sample_split = load_sample_split(
        split_path
    )

    vocabulary = load_vocabulary(
        vocabulary_path
    )

    if len(vocabulary) != EXPECTED_SPECIFIC_OTUS:
        raise ValueError(
            f"The vocabulary contains {len(vocabulary)} OTUs; "
            f"expected {EXPECTED_SPECIFIC_OTUS}."
        )

    train_ids = set(
        sample_split.loc[
            sample_split["split"] == "train",
            "sample_id",
        ].astype(str)
    )

    valid_ids = set(
        sample_split.loc[
            sample_split["split"] == "valid",
            "sample_id",
        ].astype(str)
    )

    test_ids = set(
        sample_split.loc[
            sample_split["split"] == "test",
            "sample_id",
        ].astype(str)
    )

    quality_records = []

    quality_records.append(
        validate_dataset_structure(
            dataset=train_data,
            split_name="train",
            expected_sample_ids=train_ids,
        )
    )

    quality_records.append(
        validate_dataset_structure(
            dataset=valid_data,
            split_name="valid",
            expected_sample_ids=valid_ids,
        )
    )

    quality_records.append(
        validate_dataset_structure(
            dataset=test_data,
            split_name="test",
            expected_sample_ids=test_ids,
        )
    )

    validate_no_sample_overlap(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )

    validate_training_statistics_consistency(
        datasets=[
            train_data,
            valid_data,
            test_data,
        ]
    )

    quality_summary = pd.DataFrame(
        quality_records
    )

    depth_summary = create_depth_summary(
        sample_split
    )

    quality_output = out(
        "07_stage1_quality_summary.csv",
        PREPARED_DIR,
    )

    quality_summary.to_csv(
        quality_output,
        index=False,
    )

    analysis_config = {
        "project_id": TARGET_PROJECT_ID,
        "classification": TARGET_CLASSIFICATION,
        "minimum_reference_depth": MIN_REFERENCE_DEPTH,
        "shallow_depth": SHALLOW_DEPTH,
        "subsample_repeats": N_SUBSAMPLE_REPEATS,
        "train_fraction": TRAIN_SIZE,
        "validation_fraction": VALID_SIZE,
        "test_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "pseudocount": PSEUDOCOUNT,
        "eligible_reference_samples": int(
            sample_split["sample_id"].nunique()
        ),
        "train_samples": len(train_ids),
        "valid_samples": len(valid_ids),
        "test_samples": len(test_ids),
        "specific_otu_count": len(vocabulary),
        "model_components_including_other": EXPECTED_COMPONENTS,
        "train_rows": len(train_data),
        "valid_rows": len(valid_data),
        "test_rows": len(test_data),
        "reference_target_definition": (
            "Within-sample OTU count divided by the sum of all "
            "OTU counts in the same eligible high-depth sample."
        ),
        "shallow_simulation": (
            "Multinomial sampling from the high-depth reference "
            "relative-abundance vector."
        ),
        "leakage_controls": [
            "Biological samples were split before shallow repeats "
            "were generated.",
            "All shallow repeats from one sample remain in one split.",
            "The OTU vocabulary was selected from training samples only.",
            "OTU mean abundance, prevalence, standard deviation and "
            "maximum abundance were calculated from training samples only.",
            "Validation and test samples were not used to define "
            "model features or OTU inclusion.",
        ],
        "train_file": str(Path(train_path)),
        "valid_file": str(Path(valid_path)),
        "test_file": str(Path(test_path)),
        "sample_split_file": str(Path(split_path)),
        "otu_vocabulary_file": str(Path(vocabulary_path)),
    }

    config_output = out(
        "analysis_config.json",
        PREPARED_DIR,
    )

    with open(
        config_output,
        "w",
        encoding="utf-8",
    ) as config_file:
        json.dump(
            analysis_config,
            config_file,
            ensure_ascii=False,
            indent=2,
        )

    readme_text = f"""
Stage 1 data preparation completed
=================================

1. Study scope
--------------

Project:
    {TARGET_PROJECT_ID}

Environmental classification:
    {TARGET_CLASSIFICATION}

Eligible high-depth reference samples:
    {sample_split["sample_id"].nunique()}

Minimum high-depth reference sequencing depth:
    {MIN_REFERENCE_DEPTH:,} reads


2. Prediction task
------------------

Input:
    Features obtained from simulated shallow 16S sequencing observations.

Target:
    The high-depth observed reference relative abundance for the same
    biological sample and model component.

The target is a high-depth sequencing-derived reference observation.
It is not absolute microbial abundance and is not assumed to be a
noise-free biological truth.


3. Shallow sequencing simulation
---------------------------------

Shallow depth:
    {SHALLOW_DEPTH:,} reads

Subsampling repeats per biological sample:
    {N_SUBSAMPLE_REPEATS}

Simulation method:
    Multinomial sampling from each sample's high-depth reference
    relative-abundance vector.


4. Biological-sample split
--------------------------

Training samples:
    {len(train_ids)}

Validation samples:
    {len(valid_ids)}

Test samples:
    {len(test_ids)}

Samples were split before shallow simulations were generated.
All repeats from one biological sample therefore remain in one subset.


5. OTU representation
---------------------

Specific selected OTUs:
    {len(vocabulary)}

Aggregated components:
    1 OTHER component

Total model components:
    {EXPECTED_COMPONENTS}

The OTU vocabulary was selected using training samples only.
All non-selected OTUs were combined into OTHER.


6. Training-only OTU features
-----------------------------

The following OTU-level features were calculated from training samples only:

    otu_mean_ra_train
    otu_prevalence_train
    otu_std_ra_train
    otu_max_ra_train


7. Prepared model datasets
--------------------------

Training rows:
    {len(train_data):,}

Validation rows:
    {len(valid_data):,}

Test rows:
    {len(test_data):,}


8. Interpretation limits
------------------------

The analysis may be interpreted as recovery of high-depth observed
relative abundance from shallow 16S observations for new independent
samples from the same seawater study and fixed OTU vocabulary.

The analysis must not be interpreted as:

1. prediction of absolute microbial abundance;
2. prediction of a noise-free biological truth;
3. prediction of completely new habitats;
4. prediction of taxonomic units absent from the training-derived
   vocabulary;
5. evidence that high-depth sequencing perfectly represents the
   underlying community.


9. Next modelling stage
-----------------------

The prepared datasets can be used to compare:

1. raw shallow relative abundance;
2. training-mean relative-abundance baseline;
3. Random Forest regression;
4. XGBoost regression.
"""

    readme_output = out(
        "README_stage1.txt",
        PREPARED_DIR,
    )

    with open(
        readme_output,
        "w",
        encoding="utf-8",
    ) as readme_file:
        readme_file.write(
            readme_text.strip()
            + "\n"
        )

    print("\nStage 1 quality summary:")
    print(
        quality_summary.to_string(
            index=False,
            formatters={
                "zero_rate_in_shallow": "{:.6f}".format,
                "mean_shallow_richness": "{:.2f}".format,
                "min_shallow_ra_sum": "{:.12f}".format,
                "max_shallow_ra_sum": "{:.12f}".format,
                "min_target_ra_sum": "{:.12f}".format,
                "max_target_ra_sum": "{:.12f}".format,
            },
        )
    )

    print("\nReference sequencing-depth summary:")
    print(
        depth_summary.to_string(
            index=False,
            formatters={
                "median_reference_reads": "{:.1f}".format,
                "mean_reference_reads": "{:.1f}".format,
            },
        )
    )

    print("\nAll Stage 1 checks passed:")
    print(
        "  train, validation and test samples do not overlap"
    )
    print(
        "  every biological sample has exactly five shallow repeats"
    )
    print(
        "  every shallow observation contains 255 components"
    )
    print(
        "  every shallow count vector sums to 2,000 reads"
    )
    print(
        "  every shallow relative-abundance vector sums to one"
    )
    print(
        "  every target reference vector sums to one"
    )
    print(
        "  training-derived OTU features are constant across datasets"
    )
    print(
        "  exactly one OTHER component is present"
    )
    print(
        "  no missing values were found"
    )

    print("\nSaved files:")
    print(f"  {quality_output}")
    print(f"  {config_output}")
    print(f"  {readme_output}")

    print(
        "\nStep 8 completed successfully.\n"
        "Stage 1 data preparation is now complete."
    )


if __name__ == "__main__":

    default_train = (
        PREPARED_DIR
        / "06_model_train.pkl.gz"
    )

    default_valid = (
        PREPARED_DIR
        / "06_model_valid.pkl.gz"
    )

    default_test = (
        PREPARED_DIR
        / "06_model_test.pkl.gz"
    )

    default_split = (
        PREPARED_DIR
        / "02_sample_split.csv"
    )

    default_vocabulary = (
        PREPARED_DIR
        / "03_selected_otu_vocabulary.csv"
    )

    train_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else default_train
    )

    valid_path = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else default_valid
    )

    test_path = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else default_test
    )

    split_path = (
        Path(sys.argv[4])
        if len(sys.argv) > 4
        else default_split
    )

    vocabulary_path = (
        Path(sys.argv[5])
        if len(sys.argv) > 5
        else default_vocabulary
    )

    main(
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        split_path=split_path,
        vocabulary_path=vocabulary_path,
    )