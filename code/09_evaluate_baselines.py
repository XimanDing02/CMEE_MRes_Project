#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Author: Ximan Ding xmding02@163.com
# Script: 09_evaluate_baselines.py
# Description: Construct and evaluate two baseline methods for recovering
#              high-depth reference relative abundance from shallow 16S data.
# Arguments: None. Project paths are defined relative to this script.
# Date: July 2026

"""
Evaluate two baseline methods:
1. Raw shallow relative abundance
2. Training-set mean relative abundance

The script reads Stage-1 datasets, checks leakage, calculates evaluation
metrics, and saves baseline predictions and summaries.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. Paths and settings
# ============================================================

# Input: results/prepared
# Output: results/baseline
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent

PREPARED_DIR = PROJECT_ROOT / "results" / "prepared"
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = PREPARED_DIR / "06_model_train.pkl.gz"
VALID_FILE = PREPARED_DIR / "06_model_valid.pkl.gz"
TEST_FILE = PREPARED_DIR / "06_model_test.pkl.gz"

OUTPUT_DIR = BASELINE_DIR

PSEUDOCOUNT = 1e-8
MIN_PREDICTION = 0.0
ONE_READ_THRESHOLD_FALLBACK = 1 / 2000


# ============================================================
# 2. Helper functions
# ============================================================

def require_file(file_path: Path) -> None:
    """Check that an input file exists."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}\n"
            "Please confirm that all Stage-1 scripts completed successfully."
        )


def normalize_predictions_within_sample(
    dataframe: pd.DataFrame,
    prediction_column: str,
) -> pd.Series:
    """Normalise predictions within each sample and repeat."""
    prediction = (
        pd.to_numeric(dataframe[prediction_column], errors="coerce")
        .fillna(0.0)
        .clip(lower=MIN_PREDICTION)
    )

    group_sum = prediction.groupby(
        [dataframe["sample_id"], dataframe["subsample_repeat"]],
        observed=True,
    ).transform("sum")

    if (group_sum <= 0).any():
        raise ValueError(
            f"{prediction_column} has a non-positive total in at least one "
            "sample, so compositional normalisation cannot be performed."
        )

    return prediction / group_sum


def safe_spearman(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Calculate Spearman correlation safely."""
    if (
        len(true_values) == 0
        or np.all(true_values == true_values[0])
        or np.all(predicted_values == predicted_values[0])
    ):
        return np.nan

    correlation, _ = spearmanr(true_values, predicted_values)
    return float(correlation)


def shannon_index(values: np.ndarray) -> float:
    """Calculate Shannon diversity."""
    values = np.asarray(values, dtype=float)
    values = values[values > 0]

    if len(values) == 0:
        return 0.0

    values = values / values.sum()
    return float(-np.sum(values * np.log(values)))


def simpson_index(values: np.ndarray) -> float:
    """Calculate Simpson diversity."""
    values = np.asarray(values, dtype=float)

    if values.sum() <= 0:
        return 0.0

    values = values / values.sum()
    return float(1.0 - np.sum(values ** 2))


def bray_curtis_for_compositions(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Calculate Bray-Curtis distance."""
    return float(
        0.5 * np.sum(np.abs(true_values - predicted_values))
    )


def jensen_shannon_distance(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Calculate Jensen-Shannon distance."""
    true_values = np.clip(
        np.asarray(true_values, dtype=float),
        PSEUDOCOUNT,
        None,
    )
    predicted_values = np.clip(
        np.asarray(predicted_values, dtype=float),
        PSEUDOCOUNT,
        None,
    )

    true_values = true_values / true_values.sum()
    predicted_values = predicted_values / predicted_values.sum()

    return float(
        jensenshannon(true_values, predicted_values, base=2)
    )


def calculate_row_metrics(
    dataframe: pd.DataFrame,
    prediction_column: str,
    split_name: str,
    method_name: str,
    evaluation_scope: str,
) -> dict:
    """Calculate metrics across sample-OTU rows."""
    if evaluation_scope == "all_components":
        current = dataframe.copy()
    elif evaluation_scope == "specific_otus":
        current = dataframe.loc[dataframe["is_other"] == 0].copy()
    else:
        raise ValueError(
            f"Unknown evaluation scope: {evaluation_scope}"
        )

    true_values = current["target_reference_ra"].to_numpy(dtype=float)
    predicted_values = current[prediction_column].to_numpy(dtype=float)

    true_log = np.log10(true_values + PSEUDOCOUNT)
    predicted_log = np.log10(predicted_values + PSEUDOCOUNT)

    return {
        "split": split_name,
        "method": method_name,
        "evaluation_scope": evaluation_scope,
        "number_of_rows": len(current),
        "MAE_RA": mean_absolute_error(true_values, predicted_values),
        "RMSE_RA": np.sqrt(
            mean_squared_error(true_values, predicted_values)
        ),
        "R2_RA": r2_score(true_values, predicted_values),
        "MAE_log10_RA": mean_absolute_error(true_log, predicted_log),
        "RMSE_log10_RA": np.sqrt(
            mean_squared_error(true_log, predicted_log)
        ),
        "Spearman_RA": safe_spearman(true_values, predicted_values),
    }


def calculate_sample_metrics(
    dataframe: pd.DataFrame,
    prediction_column: str,
    split_name: str,
    method_name: str,
) -> pd.DataFrame:
    """Calculate sample-level composition, RAD, and diversity metrics."""
    sample_records = []

    for (sample_id, repeat_id), current in dataframe.groupby(
        ["sample_id", "subsample_repeat"],
        observed=True,
        sort=False,
    ):
        true_values = current["target_reference_ra"].to_numpy(dtype=float)
        predicted_values = current[prediction_column].to_numpy(dtype=float)

        true_values = np.clip(true_values, 0, None)
        predicted_values = np.clip(predicted_values, 0, None)

        true_values = true_values / true_values.sum()
        predicted_values = predicted_values / predicted_values.sum()

        composition_mae = float(
            np.mean(np.abs(true_values - predicted_values))
        )
        composition_rmse = float(
            np.sqrt(np.mean((true_values - predicted_values) ** 2))
        )
        otu_spearman = safe_spearman(true_values, predicted_values)
        bray_curtis = bray_curtis_for_compositions(
            true_values,
            predicted_values,
        )
        js_distance = jensen_shannon_distance(
            true_values,
            predicted_values,
        )

        # Compare RAD shape after sorting abundances.
        true_rad = np.sort(true_values)[::-1]
        predicted_rad = np.sort(predicted_values)[::-1]

        true_rad_log = np.log10(true_rad + PSEUDOCOUNT)
        predicted_rad_log = np.log10(predicted_rad + PSEUDOCOUNT)

        rad_mae = float(
            np.mean(np.abs(true_rad - predicted_rad))
        )
        rad_rmse = float(
            np.sqrt(np.mean((true_rad - predicted_rad) ** 2))
        )
        rad_mae_log = float(
            np.mean(np.abs(true_rad_log - predicted_rad_log))
        )
        rad_rmse_log = float(
            np.sqrt(np.mean((true_rad_log - predicted_rad_log) ** 2))
        )
        rad_spearman = safe_spearman(true_rad, predicted_rad)

        top_1_error = float(
            abs(true_rad[:1].sum() - predicted_rad[:1].sum())
        )
        top_5_error = float(
            abs(true_rad[:5].sum() - predicted_rad[:5].sum())
        )
        top_10_error = float(
            abs(true_rad[:10].sum() - predicted_rad[:10].sum())
        )

        true_shannon = shannon_index(true_values)
        predicted_shannon = shannon_index(predicted_values)
        true_simpson = simpson_index(true_values)
        predicted_simpson = simpson_index(predicted_values)

        shallow_depth = int(current["shallow_depth"].iloc[0])
        one_read_threshold = (
            1 / shallow_depth
            if shallow_depth > 0
            else ONE_READ_THRESHOLD_FALLBACK
        )

        true_richness_1read = int(
            (true_values >= one_read_threshold).sum()
        )
        predicted_richness_1read = int(
            (predicted_values >= one_read_threshold).sum()
        )

        sample_records.append(
            {
                "split": split_name,
                "method": method_name,
                "sample_id": sample_id,
                "subsample_repeat": repeat_id,
                "shallow_depth": shallow_depth,
                "number_of_components": len(current),
                "composition_MAE_RA": composition_mae,
                "composition_RMSE_RA": composition_rmse,
                "composition_Spearman": otu_spearman,
                "Bray_Curtis": bray_curtis,
                "Jensen_Shannon_distance": js_distance,
                "RAD_MAE_RA": rad_mae,
                "RAD_RMSE_RA": rad_rmse,
                "RAD_MAE_log10_RA": rad_mae_log,
                "RAD_RMSE_log10_RA": rad_rmse_log,
                "RAD_Spearman": rad_spearman,
                "Top1_cumulative_error": top_1_error,
                "Top5_cumulative_error": top_5_error,
                "Top10_cumulative_error": top_10_error,
                "true_Shannon": true_shannon,
                "predicted_Shannon": predicted_shannon,
                "Shannon_absolute_error": abs(
                    true_shannon - predicted_shannon
                ),
                "true_Simpson": true_simpson,
                "predicted_Simpson": predicted_simpson,
                "Simpson_absolute_error": abs(
                    true_simpson - predicted_simpson
                ),
                "true_richness_1read": true_richness_1read,
                "predicted_richness_1read": predicted_richness_1read,
                "richness_1read_absolute_error": abs(
                    true_richness_1read
                    - predicted_richness_1read
                ),
            }
        )

    return pd.DataFrame(sample_records)


def summarize_sample_metrics(
    sample_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise sample-level metrics."""
    metric_columns = [
        "composition_MAE_RA",
        "composition_RMSE_RA",
        "composition_Spearman",
        "Bray_Curtis",
        "Jensen_Shannon_distance",
        "RAD_MAE_RA",
        "RAD_RMSE_RA",
        "RAD_MAE_log10_RA",
        "RAD_RMSE_log10_RA",
        "RAD_Spearman",
        "Top1_cumulative_error",
        "Top5_cumulative_error",
        "Top10_cumulative_error",
        "Shannon_absolute_error",
        "Simpson_absolute_error",
        "richness_1read_absolute_error",
    ]

    summary_parts = []

    for (split_name, method_name), current in sample_metrics.groupby(
        ["split", "method"],
        observed=True,
        sort=False,
    ):
        for metric in metric_columns:
            values = (
                pd.to_numeric(current[metric], errors="coerce")
                .dropna()
            )

            summary_parts.append(
                {
                    "split": split_name,
                    "method": method_name,
                    "metric": metric,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                }
            )

    return pd.DataFrame(summary_parts)


# ============================================================
# 3. Read data
# ============================================================

print("=" * 90)
print("Step 1: Read the datasets generated in Stage 1")
print("=" * 90)

for required_file in [TRAIN_FILE, VALID_FILE, TEST_FILE]:
    require_file(required_file)

train_df = pd.read_pickle(TRAIN_FILE)
valid_df = pd.read_pickle(VALID_FILE)
test_df = pd.read_pickle(TEST_FILE)

print("Training dataset shape:", train_df.shape)
print("Validation dataset shape:", valid_df.shape)
print("Test dataset shape:", test_df.shape)


# ============================================================
# 4. Check data integrity
# ============================================================

print("\n" + "=" * 90)
print("Step 2: Check required columns and sample-level leakage")
print("=" * 90)

required_columns = [
    "sample_id",
    "split",
    "subsample_repeat",
    "otu_id",
    "shallow_depth",
    "shallow_count",
    "shallow_ra",
    "otu_mean_ra_train",
    "target_reference_ra",
    "is_other",
]

for dataset_name, dataframe in [
    ("train", train_df),
    ("valid", valid_df),
    ("test", test_df),
]:
    missing_columns = sorted(
        set(required_columns) - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )

train_samples = set(
    train_df["sample_id"].astype(str).unique()
)
valid_samples = set(
    valid_df["sample_id"].astype(str).unique()
)
test_samples = set(
    test_df["sample_id"].astype(str).unique()
)

if train_samples & valid_samples:
    raise ValueError(
        "The training and validation sets contain overlapping sample_id values."
    )

if train_samples & test_samples:
    raise ValueError(
        "The training and test sets contain overlapping sample_id values."
    )

if valid_samples & test_samples:
    raise ValueError(
        "The validation and test sets contain overlapping sample_id values."
    )

print("Number of training samples:", len(train_samples))
print("Number of validation samples:", len(valid_samples))
print("Number of test samples:", len(test_samples))
print("Sample split check passed: no sample_id leakage was detected.")


# ============================================================
# 5. Build baseline predictions
# ============================================================

print("\n" + "=" * 90)
print("Step 3: Construct the two baseline predictions")
print("=" * 90)

datasets = {
    "train": train_df,
    "valid": valid_df,
    "test": test_df,
}

for split_name, dataframe in datasets.items():
    dataframe["pred_raw_shallow_ra"] = (
        normalize_predictions_within_sample(
            dataframe,
            "shallow_ra",
        )
    )

    dataframe["pred_training_mean_ra"] = (
        normalize_predictions_within_sample(
            dataframe,
            "otu_mean_ra_train",
        )
    )

    prediction_sum_check = dataframe.groupby(
        ["sample_id", "subsample_repeat"],
        observed=True,
    )[
        [
            "target_reference_ra",
            "pred_raw_shallow_ra",
            "pred_training_mean_ra",
        ]
    ].sum()

    if not np.allclose(
        prediction_sum_check.to_numpy(),
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            f"At least one target or predicted composition in "
            f"{split_name} does not sum to one."
        )

    print(
        f"{split_name}: both baselines were constructed successfully, "
        "and every predicted composition sums to one."
    )


# ============================================================
# 6. Calculate row-level metrics
# ============================================================

print("\n" + "=" * 90)
print("Step 4: Calculate OTU row-level metrics")
print("=" * 90)

method_columns = {
    "Raw shallow RA": "pred_raw_shallow_ra",
    "Training-mean RA": "pred_training_mean_ra",
}

row_metric_records = []

for split_name in ["train", "valid", "test"]:
    dataframe = datasets[split_name]

    for method_name, prediction_column in method_columns.items():
        for evaluation_scope in [
            "all_components",
            "specific_otus",
        ]:
            row_metric_records.append(
                calculate_row_metrics(
                    dataframe=dataframe,
                    prediction_column=prediction_column,
                    split_name=split_name,
                    method_name=method_name,
                    evaluation_scope=evaluation_scope,
                )
            )

row_metrics = pd.DataFrame(row_metric_records)

row_metrics.to_csv(
    BASELINE_DIR / "01_baseline_row_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

print(row_metrics.to_string(index=False))


# ============================================================
# 7. Calculate sample-level metrics
# ============================================================

print("\n" + "=" * 90)
print(
    "Step 5: Calculate sample-level composition, RAD, "
    "and diversity metrics"
)
print("=" * 90)

sample_metric_parts = []

for split_name in ["train", "valid", "test"]:
    dataframe = datasets[split_name]

    for method_name, prediction_column in method_columns.items():
        current_metrics = calculate_sample_metrics(
            dataframe=dataframe,
            prediction_column=prediction_column,
            split_name=split_name,
            method_name=method_name,
        )

        sample_metric_parts.append(current_metrics)

        print(
            f"{split_name} - {method_name}: evaluated "
            f"{len(current_metrics)} shallow replicates."
        )

sample_metrics = pd.concat(
    sample_metric_parts,
    ignore_index=True,
)

sample_metrics.to_csv(
    BASELINE_DIR / "02_baseline_sample_metrics.csv",
    index=False,
    encoding="utf-8-sig",
)

sample_summary = summarize_sample_metrics(sample_metrics)

sample_summary.to_csv(
    OUTPUT_DIR / "03_baseline_sample_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 8. Save predictions
# ============================================================

print("\n" + "=" * 90)
print("Step 6: Save baseline prediction datasets")
print("=" * 90)

prediction_columns_to_save = [
    "sample_id",
    "split",
    "subsample_repeat",
    "otu_id",
    "shallow_depth",
    "shallow_count",
    "shallow_ra",
    "target_reference_ra",
    "is_other",
    "pred_raw_shallow_ra",
    "pred_training_mean_ra",
]

prediction_output_files = {
    "train": OUTPUT_DIR / "04_baseline_predictions_train.pkl.gz",
    "valid": OUTPUT_DIR / "05_baseline_predictions_valid.pkl.gz",
    "test": OUTPUT_DIR / "06_baseline_predictions_test.pkl.gz",
}

for split_name, dataframe in datasets.items():
    output_file = prediction_output_files[split_name]

    dataframe[prediction_columns_to_save].to_pickle(
        output_file,
        compression="gzip",
    )

    print(
        f"{split_name} baseline predictions saved to: "
        f"{output_file}"
    )


# ============================================================
# 9. Save configuration
# ============================================================

config = {
    "prepared_directory": str(PREPARED_DIR),
    "baseline_directory": str(BASELINE_DIR),
    "train_file": str(TRAIN_FILE),
    "valid_file": str(VALID_FILE),
    "test_file": str(TEST_FILE),
    "pseudo_count": PSEUDOCOUNT,
    "methods": list(method_columns.keys()),
    "note": (
        "This stage evaluates only the Raw shallow RA and "
        "Training-mean RA baselines. Random Forest and XGBoost "
        "are not trained in this script."
    ),
}

with open(
    OUTPUT_DIR / "07_baseline_config.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        config,
        file,
        ensure_ascii=False,
        indent=2,
    )


print("\n" + "=" * 90)
print("Stage 2 baseline evaluation completed")
print("=" * 90)

print(
    "Review the test-set results in:\n"
    "1. 01_baseline_row_metrics.csv\n"
    "2. 02_baseline_sample_metrics.csv\n"
    "3. 03_baseline_sample_summary.csv\n"
)

print(
    "The next stage will train Random Forest and XGBoost using "
    "the same data split and evaluation framework."
)


if __name__ == "__main__":
    pass
