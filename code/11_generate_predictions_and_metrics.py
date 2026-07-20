#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Ximan Ding
# Script: 11_generate_predictions_and_metrics.py
# Description:
#    Load the trained Random Forest and XGBoost models, generate predictions
#    for four methods, normalise predictions within each sample, and calculate
#    row-level, sample-level, abundance-stratum and shallow-zero recovery metrics.
#
#    Compared methods:
#    1. Raw shallow RA
#    2. Training-mean RA
#    3. Random Forest
#    4. XGBoost
#
# Arguments:
#    None. Project paths are resolved relative to this script.
#
# Inputs:
#    results/prepared/06_model_train.pkl.gz
#    results/prepared/06_model_valid.pkl.gz
#    results/prepared/06_model_test.pkl.gz
#    results/models/random_forest_model.joblib
#    results/models/xgboost_model.joblib
#
# Outputs:
#    results/intermediate/11_predictions_train.pkl.gz
#    results/intermediate/11_predictions_valid.pkl.gz
#    results/intermediate/11_predictions_test.pkl.gz
#    results/intermediate/11_predictions_*_preview.csv
#    results/intermediate/11_row_metrics.csv
#    results/intermediate/11_sample_metrics.csv
#    results/intermediate/11_sample_summary.csv
#    results/intermediate/11_abundance_stratum_metrics.csv
#    results/intermediate/11_zero_recovery_metrics.csv
#    results/intermediate/11_prediction_config.json
#
# Date: July 2026

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREPARED_DIR = PROJECT_ROOT / "results" / "prepared"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
INTERMEDIATE_DIR = PROJECT_ROOT / "results" / "intermediate"
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = PREPARED_DIR / "06_model_train.pkl.gz"
VALID_FILE = PREPARED_DIR / "06_model_valid.pkl.gz"
TEST_FILE = PREPARED_DIR / "06_model_test.pkl.gz"

RF_MODEL_FILE = MODEL_DIR / "random_forest_model.joblib"
XGB_MODEL_FILE = MODEL_DIR / "xgboost_model.joblib"

ROW_METRICS_FILE = INTERMEDIATE_DIR / "11_row_metrics.csv"
SAMPLE_METRICS_FILE = INTERMEDIATE_DIR / "11_sample_metrics.csv"
SAMPLE_SUMMARY_FILE = INTERMEDIATE_DIR / "11_sample_summary.csv"
ABUNDANCE_METRICS_FILE = (
    INTERMEDIATE_DIR / "11_abundance_stratum_metrics.csv"
)
ZERO_RECOVERY_FILE = (
    INTERMEDIATE_DIR / "11_zero_recovery_metrics.csv"
)
CONFIG_FILE = INTERMEDIATE_DIR / "11_prediction_config.json"


# ============================================================
# 2. Global settings
# ============================================================

PSEUDOCOUNT = 1e-8
MIN_LOG_PREDICTION = np.log10(PSEUDOCOUNT)
MAX_LOG_PREDICTION = 0.0

SAVE_TRAIN_PREDICTIONS = True

FEATURE_COLUMNS = [
    "log1p_shallow_count",
    "log10_shallow_ra",
    "zero_in_shallow",
    "shallow_rank_norm",
    "shallow_richness",
    "otu_mean_ra_train",
    "otu_prevalence_train",
    "otu_std_ra_train",
    "otu_max_ra_train",
    "is_other",
]

TARGET_COLUMN = "target_log10_reference_ra"

METHOD_COLUMNS = {
    "Raw shallow RA": "pred_raw_shallow_ra",
    "Training-mean RA": "pred_training_mean_ra",
    "Random Forest": "pred_random_forest_ra",
    "XGBoost": "pred_xgboost_ra",
}


# ============================================================
# 3. Utility checks
# ============================================================

def print_section(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def require_file(file_path: Path) -> None:
    """Raise an informative error if an input file is missing."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{file_path}"
        )


def check_required_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Check columns needed for prediction and evaluation."""
    required_columns = set(
        FEATURE_COLUMNS
        + [
            TARGET_COLUMN,
            "target_reference_ra",
            "sample_id",
            "split",
            "subsample_repeat",
            "otu_id",
            "shallow_depth",
            "shallow_count",
            "shallow_ra",
            "shallow_rank",
            "is_other",
        ]
    )

    missing_columns = sorted(
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns:\n"
            f"{missing_columns}"
        )


def save_json(data: dict[str, Any], output_file: Path) -> None:
    """Save a dictionary as readable JSON."""
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# 4. Prediction conversion and normalisation
# ============================================================

def inverse_log_prediction(
    predicted_log10_ra: np.ndarray,
) -> np.ndarray:
    """Convert log10(RA + pseudocount) predictions back to RA."""
    predicted_log10_ra = np.asarray(
        predicted_log10_ra,
        dtype=float,
    )

    predicted_log10_ra = np.clip(
        predicted_log10_ra,
        MIN_LOG_PREDICTION,
        MAX_LOG_PREDICTION,
    )

    predicted_ra = (
        np.power(10.0, predicted_log10_ra)
        - PSEUDOCOUNT
    )

    return np.clip(predicted_ra, 0.0, None)


def normalize_array_by_sample(
    dataframe: pd.DataFrame,
    prediction_values: np.ndarray,
) -> np.ndarray:
    """
    Normalise predictions within sample_id × subsample_repeat.

    If a group sums to zero, Raw shallow RA is used as a fallback.
    """
    temp = dataframe[
        ["sample_id", "subsample_repeat", "shallow_ra"]
    ].copy()

    temp["_prediction"] = np.asarray(
        prediction_values,
        dtype=float,
    )

    temp["_prediction"] = (
        temp["_prediction"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    group_columns = ["sample_id", "subsample_repeat"]

    prediction_sum = temp.groupby(
        group_columns,
        observed=True,
    )["_prediction"].transform("sum")

    zero_sum_mask = prediction_sum <= 0

    if zero_sum_mask.any():
        print(
            "Warning: some prediction groups summed to zero; "
            "Raw shallow RA was used as a fallback."
        )

        temp.loc[
            zero_sum_mask,
            "_prediction",
        ] = temp.loc[
            zero_sum_mask,
            "shallow_ra",
        ]

        prediction_sum = temp.groupby(
            group_columns,
            observed=True,
        )["_prediction"].transform("sum")

    if (prediction_sum <= 0).any():
        raise ValueError(
            "At least one sample group still has a zero total "
            "after fallback normalisation."
        )

    return (
        temp["_prediction"] / prediction_sum
    ).to_numpy(dtype=float)


def create_all_method_predictions(
    dataframe: pd.DataFrame,
    rf_model: Any,
    xgb_model: Any,
) -> pd.DataFrame:
    """Generate predictions from the two baselines and two ML models."""
    result = dataframe.copy()

    result["pred_raw_shallow_ra"] = normalize_array_by_sample(
        result,
        result["shallow_ra"].to_numpy(dtype=float),
    )

    result["pred_training_mean_ra"] = normalize_array_by_sample(
        result,
        result["otu_mean_ra_train"].to_numpy(dtype=float),
    )

    X = result[FEATURE_COLUMNS].astype(np.float32)

    rf_pred_log = rf_model.predict(X)
    rf_pred_ra = inverse_log_prediction(rf_pred_log)
    result["pred_random_forest_ra"] = normalize_array_by_sample(
        result,
        rf_pred_ra,
    )

    xgb_pred_log = xgb_model.predict(X)
    xgb_pred_ra = inverse_log_prediction(xgb_pred_log)
    result["pred_xgboost_ra"] = normalize_array_by_sample(
        result,
        xgb_pred_ra,
    )

    return result


# ============================================================
# 5. Evaluation functions
# ============================================================

def safe_spearman(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Calculate Spearman correlation safely."""
    true_values = np.asarray(true_values, dtype=float)
    predicted_values = np.asarray(
        predicted_values,
        dtype=float,
    )

    if (
        len(true_values) < 2
        or np.all(true_values == true_values[0])
        or np.all(predicted_values == predicted_values[0])
    ):
        return np.nan

    correlation, _ = spearmanr(
        true_values,
        predicted_values,
    )

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
    """Calculate Simpson diversity in 1-D form."""
    values = np.asarray(values, dtype=float)

    if values.sum() <= 0:
        return 0.0

    values = values / values.sum()
    return float(1.0 - np.sum(values ** 2))


def bray_curtis(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> float:
    """Calculate Bray-Curtis distance for two closed compositions."""
    return float(
        0.5
        * np.sum(
            np.abs(
                np.asarray(true_values)
                - np.asarray(predicted_values)
            )
        )
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
        jensenshannon(
            true_values,
            predicted_values,
            base=2,
        )
    )


def calculate_row_metrics(
    dataframe: pd.DataFrame,
    method_name: str,
    prediction_column: str,
    split_name: str,
    scope_name: str,
) -> dict[str, Any]:
    """Calculate OTU row-level metrics."""
    if scope_name == "all_components":
        current = dataframe
    elif scope_name == "specific_otus":
        current = dataframe.loc[dataframe["is_other"] == 0]
    else:
        raise ValueError(f"Unknown scope: {scope_name}")

    y_true = current["target_reference_ra"].to_numpy(dtype=float)
    y_pred = current[prediction_column].to_numpy(dtype=float)

    y_true_log = np.log10(y_true + PSEUDOCOUNT)
    y_pred_log = np.log10(y_pred + PSEUDOCOUNT)

    return {
        "split": split_name,
        "method": method_name,
        "evaluation_scope": scope_name,
        "number_of_rows": len(current),
        "MAE_RA": mean_absolute_error(y_true, y_pred),
        "RMSE_RA": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "R2_RA": r2_score(y_true, y_pred),
        "MAE_log10_RA": mean_absolute_error(
            y_true_log,
            y_pred_log,
        ),
        "RMSE_log10_RA": np.sqrt(
            mean_squared_error(
                y_true_log,
                y_pred_log,
            )
        ),
        "Spearman_RA": safe_spearman(
            y_true,
            y_pred,
        ),
    }


def assign_abundance_group(
    target_ra: pd.Series,
) -> pd.Series:
    """Assign reference-RA abundance strata."""
    conditions = [
        target_ra >= 1e-2,
        (target_ra >= 1e-3) & (target_ra < 1e-2),
        (target_ra >= 1e-4) & (target_ra < 1e-3),
        (target_ra > 0) & (target_ra < 1e-4),
        target_ra == 0,
    ]

    choices = [
        "Dominant (>=1e-2)",
        "Common (1e-3 to 1e-2)",
        "Rare (1e-4 to 1e-3)",
        "Very rare (<1e-4)",
        "Reference zero",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="Unknown",
        ),
        index=target_ra.index,
    )


def calculate_abundance_stratum_metrics(
    dataframe: pd.DataFrame,
    method_name: str,
    prediction_column: str,
    split_name: str,
) -> pd.DataFrame:
    """Evaluate specific OTUs within reference abundance strata."""
    current = dataframe.loc[
        dataframe["is_other"] == 0
    ].copy()

    current["abundance_group"] = assign_abundance_group(
        current["target_reference_ra"]
    )

    abundance_order = [
        "Dominant (>=1e-2)",
        "Common (1e-3 to 1e-2)",
        "Rare (1e-4 to 1e-3)",
        "Very rare (<1e-4)",
        "Reference zero",
    ]

    records: list[dict[str, Any]] = []

    for abundance_group in abundance_order:
        group_data = current.loc[
            current["abundance_group"] == abundance_group
        ]

        if group_data.empty:
            continue

        y_true = group_data[
            "target_reference_ra"
        ].to_numpy(dtype=float)

        y_pred = group_data[
            prediction_column
        ].to_numpy(dtype=float)

        y_true_log = np.log10(y_true + PSEUDOCOUNT)
        y_pred_log = np.log10(y_pred + PSEUDOCOUNT)

        records.append(
            {
                "split": split_name,
                "method": method_name,
                "abundance_group": abundance_group,
                "number_of_rows": len(group_data),
                "MAE_RA": mean_absolute_error(
                    y_true,
                    y_pred,
                ),
                "RMSE_RA": np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                ),
                "MAE_log10_RA": mean_absolute_error(
                    y_true_log,
                    y_pred_log,
                ),
                "RMSE_log10_RA": np.sqrt(
                    mean_squared_error(
                        y_true_log,
                        y_pred_log,
                    )
                ),
                "Spearman_RA": safe_spearman(
                    y_true,
                    y_pred,
                ),
            }
        )

    return pd.DataFrame(records)


def calculate_zero_recovery_metrics(
    dataframe: pd.DataFrame,
    method_name: str,
    prediction_column: str,
    split_name: str,
) -> dict[str, Any]:
    """Evaluate shallow-zero but reference-positive specific OTUs."""
    current = dataframe.loc[
        (dataframe["is_other"] == 0)
        & (dataframe["zero_in_shallow"] == 1)
        & (dataframe["target_reference_ra"] > 0)
    ]

    if current.empty:
        return {
            "split": split_name,
            "method": method_name,
            "number_of_rows": 0,
            "MAE_RA": np.nan,
            "RMSE_RA": np.nan,
            "MAE_log10_RA": np.nan,
            "RMSE_log10_RA": np.nan,
            "positive_prediction_rate": np.nan,
        }

    y_true = current["target_reference_ra"].to_numpy(dtype=float)
    y_pred = current[prediction_column].to_numpy(dtype=float)

    y_true_log = np.log10(y_true + PSEUDOCOUNT)
    y_pred_log = np.log10(y_pred + PSEUDOCOUNT)

    return {
        "split": split_name,
        "method": method_name,
        "number_of_rows": len(current),
        "MAE_RA": mean_absolute_error(y_true, y_pred),
        "RMSE_RA": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "MAE_log10_RA": mean_absolute_error(
            y_true_log,
            y_pred_log,
        ),
        "RMSE_log10_RA": np.sqrt(
            mean_squared_error(
                y_true_log,
                y_pred_log,
            )
        ),
        "positive_prediction_rate": float((y_pred > 0).mean()),
    }


def calculate_sample_metrics(
    dataframe: pd.DataFrame,
    method_name: str,
    prediction_column: str,
    split_name: str,
) -> pd.DataFrame:
    """
    Calculate sample-level composition, RAD and diversity metrics.

    Full composition distances include OTHER.
    RAD and alpha-diversity metrics exclude OTHER.
    """
    records: list[dict[str, Any]] = []

    for (
        sample_id,
        repeat_id,
    ), current in dataframe.groupby(
        ["sample_id", "subsample_repeat"],
        observed=True,
        sort=False,
    ):
        true_all = current[
            "target_reference_ra"
        ].to_numpy(dtype=float)

        pred_all = current[
            prediction_column
        ].to_numpy(dtype=float)

        true_all = np.clip(true_all, 0.0, None)
        pred_all = np.clip(pred_all, 0.0, None)

        if true_all.sum() <= 0 or pred_all.sum() <= 0:
            raise ValueError(
                f"Invalid complete composition for sample "
                f"{sample_id}, repeat {repeat_id}."
            )

        true_all = true_all / true_all.sum()
        pred_all = pred_all / pred_all.sum()

        composition_mae = float(
            np.mean(np.abs(true_all - pred_all))
        )
        composition_rmse = float(
            np.sqrt(
                np.mean((true_all - pred_all) ** 2)
            )
        )
        composition_spearman = safe_spearman(
            true_all,
            pred_all,
        )
        bc_distance = bray_curtis(
            true_all,
            pred_all,
        )
        js_distance = jensen_shannon_distance(
            true_all,
            pred_all,
        )

        specific = current.loc[current["is_other"] == 0]

        true_specific = specific[
            "target_reference_ra"
        ].to_numpy(dtype=float)

        pred_specific = specific[
            prediction_column
        ].to_numpy(dtype=float)

        true_specific = np.clip(
            true_specific,
            0.0,
            None,
        )
        pred_specific = np.clip(
            pred_specific,
            0.0,
            None,
        )

        if true_specific.sum() <= 0:
            raise ValueError(
                f"Specific-OTU reference RA sums to zero for "
                f"sample {sample_id}, repeat {repeat_id}."
            )

        true_specific = (
            true_specific / true_specific.sum()
        )

        if pred_specific.sum() <= 0:
            fallback = specific[
                "shallow_ra"
            ].to_numpy(dtype=float)

            if fallback.sum() <= 0:
                fallback = np.ones_like(
                    fallback,
                    dtype=float,
                )

            pred_specific = fallback

        pred_specific = (
            pred_specific / pred_specific.sum()
        )

        true_rad = np.sort(true_specific)[::-1]
        pred_rad = np.sort(pred_specific)[::-1]

        true_rad_log = np.log10(
            true_rad + PSEUDOCOUNT
        )
        pred_rad_log = np.log10(
            pred_rad + PSEUDOCOUNT
        )

        rad_mae = float(
            np.mean(np.abs(true_rad - pred_rad))
        )
        rad_rmse = float(
            np.sqrt(
                np.mean((true_rad - pred_rad) ** 2)
            )
        )
        rad_mae_log = float(
            np.mean(
                np.abs(true_rad_log - pred_rad_log)
            )
        )
        rad_rmse_log = float(
            np.sqrt(
                np.mean(
                    (true_rad_log - pred_rad_log) ** 2
                )
            )
        )

        top1_error = float(
            abs(true_rad[:1].sum() - pred_rad[:1].sum())
        )
        top5_error = float(
            abs(true_rad[:5].sum() - pred_rad[:5].sum())
        )
        top10_error = float(
            abs(
                true_rad[:10].sum()
                - pred_rad[:10].sum()
            )
        )

        n_components = len(true_rad)
        head_end = max(
            1,
            int(np.ceil(n_components * 0.05)),
        )
        middle_end = max(
            head_end + 1,
            int(np.ceil(n_components * 0.50)),
        )
        middle_end = min(middle_end, n_components)

        head_mae_log = float(
            np.mean(
                np.abs(
                    true_rad_log[:head_end]
                    - pred_rad_log[:head_end]
                )
            )
        )

        middle_mae_log = float(
            np.mean(
                np.abs(
                    true_rad_log[head_end:middle_end]
                    - pred_rad_log[head_end:middle_end]
                )
            )
        )

        if middle_end < n_components:
            tail_mae_log = float(
                np.mean(
                    np.abs(
                        true_rad_log[middle_end:]
                        - pred_rad_log[middle_end:]
                    )
                )
            )
        else:
            tail_mae_log = np.nan

        true_shannon = shannon_index(true_specific)
        pred_shannon = shannon_index(pred_specific)

        true_simpson = simpson_index(true_specific)
        pred_simpson = simpson_index(pred_specific)

        shallow_depth = int(
            current["shallow_depth"].iloc[0]
        )

        one_read_threshold = (
            1.0 / shallow_depth
            if shallow_depth > 0
            else 1.0 / 2000
        )

        true_richness_1read = int(
            (true_specific >= one_read_threshold).sum()
        )
        pred_richness_1read = int(
            (pred_specific >= one_read_threshold).sum()
        )

        records.append(
            {
                "split": split_name,
                "method": method_name,
                "sample_id": sample_id,
                "subsample_repeat": repeat_id,
                "shallow_depth": shallow_depth,
                "composition_MAE_RA": composition_mae,
                "composition_RMSE_RA": composition_rmse,
                "composition_Spearman": composition_spearman,
                "Bray_Curtis": bc_distance,
                "Jensen_Shannon_distance": js_distance,
                "RAD_MAE_RA_specific": rad_mae,
                "RAD_RMSE_RA_specific": rad_rmse,
                "RAD_MAE_log10_RA_specific": rad_mae_log,
                "RAD_RMSE_log10_RA_specific": rad_rmse_log,
                "RAD_head_MAE_log10": head_mae_log,
                "RAD_middle_MAE_log10": middle_mae_log,
                "RAD_tail_MAE_log10": tail_mae_log,
                "Top1_cumulative_error": top1_error,
                "Top5_cumulative_error": top5_error,
                "Top10_cumulative_error": top10_error,
                "true_Shannon_specific": true_shannon,
                "predicted_Shannon_specific": pred_shannon,
                "Shannon_absolute_error": abs(
                    true_shannon - pred_shannon
                ),
                "true_Simpson_specific": true_simpson,
                "predicted_Simpson_specific": pred_simpson,
                "Simpson_absolute_error": abs(
                    true_simpson - pred_simpson
                ),
                "true_richness_1read_specific": (
                    true_richness_1read
                ),
                "predicted_richness_1read_specific": (
                    pred_richness_1read
                ),
                "richness_1read_absolute_error": abs(
                    true_richness_1read
                    - pred_richness_1read
                ),
            }
        )

    return pd.DataFrame(records)


def summarize_sample_metrics(
    sample_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise sample-level metrics by split and method."""
    identifier_columns = {
        "split",
        "method",
        "sample_id",
        "subsample_repeat",
        "shallow_depth",
    }

    metric_columns = [
        column
        for column in sample_metrics.columns
        if column not in identifier_columns
    ]

    records: list[dict[str, Any]] = []

    for (
        split_name,
        method_name,
    ), current in sample_metrics.groupby(
        ["split", "method"],
        observed=True,
        sort=False,
    ):
        for metric in metric_columns:
            values = pd.to_numeric(
                current[metric],
                errors="coerce",
            ).dropna()

            records.append(
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

    return pd.DataFrame(records)


# ============================================================
# 6. Main workflow
# ============================================================

def main() -> None:
    """Generate predictions and evaluation metrics."""
    overall_start_time = time.time()

    print_section("Step 1: Read prepared datasets and trained models")

    for file_path in [
        TRAIN_FILE,
        VALID_FILE,
        TEST_FILE,
        RF_MODEL_FILE,
        XGB_MODEL_FILE,
    ]:
        require_file(file_path)

    train_df = pd.read_pickle(TRAIN_FILE)
    valid_df = pd.read_pickle(VALID_FILE)
    test_df = pd.read_pickle(TEST_FILE)

    for dataframe, dataset_name in [
        (train_df, "Training dataset"),
        (valid_df, "Validation dataset"),
        (test_df, "Test dataset"),
    ]:
        check_required_columns(dataframe, dataset_name)

    rf_model = joblib.load(RF_MODEL_FILE)
    xgb_model = joblib.load(XGB_MODEL_FILE)

    print(f"Training dataset shape:   {train_df.shape}")
    print(f"Validation dataset shape: {valid_df.shape}")
    print(f"Test dataset shape:       {test_df.shape}")
    print("Model files loaded successfully.")

    print_section("Step 2: Generate predictions for four methods")

    datasets_to_predict: dict[str, pd.DataFrame] = {
        "valid": valid_df,
        "test": test_df,
    }

    if SAVE_TRAIN_PREDICTIONS:
        datasets_to_predict["train"] = train_df

    prediction_datasets: dict[str, pd.DataFrame] = {}

    columns_to_save = [
        "sample_id",
        "split",
        "subsample_repeat",
        "otu_id",
        "shallow_depth",
        "shallow_count",
        "shallow_ra",
        "log10_shallow_ra",
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
        "pred_raw_shallow_ra",
        "pred_training_mean_ra",
        "pred_random_forest_ra",
        "pred_xgboost_ra",
    ]

    prediction_files: dict[str, str] = {}

    for split_name, dataframe in datasets_to_predict.items():
        print(f"\nGenerating predictions for: {split_name}")

        prediction_data = create_all_method_predictions(
            dataframe=dataframe,
            rf_model=rf_model,
            xgb_model=xgb_model,
        )

        prediction_datasets[split_name] = prediction_data

        output_file = (
            INTERMEDIATE_DIR
            / f"11_predictions_{split_name}.pkl.gz"
        )
        preview_file = (
            INTERMEDIATE_DIR
            / f"11_predictions_{split_name}_preview.csv"
        )

        prediction_data[columns_to_save].to_pickle(
            output_file,
            compression="gzip",
        )

        prediction_data[
            columns_to_save
        ].head(5000).to_csv(
            preview_file,
            index=False,
            encoding="utf-8-sig",
        )

        prediction_files[split_name] = str(output_file)

        print(f"Saved: {output_file}")

    print_section("Step 3: Evaluate all four methods")

    row_metric_records: list[dict[str, Any]] = []
    sample_metric_parts: list[pd.DataFrame] = []
    abundance_metric_parts: list[pd.DataFrame] = []
    zero_recovery_records: list[dict[str, Any]] = []

    # Final independent evaluation focuses on test.
    # Valid is retained as an exploratory result.
    for split_name in ["valid", "test"]:
        current_data = prediction_datasets[split_name]

        for method_name, prediction_column in METHOD_COLUMNS.items():
            for scope_name in [
                "all_components",
                "specific_otus",
            ]:
                row_metric_records.append(
                    calculate_row_metrics(
                        dataframe=current_data,
                        method_name=method_name,
                        prediction_column=prediction_column,
                        split_name=split_name,
                        scope_name=scope_name,
                    )
                )

            sample_metric_parts.append(
                calculate_sample_metrics(
                    dataframe=current_data,
                    method_name=method_name,
                    prediction_column=prediction_column,
                    split_name=split_name,
                )
            )

            abundance_metric_parts.append(
                calculate_abundance_stratum_metrics(
                    dataframe=current_data,
                    method_name=method_name,
                    prediction_column=prediction_column,
                    split_name=split_name,
                )
            )

            zero_recovery_records.append(
                calculate_zero_recovery_metrics(
                    dataframe=current_data,
                    method_name=method_name,
                    prediction_column=prediction_column,
                    split_name=split_name,
                )
            )

            print(
                f"Completed: {split_name} - {method_name}"
            )

    print_section("Step 4: Save evaluation tables")

    row_metrics = pd.DataFrame(row_metric_records)
    row_metrics.to_csv(
        ROW_METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    sample_metrics = pd.concat(
        sample_metric_parts,
        ignore_index=True,
    )
    sample_metrics.to_csv(
        SAMPLE_METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    sample_summary = summarize_sample_metrics(
        sample_metrics
    )
    sample_summary.to_csv(
        SAMPLE_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    abundance_metrics = pd.concat(
        abundance_metric_parts,
        ignore_index=True,
    )
    abundance_metrics.to_csv(
        ABUNDANCE_METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    zero_recovery_metrics = pd.DataFrame(
        zero_recovery_records
    )
    zero_recovery_metrics.to_csv(
        ZERO_RECOVERY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_section("Step 5: Save prediction configuration")

    total_runtime_minutes = (
        time.time() - overall_start_time
    ) / 60

    run_config: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "prepared_directory": str(PREPARED_DIR),
        "model_directory": str(MODEL_DIR),
        "intermediate_directory": str(INTERMEDIATE_DIR),
        "train_file": str(TRAIN_FILE),
        "valid_file": str(VALID_FILE),
        "test_file": str(TEST_FILE),
        "random_forest_model_file": str(RF_MODEL_FILE),
        "xgboost_model_file": str(XGB_MODEL_FILE),
        "prediction_files": prediction_files,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "prediction_methods": list(METHOD_COLUMNS.keys()),
        "pseudo_count": PSEUDOCOUNT,
        "save_train_predictions": SAVE_TRAIN_PREDICTIONS,
        "total_runtime_minutes": total_runtime_minutes,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "joblib_version": joblib.__version__,
        "note": (
            "Final model performance conclusions should use the "
            "independent test split. Validation results are retained "
            "for exploratory checks only because validation data were "
            "used during model selection."
        ),
    }

    save_json(run_config, CONFIG_FILE)

    print_section("Prediction and evaluation completed successfully")

    created_files = [
        ROW_METRICS_FILE,
        SAMPLE_METRICS_FILE,
        SAMPLE_SUMMARY_FILE,
        ABUNDANCE_METRICS_FILE,
        ZERO_RECOVERY_FILE,
        CONFIG_FILE,
    ]

    for split_name in datasets_to_predict:
        created_files.extend(
            [
                INTERMEDIATE_DIR
                / f"11_predictions_{split_name}.pkl.gz",
                INTERMEDIATE_DIR
                / f"11_predictions_{split_name}_preview.csv",
            ]
        )

    print("Created files:")
    for output_file in created_files:
        print(
            f"  - {output_file.relative_to(PROJECT_ROOT)}"
        )

    print(
        f"\nTotal runtime: {total_runtime_minutes:.2f} minutes"
    )


if __name__ == "__main__":
    main()