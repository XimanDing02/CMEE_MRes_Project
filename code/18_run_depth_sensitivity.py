#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Author: Ximan Ding
# Script: 18_run_depth_sensitivity.py
# Description:
#    Run sequencing-depth sensitivity analyses for shallow-to-high-depth
#    16S relative-abundance recovery.
#
#    For each shallow sequencing depth (500, 1,000, 2,000 and 5,000 reads),
#    this script:
#    1. Reuses the fixed Stage-1 biological-sample split;
#    2. Regenerates shallow multinomial sequencing simulations;
#    3. Trains Random Forest and XGBoost models independently;
#    4. Uses the validation set for XGBoost early stopping;
#    5. Refits final models using the combined training and validation data;
#    6. Evaluates Raw shallow RA, Training-mean RA, Random Forest and XGBoost
#       on the independent test set;
#    7. Calculates composition, rank-abundance, alpha-diversity, richness,
#       abundance-stratum and shallow-zero recovery metrics;
#    8. Exports per-depth results and combined cross-depth summaries;
#    9. Supports resuming completed depths.
#
# Inputs:
#    results/prepared/02_sample_split.csv
#    results/prepared/04_reference_ra_matrix.pkl.gz
#    results/prepared/05_model_otu_train_statistics.csv
#
# Outputs:
#    results/depth_sensitivity/depth_500/
#    results/depth_sensitivity/depth_1000/
#    results/depth_sensitivity/depth_2000/
#    results/depth_sensitivity/depth_5000/
#    results/depth_sensitivity/01_depth_row_metrics.csv
#    results/depth_sensitivity/02_depth_sample_repeat_metrics.csv
#    results/depth_sensitivity/03_depth_sample_level_metrics.csv
#    results/depth_sensitivity/04_depth_performance_summary.csv
#    results/depth_sensitivity/05_depth_abundance_stratum_metrics.csv
#    results/depth_sensitivity/06_depth_zero_recovery_metrics.csv
#    results/depth_sensitivity/07_depth_model_selection_summary.csv
#    results/depth_sensitivity/08_depth_improvement_vs_raw.csv
#    results/depth_sensitivity/09_depth_sensitivity_config.json
#
# Notes:
#    - The 2,000-read sensitivity run is independently resimulated and refitted.
#      It is therefore not expected to reproduce the main 2,000-read analysis
#      exactly.
#    - High-depth relative abundance is treated as a reference observation,
#      not as error-free biological truth or absolute abundance.


from __future__ import annotations

from pathlib import Path
import gc
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# =============================================================================
# 1. Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
PREPARED_DIR = RESULTS_DIR / "prepared"
OUTPUT_DIR = RESULTS_DIR / "depth_sensitivity"

REFERENCE_MATRIX_FILE = PREPARED_DIR / "04_reference_ra_matrix.pkl.gz"
SAMPLE_SPLIT_FILE = PREPARED_DIR / "02_sample_split.csv"
OTU_STATISTICS_FILE = PREPARED_DIR / "05_model_otu_train_statistics.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. Analysis settings
# =============================================================================

DEPTHS = [500, 1000, 2000, 5000]
N_SUBSAMPLE_REPEATS = 5

RESUME_COMPLETED_DEPTHS = True
SAVE_SIMULATED_DATASETS = False
SAVE_TEST_PREDICTIONS = True

RANDOM_STATE = 42
PSEUDOCOUNT = 1e-8
FIXED_RICHNESS_THRESHOLD = 1e-4
N_BOOTSTRAP = 2000


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


RF_PARAMS = {
    "n_estimators": 180,
    "max_depth": 18,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": 0.8,
    "bootstrap": True,
    "max_samples": 0.70,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": 0,
}


XGB_INITIAL_PARAMS = {
    "n_estimators": 1600,
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.80,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.0,
    "reg_lambda": 2.0,
    "gamma": 0.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "early_stopping_rounds": 60,
}


METHOD_COLUMNS = {
    "Raw shallow RA": "pred_raw_shallow_ra",
    "Training-mean RA": "pred_training_mean_ra",
    "Random Forest": "pred_random_forest_ra",
    "XGBoost": "pred_xgboost_ra",
}


SENSITIVITY_METRICS = {
    "Bray_Curtis": "lower",
    "Jensen_Shannon_distance": "lower",
    "RAD_RMSE_log10_RA_specific": "lower",
    "RAD_tail_MAE_log10": "lower",
    "Shannon_absolute_error": "lower",
    "Simpson_absolute_error": "lower",
    "richness_1read_absolute_error": "lower",
    "richness_fixed_absolute_error": "lower",
}


# =============================================================================
# 3. General utility functions
# =============================================================================

def print_step(title: str) -> None:
    """Print a clearly separated workflow step."""
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def require_file(path: Path) -> None:
    """Raise a clear error if an expected input file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            "Required file not found:\n"
            f"{path}\n\n"
            "Run the relevant Stage-1 scripts before this analysis."
        )


def save_pickle(dataframe: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as a gzip-compressed pickle file."""
    dataframe.to_pickle(path, compression="gzip")
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"Saved: {path.name} ({size_mb:.2f} MB)")


def safe_spearman(y_true, y_pred) -> float:
    """Calculate Spearman correlation while handling constant vectors."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if (
        len(y_true) < 2
        or np.all(y_true == y_true[0])
        or np.all(y_pred == y_pred[0])
    ):
        return np.nan

    value, _ = spearmanr(y_true, y_pred)
    return float(value)


def shannon_index(values) -> float:
    """Calculate the Shannon diversity index."""
    values = np.asarray(values, dtype=float)
    values = values[values > 0]

    if len(values) == 0:
        return 0.0

    values = values / values.sum()
    return float(-np.sum(values * np.log(values)))


def simpson_index(values) -> float:
    """Calculate the Gini-Simpson diversity index."""
    values = np.asarray(values, dtype=float)

    if values.sum() <= 0:
        return 0.0

    values = values / values.sum()
    return float(1.0 - np.sum(values ** 2))


def bray_curtis(y_true, y_pred) -> float:
    """
    Calculate Bray-Curtis distance for two compositions that each sum to one.

    For closed compositions, Bray-Curtis simplifies to half the L1 distance.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(0.5 * np.sum(np.abs(y_true - y_pred)))


def js_distance(y_true, y_pred) -> float:
    """Calculate Jensen-Shannon distance using base-2 logarithms."""
    y_true = np.clip(np.asarray(y_true, dtype=float), PSEUDOCOUNT, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), PSEUDOCOUNT, None)

    y_true = y_true / y_true.sum()
    y_pred = y_pred / y_pred.sum()

    return float(jensenshannon(y_true, y_pred, base=2))


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    dataframe_name: str,
) -> None:
    """Check that a DataFrame contains all required columns."""
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(
            f"{dataframe_name} is missing required columns: {missing}"
        )


def validate_composition_sums(
    dataframe: pd.DataFrame,
    column: str,
    tolerance: float = 1e-6,
) -> None:
    """Confirm that each sample-repeat composition sums to approximately one."""
    sums = (
        dataframe
        .groupby(
            ["sample_id", "subsample_repeat"],
            observed=True,
        )[column]
        .sum()
    )

    if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=tolerance):
        raise ValueError(
            f"Composition validation failed for column '{column}'. "
            f"Observed range: {sums.min():.8f} to {sums.max():.8f}"
        )


# =============================================================================
# 4. Simulate supervised data at one sequencing depth
# =============================================================================

def create_depth_dataset(
    sample_ids,
    split_name: str,
    reference_matrix: pd.DataFrame,
    otu_statistics: pd.DataFrame,
    depth: int,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate OTU-level supervised modelling rows at a selected shallow depth.

    Each biological sample is independently simulated `repeats` times from
    its fixed high-depth reference relative-abundance composition using a
    multinomial distribution.
    """
    rng = np.random.default_rng(seed)

    otu_ids = reference_matrix.columns.astype(str).to_numpy()
    n_otus = len(otu_ids)

    stats = (
        otu_statistics
        .set_index("otu_id")
        .reindex(otu_ids)
    )

    stat_columns = [
        "otu_mean_ra_train",
        "otu_prevalence_train",
        "otu_std_ra_train",
        "otu_max_ra_train",
    ]

    if stats[stat_columns].isna().any().any():
        missing_otus = stats.index[
            stats[stat_columns].isna().any(axis=1)
        ].tolist()

        raise ValueError(
            "Some OTUs are missing training-set statistics. "
            f"Example missing OTUs: {missing_otus[:10]}"
        )

    mean_ra = stats["otu_mean_ra_train"].to_numpy(dtype=float)
    prevalence = stats["otu_prevalence_train"].to_numpy(dtype=float)
    std_ra = stats["otu_std_ra_train"].to_numpy(dtype=float)
    max_ra = stats["otu_max_ra_train"].to_numpy(dtype=float)
    is_other = (otu_ids == "OTHER").astype(np.int8)

    parts = []
    sample_ids = list(sample_ids)

    for sample_number, sample_id in enumerate(sample_ids, start=1):
        reference_values = reference_matrix.loc[sample_id].to_numpy(dtype=float)
        reference_values = np.clip(reference_values, 0.0, None)

        reference_sum = reference_values.sum()
        if reference_sum <= 0:
            raise ValueError(
                f"Reference composition has a non-positive sum for sample {sample_id}."
            )

        reference_values = reference_values / reference_sum
        target_log = np.log10(reference_values + PSEUDOCOUNT)

        for repeat_id in range(1, repeats + 1):
            counts = rng.multinomial(depth, reference_values)
            shallow_ra = counts / depth
            zero_flag = (counts == 0).astype(np.int8)
            richness = int((counts > 0).sum())

            ranks = np.full(
                n_otus,
                richness + 1,
                dtype=np.int32,
            )

            positive_indices = np.flatnonzero(counts > 0)

            if len(positive_indices) > 0:
                ordered_indices = positive_indices[
                    np.argsort(
                        -shallow_ra[positive_indices],
                        kind="mergesort",
                    )
                ]
                ranks[ordered_indices] = np.arange(
                    1,
                    len(ordered_indices) + 1,
                )

            rank_norm = ranks / (richness + 1)

            parts.append(
                pd.DataFrame(
                    {
                        "sample_id": sample_id,
                        "split": split_name,
                        "subsample_repeat": repeat_id,
                        "otu_id": otu_ids,
                        "shallow_depth": depth,
                        "shallow_count": counts.astype(np.int32),
                        "shallow_ra": shallow_ra.astype(np.float32),
                        "log10_shallow_ra": np.log10(
                            shallow_ra + PSEUDOCOUNT
                        ).astype(np.float32),
                        "log1p_shallow_count": np.log1p(
                            counts
                        ).astype(np.float32),
                        "zero_in_shallow": zero_flag,
                        "shallow_rank": ranks,
                        "shallow_rank_norm": rank_norm.astype(np.float32),
                        "shallow_richness": np.full(
                            n_otus,
                            richness,
                            dtype=np.int16,
                        ),
                        "otu_mean_ra_train": mean_ra.astype(np.float32),
                        "otu_prevalence_train": prevalence.astype(np.float32),
                        "otu_std_ra_train": std_ra.astype(np.float32),
                        "otu_max_ra_train": max_ra.astype(np.float32),
                        "target_reference_ra": reference_values.astype(np.float32),
                        "target_log10_reference_ra": target_log.astype(np.float32),
                        "is_other": is_other,
                    }
                )
            )

        if sample_number % 30 == 0 or sample_number == len(sample_ids):
            print(
                f"Depth {depth:,} | {split_name:<5} | "
                f"{sample_number}/{len(sample_ids)} samples generated"
            )

    result = pd.concat(parts, ignore_index=True)

    expected_rows = len(sample_ids) * repeats * n_otus
    if len(result) != expected_rows:
        raise RuntimeError(
            f"Unexpected number of rows for {split_name} at depth {depth}. "
            f"Expected {expected_rows:,}, observed {len(result):,}."
        )

    validate_composition_sums(result, "shallow_ra")
    validate_composition_sums(result, "target_reference_ra")

    return result


# =============================================================================
# 5. Convert model outputs to closed relative-abundance compositions
# =============================================================================

def inverse_log_prediction(values) -> np.ndarray:
    """Back-transform log10 predictions to the non-negative RA scale."""
    values = np.asarray(values, dtype=float)
    values = np.clip(values, np.log10(PSEUDOCOUNT), 0.0)
    values = np.power(10.0, values) - PSEUDOCOUNT
    return np.clip(values, 0.0, None)


def normalize_by_sample(
    dataframe: pd.DataFrame,
    values,
) -> np.ndarray:
    """
    Normalize non-negative predictions within each sample-repeat composition.

    If an entire predicted composition sums to zero, Raw shallow RA is used as
    a fallback for that sample-repeat.
    """
    temporary = dataframe[
        ["sample_id", "subsample_repeat", "shallow_ra"]
    ].copy()

    prediction_values = np.asarray(values, dtype=float)
    prediction_values = np.nan_to_num(
        prediction_values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    prediction_values = np.clip(prediction_values, 0.0, None)

    temporary["_prediction"] = prediction_values

    group_columns = ["sample_id", "subsample_repeat"]
    group_sum = temporary.groupby(
        group_columns,
        observed=True,
    )["_prediction"].transform("sum")

    zero_sum_mask = group_sum <= 0

    if zero_sum_mask.any():
        temporary.loc[zero_sum_mask, "_prediction"] = temporary.loc[
            zero_sum_mask,
            "shallow_ra",
        ]

        group_sum = temporary.groupby(
            group_columns,
            observed=True,
        )["_prediction"].transform("sum")

    if (group_sum <= 0).any():
        raise ValueError(
            "At least one sample-repeat composition still has a non-positive "
            "prediction sum after fallback normalization."
        )

    return (
        temporary["_prediction"] / group_sum
    ).to_numpy(dtype=float)


def create_predictions(
    dataframe: pd.DataFrame,
    rf_model: RandomForestRegressor,
    xgb_model: XGBRegressor,
) -> pd.DataFrame:
    """Generate predictions from both baselines and both ML models."""
    result = dataframe.copy()

    result["pred_raw_shallow_ra"] = normalize_by_sample(
        result,
        result["shallow_ra"].to_numpy(dtype=float),
    )

    result["pred_training_mean_ra"] = normalize_by_sample(
        result,
        result["otu_mean_ra_train"].to_numpy(dtype=float),
    )

    features = result[FEATURE_COLUMNS].astype(np.float32)

    result["pred_random_forest_ra"] = normalize_by_sample(
        result,
        inverse_log_prediction(rf_model.predict(features)),
    )

    result["pred_xgboost_ra"] = normalize_by_sample(
        result,
        inverse_log_prediction(xgb_model.predict(features)),
    )

    for prediction_column in METHOD_COLUMNS.values():
        validate_composition_sums(result, prediction_column)

    return result


# =============================================================================
# 6. Evaluation functions
# =============================================================================

def calculate_row_metrics(
    dataframe: pd.DataFrame,
    depth: int,
    method: str,
    prediction_column: str,
    scope: str,
) -> dict:
    """Calculate OTU-row-level metrics at one evaluation scope."""
    if scope == "all_components":
        current = dataframe
    elif scope == "specific_otus":
        current = dataframe.loc[dataframe["is_other"] == 0]
    else:
        raise ValueError(f"Unknown evaluation scope: {scope}")

    y_true = current["target_reference_ra"].to_numpy(dtype=float)
    y_pred = current[prediction_column].to_numpy(dtype=float)

    y_true_log = np.log10(y_true + PSEUDOCOUNT)
    y_pred_log = np.log10(y_pred + PSEUDOCOUNT)

    return {
        "depth": depth,
        "method": method,
        "evaluation_scope": scope,
        "number_of_rows": len(current),
        "MAE_RA": mean_absolute_error(y_true, y_pred),
        "RMSE_RA": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2_RA": r2_score(y_true, y_pred),
        "MAE_log10_RA": mean_absolute_error(y_true_log, y_pred_log),
        "RMSE_log10_RA": np.sqrt(
            mean_squared_error(y_true_log, y_pred_log)
        ),
        "Spearman_RA": safe_spearman(y_true, y_pred),
    }


def calculate_sample_repeat_metrics(
    dataframe: pd.DataFrame,
    depth: int,
    method: str,
    prediction_column: str,
) -> pd.DataFrame:
    """
    Calculate one set of ecological metrics per test sample and repeat.

    OTHER is included for whole-composition distances and excluded for RAD,
    alpha-diversity and richness calculations.
    """
    records = []

    grouped = dataframe.groupby(
        ["sample_id", "subsample_repeat"],
        observed=True,
        sort=False,
    )

    for (sample_id, repeat_id), current in grouped:
        true_all = current["target_reference_ra"].to_numpy(dtype=float)
        pred_all = current[prediction_column].to_numpy(dtype=float)

        true_all = np.clip(true_all, 0.0, None)
        pred_all = np.clip(pred_all, 0.0, None)

        true_all = true_all / true_all.sum()
        pred_all = pred_all / pred_all.sum()

        specific = current.loc[current["is_other"] == 0]

        true_specific = specific[
            "target_reference_ra"
        ].to_numpy(dtype=float)

        pred_specific = specific[
            prediction_column
        ].to_numpy(dtype=float)

        true_specific = np.clip(true_specific, 0.0, None)
        pred_specific = np.clip(pred_specific, 0.0, None)

        if true_specific.sum() <= 0:
            raise ValueError(
                f"Specific-OTU reference composition sums to zero for "
                f"sample {sample_id}, repeat {repeat_id}."
            )

        true_specific = true_specific / true_specific.sum()

        if pred_specific.sum() <= 0:
            pred_specific = specific["shallow_ra"].to_numpy(dtype=float)

        if pred_specific.sum() <= 0:
            pred_specific = np.ones_like(pred_specific, dtype=float)

        pred_specific = pred_specific / pred_specific.sum()

        true_rad = np.sort(true_specific)[::-1]
        pred_rad = np.sort(pred_specific)[::-1]

        true_rad_log = np.log10(true_rad + PSEUDOCOUNT)
        pred_rad_log = np.log10(pred_rad + PSEUDOCOUNT)

        n_components = len(true_rad)
        head_end = max(1, int(np.ceil(n_components * 0.05)))
        middle_end = min(
            n_components,
            max(
                head_end + 1,
                int(np.ceil(n_components * 0.50)),
            ),
        )

        tail_error = (
            float(
                np.mean(
                    np.abs(
                        true_rad_log[middle_end:]
                        - pred_rad_log[middle_end:]
                    )
                )
            )
            if middle_end < n_components
            else np.nan
        )

        true_shannon = shannon_index(true_specific)
        pred_shannon = shannon_index(pred_specific)

        true_simpson = simpson_index(true_specific)
        pred_simpson = simpson_index(pred_specific)

        one_read_threshold = 1.0 / depth

        true_richness_1read = int(
            (true_specific >= one_read_threshold).sum()
        )
        pred_richness_1read = int(
            (pred_specific >= one_read_threshold).sum()
        )

        true_richness_fixed = int(
            (true_specific >= FIXED_RICHNESS_THRESHOLD).sum()
        )
        pred_richness_fixed = int(
            (pred_specific >= FIXED_RICHNESS_THRESHOLD).sum()
        )

        records.append(
            {
                "depth": depth,
                "method": method,
                "sample_id": sample_id,
                "subsample_repeat": repeat_id,
                "composition_RMSE_RA": float(
                    np.sqrt(np.mean((true_all - pred_all) ** 2))
                ),
                "composition_Spearman": safe_spearman(
                    true_all,
                    pred_all,
                ),
                "Bray_Curtis": bray_curtis(true_all, pred_all),
                "Jensen_Shannon_distance": js_distance(
                    true_all,
                    pred_all,
                ),
                "RAD_MAE_log10_RA_specific": float(
                    np.mean(
                        np.abs(true_rad_log - pred_rad_log)
                    )
                ),
                "RAD_RMSE_log10_RA_specific": float(
                    np.sqrt(
                        np.mean(
                            (true_rad_log - pred_rad_log) ** 2
                        )
                    )
                ),
                "RAD_head_MAE_log10": float(
                    np.mean(
                        np.abs(
                            true_rad_log[:head_end]
                            - pred_rad_log[:head_end]
                        )
                    )
                ),
                "RAD_middle_MAE_log10": float(
                    np.mean(
                        np.abs(
                            true_rad_log[head_end:middle_end]
                            - pred_rad_log[head_end:middle_end]
                        )
                    )
                ),
                "RAD_tail_MAE_log10": tail_error,
                "Shannon_absolute_error": abs(
                    true_shannon - pred_shannon
                ),
                "Simpson_absolute_error": abs(
                    true_simpson - pred_simpson
                ),
                "true_richness_1read": true_richness_1read,
                "predicted_richness_1read": pred_richness_1read,
                "richness_1read_absolute_error": abs(
                    true_richness_1read - pred_richness_1read
                ),
                "true_richness_fixed": true_richness_fixed,
                "predicted_richness_fixed": pred_richness_fixed,
                "richness_fixed_absolute_error": abs(
                    true_richness_fixed - pred_richness_fixed
                ),
            }
        )

    return pd.DataFrame(records)


def assign_abundance_group(target_ra: pd.Series) -> pd.Series:
    """Assign reference-abundance strata to specific OTU rows."""
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


def calculate_abundance_metrics(
    dataframe: pd.DataFrame,
    depth: int,
    method: str,
    prediction_column: str,
) -> pd.DataFrame:
    """Calculate row-level errors within reference-abundance strata."""
    current = dataframe.loc[
        dataframe["is_other"] == 0
    ].copy()

    current["abundance_group"] = assign_abundance_group(
        current["target_reference_ra"]
    )

    records = []

    for group_name, group_data in current.groupby(
        "abundance_group",
        observed=True,
        sort=False,
    ):
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
                "depth": depth,
                "method": method,
                "abundance_group": group_name,
                "number_of_rows": len(group_data),
                "MAE_RA": mean_absolute_error(y_true, y_pred),
                "RMSE_RA": np.sqrt(
                    mean_squared_error(y_true, y_pred)
                ),
                "MAE_log10_RA": mean_absolute_error(
                    y_true_log,
                    y_pred_log,
                ),
                "RMSE_log10_RA": np.sqrt(
                    mean_squared_error(y_true_log, y_pred_log)
                ),
                "Spearman_RA": safe_spearman(y_true, y_pred),
            }
        )

    return pd.DataFrame(records)


def calculate_zero_metrics(
    dataframe: pd.DataFrame,
    depth: int,
    method: str,
    prediction_column: str,
) -> dict:
    """
    Evaluate shallow sampling zeros with positive reference abundance.

    OTHER is excluded because it is a pooled residual component rather than a
    specific biological OTU.
    """
    current = dataframe.loc[
        (dataframe["is_other"] == 0)
        & (dataframe["zero_in_shallow"] == 1)
        & (dataframe["target_reference_ra"] > 0)
    ]

    if current.empty:
        return {
            "depth": depth,
            "method": method,
            "number_of_rows": 0,
            "MAE_RA": np.nan,
            "RMSE_RA": np.nan,
            "MAE_log10_RA": np.nan,
            "RMSE_log10_RA": np.nan,
            "positive_prediction_rate": np.nan,
        }

    y_true = current[
        "target_reference_ra"
    ].to_numpy(dtype=float)

    y_pred = current[
        prediction_column
    ].to_numpy(dtype=float)

    y_true_log = np.log10(y_true + PSEUDOCOUNT)
    y_pred_log = np.log10(y_pred + PSEUDOCOUNT)

    return {
        "depth": depth,
        "method": method,
        "number_of_rows": len(current),
        "MAE_RA": mean_absolute_error(y_true, y_pred),
        "RMSE_RA": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE_log10_RA": mean_absolute_error(y_true_log, y_pred_log),
        "RMSE_log10_RA": np.sqrt(
            mean_squared_error(y_true_log, y_pred_log)
        ),
        "positive_prediction_rate": float((y_pred > 0).mean()),
    }


def summarize_sample_metrics(
    sample_repeat_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Average the five technical repeats within biological samples, then
    summarize variation across independent biological samples.
    """
    excluded_columns = {
        "depth",
        "method",
        "sample_id",
        "subsample_repeat",
    }

    metric_columns = [
        column
        for column in sample_repeat_metrics.columns
        if column not in excluded_columns
    ]

    sample_level = (
        sample_repeat_metrics
        .groupby(
            ["depth", "method", "sample_id"],
            as_index=False,
            observed=True,
        )[metric_columns]
        .mean()
    )

    records = []

    for (depth, method), current in sample_level.groupby(
        ["depth", "method"],
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
                    "depth": depth,
                    "method": method,
                    "metric": metric,
                    "n_samples": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                }
            )

    return sample_level, pd.DataFrame(records)


def bootstrap_mean_ci(
    values,
    seed: int,
) -> tuple[float, float, float]:
    """Calculate a non-parametric bootstrap CI for the sample mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    n_values = len(values)
    bootstrap_means = np.empty(N_BOOTSTRAP, dtype=float)

    for bootstrap_index in range(N_BOOTSTRAP):
        bootstrap_means[bootstrap_index] = values[
            rng.integers(0, n_values, size=n_values)
        ].mean()

    low, high = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    return (
        float(values.mean()),
        float(low),
        float(high),
    )


def calculate_improvement_vs_raw(
    sample_level_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate paired biological-sample improvements over Raw shallow RA.

    Positive oriented improvement always means the ML model performed better.
    """
    records = []
    seed_counter = 0

    for depth in sorted(
        sample_level_metrics["depth"].unique()
    ):
        depth_data = sample_level_metrics.loc[
            sample_level_metrics["depth"] == depth
        ]

        for metric, direction in SENSITIVITY_METRICS.items():
            if metric not in depth_data.columns:
                continue

            wide = depth_data.pivot(
                index="sample_id",
                columns="method",
                values=metric,
            )

            if "Raw shallow RA" not in wide.columns:
                continue

            for model_name in ["Random Forest", "XGBoost"]:
                if model_name not in wide.columns:
                    continue

                paired = wide[
                    ["Raw shallow RA", model_name]
                ].dropna()

                raw_values = paired["Raw shallow RA"]
                model_values = paired[model_name]

                if direction == "lower":
                    improvement = raw_values - model_values
                else:
                    improvement = model_values - raw_values

                (
                    mean_improvement,
                    ci_low,
                    ci_high,
                ) = bootstrap_mean_ci(
                    improvement.to_numpy(),
                    RANDOM_STATE + seed_counter,
                )
                seed_counter += 1

                raw_mean = float(raw_values.mean())
                model_mean = float(model_values.mean())

                if abs(raw_mean) > 1e-15:
                    if direction == "lower":
                        relative_improvement = (
                            (raw_mean - model_mean)
                            / abs(raw_mean)
                            * 100
                        )
                    else:
                        relative_improvement = (
                            (model_mean - raw_mean)
                            / abs(raw_mean)
                            * 100
                        )
                else:
                    relative_improvement = np.nan

                records.append(
                    {
                        "depth": depth,
                        "metric": metric,
                        "direction": direction,
                        "model": model_name,
                        "n_samples": len(paired),
                        "raw_mean": raw_mean,
                        "model_mean": model_mean,
                        "mean_oriented_improvement": mean_improvement,
                        "improvement_ci_low": ci_low,
                        "improvement_ci_high": ci_high,
                        "relative_improvement_percent": relative_improvement,
                        "sample_win_rate": float(
                            (improvement > 0).mean()
                        ),
                    }
                )

    return pd.DataFrame(records)


# =============================================================================
# 7. Read and validate fixed Stage-1 inputs
# =============================================================================

def read_inputs():
    """Read the fixed reference matrix, sample split and OTU statistics."""
    print_step(
        "Step 1: Read the reference matrix, sample split and OTU statistics"
    )

    for input_file in [
        REFERENCE_MATRIX_FILE,
        SAMPLE_SPLIT_FILE,
        OTU_STATISTICS_FILE,
    ]:
        require_file(input_file)

    reference_matrix = pd.read_pickle(
        REFERENCE_MATRIX_FILE
    )

    if "sample_id" in reference_matrix.columns:
        reference_matrix = reference_matrix.set_index("sample_id")

    reference_matrix.index = (
        reference_matrix.index.astype(str)
    )
    reference_matrix.columns = (
        reference_matrix.columns.astype(str)
    )

    if reference_matrix.index.has_duplicates:
        duplicated = (
            reference_matrix.index[
                reference_matrix.index.duplicated()
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            "Reference matrix contains duplicated sample IDs. "
            f"Examples: {duplicated[:10]}"
        )

    sample_split = pd.read_csv(
        SAMPLE_SPLIT_FILE,
        dtype={
            "sample_id": str,
            "split": str,
        },
    )

    validate_columns(
        sample_split,
        ["sample_id", "split"],
        "Sample split table",
    )

    otu_statistics = pd.read_csv(
        OTU_STATISTICS_FILE,
        dtype={"otu_id": str},
    )

    validate_columns(
        otu_statistics,
        [
            "otu_id",
            "otu_mean_ra_train",
            "otu_prevalence_train",
            "otu_std_ra_train",
            "otu_max_ra_train",
        ],
        "OTU statistics table",
    )

    sample_split["split"] = (
        sample_split["split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    allowed_splits = {"train", "valid", "test"}
    observed_splits = set(sample_split["split"].unique())

    unexpected_splits = observed_splits - allowed_splits
    if unexpected_splits:
        raise ValueError(
            f"Unexpected split labels: {sorted(unexpected_splits)}"
        )

    if sample_split["sample_id"].duplicated().any():
        duplicated = (
            sample_split.loc[
                sample_split["sample_id"].duplicated(),
                "sample_id",
            ]
            .unique()
            .tolist()
        )
        raise ValueError(
            "Sample split table contains duplicated sample IDs. "
            f"Examples: {duplicated[:10]}"
        )

    train_ids = sample_split.loc[
        sample_split["split"] == "train",
        "sample_id",
    ].tolist()

    valid_ids = sample_split.loc[
        sample_split["split"] == "valid",
        "sample_id",
    ].tolist()

    test_ids = sample_split.loc[
        sample_split["split"] == "test",
        "sample_id",
    ].tolist()

    missing_samples = (
        set(train_ids + valid_ids + test_ids)
        - set(reference_matrix.index)
    )

    if missing_samples:
        raise ValueError(
            "Some split samples are missing from the reference matrix. "
            f"Examples: {sorted(missing_samples)[:10]}"
        )

    missing_statistics = (
        set(reference_matrix.columns)
        - set(otu_statistics["otu_id"])
    )

    if missing_statistics:
        raise ValueError(
            "Some reference-matrix components are missing OTU statistics. "
            f"Examples: {sorted(missing_statistics)[:10]}"
        )

    row_sums = reference_matrix.sum(axis=1).to_numpy(dtype=float)

    if not np.all(np.isfinite(row_sums)):
        raise ValueError(
            "Reference matrix contains non-finite row sums."
        )

    if np.any(row_sums <= 0):
        raise ValueError(
            "Reference matrix contains rows with non-positive sums."
        )

    reference_matrix = reference_matrix.div(
        reference_matrix.sum(axis=1),
        axis=0,
    )

    print(f"Project root:      {PROJECT_ROOT}")
    print(f"Reference matrix:  {reference_matrix.shape}")
    print(f"Training samples:  {len(train_ids)}")
    print(f"Validation samples:{len(valid_ids)}")
    print(f"Test samples:      {len(test_ids)}")
    print(f"Components:        {reference_matrix.shape[1]}")
    print(f"Depths:            {DEPTHS}")
    print(f"Repeats per sample:{N_SUBSAMPLE_REPEATS}")

    return (
        reference_matrix,
        sample_split,
        otu_statistics,
        train_ids,
        valid_ids,
        test_ids,
    )


# =============================================================================
# 8. Run one sequencing depth
# =============================================================================

def build_depth_paths(depth: int) -> tuple[Path, Path, dict[str, Path], Path]:
    """Construct output paths for one sequencing-depth experiment."""
    depth_directory = OUTPUT_DIR / f"depth_{depth}"
    model_directory = depth_directory / "models"

    depth_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    output_files = {
        "row": depth_directory / "01_test_row_metrics.csv",
        "repeat": depth_directory / "02_test_sample_repeat_metrics.csv",
        "sample": depth_directory / "03_test_sample_level_metrics.csv",
        "summary": depth_directory / "04_test_performance_summary.csv",
        "abundance": depth_directory / "05_abundance_stratum_metrics.csv",
        "zero": depth_directory / "06_zero_recovery_metrics.csv",
        "selection": depth_directory / "07_model_selection_summary.csv",
    }

    completion_file = depth_directory / "depth_complete.json"

    return (
        depth_directory,
        model_directory,
        output_files,
        completion_file,
    )


def depth_is_complete(
    output_files: dict[str, Path],
    completion_file: Path,
) -> bool:
    """Check whether one depth can be safely loaded instead of recomputed."""
    return (
        RESUME_COMPLETED_DEPTHS
        and completion_file.exists()
        and all(path.exists() for path in output_files.values())
    )


def load_completed_depth(
    output_files: dict[str, Path],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Load previously completed per-depth result tables."""
    return (
        pd.read_csv(output_files["row"]),
        pd.read_csv(output_files["repeat"]),
        pd.read_csv(output_files["sample"]),
        pd.read_csv(output_files["summary"]),
        pd.read_csv(output_files["abundance"]),
        pd.read_csv(output_files["zero"]),
        pd.read_csv(output_files["selection"]),
    )


def run_one_depth(
    depth: int,
    reference_matrix: pd.DataFrame,
    otu_statistics: pd.DataFrame,
    train_ids: list[str],
    valid_ids: list[str],
    test_ids: list[str],
):
    """Run simulation, model fitting, prediction and evaluation at one depth."""
    (
        depth_directory,
        model_directory,
        output_files,
        completion_file,
    ) = build_depth_paths(depth)

    if depth_is_complete(output_files, completion_file):
        print(
            f"Depth {depth:,} reads is already complete. "
            "Loading existing result tables."
        )
        return load_completed_depth(output_files)

    depth_start = time.time()

    print("\nGenerating shallow supervised datasets...")

    train_data = create_depth_dataset(
        train_ids,
        "train",
        reference_matrix,
        otu_statistics,
        depth,
        N_SUBSAMPLE_REPEATS,
        RANDOM_STATE + depth * 10 + 1,
    )

    valid_data = create_depth_dataset(
        valid_ids,
        "valid",
        reference_matrix,
        otu_statistics,
        depth,
        N_SUBSAMPLE_REPEATS,
        RANDOM_STATE + depth * 10 + 2,
    )

    test_data = create_depth_dataset(
        test_ids,
        "test",
        reference_matrix,
        otu_statistics,
        depth,
        N_SUBSAMPLE_REPEATS,
        RANDOM_STATE + depth * 10 + 3,
    )

    print(
        f"Training rows:   {len(train_data):,}\n"
        f"Validation rows: {len(valid_data):,}\n"
        f"Test rows:       {len(test_data):,}"
    )

    if SAVE_SIMULATED_DATASETS:
        save_pickle(
            train_data,
            depth_directory / "simulated_train.pkl.gz",
        )
        save_pickle(
            valid_data,
            depth_directory / "simulated_valid.pkl.gz",
        )
        save_pickle(
            test_data,
            depth_directory / "simulated_test.pkl.gz",
        )

    X_train = train_data[FEATURE_COLUMNS].astype(np.float32)
    y_train = train_data[TARGET_COLUMN].astype(np.float32)

    X_valid = valid_data[FEATURE_COLUMNS].astype(np.float32)
    y_valid = valid_data[TARGET_COLUMN].astype(np.float32)

    print("\nTraining the first Random Forest model...")

    rf_start = time.time()

    rf_first = RandomForestRegressor(**RF_PARAMS)
    rf_first.fit(X_train, y_train)

    rf_valid_prediction = rf_first.predict(X_valid)

    rf_valid_rmse = float(
        np.sqrt(
            mean_squared_error(
                y_valid,
                rf_valid_prediction,
            )
        )
    )

    rf_minutes = (time.time() - rf_start) / 60

    print(
        f"Random Forest validation log10-RMSE: "
        f"{rf_valid_rmse:.6f}"
    )
    print(
        f"Random Forest first-fit time: "
        f"{rf_minutes:.2f} minutes"
    )

    print("\nTraining XGBoost with validation early stopping...")

    xgb_start = time.time()

    xgb_first = XGBRegressor(**XGB_INITIAL_PARAMS)

    xgb_first.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_valid, y_valid),
        ],
        verbose=100,
    )

    if (
        hasattr(xgb_first, "best_iteration")
        and xgb_first.best_iteration is not None
    ):
        best_iteration = int(xgb_first.best_iteration) + 1
    else:
        best_iteration = int(
            XGB_INITIAL_PARAMS["n_estimators"]
        )

    xgb_valid_prediction = xgb_first.predict(X_valid)

    xgb_valid_rmse = float(
        np.sqrt(
            mean_squared_error(
                y_valid,
                xgb_valid_prediction,
            )
        )
    )

    xgb_minutes = (time.time() - xgb_start) / 60

    print(f"XGBoost best tree count: {best_iteration}")
    print(
        f"XGBoost validation log10-RMSE: "
        f"{xgb_valid_rmse:.6f}"
    )
    print(
        f"XGBoost first-fit time: "
        f"{xgb_minutes:.2f} minutes"
    )

    print("\nRefitting final models using train + validation data...")

    combined_data = pd.concat(
        [train_data, valid_data],
        ignore_index=True,
    )

    X_combined = combined_data[
        FEATURE_COLUMNS
    ].astype(np.float32)

    y_combined = combined_data[
        TARGET_COLUMN
    ].astype(np.float32)

    final_rf = RandomForestRegressor(**RF_PARAMS)
    final_rf.fit(X_combined, y_combined)

    joblib.dump(
        final_rf,
        model_directory / "random_forest_model.joblib",
        compress=3,
    )

    final_xgb_params = XGB_INITIAL_PARAMS.copy()
    final_xgb_params.pop(
        "early_stopping_rounds",
        None,
    )
    final_xgb_params["n_estimators"] = best_iteration

    final_xgb = XGBRegressor(**final_xgb_params)
    final_xgb.fit(
        X_combined,
        y_combined,
        verbose=False,
    )

    joblib.dump(
        final_xgb,
        model_directory / "xgboost_model.joblib",
        compress=3,
    )

    print("\nGenerating independent test-set predictions...")

    test_predictions = create_predictions(
        test_data,
        final_rf,
        final_xgb,
    )

    if SAVE_TEST_PREDICTIONS:
        prediction_columns = [
            "sample_id",
            "split",
            "subsample_repeat",
            "otu_id",
            "shallow_depth",
            "shallow_count",
            "shallow_ra",
            "zero_in_shallow",
            "shallow_rank",
            "shallow_rank_norm",
            "shallow_richness",
            "target_reference_ra",
            "target_log10_reference_ra",
            "is_other",
            "pred_raw_shallow_ra",
            "pred_training_mean_ra",
            "pred_random_forest_ra",
            "pred_xgboost_ra",
        ]

        save_pickle(
            test_predictions[prediction_columns],
            depth_directory / "08_test_predictions.pkl.gz",
        )

    print("\nCalculating test-set evaluation metrics...")

    row_records = []
    sample_parts = []
    abundance_parts = []
    zero_records = []

    for method, prediction_column in METHOD_COLUMNS.items():
        print(f"Evaluating: {method}")

        for scope in [
            "all_components",
            "specific_otus",
        ]:
            row_records.append(
                calculate_row_metrics(
                    test_predictions,
                    depth,
                    method,
                    prediction_column,
                    scope,
                )
            )

        sample_parts.append(
            calculate_sample_repeat_metrics(
                test_predictions,
                depth,
                method,
                prediction_column,
            )
        )

        abundance_parts.append(
            calculate_abundance_metrics(
                test_predictions,
                depth,
                method,
                prediction_column,
            )
        )

        zero_records.append(
            calculate_zero_metrics(
                test_predictions,
                depth,
                method,
                prediction_column,
            )
        )

    row_metrics = pd.DataFrame(row_records)

    sample_repeat_metrics = pd.concat(
        sample_parts,
        ignore_index=True,
    )

    (
        sample_level_metrics,
        performance_summary,
    ) = summarize_sample_metrics(
        sample_repeat_metrics
    )

    abundance_metrics = pd.concat(
        abundance_parts,
        ignore_index=True,
    )

    zero_metrics = pd.DataFrame(zero_records)

    total_depth_minutes = (
        time.time() - depth_start
    ) / 60

    selection_summary = pd.DataFrame(
        [
            {
                "depth": depth,
                "rf_validation_log_rmse": rf_valid_rmse,
                "xgb_validation_log_rmse": xgb_valid_rmse,
                "xgb_best_iteration": best_iteration,
                "rf_first_fit_minutes": rf_minutes,
                "xgb_first_fit_minutes": xgb_minutes,
                "total_depth_minutes": total_depth_minutes,
            }
        ]
    )

    row_metrics.to_csv(
        output_files["row"],
        index=False,
        encoding="utf-8-sig",
    )

    sample_repeat_metrics.to_csv(
        output_files["repeat"],
        index=False,
        encoding="utf-8-sig",
    )

    sample_level_metrics.to_csv(
        output_files["sample"],
        index=False,
        encoding="utf-8-sig",
    )

    performance_summary.to_csv(
        output_files["summary"],
        index=False,
        encoding="utf-8-sig",
    )

    abundance_metrics.to_csv(
        output_files["abundance"],
        index=False,
        encoding="utf-8-sig",
    )

    zero_metrics.to_csv(
        output_files["zero"],
        index=False,
        encoding="utf-8-sig",
    )

    selection_summary.to_csv(
        output_files["selection"],
        index=False,
        encoding="utf-8-sig",
    )

    completion_info = {
        "depth": int(depth),
        "status": "complete",
        "subsample_repeats": int(N_SUBSAMPLE_REPEATS),
        "training_samples": int(len(train_ids)),
        "validation_samples": int(len(valid_ids)),
        "test_samples": int(len(test_ids)),
        "components": int(reference_matrix.shape[1]),
        "xgb_best_iteration": int(best_iteration),
        "total_minutes": float(total_depth_minutes),
    }

    with open(
        completion_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            completion_info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    del train_data
    del valid_data
    del test_data
    del combined_data
    del X_train
    del y_train
    del X_valid
    del y_valid
    del X_combined
    del y_combined
    del rf_first
    del xgb_first
    del final_rf
    del final_xgb
    del test_predictions

    gc.collect()

    print(
        f"\nDepth {depth:,} reads completed in "
        f"{total_depth_minutes:.2f} minutes."
    )

    return (
        row_metrics,
        sample_repeat_metrics,
        sample_level_metrics,
        performance_summary,
        abundance_metrics,
        zero_metrics,
        selection_summary,
    )


# =============================================================================
# 9. Combine results across all sequencing depths
# =============================================================================

def save_combined_results(
    all_row_metrics,
    all_sample_repeat_metrics,
    all_sample_level_metrics,
    all_summaries,
    all_abundance_metrics,
    all_zero_metrics,
    all_model_selection,
) -> None:
    """Combine and export results from all completed sequencing depths."""
    print_step(
        "Step 3: Combine results across all sequencing depths"
    )

    combined_row = pd.concat(
        all_row_metrics,
        ignore_index=True,
    )

    combined_repeat = pd.concat(
        all_sample_repeat_metrics,
        ignore_index=True,
    )

    combined_sample = pd.concat(
        all_sample_level_metrics,
        ignore_index=True,
    )

    combined_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    combined_abundance = pd.concat(
        all_abundance_metrics,
        ignore_index=True,
    )

    combined_zero = pd.concat(
        all_zero_metrics,
        ignore_index=True,
    )

    combined_selection = pd.concat(
        all_model_selection,
        ignore_index=True,
    )

    improvement_vs_raw = calculate_improvement_vs_raw(
        combined_sample
    )

    combined_row.to_csv(
        OUTPUT_DIR / "01_depth_row_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_repeat.to_csv(
        OUTPUT_DIR / "02_depth_sample_repeat_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_sample.to_csv(
        OUTPUT_DIR / "03_depth_sample_level_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_summary.to_csv(
        OUTPUT_DIR / "04_depth_performance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_abundance.to_csv(
        OUTPUT_DIR / "05_depth_abundance_stratum_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_zero.to_csv(
        OUTPUT_DIR / "06_depth_zero_recovery_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_selection.to_csv(
        OUTPUT_DIR / "07_depth_model_selection_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    improvement_vs_raw.to_csv(
        OUTPUT_DIR / "08_depth_improvement_vs_raw.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "project_root": str(PROJECT_ROOT),
        "reference_matrix_file": str(REFERENCE_MATRIX_FILE),
        "sample_split_file": str(SAMPLE_SPLIT_FILE),
        "otu_statistics_file": str(OTU_STATISTICS_FILE),
        "output_directory": str(OUTPUT_DIR),
        "depths": DEPTHS,
        "subsample_repeats": N_SUBSAMPLE_REPEATS,
        "random_state": RANDOM_STATE,
        "pseudo_count": PSEUDOCOUNT,
        "fixed_richness_threshold": FIXED_RICHNESS_THRESHOLD,
        "bootstrap_iterations": N_BOOTSTRAP,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "methods": list(METHOD_COLUMNS.keys()),
        "rf_parameters": RF_PARAMS,
        "xgb_initial_parameters": XGB_INITIAL_PARAMS,
        "resume_completed_depths": RESUME_COMPLETED_DEPTHS,
        "save_simulated_datasets": SAVE_SIMULATED_DATASETS,
        "save_test_predictions": SAVE_TEST_PREDICTIONS,
        "note": (
            "Each sequencing depth is independently resimulated and refitted. "
            "The 2,000-read sensitivity result is not expected to reproduce "
            "the main analysis exactly."
        ),
    }

    with open(
        OUTPUT_DIR / "09_depth_sensitivity_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Combined outputs saved:")
    print(f"  {OUTPUT_DIR / '01_depth_row_metrics.csv'}")
    print(f"  {OUTPUT_DIR / '02_depth_sample_repeat_metrics.csv'}")
    print(f"  {OUTPUT_DIR / '03_depth_sample_level_metrics.csv'}")
    print(f"  {OUTPUT_DIR / '04_depth_performance_summary.csv'}")
    print(f"  {OUTPUT_DIR / '05_depth_abundance_stratum_metrics.csv'}")
    print(f"  {OUTPUT_DIR / '06_depth_zero_recovery_metrics.csv'}")
    print(f"  {OUTPUT_DIR / '07_depth_model_selection_summary.csv'}")
    print(f"  {OUTPUT_DIR / '08_depth_improvement_vs_raw.csv'}")
    print(f"  {OUTPUT_DIR / '09_depth_sensitivity_config.json'}")


# =============================================================================
# 10. Main workflow
# =============================================================================

def main() -> None:
    """Run the complete multi-depth sensitivity analysis."""
    overall_start = time.time()

    (
        reference_matrix,
        _sample_split,
        otu_statistics,
        train_ids,
        valid_ids,
        test_ids,
    ) = read_inputs()

    print_step(
        "Step 2: Simulate, fit and evaluate models at each sequencing depth"
    )

    all_row_metrics = []
    all_sample_repeat_metrics = []
    all_sample_level_metrics = []
    all_summaries = []
    all_abundance_metrics = []
    all_zero_metrics = []
    all_model_selection = []

    for depth_number, depth in enumerate(
        DEPTHS,
        start=1,
    ):
        print("\n" + "#" * 100)
        print(
            f"Depth {depth:,} reads "
            f"({depth_number}/{len(DEPTHS)})"
        )
        print("#" * 100)

        (
            row_metrics,
            sample_repeat_metrics,
            sample_level_metrics,
            performance_summary,
            abundance_metrics,
            zero_metrics,
            model_selection,
        ) = run_one_depth(
            depth=depth,
            reference_matrix=reference_matrix,
            otu_statistics=otu_statistics,
            train_ids=train_ids,
            valid_ids=valid_ids,
            test_ids=test_ids,
        )

        all_row_metrics.append(row_metrics)
        all_sample_repeat_metrics.append(
            sample_repeat_metrics
        )
        all_sample_level_metrics.append(
            sample_level_metrics
        )
        all_summaries.append(performance_summary)
        all_abundance_metrics.append(
            abundance_metrics
        )
        all_zero_metrics.append(zero_metrics)
        all_model_selection.append(model_selection)

    save_combined_results(
        all_row_metrics=all_row_metrics,
        all_sample_repeat_metrics=all_sample_repeat_metrics,
        all_sample_level_metrics=all_sample_level_metrics,
        all_summaries=all_summaries,
        all_abundance_metrics=all_abundance_metrics,
        all_zero_metrics=all_zero_metrics,
        all_model_selection=all_model_selection,
    )

    total_minutes = (time.time() - overall_start) / 60

    print_step("Depth-sensitivity analysis completed")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total runtime:    {total_minutes:.2f} minutes")
    print(
        "Next script:     code/19_plot_depth_sensitivity.py"
    )


if __name__ == "__main__":
    main()
