#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Ximan Ding
# Script: 10_train_models.py
# Description:
#    Train Random Forest and XGBoost models for shallow-to-high-depth
#    16S relative abundance prediction.
#
#    This script:
#    1. Reads prepared training, validation and test datasets;
#    2. Checks required columns and sample-level data leakage;
#    3. Trains a first Random Forest model on the training set;
#    4. Trains a first XGBoost model with validation-set early stopping;
#    5. Combines training and validation data;
#    6. Retrains the final Random Forest and XGBoost models;
#    7. Saves model files, validation metrics, feature importance,
#       and a reproducible training configuration.
#
#    This script does not generate final predictions or test-set metrics.
#    Those tasks are handled by 11_generate_predictions_and_metrics.py.
#
#Arguments:
#    None. Project paths are resolved relative to this script.
#
#Inputs:
#    results/prepared/06_model_train.pkl.gz
#    results/prepared/06_model_valid.pkl.gz
#    results/prepared/06_model_test.pkl.gz
#
#Outputs:
#    results/models/random_forest_model.joblib
#    results/models/xgboost_model.joblib
#    results/models/model_validation_metrics.csv
#    results/models/model_feature_importance.csv
#    results/models/model_training_config.json
#
#Date: July 2026


# ============================================================
# 0. Import packages
# ============================================================

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
import sklearn
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


# ============================================================
# 1. Project paths
# ============================================================

# This file is expected to be inside:
# CMEE_MRES_PROJECT/code/10_train_models.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREPARED_DIR = PROJECT_ROOT / "results" / "prepared"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = PREPARED_DIR / "06_model_train.pkl.gz"
VALID_FILE = PREPARED_DIR / "06_model_valid.pkl.gz"
TEST_FILE = PREPARED_DIR / "06_model_test.pkl.gz"

RF_MODEL_FILE = MODEL_DIR / "random_forest_model.joblib"
XGB_MODEL_FILE = MODEL_DIR / "xgboost_model.joblib"
VALIDATION_METRICS_FILE = MODEL_DIR / "model_validation_metrics.csv"
FEATURE_IMPORTANCE_FILE = MODEL_DIR / "model_feature_importance.csv"
CONFIG_FILE = MODEL_DIR / "model_training_config.json"


# ============================================================
# 2. Global settings
# ============================================================

RANDOM_STATE = 42
PSEUDOCOUNT = 1e-8

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

RF_PARAMS: dict[str, Any] = {
    "n_estimators": 220,
    "max_depth": 18,
    "min_samples_split": 4,
    "min_samples_leaf": 2,
    "max_features": 0.8,
    "bootstrap": True,
    "max_samples": 0.70,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": 1,
}

XGB_INITIAL_PARAMS: dict[str, Any] = {
    "n_estimators": 2000,
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
    "early_stopping_rounds": 80,
}


# ============================================================
# 3. Utility functions
# ============================================================

def print_section(title: str) -> None:
    """Print a clear section heading."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def require_file(file_path: Path) -> None:
    """Raise an informative error if an input file does not exist."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required input file was not found:\n{file_path}\n\n"
            "Run Stage 1 scripts 01-08 before training the models."
        )


def check_required_columns(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Check that a prepared dataset contains all modelling columns."""
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


def check_missing_values(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Check model features and target for missing or infinite values."""
    columns_to_check = FEATURE_COLUMNS + [TARGET_COLUMN]

    numeric_data = dataframe[columns_to_check].apply(
        pd.to_numeric,
        errors="coerce",
    )

    missing_counts = numeric_data.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        raise ValueError(
            f"{dataset_name} contains missing or non-numeric values:\n"
            f"{missing_counts.to_string()}"
        )

    values = numeric_data.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            f"{dataset_name} contains infinite values in model columns."
        )


def check_sample_leakage(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Ensure biological sample IDs do not overlap across subsets."""
    train_samples = set(train_df["sample_id"].astype(str).unique())
    valid_samples = set(valid_df["sample_id"].astype(str).unique())
    test_samples = set(test_df["sample_id"].astype(str).unique())

    overlaps = {
        "train-valid": train_samples & valid_samples,
        "train-test": train_samples & test_samples,
        "valid-test": valid_samples & test_samples,
    }

    non_empty_overlaps = {
        name: values
        for name, values in overlaps.items()
        if values
    }

    if non_empty_overlaps:
        overlap_summary = {
            name: sorted(values)[:10]
            for name, values in non_empty_overlaps.items()
        }
        raise ValueError(
            "Sample-level data leakage was detected:\n"
            f"{overlap_summary}"
        )

    print("Sample leakage check passed.")
    print(f"Training biological samples:   {len(train_samples)}")
    print(f"Validation biological samples: {len(valid_samples)}")
    print(f"Test biological samples:       {len(test_samples)}")


def calculate_rmse(
    true_values: np.ndarray | pd.Series,
    predicted_values: np.ndarray,
) -> float:
    """Calculate root mean squared error."""
    return float(
        np.sqrt(
            mean_squared_error(
                true_values,
                predicted_values,
            )
        )
    )


def save_json(data: dict[str, Any], output_file: Path) -> None:
    """Save a dictionary as readable UTF-8 JSON."""
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# 4. Main workflow
# ============================================================

def main() -> None:
    """Run model training and save all training-stage outputs."""

    overall_start_time = time.time()

    # --------------------------------------------------------
    # Step 1: Read prepared data
    # --------------------------------------------------------
    print_section("Step 1: Read prepared modelling datasets")

    for file_path in [TRAIN_FILE, VALID_FILE, TEST_FILE]:
        require_file(file_path)

    train_df = pd.read_pickle(TRAIN_FILE)
    valid_df = pd.read_pickle(VALID_FILE)
    test_df = pd.read_pickle(TEST_FILE)

    print(f"Training dataset shape:   {train_df.shape}")
    print(f"Validation dataset shape: {valid_df.shape}")
    print(f"Test dataset shape:       {test_df.shape}")

    # --------------------------------------------------------
    # Step 2: Validate datasets
    # --------------------------------------------------------
    print_section("Step 2: Check columns, values and sample leakage")

    for dataframe, dataset_name in [
        (train_df, "Training dataset"),
        (valid_df, "Validation dataset"),
        (test_df, "Test dataset"),
    ]:
        check_required_columns(dataframe, dataset_name)
        check_missing_values(dataframe, dataset_name)

    check_sample_leakage(train_df, valid_df, test_df)

    # --------------------------------------------------------
    # Step 3: Prepare feature matrices and targets
    # --------------------------------------------------------
    print_section("Step 3: Prepare model features and targets")

    X_train = train_df[FEATURE_COLUMNS].astype(np.float32)
    y_train = train_df[TARGET_COLUMN].astype(np.float32)

    X_valid = valid_df[FEATURE_COLUMNS].astype(np.float32)
    y_valid = valid_df[TARGET_COLUMN].astype(np.float32)

    print("Model features:")
    for feature_name in FEATURE_COLUMNS:
        print(f"  - {feature_name}")

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_valid shape: {X_valid.shape}")

    # --------------------------------------------------------
    # Step 4: First Random Forest fit
    # --------------------------------------------------------
    print_section(
        "Step 4: Train the first Random Forest model "
        "using the training set"
    )

    rf_start_time = time.time()

    rf_train_model = RandomForestRegressor(**RF_PARAMS)
    rf_train_model.fit(X_train, y_train)

    rf_valid_prediction = rf_train_model.predict(X_valid)
    rf_validation_rmse = calculate_rmse(
        y_valid,
        rf_valid_prediction,
    )

    rf_first_fit_minutes = (time.time() - rf_start_time) / 60

    print(
        f"Random Forest validation log10-RMSE: "
        f"{rf_validation_rmse:.6f}"
    )
    print(
        f"Random Forest first-fit time: "
        f"{rf_first_fit_minutes:.2f} minutes"
    )

    # --------------------------------------------------------
    # Step 5: First XGBoost fit with early stopping
    # --------------------------------------------------------
    print_section(
        "Step 5: Train the first XGBoost model "
        "with validation early stopping"
    )

    xgb_start_time = time.time()

    xgb_train_model = XGBRegressor(**XGB_INITIAL_PARAMS)
    xgb_train_model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_valid, y_valid),
        ],
        verbose=50,
    )

    best_iteration_attribute = getattr(
        xgb_train_model,
        "best_iteration",
        None,
    )

    if best_iteration_attribute is None:
        best_iteration = int(
            XGB_INITIAL_PARAMS["n_estimators"]
        )
        print(
            "Warning: XGBoost did not report best_iteration; "
            "the configured n_estimators value will be used."
        )
    else:
        # best_iteration is zero-indexed in XGBoost.
        best_iteration = int(best_iteration_attribute) + 1

    xgb_valid_prediction = xgb_train_model.predict(X_valid)
    xgb_validation_rmse = calculate_rmse(
        y_valid,
        xgb_valid_prediction,
    )

    xgb_first_fit_minutes = (time.time() - xgb_start_time) / 60

    print(f"XGBoost selected tree count: {best_iteration}")
    print(
        f"XGBoost validation log10-RMSE: "
        f"{xgb_validation_rmse:.6f}"
    )
    print(
        f"XGBoost first-fit time: "
        f"{xgb_first_fit_minutes:.2f} minutes"
    )

    # --------------------------------------------------------
    # Step 6: Combine training and validation data
    # --------------------------------------------------------
    print_section(
        "Step 6: Combine training and validation data "
        "for final model fitting"
    )

    combined_df = pd.concat(
        [train_df, valid_df],
        ignore_index=True,
    )

    X_combined = combined_df[FEATURE_COLUMNS].astype(np.float32)
    y_combined = combined_df[TARGET_COLUMN].astype(np.float32)

    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"X_combined shape:       {X_combined.shape}")

    # --------------------------------------------------------
    # Step 7: Fit and save the final Random Forest
    # --------------------------------------------------------
    print_section("Step 7: Fit and save the final Random Forest")

    final_rf_start_time = time.time()

    final_rf_model = RandomForestRegressor(**RF_PARAMS)
    final_rf_model.fit(X_combined, y_combined)

    joblib.dump(
        final_rf_model,
        RF_MODEL_FILE,
        compress=3,
    )

    final_rf_fit_minutes = (
        time.time() - final_rf_start_time
    ) / 60

    print(f"Saved Random Forest model:\n{RF_MODEL_FILE}")
    print(
        f"Final Random Forest fit time: "
        f"{final_rf_fit_minutes:.2f} minutes"
    )

    # --------------------------------------------------------
    # Step 8: Fit and save the final XGBoost
    # --------------------------------------------------------
    print_section("Step 8: Fit and save the final XGBoost model")

    final_xgb_params = XGB_INITIAL_PARAMS.copy()
    final_xgb_params.pop("early_stopping_rounds", None)
    final_xgb_params["n_estimators"] = best_iteration

    final_xgb_start_time = time.time()

    final_xgb_model = XGBRegressor(**final_xgb_params)
    final_xgb_model.fit(
        X_combined,
        y_combined,
        verbose=False,
    )

    joblib.dump(
        final_xgb_model,
        XGB_MODEL_FILE,
        compress=3,
    )

    final_xgb_fit_minutes = (
        time.time() - final_xgb_start_time
    ) / 60

    print(f"Saved XGBoost model:\n{XGB_MODEL_FILE}")
    print(
        f"Final XGBoost fit time: "
        f"{final_xgb_fit_minutes:.2f} minutes"
    )

    # --------------------------------------------------------
    # Step 9: Save validation metrics
    # --------------------------------------------------------
    print_section("Step 9: Save validation metrics")

    validation_metrics = pd.DataFrame(
        [
            {
                "model": "Random Forest",
                "validation_metric": "RMSE",
                "validation_scale": "log10_reference_ra",
                "validation_value": rf_validation_rmse,
                "selected_n_estimators": RF_PARAMS[
                    "n_estimators"
                ],
                "first_fit_minutes": rf_first_fit_minutes,
            },
            {
                "model": "XGBoost",
                "validation_metric": "RMSE",
                "validation_scale": "log10_reference_ra",
                "validation_value": xgb_validation_rmse,
                "selected_n_estimators": best_iteration,
                "first_fit_minutes": xgb_first_fit_minutes,
            },
        ]
    )

    validation_metrics.to_csv(
        VALIDATION_METRICS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved validation metrics:\n"
        f"{VALIDATION_METRICS_FILE}"
    )

    # --------------------------------------------------------
    # Step 10: Save built-in feature importance
    # --------------------------------------------------------
    print_section("Step 10: Save built-in feature importance")

    rf_importance = pd.DataFrame(
        {
            "model": "Random Forest",
            "feature": FEATURE_COLUMNS,
            "importance": final_rf_model.feature_importances_,
        }
    )

    xgb_importance = pd.DataFrame(
        {
            "model": "XGBoost",
            "feature": FEATURE_COLUMNS,
            "importance": final_xgb_model.feature_importances_,
        }
    )

    feature_importance = pd.concat(
        [rf_importance, xgb_importance],
        ignore_index=True,
    )

    feature_importance["rank_within_model"] = (
        feature_importance.groupby("model")["importance"]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    feature_importance = feature_importance.sort_values(
        ["model", "rank_within_model", "feature"]
    )

    feature_importance.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved feature importance:\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )

    # --------------------------------------------------------
    # Step 11: Save reproducible training configuration
    # --------------------------------------------------------
    print_section("Step 11: Save model training configuration")

    total_runtime_minutes = (
        time.time() - overall_start_time
    ) / 60

    run_config: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "prepared_directory": str(PREPARED_DIR),
        "model_directory": str(MODEL_DIR),
        "train_file": str(TRAIN_FILE),
        "valid_file": str(VALID_FILE),
        "test_file": str(TEST_FILE),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "pseudo_count": PSEUDOCOUNT,
        "random_state": RANDOM_STATE,
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "training_samples": int(
            train_df["sample_id"].nunique()
        ),
        "validation_samples": int(
            valid_df["sample_id"].nunique()
        ),
        "test_samples": int(
            test_df["sample_id"].nunique()
        ),
        "rf_parameters": RF_PARAMS,
        "xgb_initial_parameters": XGB_INITIAL_PARAMS,
        "xgb_best_iteration": best_iteration,
        "xgb_final_parameters": final_xgb_params,
        "rf_validation_log10_rmse": rf_validation_rmse,
        "xgb_validation_log10_rmse": xgb_validation_rmse,
        "rf_first_fit_minutes": rf_first_fit_minutes,
        "xgb_first_fit_minutes": xgb_first_fit_minutes,
        "final_rf_fit_minutes": final_rf_fit_minutes,
        "final_xgb_fit_minutes": final_xgb_fit_minutes,
        "total_runtime_minutes": total_runtime_minutes,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "joblib_version": joblib.__version__,
        "note": (
            "Validation data were used for model selection and "
            "XGBoost early stopping. Final models were fitted using "
            "the combined training and validation data. The test set "
            "was not used for fitting or model selection."
        ),
    }

    save_json(run_config, CONFIG_FILE)

    print(f"Saved training configuration:\n{CONFIG_FILE}")

    # --------------------------------------------------------
    # Completion summary
    # --------------------------------------------------------
    print_section("Model training completed successfully")

    print("Created files:")
    for output_file in [
        RF_MODEL_FILE,
        XGB_MODEL_FILE,
        VALIDATION_METRICS_FILE,
        FEATURE_IMPORTANCE_FILE,
        CONFIG_FILE,
    ]:
        print(f"  - {output_file.relative_to(PROJECT_ROOT)}")

    print(
        f"\nTotal runtime: {total_runtime_minutes:.2f} minutes"
    )
    print(
        "\nNext script:\n"
        "  code/11_generate_predictions_and_metrics.py"
    )


if __name__ == "__main__":
    main()