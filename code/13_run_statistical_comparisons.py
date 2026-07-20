# -*- coding: utf-8 -*-
# Author: Ximan Ding
# Script: 13_run_statistical_comparisons.py
# Description:
#     Run formal paired statistical comparisons among four prediction methods
#     using the independent test-set biological samples prepared by
#     12_prepare_statistical_tables.py.
#
#     This script:
#     1. Reads the biological-sample-level test metrics.
#     2. Runs Friedman overall tests for each metric.
#     3. Runs all paired Wilcoxon signed-rank tests.
#     4. Applies Holm correction within each metric.
#     5. Calculates paired rank-biserial effect sizes.
#     6. Calculates bootstrap 95% confidence intervals for paired improvements.
#     7. Extracts the key model comparisons used in the report.
#     8. Creates a long-format paired-improvement table for later plotting.
#
# Inputs:
#     results/intermediate/12_test_sample_level_metrics.csv
#     results/final/statistics/12_statistical_data_config.json
#
# Outputs:
#     results/final/statistics/13_friedman_overall_tests.csv
#     results/final/statistics/13_pairwise_wilcoxon_holm.csv
#     results/final/statistics/13_key_model_comparisons.csv
#     results/final/statistics/13_primary_key_model_comparisons.csv
#     results/intermediate/13_paired_sample_improvements.csv
#     results/final/statistics/13_statistical_analysis_config.json
#
# Date: July 2026

# =============================================================================
# 0. Import packages
# =============================================================================

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import (
    friedmanchisquare,
    rankdata,
    wilcoxon,
)


# =============================================================================
# 1. Project paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INTERMEDIATE_DIR = PROJECT_ROOT / "results" / "intermediate"
STATISTICS_DIR = PROJECT_ROOT / "results" / "statistics"

INTERMEDIATE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

STATISTICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SAMPLE_LEVEL_FILE = (
    INTERMEDIATE_DIR
    / "12_test_sample_level_metrics.csv"
)

STATISTICAL_DATA_CONFIG_FILE = (
    STATISTICS_DIR
    / "12_statistical_data_config.json"
)

FRIEDMAN_OUTPUT = (
    STATISTICS_DIR
    / "13_friedman_overall_tests.csv"
)

PAIRWISE_OUTPUT = (
    STATISTICS_DIR
    / "13_pairwise_wilcoxon_holm.csv"
)

KEY_COMPARISONS_OUTPUT = (
    STATISTICS_DIR
    / "13_key_model_comparisons.csv"
)

PRIMARY_KEY_COMPARISONS_OUTPUT = (
    STATISTICS_DIR
    / "13_primary_key_model_comparisons.csv"
)

PAIRED_IMPROVEMENTS_OUTPUT = (
    INTERMEDIATE_DIR
    / "13_paired_sample_improvements.csv"
)

CONFIG_OUTPUT = (
    STATISTICS_DIR
    / "13_statistical_analysis_config.json"
)


# =============================================================================
# 2. Statistical settings
# =============================================================================

TARGET_SPLIT = "test"

METHOD_ORDER = [
    "Raw shallow RA",
    "Training-mean RA",
    "Random Forest",
    "XGBoost",
]

METRIC_DIRECTIONS = {
    "composition_RMSE_RA": "lower",
    "composition_Spearman": "higher",
    "Bray_Curtis": "lower",
    "Jensen_Shannon_distance": "lower",
    "RAD_RMSE_log10_RA_specific": "lower",
    "RAD_head_MAE_log10": "lower",
    "RAD_middle_MAE_log10": "lower",
    "RAD_tail_MAE_log10": "lower",
    "Top1_cumulative_error": "lower",
    "Top5_cumulative_error": "lower",
    "Top10_cumulative_error": "lower",
    "Shannon_absolute_error": "lower",
    "Simpson_absolute_error": "lower",
    "richness_1read_absolute_error": "lower",
}

PRIMARY_METRICS = [
    "Bray_Curtis",
    "Jensen_Shannon_distance",
    "RAD_RMSE_log10_RA_specific",
    "RAD_tail_MAE_log10",
    "Shannon_absolute_error",
    "richness_1read_absolute_error",
]

KEY_COMPARISON_NAMES = [
    "Random Forest vs Raw shallow RA",
    "XGBoost vs Raw shallow RA",
    "XGBoost vs Random Forest",
]

N_BOOTSTRAP = 5000
RANDOM_STATE = 42
ALPHA = 0.05


# =============================================================================
# 3. Helper functions
# =============================================================================

def print_step(step_number: int, title: str) -> None:
    """Print a consistent progress heading."""

    print("\n" + "=" * 92)
    print(f"Step {step_number}: {title}")
    print("=" * 92)


def require_file(file_path: Path) -> None:
    """Raise a clear error when a required file is missing."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}\n"
            "Run code/12_prepare_statistical_tables.py first."
        )


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Check that all required columns exist."""

    missing = sorted(
        set(required_columns) - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}"
        )


def available_metric_directions(
    dataframe: pd.DataFrame,
) -> dict[str, str]:
    """Keep configured metrics that are present in the table."""

    available = {
        metric: direction
        for metric, direction in METRIC_DIRECTIONS.items()
        if metric in dataframe.columns
    }

    if not available:
        raise ValueError(
            "None of the configured metrics were found in "
            "12_test_sample_level_metrics.csv."
        )

    missing = [
        metric
        for metric in METRIC_DIRECTIONS
        if metric not in dataframe.columns
    ]

    if missing:
        print(
            "Warning: the following configured metrics were not found "
            "and will be skipped:"
        )
        for metric in missing:
            print(f"  - {metric}")

    return available


def holm_adjust(
    p_values: np.ndarray,
) -> np.ndarray:
    """
    Apply the Holm-Bonferroni multiple-testing correction.

    Missing p-values remain missing.
    """

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
        dtype=float,
    )

    valid_mask = np.isfinite(p_values)

    if valid_mask.sum() == 0:
        return adjusted

    valid_p = p_values[valid_mask]
    m = len(valid_p)

    order = np.argsort(valid_p)
    sorted_p = valid_p[order]

    sorted_adjusted = np.empty(
        m,
        dtype=float,
    )

    running_max = 0.0

    for position, p_value in enumerate(sorted_p):
        multiplier = m - position

        current_adjusted = min(
            p_value * multiplier,
            1.0,
        )

        running_max = max(
            running_max,
            current_adjusted,
        )

        sorted_adjusted[position] = running_max

    reverse_order = np.empty_like(order)
    reverse_order[order] = np.arange(m)

    valid_adjusted = (
        sorted_adjusted[reverse_order]
    )

    adjusted[valid_mask] = valid_adjusted

    return adjusted


def format_p_value(
    p_value: float,
) -> str:
    """Format a p-value for report tables."""

    if pd.isna(p_value):
        return "NA"

    if p_value < 0.001:
        return "<0.001"

    return f"{p_value:.3f}"


def significance_label(
    p_value: float,
) -> str:
    """Return a conventional significance label."""

    if pd.isna(p_value):
        return "NA"

    if p_value < 0.001:
        return "***"

    if p_value < 0.01:
        return "**"

    if p_value < 0.05:
        return "*"

    return "ns"


def rank_biserial_from_oriented_difference(
    oriented_difference: np.ndarray,
) -> float:
    """
    Calculate paired rank-biserial correlation.

    Positive values mean method_B is better than method_A.
    """

    differences = np.asarray(
        oriented_difference,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
        & (differences != 0)
    ]

    if len(differences) == 0:
        return 0.0

    ranks = rankdata(
        np.abs(differences),
        method="average",
    )

    positive_rank_sum = ranks[
        differences > 0
    ].sum()

    negative_rank_sum = ranks[
        differences < 0
    ].sum()

    total_rank_sum = ranks.sum()

    if total_rank_sum == 0:
        return 0.0

    return float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / total_rank_sum
    )


def bootstrap_paired_improvement(
    oriented_difference: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> dict[str, float]:
    """
    Bootstrap paired improvement values at the biological-sample level.

    Positive values mean method_B is better than method_A.
    """

    differences = np.asarray(
        oriented_difference,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) == 0:
        return {
            "mean_improvement": np.nan,
            "mean_ci_low": np.nan,
            "mean_ci_high": np.nan,
            "median_improvement": np.nan,
            "median_ci_low": np.nan,
            "median_ci_high": np.nan,
        }

    rng = np.random.default_rng(
        random_state
    )

    n_samples = len(differences)

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    bootstrap_medians = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for bootstrap_index in range(
        n_bootstrap
    ):
        sampled_indices = rng.integers(
            0,
            n_samples,
            size=n_samples,
        )

        sampled_difference = (
            differences[sampled_indices]
        )

        bootstrap_means[
            bootstrap_index
        ] = np.mean(sampled_difference)

        bootstrap_medians[
            bootstrap_index
        ] = np.median(sampled_difference)

    mean_ci_low, mean_ci_high = np.percentile(
        bootstrap_means,
        [2.5, 97.5],
    )

    median_ci_low, median_ci_high = np.percentile(
        bootstrap_medians,
        [2.5, 97.5],
    )

    return {
        "mean_improvement": float(
            np.mean(differences)
        ),
        "mean_ci_low": float(mean_ci_low),
        "mean_ci_high": float(mean_ci_high),
        "median_improvement": float(
            np.median(differences)
        ),
        "median_ci_low": float(
            median_ci_low
        ),
        "median_ci_high": float(
            median_ci_high
        ),
    }


def calculate_relative_improvement_percent(
    method_a_values: np.ndarray,
    method_b_values: np.ndarray,
    direction: str,
) -> float:
    """
    Calculate the mean relative improvement of method_B over method_A.

    Positive values mean method_B is better.
    """

    mean_a = float(
        np.nanmean(method_a_values)
    )

    mean_b = float(
        np.nanmean(method_b_values)
    )

    denominator = abs(mean_a)

    if denominator <= 1e-15:
        return np.nan

    if direction == "lower":
        return float(
            (mean_a - mean_b)
            / denominator
            * 100.0
        )

    if direction == "higher":
        return float(
            (mean_b - mean_a)
            / denominator
            * 100.0
        )

    raise ValueError(
        f"Unknown metric direction: {direction}"
    )


def make_json_serialisable(
    value: Any,
) -> Any:
    """Convert values into JSON-compatible objects."""

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): make_json_serialisable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_serialisable(item)
            for item in value
        ]

    return value


# =============================================================================
# 4. Main statistical analysis
# =============================================================================

def main() -> None:
    """Run the formal paired statistical comparisons."""

    overall_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Step 1: Read the biological-sample-level table
    # -------------------------------------------------------------------------

    print_step(
        1,
        "Read biological-sample-level test metrics",
    )

    require_file(SAMPLE_LEVEL_FILE)

    sample_level_metrics = pd.read_csv(
        SAMPLE_LEVEL_FILE
    )

    print(
        f"Sample-level table shape: "
        f"{sample_level_metrics.shape}"
    )

    # -------------------------------------------------------------------------
    # Step 2: Validate methods, samples and metrics
    # -------------------------------------------------------------------------

    print_step(
        2,
        "Check methods, samples and metric completeness",
    )

    require_columns(
        sample_level_metrics,
        [
            "split",
            "method",
            "sample_id",
        ],
        "12_test_sample_level_metrics.csv",
    )

    test_data = sample_level_metrics.loc[
        sample_level_metrics["split"] == TARGET_SPLIT
    ].copy()

    if test_data.empty:
        raise ValueError(
            "No test rows were found in "
            "12_test_sample_level_metrics.csv."
        )

    observed_methods = set(
        test_data["method"]
        .dropna()
        .astype(str)
        .unique()
    )

    missing_methods = [
        method
        for method in METHOD_ORDER
        if method not in observed_methods
    ]

    if missing_methods:
        raise ValueError(
            f"Missing prediction methods: {missing_methods}"
        )

    metric_directions = (
        available_metric_directions(test_data)
    )

    n_samples = int(
        test_data["sample_id"].nunique()
    )

    expected_rows = (
        n_samples * len(METHOD_ORDER)
    )

    if len(test_data) != expected_rows:
        raise ValueError(
            "The test table is not a complete sample × method grid. "
            f"Expected {expected_rows} rows but found "
            f"{len(test_data)}."
        )

    print(
        f"Biological samples: {n_samples}"
    )
    print(
        f"Methods:            {len(METHOD_ORDER)}"
    )
    print(
        f"Metrics:            {len(metric_directions)}"
    )
    print(
        f"Expected rows:       {expected_rows}"
    )

    # -------------------------------------------------------------------------
    # Step 3: Run Friedman overall tests
    # -------------------------------------------------------------------------

    print_step(
        3,
        "Run Friedman overall tests",
    )

    friedman_records: list[dict[str, Any]] = []

    for metric, direction in (
        metric_directions.items()
    ):
        metric_wide = (
            test_data
            .pivot(
                index="sample_id",
                columns="method",
                values=metric,
            )
            .reindex(columns=METHOD_ORDER)
            .dropna()
        )

        if len(metric_wide) < 3:
            statistic = np.nan
            p_value = np.nan
        else:
            statistic, p_value = (
                friedmanchisquare(
                    *[
                        metric_wide[
                            method
                        ].to_numpy(dtype=float)
                        for method in METHOD_ORDER
                    ]
                )
            )

        friedman_records.append(
            {
                "metric": metric,
                "direction": direction,
                "n_paired_samples": int(
                    len(metric_wide)
                ),
                "friedman_chi_square": (
                    float(statistic)
                    if np.isfinite(statistic)
                    else np.nan
                ),
                "friedman_p_raw": (
                    float(p_value)
                    if np.isfinite(p_value)
                    else np.nan
                ),
            }
        )

    friedman_results = pd.DataFrame(
        friedman_records
    )

    friedman_results["friedman_p_holm"] = (
        holm_adjust(
            friedman_results[
                "friedman_p_raw"
            ].to_numpy(dtype=float)
        )
    )

    friedman_results[
        "friedman_p_holm_formatted"
    ] = friedman_results[
        "friedman_p_holm"
    ].apply(format_p_value)

    friedman_results[
        "significance"
    ] = friedman_results[
        "friedman_p_holm"
    ].apply(significance_label)

    friedman_results.to_csv(
        FRIEDMAN_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved:")
    print(FRIEDMAN_OUTPUT)

    print(
        friedman_results[
            [
                "metric",
                "n_paired_samples",
                "friedman_chi_square",
                "friedman_p_holm_formatted",
                "significance",
            ]
        ].to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Step 4: Run pairwise Wilcoxon signed-rank tests
    # -------------------------------------------------------------------------

    print_step(
        4,
        "Run pairwise Wilcoxon signed-rank tests",
    )

    method_pairs = list(
        combinations(
            METHOD_ORDER,
            2,
        )
    )

    pairwise_records: list[dict[str, Any]] = []
    bootstrap_seed_counter = 0

    for metric, direction in (
        metric_directions.items()
    ):
        metric_wide = (
            test_data
            .pivot(
                index="sample_id",
                columns="method",
                values=metric,
            )
            .reindex(columns=METHOD_ORDER)
        )

        metric_start_index = len(
            pairwise_records
        )

        for method_a, method_b in method_pairs:

            paired = metric_wide[
                [
                    method_a,
                    method_b,
                ]
            ].dropna()

            values_a = paired[
                method_a
            ].to_numpy(dtype=float)

            values_b = paired[
                method_b
            ].to_numpy(dtype=float)

            if direction == "lower":
                oriented_difference = (
                    values_a - values_b
                )
            else:
                oriented_difference = (
                    values_b - values_a
                )

            nonzero_difference = (
                oriented_difference[
                    oriented_difference != 0
                ]
            )

            if (
                len(oriented_difference) < 2
                or len(nonzero_difference) == 0
            ):
                wilcoxon_statistic = 0.0
                p_raw = 1.0
            else:
                try:
                    wilcoxon_result = wilcoxon(
                        oriented_difference,
                        zero_method="wilcox",
                        alternative="two-sided",
                        method="auto",
                    )

                    wilcoxon_statistic = float(
                        wilcoxon_result.statistic
                    )

                    p_raw = float(
                        wilcoxon_result.pvalue
                    )

                except ValueError:
                    wilcoxon_statistic = np.nan
                    p_raw = np.nan

            effect_size = (
                rank_biserial_from_oriented_difference(
                    oriented_difference
                )
            )

            bootstrap_result = (
                bootstrap_paired_improvement(
                    oriented_difference=(
                        oriented_difference
                    ),
                    n_bootstrap=N_BOOTSTRAP,
                    random_state=(
                        RANDOM_STATE
                        + bootstrap_seed_counter
                    ),
                )
            )

            bootstrap_seed_counter += 1

            relative_improvement = (
                calculate_relative_improvement_percent(
                    method_a_values=values_a,
                    method_b_values=values_b,
                    direction=direction,
                )
            )

            pairwise_records.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "method_A": method_a,
                    "method_B": method_b,
                    "comparison": (
                        f"{method_b} vs {method_a}"
                    ),
                    "n_paired_samples": int(
                        len(paired)
                    ),
                    "mean_method_A": float(
                        np.mean(values_a)
                    ),
                    "mean_method_B": float(
                        np.mean(values_b)
                    ),
                    "mean_oriented_improvement_B_over_A": (
                        bootstrap_result[
                            "mean_improvement"
                        ]
                    ),
                    "mean_improvement_ci_low": (
                        bootstrap_result[
                            "mean_ci_low"
                        ]
                    ),
                    "mean_improvement_ci_high": (
                        bootstrap_result[
                            "mean_ci_high"
                        ]
                    ),
                    "median_oriented_improvement_B_over_A": (
                        bootstrap_result[
                            "median_improvement"
                        ]
                    ),
                    "median_improvement_ci_low": (
                        bootstrap_result[
                            "median_ci_low"
                        ]
                    ),
                    "median_improvement_ci_high": (
                        bootstrap_result[
                            "median_ci_high"
                        ]
                    ),
                    "relative_improvement_percent_B_over_A": (
                        relative_improvement
                    ),
                    "rank_biserial_effect_B_over_A": (
                        effect_size
                    ),
                    "wilcoxon_statistic": (
                        wilcoxon_statistic
                    ),
                    "p_raw": p_raw,
                }
            )

        metric_end_index = len(
            pairwise_records
        )

        metric_raw_p = np.asarray(
            [
                pairwise_records[index]["p_raw"]
                for index in range(
                    metric_start_index,
                    metric_end_index,
                )
            ],
            dtype=float,
        )

        metric_adjusted_p = holm_adjust(
            metric_raw_p
        )

        for local_index, adjusted_p in enumerate(
            metric_adjusted_p
        ):
            global_index = (
                metric_start_index
                + local_index
            )

            pairwise_records[
                global_index
            ]["p_holm_within_metric"] = (
                adjusted_p
            )

    pairwise_results = pd.DataFrame(
        pairwise_records
    )

    pairwise_results[
        "p_holm_formatted"
    ] = pairwise_results[
        "p_holm_within_metric"
    ].apply(format_p_value)

    pairwise_results[
        "significance"
    ] = pairwise_results[
        "p_holm_within_metric"
    ].apply(significance_label)

    pairwise_results.to_csv(
        PAIRWISE_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved:")
    print(PAIRWISE_OUTPUT)

    # -------------------------------------------------------------------------
    # Step 5: Extract key comparisons
    # -------------------------------------------------------------------------

    print_step(
        5,
        "Extract key model comparisons",
    )

    key_comparisons = pairwise_results.loc[
        pairwise_results["comparison"].isin(
            KEY_COMPARISON_NAMES
        )
    ].copy()

    key_comparisons.to_csv(
        KEY_COMPARISONS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    available_primary_metrics = [
        metric
        for metric in PRIMARY_METRICS
        if metric in metric_directions
    ]

    primary_key_comparisons = (
        key_comparisons.loc[
            key_comparisons["metric"].isin(
                available_primary_metrics
            )
        ]
        .copy()
    )

    primary_key_comparisons.to_csv(
        PRIMARY_KEY_COMPARISONS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved:")
    print(KEY_COMPARISONS_OUTPUT)
    print(PRIMARY_KEY_COMPARISONS_OUTPUT)

    if not primary_key_comparisons.empty:
        print(
            "\nPrimary key-comparison preview:\n"
        )

        print(
            primary_key_comparisons[
                [
                    "metric",
                    "comparison",
                    "relative_improvement_percent_B_over_A",
                    "rank_biserial_effect_B_over_A",
                    "p_holm_formatted",
                    "significance",
                ]
            ].to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # Step 6: Create paired-improvement long table for plotting
    # -------------------------------------------------------------------------

    print_step(
        6,
        "Create paired-sample improvement table for plotting",
    )

    paired_difference_records: list[
        dict[str, Any]
    ] = []

    for metric, direction in (
        metric_directions.items()
    ):
        metric_wide = (
            test_data
            .pivot(
                index="sample_id",
                columns="method",
                values=metric,
            )
            .reindex(columns=METHOD_ORDER)
        )

        for method_a, method_b in method_pairs:

            paired = metric_wide[
                [
                    method_a,
                    method_b,
                ]
            ].dropna()

            for sample_id, row in paired.iterrows():

                value_a = float(row[method_a])
                value_b = float(row[method_b])

                if direction == "lower":
                    oriented_improvement = (
                        value_a - value_b
                    )
                else:
                    oriented_improvement = (
                        value_b - value_a
                    )

                paired_difference_records.append(
                    {
                        "metric": metric,
                        "direction": direction,
                        "sample_id": sample_id,
                        "method_A": method_a,
                        "method_B": method_b,
                        "comparison": (
                            f"{method_b} vs "
                            f"{method_a}"
                        ),
                        "value_method_A": value_a,
                        "value_method_B": value_b,
                        "oriented_improvement_B_over_A": (
                            oriented_improvement
                        ),
                        "method_B_better": (
                            oriented_improvement > 0
                        ),
                    }
                )

    paired_improvements = pd.DataFrame(
        paired_difference_records
    )

    paired_improvements.to_csv(
        PAIRED_IMPROVEMENTS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved:")
    print(PAIRED_IMPROVEMENTS_OUTPUT)

    # -------------------------------------------------------------------------
    # Step 7: Save analysis configuration
    # -------------------------------------------------------------------------

    print_step(
        7,
        "Save statistical-analysis configuration",
    )

    config = {
        "target_split": TARGET_SPLIT,
        "input_sample_level_file": (
            SAMPLE_LEVEL_FILE
        ),
        "statistical_data_config_file": (
            STATISTICAL_DATA_CONFIG_FILE
        ),
        "method_order": METHOD_ORDER,
        "metric_directions": (
            metric_directions
        ),
        "primary_metrics": (
            available_primary_metrics
        ),
        "key_comparisons": (
            KEY_COMPARISON_NAMES
        ),
        "n_test_biological_samples": (
            n_samples
        ),
        "n_bootstrap": N_BOOTSTRAP,
        "random_state": RANDOM_STATE,
        "alpha": ALPHA,
        "statistical_unit": (
            "Biological sample_id. Five shallow-subsampling "
            "repeats had already been averaged within each "
            "sample_id × method by script 12."
        ),
        "overall_test": (
            "Friedman test across four paired methods"
        ),
        "pairwise_test": (
            "Paired two-sided Wilcoxon signed-rank test"
        ),
        "multiple_testing": (
            "Holm correction across the six pairwise "
            "comparisons within each metric"
        ),
        "friedman_multiple_testing": (
            "Holm correction across all tested metrics"
        ),
        "effect_size": (
            "Paired rank-biserial correlation. Positive values "
            "mean method_B is better than method_A."
        ),
        "bootstrap": (
            "Biological-sample-level percentile bootstrap "
            "95% confidence intervals for mean and median "
            "oriented improvements."
        ),
        "improvement_direction": (
            "Positive oriented improvement and positive relative "
            "improvement percentage indicate method_B is better."
        ),
        "output_files": {
            "friedman_tests": (
                FRIEDMAN_OUTPUT
            ),
            "pairwise_tests": (
                PAIRWISE_OUTPUT
            ),
            "key_comparisons": (
                KEY_COMPARISONS_OUTPUT
            ),
            "primary_key_comparisons": (
                PRIMARY_KEY_COMPARISONS_OUTPUT
            ),
            "paired_improvements": (
                PAIRED_IMPROVEMENTS_OUTPUT
            ),
        },
    }

    with open(
        CONFIG_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_serialisable(config),
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("Saved:")
    print(CONFIG_OUTPUT)

    # -------------------------------------------------------------------------
    # Completion message
    # -------------------------------------------------------------------------

    total_minutes = (
        time.perf_counter() - overall_start
    ) / 60.0

    print("\n" + "=" * 92)
    print("Statistical comparisons completed successfully")
    print("=" * 92)

    print(
        "\nCreated files:\n"
        "  - results/statistics/"
        "13_friedman_overall_tests.csv\n"
        "  - results/statistics/"
        "13_pairwise_wilcoxon_holm.csv\n"
        "  - results/statistics/"
        "13_key_model_comparisons.csv\n"
        "  - results/statistics/"
        "13_primary_key_model_comparisons.csv\n"
        "  - results/intermediate/"
        "13_paired_sample_improvements.csv\n"
        "  - results/statistics/"
        "13_statistical_analysis_config.json\n"
    )

    print(
        f"Total runtime: "
        f"{total_minutes:.2f} minutes"
    )

    print(
        "\nNext stage:\n"
        "  Create the model-performance figures from the "
        "prepared statistical tables."
    )


if __name__ == "__main__":
    main()