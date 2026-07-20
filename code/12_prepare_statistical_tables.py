# -*- coding: utf-8 -*-
#
# Author: Ximan Ding
# Script: 12_prepare_statistical_tables.py
# Description:
#     Prepare independent-test statistical tables from the prediction and
#     evaluation outputs created by
#     11_generate_predictions_and_metrics.py.
#
#     This script does not train models and does not run significance tests.
#
#     It:
#     1. Reads sample-level, abundance-stratum and shallow-zero metrics.
#     2. Keeps the independent test split only.
#     3. Checks that all four prediction methods are present.
#     4. Averages the five simulated shallow-sequencing repeats within each
#        biological sample and method.
#     5. Creates publication-ready descriptive summaries.
#     6. Calculates abundance-stratum and shallow-zero improvements relative
#        to Raw shallow RA.
#
# Inputs:
#     results/intermediate/11_sample_metrics.csv
#     results/intermediate/11_abundance_stratum_metrics.csv
#     results/intermediate/11_zero_recovery_metrics.csv
#
# Outputs:
#     results/intermediate/12_test_sample_level_metrics.csv
#     results/final/statistics/12_test_performance_summary.csv
#     results/final/statistics/12_primary_metrics_summary.csv
#     results/final/statistics/12_abundance_group_improvement_vs_raw.csv
#     results/final/statistics/12_test_zero_recovery_results.csv
#     results/final/statistics/12_zero_recovery_improvement_vs_raw.csv
#     results/final/statistics/12_statistical_data_config.json
#
# Date: July 2026


# =============================================================================
# 0. Import packages
# =============================================================================

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

SAMPLE_METRICS_FILE = (
    INTERMEDIATE_DIR / "11_sample_metrics.csv"
)

ABUNDANCE_METRICS_FILE = (
    INTERMEDIATE_DIR
    / "11_abundance_stratum_metrics.csv"
)

ZERO_RECOVERY_FILE = (
    INTERMEDIATE_DIR
    / "11_zero_recovery_metrics.csv"
)

SAMPLE_LEVEL_OUTPUT = (
    INTERMEDIATE_DIR
    / "12_test_sample_level_metrics.csv"
)

PERFORMANCE_SUMMARY_OUTPUT = (
    STATISTICS_DIR
    / "12_test_performance_summary.csv"
)

PRIMARY_SUMMARY_OUTPUT = (
    STATISTICS_DIR
    / "12_primary_metrics_summary.csv"
)

ABUNDANCE_IMPROVEMENT_OUTPUT = (
    STATISTICS_DIR
    / "12_abundance_group_improvement_vs_raw.csv"
)

ZERO_RESULTS_OUTPUT = (
    STATISTICS_DIR
    / "12_test_zero_recovery_results.csv"
)

ZERO_IMPROVEMENT_OUTPUT = (
    STATISTICS_DIR
    / "12_zero_recovery_improvement_vs_raw.csv"
)

CONFIG_OUTPUT = (
    STATISTICS_DIR
    / "12_statistical_data_config.json"
)


# =============================================================================
# 2. Analysis settings
# =============================================================================

TARGET_SPLIT = "test"

METHOD_ORDER = [
    "Raw shallow RA",
    "Training-mean RA",
    "Random Forest",
    "XGBoost",
]

METHOD_SHORT_NAMES = {
    "Raw shallow RA": "Raw",
    "Training-mean RA": "Mean",
    "Random Forest": "RF",
    "XGBoost": "XGB",
}

# lower = smaller values are better
# higher = larger values are better
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

ABUNDANCE_METRIC_DIRECTIONS = {
    "MAE_RA": "lower",
    "RMSE_RA": "lower",
    "MAE_log10_RA": "lower",
    "RMSE_log10_RA": "lower",
    "Spearman_RA": "higher",
}

ZERO_METRIC_DIRECTIONS = {
    "MAE_RA": "lower",
    "RMSE_RA": "lower",
    "MAE_log10_RA": "lower",
    "RMSE_log10_RA": "lower",
}


# =============================================================================
# 3. Helper functions
# =============================================================================

def print_step(step_number: int, title: str) -> None:
    """Print a consistent progress heading."""

    print("\n" + "=" * 88)
    print(f"Step {step_number}: {title}")
    print("=" * 88)


def require_file(file_path: Path) -> None:
    """Raise a clear error when an input file does not exist."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}\n"
            "Run code/11_generate_predictions_and_metrics.py first."
        )


def require_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Check required columns in an input table."""

    missing = sorted(
        set(required_columns) - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            f"{table_name} is missing required columns: {missing}"
        )


def available_metric_directions(
    dataframe: pd.DataFrame,
    configured_directions: dict[str, str],
) -> dict[str, str]:
    """
    Keep configured metrics that are present in the input table.

    At least one configured metric must be available.
    """

    available = {
        metric: direction
        for metric, direction in configured_directions.items()
        if metric in dataframe.columns
    }

    if not available:
        raise ValueError(
            "None of the configured statistical metrics were found. "
            f"Available columns are:\n{list(dataframe.columns)}"
        )

    missing = [
        metric
        for metric in configured_directions
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


def calculate_relative_improvement(
    baseline_value: float,
    model_value: float,
    direction: str,
) -> float:
    """
    Calculate model improvement relative to a baseline.

    Positive values always mean that the model is better.
    """

    if not np.isfinite(baseline_value):
        return np.nan

    if not np.isfinite(model_value):
        return np.nan

    denominator = abs(float(baseline_value))

    if denominator <= 1e-15:
        return np.nan

    if direction == "lower":
        return float(
            (baseline_value - model_value)
            / denominator
            * 100.0
        )

    if direction == "higher":
        return float(
            (model_value - baseline_value)
            / denominator
            * 100.0
        )

    raise ValueError(
        f"Unknown metric direction: {direction}"
    )


def build_performance_summary(
    sample_level_data: pd.DataFrame,
    metric_directions: dict[str, str],
) -> pd.DataFrame:
    """Create publication-ready descriptive statistics."""

    records: list[dict[str, Any]] = []

    for metric, direction in metric_directions.items():

        method_means = (
            sample_level_data
            .groupby(
                "method",
                observed=True,
            )[metric]
            .mean()
            .reindex(METHOD_ORDER)
        )

        ranks = method_means.rank(
            method="min",
            ascending=(direction == "lower"),
        )

        for method in METHOD_ORDER:

            values = pd.to_numeric(
                sample_level_data.loc[
                    sample_level_data["method"] == method,
                    metric,
                ],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            mean_value = float(values.mean())
            standard_deviation = float(
                values.std(ddof=1)
            )
            median_value = float(values.median())
            q25 = float(values.quantile(0.25))
            q75 = float(values.quantile(0.75))

            method_rank = (
                int(ranks.loc[method])
                if pd.notna(ranks.loc[method])
                else np.nan
            )

            records.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "method": method,
                    "method_short": (
                        METHOD_SHORT_NAMES[method]
                    ),
                    "n_biological_samples": int(
                        len(values)
                    ),
                    "mean": mean_value,
                    "standard_deviation": (
                        standard_deviation
                    ),
                    "median": median_value,
                    "q25": q25,
                    "q75": q75,
                    "mean_plus_minus_sd": (
                        f"{mean_value:.6g} ± "
                        f"{standard_deviation:.6g}"
                    ),
                    "median_IQR": (
                        f"{median_value:.6g} "
                        f"[{q25:.6g}, {q75:.6g}]"
                    ),
                    "rank_by_mean": method_rank,
                    "is_best_by_mean": (
                        method_rank == 1
                    ),
                }
            )

    return pd.DataFrame(records)


def make_json_serialisable(value: Any) -> Any:
    """Convert values to JSON-compatible objects."""

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
# 4. Main analysis
# =============================================================================

def main() -> None:
    """Prepare independent-test statistical tables."""

    overall_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Step 1: Read the outputs produced by script 11
    # -------------------------------------------------------------------------

    print_step(
        1,
        "Read prediction and evaluation tables",
    )

    for input_file in [
        SAMPLE_METRICS_FILE,
        ABUNDANCE_METRICS_FILE,
        ZERO_RECOVERY_FILE,
    ]:
        require_file(input_file)

    sample_metrics = pd.read_csv(
        SAMPLE_METRICS_FILE
    )

    abundance_metrics = pd.read_csv(
        ABUNDANCE_METRICS_FILE
    )

    zero_recovery_metrics = pd.read_csv(
        ZERO_RECOVERY_FILE
    )

    print(
        f"Sample metrics shape:             "
        f"{sample_metrics.shape}"
    )
    print(
        f"Abundance-stratum metrics shape: "
        f"{abundance_metrics.shape}"
    )
    print(
        f"Zero-recovery metrics shape:      "
        f"{zero_recovery_metrics.shape}"
    )

    # -------------------------------------------------------------------------
    # Step 2: Check required identifiers and available metrics
    # -------------------------------------------------------------------------

    print_step(
        2,
        "Check columns and identify available metrics",
    )

    require_columns(
        sample_metrics,
        [
            "split",
            "method",
            "sample_id",
            "subsample_repeat",
        ],
        "11_sample_metrics.csv",
    )

    require_columns(
        abundance_metrics,
        [
            "split",
            "method",
            "abundance_group",
        ],
        "11_abundance_stratum_metrics.csv",
    )

    require_columns(
        zero_recovery_metrics,
        [
            "split",
            "method",
        ],
        "11_zero_recovery_metrics.csv",
    )

    sample_metric_directions = (
        available_metric_directions(
            sample_metrics,
            METRIC_DIRECTIONS,
        )
    )

    abundance_metric_directions = (
        available_metric_directions(
            abundance_metrics,
            ABUNDANCE_METRIC_DIRECTIONS,
        )
    )

    zero_metric_directions = (
        available_metric_directions(
            zero_recovery_metrics,
            ZERO_METRIC_DIRECTIONS,
        )
    )

    print(
        f"Available sample-level metrics: "
        f"{len(sample_metric_directions)}"
    )

    for metric in sample_metric_directions:
        print(f"  - {metric}")

    # -------------------------------------------------------------------------
    # Step 3: Keep the independent test split
    # -------------------------------------------------------------------------

    print_step(
        3,
        "Extract the independent test split",
    )

    test_sample_metrics = sample_metrics.loc[
        sample_metrics["split"] == TARGET_SPLIT
    ].copy()

    if test_sample_metrics.empty:
        raise ValueError(
            "No rows with split='test' were found in "
            "11_sample_metrics.csv."
        )

    observed_methods = set(
        test_sample_metrics["method"]
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
            "The test sample metrics are missing methods: "
            f"{missing_methods}"
        )

    n_test_samples = int(
        test_sample_metrics["sample_id"].nunique()
    )

    print(
        f"Test repeat-level records: "
        f"{len(test_sample_metrics):,}"
    )
    print(
        f"Test biological samples:   "
        f"{n_test_samples}"
    )
    print(
        f"Prediction methods:         "
        f"{len(observed_methods)}"
    )

    # -------------------------------------------------------------------------
    # Step 4: Check simulated repeat counts
    # -------------------------------------------------------------------------

    print_step(
        4,
        "Check shallow-sequencing repeat counts",
    )

    repeat_counts = (
        test_sample_metrics
        .groupby(
            [
                "method",
                "sample_id",
            ],
            observed=True,
        )["subsample_repeat"]
        .nunique()
    )

    print(
        repeat_counts.describe().to_string()
    )

    expected_repeat_count = int(
        repeat_counts.mode().iloc[0]
    )

    unexpected_repeat_groups = repeat_counts.loc[
        repeat_counts != expected_repeat_count
    ]

    if not unexpected_repeat_groups.empty:
        raise ValueError(
            "Not every sample_id × method group has the same "
            "number of shallow repeats.\n"
            f"Unexpected groups:\n"
            f"{unexpected_repeat_groups.head(20)}"
        )

    print(
        "All sample_id × method groups contain "
        f"{expected_repeat_count} repeats."
    )

    # -------------------------------------------------------------------------
    # Step 5: Average repeats within biological samples
    # -------------------------------------------------------------------------

    print_step(
        5,
        "Average shallow repeats within each biological sample",
    )

    metric_columns = list(
        sample_metric_directions.keys()
    )

    for metric in metric_columns:
        test_sample_metrics[metric] = pd.to_numeric(
            test_sample_metrics[metric],
            errors="coerce",
        )

    sample_level_metrics = (
        test_sample_metrics
        .groupby(
            [
                "split",
                "method",
                "sample_id",
            ],
            as_index=False,
            observed=True,
        )[metric_columns]
        .mean()
    )

    sample_level_metrics["method"] = (
        pd.Categorical(
            sample_level_metrics["method"],
            categories=METHOD_ORDER,
            ordered=True,
        )
    )

    sample_level_metrics = (
        sample_level_metrics
        .sort_values(
            [
                "sample_id",
                "method",
            ]
        )
        .reset_index(drop=True)
    )

    expected_rows = (
        n_test_samples * len(METHOD_ORDER)
    )

    if len(sample_level_metrics) != expected_rows:
        sample_method_counts = (
            sample_level_metrics
            .groupby("sample_id", observed=True)
            ["method"]
            .nunique()
        )

        incomplete_samples = (
            sample_method_counts.loc[
                sample_method_counts
                != len(METHOD_ORDER)
            ]
        )

        raise ValueError(
            "The biological-sample table is incomplete. "
            f"Expected {expected_rows} rows but found "
            f"{len(sample_level_metrics)}.\n"
            f"Incomplete samples:\n"
            f"{incomplete_samples.head(20)}"
        )

    sample_level_metrics.to_csv(
        SAMPLE_LEVEL_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Sample-level table shape: "
        f"{sample_level_metrics.shape}"
    )
    print(
        f"Expected rows confirmed: "
        f"{n_test_samples} samples × "
        f"{len(METHOD_ORDER)} methods = "
        f"{expected_rows}"
    )
    print("Saved:")
    print(SAMPLE_LEVEL_OUTPUT)

    # -------------------------------------------------------------------------
    # Step 6: Create descriptive performance summaries
    # -------------------------------------------------------------------------

    print_step(
        6,
        "Create publication-ready performance summaries",
    )

    performance_summary = build_performance_summary(
        sample_level_data=sample_level_metrics,
        metric_directions=sample_metric_directions,
    )

    performance_summary.to_csv(
        PERFORMANCE_SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    available_primary_metrics = [
        metric
        for metric in PRIMARY_METRICS
        if metric in sample_metric_directions
    ]

    primary_summary = performance_summary.loc[
        performance_summary["metric"].isin(
            available_primary_metrics
        )
    ].copy()

    primary_summary.to_csv(
        PRIMARY_SUMMARY_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved:")
    print(PERFORMANCE_SUMMARY_OUTPUT)
    print(PRIMARY_SUMMARY_OUTPUT)

    if not primary_summary.empty:
        print(
            "\nPrimary-metric preview:\n"
        )
        print(
            primary_summary[
                [
                    "metric",
                    "method",
                    "mean_plus_minus_sd",
                    "rank_by_mean",
                ]
            ]
            .to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # Step 7: Prepare abundance-stratum improvements
    # -------------------------------------------------------------------------

    print_step(
        7,
        "Prepare abundance-stratum improvements relative to Raw",
    )

    test_abundance = abundance_metrics.loc[
        abundance_metrics["split"] == TARGET_SPLIT
    ].copy()

    abundance_records: list[dict[str, Any]] = []

    if not test_abundance.empty:

        for abundance_group in (
            test_abundance["abundance_group"]
            .dropna()
            .unique()
        ):
            group_data = test_abundance.loc[
                test_abundance["abundance_group"]
                == abundance_group
            ].copy()

            for method in [
                "Random Forest",
                "XGBoost",
            ]:
                for metric, direction in (
                    abundance_metric_directions.items()
                ):

                    baseline_values = pd.to_numeric(
                        group_data.loc[
                            group_data["method"]
                            == "Raw shallow RA",
                            metric,
                        ],
                        errors="coerce",
                    ).dropna()

                    model_values = pd.to_numeric(
                        group_data.loc[
                            group_data["method"] == method,
                            metric,
                        ],
                        errors="coerce",
                    ).dropna()

                    if (
                        baseline_values.empty
                        or model_values.empty
                    ):
                        continue

                    baseline_value = float(
                        baseline_values.mean()
                    )
                    model_value = float(
                        model_values.mean()
                    )

                    abundance_records.append(
                        {
                            "split": TARGET_SPLIT,
                            "abundance_group": (
                                abundance_group
                            ),
                            "metric": metric,
                            "direction": direction,
                            "baseline_method": (
                                "Raw shallow RA"
                            ),
                            "model": method,
                            "raw_shallow_value": (
                                baseline_value
                            ),
                            "model_value": model_value,
                            "relative_improvement_percent": (
                                calculate_relative_improvement(
                                    baseline_value,
                                    model_value,
                                    direction,
                                )
                            ),
                        }
                    )

    abundance_improvement = pd.DataFrame(
        abundance_records
    )

    abundance_improvement.to_csv(
        ABUNDANCE_IMPROVEMENT_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved:")
    print(ABUNDANCE_IMPROVEMENT_OUTPUT)

    # -------------------------------------------------------------------------
    # Step 8: Prepare shallow-zero recovery summaries
    # -------------------------------------------------------------------------

    print_step(
        8,
        "Prepare shallow-zero recovery summaries",
    )

    test_zero_recovery = zero_recovery_metrics.loc[
        zero_recovery_metrics["split"] == TARGET_SPLIT
    ].copy()

    method_category = pd.CategoricalDtype(
        categories=METHOD_ORDER,
        ordered=True,
    )

    if "method" in test_zero_recovery.columns:
        test_zero_recovery["method"] = (
            test_zero_recovery["method"]
            .astype(method_category)
        )

        test_zero_recovery = (
            test_zero_recovery
            .sort_values("method")
            .reset_index(drop=True)
        )

    test_zero_recovery.to_csv(
        ZERO_RESULTS_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    zero_records: list[dict[str, Any]] = []

    for method in [
        "Random Forest",
        "XGBoost",
    ]:
        for metric, direction in (
            zero_metric_directions.items()
        ):

            baseline_values = pd.to_numeric(
                test_zero_recovery.loc[
                    test_zero_recovery["method"]
                    == "Raw shallow RA",
                    metric,
                ],
                errors="coerce",
            ).dropna()

            model_values = pd.to_numeric(
                test_zero_recovery.loc[
                    test_zero_recovery["method"]
                    == method,
                    metric,
                ],
                errors="coerce",
            ).dropna()

            if (
                baseline_values.empty
                or model_values.empty
            ):
                continue

            baseline_value = float(
                baseline_values.mean()
            )
            model_value = float(
                model_values.mean()
            )

            zero_records.append(
                {
                    "split": TARGET_SPLIT,
                    "metric": metric,
                    "direction": direction,
                    "baseline_method": (
                        "Raw shallow RA"
                    ),
                    "model": method,
                    "raw_shallow_value": (
                        baseline_value
                    ),
                    "model_value": model_value,
                    "relative_improvement_percent": (
                        calculate_relative_improvement(
                            baseline_value,
                            model_value,
                            direction,
                        )
                    ),
                }
            )

    zero_improvement = pd.DataFrame(
        zero_records
    )

    zero_improvement.to_csv(
        ZERO_IMPROVEMENT_OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved:")
    print(ZERO_RESULTS_OUTPUT)
    print(ZERO_IMPROVEMENT_OUTPUT)

    # -------------------------------------------------------------------------
    # Step 9: Save configuration
    # -------------------------------------------------------------------------

    print_step(
        9,
        "Save statistical-data configuration",
    )

    config = {
        "target_split": TARGET_SPLIT,
        "method_order": METHOD_ORDER,
        "method_short_names": METHOD_SHORT_NAMES,
        "available_sample_metric_directions": (
            sample_metric_directions
        ),
        "available_primary_metrics": (
            available_primary_metrics
        ),
        "available_abundance_metric_directions": (
            abundance_metric_directions
        ),
        "available_zero_metric_directions": (
            zero_metric_directions
        ),
        "input_files": {
            "sample_metrics": SAMPLE_METRICS_FILE,
            "abundance_stratum_metrics": (
                ABUNDANCE_METRICS_FILE
            ),
            "zero_recovery_metrics": (
                ZERO_RECOVERY_FILE
            ),
        },
        "output_files": {
            "test_sample_level_metrics": (
                SAMPLE_LEVEL_OUTPUT
            ),
            "test_performance_summary": (
                PERFORMANCE_SUMMARY_OUTPUT
            ),
            "primary_metrics_summary": (
                PRIMARY_SUMMARY_OUTPUT
            ),
            "abundance_group_improvement": (
                ABUNDANCE_IMPROVEMENT_OUTPUT
            ),
            "test_zero_recovery_results": (
                ZERO_RESULTS_OUTPUT
            ),
            "zero_recovery_improvement": (
                ZERO_IMPROVEMENT_OUTPUT
            ),
        },
        "n_test_biological_samples": (
            n_test_samples
        ),
        "shallow_repeats_per_sample_method": (
            expected_repeat_count
        ),
        "statistical_unit": (
            "Biological sample_id. Repeat-level metrics were "
            "averaged within each sample_id × method before "
            "statistical inference."
        ),
        "note": (
            "This script prepares statistical tables only. "
            "Friedman, paired Wilcoxon, Holm correction, "
            "effect sizes and bootstrap confidence intervals "
            "are performed by script 13."
        ),
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

    print("\n" + "=" * 88)
    print("Statistical table preparation completed successfully")
    print("=" * 88)

    print(
        "\nCreated files:\n"
        "  - results/intermediate/"
        "12_test_sample_level_metrics.csv\n"
        "  - results/statistics/"
        "12_test_performance_summary.csv\n"
        "  - results/statistics/"
        "12_primary_metrics_summary.csv\n"
        "  - results/statistics/"
        "12_abundance_group_improvement_vs_raw.csv\n"
        "  - results/statistics/"
        "12_test_zero_recovery_results.csv\n"
        "  - results/statistics/"
        "12_zero_recovery_improvement_vs_raw.csv\n"
        "  - results/statistics/"
        "12_statistical_data_config.json\n"
    )

    print(
        f"Total runtime: "
        f"{total_minutes:.2f} minutes"
    )

    print(
        "\nNext script:\n"
        "  code/13_run_statistical_comparisons.py"
    )


if __name__ == "__main__":
    main()