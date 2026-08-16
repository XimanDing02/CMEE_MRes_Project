#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Script: 20_prepare_revision_outputs.py
# Description:
#   Generate small supplemental outputs requested in the July 2026 feedback.
#   This script does not retrain models and does not regenerate shallow
#   simulations. It reads the outputs produced by scripts 01-19.

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
PREPARED_DIR = RESULTS_DIR / "prepared"
INTERMEDIATE_DIR = RESULTS_DIR / "intermediate"
DEPTH_DIR = RESULTS_DIR / "depth_sensitivity"
REVISION_DIR = RESULTS_DIR / "revision_outputs"
FIGURE_DIR = REVISION_DIR / "figures"

REVISION_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

MIN_REFERENCE_DEPTH = 10_000
FIXED_RICHNESS_THRESHOLD = 1e-4
PSEUDOCOUNT = 1e-8
N_BOOTSTRAP = 5_000
RANDOM_STATE = 42

METADATA_FILE = INTERMEDIATE_DIR / "crosssec_sample_metadata.csv"
ELIGIBLE_FILE = PREPARED_DIR / "01_eligible_reference_samples.csv"
SPLIT_FILE = PREPARED_DIR / "02_sample_split.csv"
SAMPLE_METRICS_FILE = INTERMEDIATE_DIR / "11_sample_metrics.csv"
TEST_PREDICTIONS_FILE = INTERMEDIATE_DIR / "11_predictions_test.pkl.gz"
DEPTH_SUMMARY_FILE = DEPTH_DIR / "04_depth_performance_summary.csv"
DEPTH_IMPROVEMENT_FILE = DEPTH_DIR / "08_depth_improvement_vs_raw.csv"

METHOD_ORDER = [
    "Raw shallow RA",
    "Training-mean RA",
    "Random Forest",
    "XGBoost",
]

METHOD_SHORT = {
    "Raw shallow RA": "Raw",
    "Training-mean RA": "Mean",
    "Random Forest": "RF",
    "XGBoost": "XGB",
}

PREDICTION_COLUMNS = {
    "Raw shallow RA": "pred_raw_shallow_ra",
    "Training-mean RA": "pred_training_mean_ra",
    "Random Forest": "pred_random_forest_ra",
    "XGBoost": "pred_xgboost_ra",
}

REVISION_TABLE_FILES = [
    "01_reference_depth_audit.csv",
    "02_signed_shannon_sample_level.csv",
    "03_signed_shannon_summary.csv",
    "04_reference_positive_shallow_zero_sample_level.csv",
    "05_reference_positive_shallow_zero_summary.csv",
    "06_depth_improvement_principal_metrics.csv",
]

SUPPLEMENTAL_FIGURE_BASENAMES = [
    "Supplementary_Figure_signed_Shannon_error",
    "Supplementary_Figure_reference_positive_shallow_zero_imputation",
    "Supplementary_Figure_depth_sensitivity_fixed_richness",
]


def require_file(path: Path) -> None:
    """Raise a clear error if an expected input is missing."""

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def bootstrap_ci(
    values: pd.Series,
    statistic: str = "median",
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Bootstrap a 95% confidence interval for a sample-level statistic."""

    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)

    if len(clean) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(random_state)
    estimates = np.empty(n_bootstrap, dtype=float)

    for index in range(n_bootstrap):
        sample = rng.choice(clean, size=len(clean), replace=True)

        if statistic == "median":
            estimates[index] = np.median(sample)
        elif statistic == "mean":
            estimates[index] = np.mean(sample)
        else:
            raise ValueError("statistic must be 'median' or 'mean'.")

    low, high = np.percentile(estimates, [2.5, 97.5])

    return float(low), float(high)


def save_figure(fig: plt.Figure, base_name: str) -> None:
    """Save a figure in PNG, PDF and SVG formats."""

    for suffix in ["png", "pdf", "svg"]:
        output = FIGURE_DIR / f"{base_name}.{suffix}"

        if suffix == "png":
            fig.savefig(output, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(output, bbox_inches="tight")

    plt.close(fig)


def add_jittered_points(
    axis: plt.Axes,
    x_position: int,
    values: np.ndarray,
    random_state: int,
) -> None:
    """Overlay jittered sample points on a boxplot."""

    rng = np.random.default_rng(random_state)
    x_values = rng.normal(
        loc=x_position,
        scale=0.045,
        size=len(values),
    )

    axis.scatter(
        x_values,
        values,
        s=13,
        alpha=0.45,
        edgecolors="none",
    )


def audit_reference_depths() -> pd.DataFrame:
    """Audit the 10,000-read eligibility threshold."""

    for path in [METADATA_FILE, ELIGIBLE_FILE, SPLIT_FILE]:
        require_file(path)

    metadata = pd.read_csv(METADATA_FILE)
    eligible = pd.read_csv(ELIGIBLE_FILE)
    split = pd.read_csv(SPLIT_FILE)

    required_metadata = {"sample_id", "calculated_total_reads"}
    required_eligible = {"sample_id", "calculated_total_reads"}
    required_split = {"sample_id", "split", "calculated_total_reads"}

    if not required_metadata.issubset(metadata.columns):
        raise ValueError("crosssec_sample_metadata.csv is missing depth columns.")
    if not required_eligible.issubset(eligible.columns):
        raise ValueError("01_eligible_reference_samples.csv is missing depth columns.")
    if not required_split.issubset(split.columns):
        raise ValueError("02_sample_split.csv is missing split/depth columns.")

    records: list[dict[str, object]] = []

    records.append(
        {
            "cohort": "Original selected project",
            "split": "all_before_filtering",
            "n_samples": int(metadata["sample_id"].nunique()),
            "minimum_reads": float(metadata["calculated_total_reads"].min()),
            "median_reads": float(metadata["calculated_total_reads"].median()),
            "mean_reads": float(metadata["calculated_total_reads"].mean()),
            "maximum_reads": float(metadata["calculated_total_reads"].max()),
            "eligibility_threshold": MIN_REFERENCE_DEPTH,
        }
    )

    records.append(
        {
            "cohort": "Eligible reference cohort",
            "split": "all_after_filtering",
            "n_samples": int(eligible["sample_id"].nunique()),
            "minimum_reads": float(eligible["calculated_total_reads"].min()),
            "median_reads": float(eligible["calculated_total_reads"].median()),
            "mean_reads": float(eligible["calculated_total_reads"].mean()),
            "maximum_reads": float(eligible["calculated_total_reads"].max()),
            "eligibility_threshold": MIN_REFERENCE_DEPTH,
        }
    )

    for split_name in ["train", "valid", "test"]:
        current = split.loc[split["split"] == split_name]

        records.append(
            {
                "cohort": "Eligible reference cohort",
                "split": split_name,
                "n_samples": int(current["sample_id"].nunique()),
                "minimum_reads": float(current["calculated_total_reads"].min()),
                "median_reads": float(current["calculated_total_reads"].median()),
                "mean_reads": float(current["calculated_total_reads"].mean()),
                "maximum_reads": float(current["calculated_total_reads"].max()),
                "eligibility_threshold": MIN_REFERENCE_DEPTH,
            }
        )

    audit = pd.DataFrame(records)
    eligible_minimum = float(eligible["calculated_total_reads"].min())
    audit["threshold_check_passed"] = eligible_minimum >= MIN_REFERENCE_DEPTH
    audit["interpretation"] = np.where(
        audit["split"] == "all_before_filtering",
        "Includes the two low-depth samples before the reference filter.",
        "After filtering, all reference samples meet the 10,000-read threshold.",
    )

    audit.to_csv(
        REVISION_DIR / "01_reference_depth_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if eligible_minimum < MIN_REFERENCE_DEPTH:
        raise ValueError(
            "At least one eligible reference sample is below 10,000 reads."
        )

    return audit


def calculate_signed_shannon() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate signed Shannon error on held-out biological samples."""

    require_file(SAMPLE_METRICS_FILE)

    sample_metrics = pd.read_csv(SAMPLE_METRICS_FILE)

    required = {
        "split",
        "method",
        "sample_id",
        "subsample_repeat",
        "true_Shannon_specific",
        "predicted_Shannon_specific",
    }

    missing = required.difference(sample_metrics.columns)

    if missing:
        raise ValueError(f"11_sample_metrics.csv is missing columns: {sorted(missing)}")

    shannon_repeat = sample_metrics.loc[
        sample_metrics["split"] == "test",
        [
            "method",
            "sample_id",
            "subsample_repeat",
            "true_Shannon_specific",
            "predicted_Shannon_specific",
        ],
    ].copy()

    shannon_repeat["Shannon_signed_error"] = (
        shannon_repeat["predicted_Shannon_specific"]
        - shannon_repeat["true_Shannon_specific"]
    )
    shannon_repeat["Shannon_absolute_error"] = (
        shannon_repeat["Shannon_signed_error"].abs()
    )

    shannon_sample = (
        shannon_repeat.groupby(
            ["method", "sample_id"],
            as_index=False,
            observed=True,
        )
        .agg(
            true_Shannon_specific=("true_Shannon_specific", "mean"),
            predicted_Shannon_specific=("predicted_Shannon_specific", "mean"),
            Shannon_signed_error=("Shannon_signed_error", "mean"),
            Shannon_absolute_error=("Shannon_absolute_error", "mean"),
            n_shallow_repeats=("subsample_repeat", "nunique"),
        )
    )

    summary_records: list[dict[str, object]] = []

    for method in METHOD_ORDER:
        current = shannon_sample.loc[shannon_sample["method"] == method]
        signed = current["Shannon_signed_error"].dropna()
        absolute = current["Shannon_absolute_error"].dropna()
        ci_low, ci_high = bootstrap_ci(signed, statistic="median")

        summary_records.append(
            {
                "method": method,
                "n_biological_samples": int(current["sample_id"].nunique()),
                "mean_signed_error": float(signed.mean()),
                "median_signed_error": float(signed.median()),
                "signed_error_q25": float(signed.quantile(0.25)),
                "signed_error_q75": float(signed.quantile(0.75)),
                "median_signed_error_ci_low": ci_low,
                "median_signed_error_ci_high": ci_high,
                "proportion_overpredicted": float((signed > 0).mean()),
                "proportion_underpredicted": float((signed < 0).mean()),
                "mean_absolute_error": float(absolute.mean()),
                "median_absolute_error": float(absolute.median()),
            }
        )

    shannon_summary = pd.DataFrame(summary_records)

    shannon_sample.to_csv(
        REVISION_DIR / "02_signed_shannon_sample_level.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shannon_summary.to_csv(
        REVISION_DIR / "03_signed_shannon_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, axis = plt.subplots(figsize=(9.0, 6.3))
    plot_values = [
        shannon_sample.loc[
            shannon_sample["method"] == method,
            "Shannon_signed_error",
        ].to_numpy(float)
        for method in METHOD_ORDER
    ]

    axis.boxplot(
        plot_values,
        tick_labels=[METHOD_SHORT[method] for method in METHOD_ORDER],
        showfliers=False,
    )

    for index, values in enumerate(plot_values, start=1):
        add_jittered_points(axis, index, values, RANDOM_STATE + index)

    axis.axhline(0.0, linestyle="--", linewidth=1.2)
    axis.set_xlabel("Method")
    axis.set_ylabel("Signed Shannon error\nH(predicted) - H(reference)")
    axis.set_title("Signed Shannon-diversity error in held-out samples")
    axis.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, SUPPLEMENTAL_FIGURE_BASENAMES[0])

    return shannon_sample, shannon_summary


def calculate_shallow_zero_imputation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate abundance imputation for reference-positive shallow zeros."""

    require_file(TEST_PREDICTIONS_FILE)

    predictions = pd.read_pickle(TEST_PREDICTIONS_FILE, compression="gzip")

    required = {
        "sample_id",
        "subsample_repeat",
        "target_reference_ra",
        "zero_in_shallow",
        "is_other",
        *PREDICTION_COLUMNS.values(),
    }

    missing = required.difference(predictions.columns)

    if missing:
        raise ValueError(f"11_predictions_test.pkl.gz is missing columns: {sorted(missing)}")

    zero_subset = predictions.loc[
        (predictions["is_other"] == 0)
        & (predictions["zero_in_shallow"] == 1)
        & (predictions["target_reference_ra"] > 0)
    ].copy()

    if zero_subset.empty:
        raise ValueError("No reference-positive shallow-zero rows were found.")

    records: list[dict[str, object]] = []

    for (sample_id, repeat_id), current in zero_subset.groupby(
        ["sample_id", "subsample_repeat"],
        observed=True,
        sort=False,
    ):
        y_true = current["target_reference_ra"].to_numpy(float)
        y_true_log = np.log10(y_true + PSEUDOCOUNT)

        for method in METHOD_ORDER:
            column = PREDICTION_COLUMNS[method]
            y_pred = np.clip(current[column].to_numpy(float), 0.0, None)
            y_pred_log = np.log10(y_pred + PSEUDOCOUNT)

            records.append(
                {
                    "method": method,
                    "sample_id": sample_id,
                    "subsample_repeat": repeat_id,
                    "n_reference_positive_shallow_zeros": len(current),
                    "MAE_RA": float(np.mean(np.abs(y_true - y_pred))),
                    "RMSE_RA": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                    "MAE_log10_RA": float(np.mean(np.abs(y_true_log - y_pred_log))),
                    "RMSE_log10_RA": float(
                        np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))
                    ),
                    "positive_prediction_rate": float((y_pred > 0).mean()),
                }
            )

    zero_repeat = pd.DataFrame(records)

    zero_sample = (
        zero_repeat.groupby(
            ["method", "sample_id"],
            as_index=False,
            observed=True,
        )
        .agg(
            n_reference_positive_shallow_zeros=(
                "n_reference_positive_shallow_zeros",
                "mean",
            ),
            MAE_RA=("MAE_RA", "mean"),
            RMSE_RA=("RMSE_RA", "mean"),
            MAE_log10_RA=("MAE_log10_RA", "mean"),
            RMSE_log10_RA=("RMSE_log10_RA", "mean"),
            positive_prediction_rate=("positive_prediction_rate", "mean"),
            n_shallow_repeats=("subsample_repeat", "nunique"),
        )
    )

    summary_records: list[dict[str, object]] = []

    for method in METHOD_ORDER:
        current = zero_sample.loc[zero_sample["method"] == method]

        record: dict[str, object] = {
            "method": method,
            "n_biological_samples": int(current["sample_id"].nunique()),
            "mean_positive_prediction_rate": float(
                current["positive_prediction_rate"].mean()
            ),
        }

        for metric in ["MAE_RA", "RMSE_RA", "MAE_log10_RA", "RMSE_log10_RA"]:
            record[f"{metric}_mean"] = float(current[metric].mean())
            record[f"{metric}_median"] = float(current[metric].median())

        summary_records.append(record)

    zero_summary = pd.DataFrame(summary_records)

    zero_sample.to_csv(
        REVISION_DIR / "04_reference_positive_shallow_zero_sample_level.csv",
        index=False,
        encoding="utf-8-sig",
    )
    zero_summary.to_csv(
        REVISION_DIR / "05_reference_positive_shallow_zero_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    zero_metrics = [
        ("MAE_RA", "MAE (RA scale)"),
        ("RMSE_RA", "RMSE (RA scale)"),
        ("MAE_log10_RA", "MAE (log10 scale)"),
        ("RMSE_log10_RA", "RMSE (log10 scale)"),
    ]

    for panel_index, (axis, (metric, title)) in enumerate(
        zip(axes.flatten(), zero_metrics)
    ):
        plot_values = [
            zero_sample.loc[zero_sample["method"] == method, metric].to_numpy(float)
            for method in METHOD_ORDER
        ]

        axis.boxplot(
            plot_values,
            tick_labels=[METHOD_SHORT[method] for method in METHOD_ORDER],
            showfliers=False,
        )

        for index, values in enumerate(plot_values, start=1):
            add_jittered_points(
                axis,
                index,
                values,
                RANDOM_STATE + index + panel_index * 10,
            )

        axis.set_title(title)
        axis.set_xlabel("Method")
        axis.set_ylabel("Prediction error")
        axis.grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle(
        "Abundance imputation for reference-positive shallow zeros",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, SUPPLEMENTAL_FIGURE_BASENAMES[1])

    return zero_sample, zero_summary


def plot_fixed_richness_depth_sensitivity() -> pd.DataFrame:
    """Create the feedback-focused depth figure using fixed richness."""

    require_file(DEPTH_SUMMARY_FILE)

    depth_summary = pd.read_csv(DEPTH_SUMMARY_FILE)

    required = {"depth", "method", "metric", "mean", "q25", "q75"}
    missing = required.difference(depth_summary.columns)

    if missing:
        raise ValueError(f"04_depth_performance_summary.csv is missing columns: {sorted(missing)}")

    principal_metrics = [
        ("Bray_Curtis", "Bray-Curtis dissimilarity"),
        ("Jensen_Shannon_distance", "Jensen-Shannon distance"),
        ("RAD_RMSE_log10_RA_specific", "RAD log10-RMSE"),
        ("RAD_tail_MAE_log10", "RAD tail log10-MAE"),
        ("Shannon_absolute_error", "Shannon absolute error"),
        (
            "richness_fixed_absolute_error",
            f"Fixed-threshold richness error\n(RA >= {FIXED_RICHNESS_THRESHOLD:g})",
        ),
    ]

    available = set(depth_summary["metric"].astype(str))
    missing_metrics = [
        metric for metric, _ in principal_metrics if metric not in available
    ]

    if missing_metrics:
        raise ValueError(
            "Depth summary is missing required feedback metrics: "
            f"{missing_metrics}"
        )

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.3))

    for axis, (metric, title) in zip(axes.flatten(), principal_metrics):
        current_metric = depth_summary.loc[
            depth_summary["metric"] == metric
        ].copy()

        for method in METHOD_ORDER:
            current = (
                current_metric.loc[current_metric["method"] == method]
                .sort_values("depth")
            )

            if current.empty:
                continue

            depth_values = current["depth"].to_numpy(float)
            mean_values = current["mean"].to_numpy(float)
            q25_values = current["q25"].to_numpy(float)
            q75_values = current["q75"].to_numpy(float)

            line = axis.plot(
                depth_values,
                mean_values,
                marker="o",
                linewidth=1.8,
                label=METHOD_SHORT[method],
            )[0]
            axis.fill_between(
                depth_values,
                q25_values,
                q75_values,
                alpha=0.10,
                color=line.get_color(),
            )

        axis.set_title(title)
        axis.set_xlabel("Simulated shallow sequencing depth (reads)")
        axis.set_ylabel("Error")
        axis.set_xticks(sorted(current_metric["depth"].unique()))
        axis.grid(linestyle="--", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.suptitle(
        "Sequencing-depth sensitivity using fixed-threshold richness",
        fontsize=16,
        fontweight="bold",
        y=1.015,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, SUPPLEMENTAL_FIGURE_BASENAMES[2])

    if DEPTH_IMPROVEMENT_FILE.exists():
        depth_improvement = pd.read_csv(DEPTH_IMPROVEMENT_FILE)
        retained = [metric for metric, _ in principal_metrics]
        depth_improvement_main = depth_improvement.loc[
            depth_improvement["metric"].isin(retained)
        ].copy()
        depth_improvement_main.to_csv(
            REVISION_DIR / "06_depth_improvement_principal_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

    return depth_summary


def save_manifest(
    depth_audit: pd.DataFrame,
    shannon_summary: pd.DataFrame,
    zero_summary: pd.DataFrame,
) -> None:
    """Save a small manifest describing the feedback outputs."""

    manifest = {
        "analysis_retrained": False,
        "source": "Outputs from scripts 01-19 in this split pipeline.",
        "reference_depth_threshold": MIN_REFERENCE_DEPTH,
        "reference_depth_check_passed": bool(
            depth_audit["threshold_check_passed"].all()
        ),
        "eligible_reference_minimum_reads": float(
            depth_audit.loc[
                depth_audit["split"] == "all_after_filtering",
                "minimum_reads",
            ].iloc[0]
        ),
        "signed_shannon_definition": "H(predicted) - H(reference)",
        "zero_analysis_title": (
            "Abundance imputation for reference-positive shallow zeros"
        ),
        "zero_analysis_scope": (
            "is_other=0, zero_in_shallow=1 and target_reference_ra>0"
        ),
        "cross_depth_richness_metric": "richness_fixed_absolute_error",
        "fixed_richness_threshold": FIXED_RICHNESS_THRESHOLD,
        "revision_output_tables": REVISION_TABLE_FILES,
        "supplemental_figure_files": SUPPLEMENTAL_FIGURE_BASENAMES,
        "signed_shannon_median_by_method": dict(
            zip(
                shannon_summary["method"],
                shannon_summary["median_signed_error"],
            )
        ),
        "zero_log10_rmse_mean_by_method": dict(
            zip(
                zero_summary["method"],
                zero_summary["RMSE_log10_RA_mean"],
            )
        ),
    }

    with open(
        REVISION_DIR / "revision_manifest.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)


def main() -> None:
    """Run all feedback-focused supplemental outputs."""

    print("=" * 88)
    print("Teacher-feedback supplemental analyses")
    print("=" * 88)

    depth_audit = audit_reference_depths()
    print("\nReference-depth audit:")
    print(depth_audit.to_string(index=False))

    _, shannon_summary = calculate_signed_shannon()
    print("\nSigned Shannon summary:")
    print(shannon_summary.to_string(index=False))

    _, zero_summary = calculate_shallow_zero_imputation()
    print("\nReference-positive shallow-zero summary:")
    print(zero_summary.to_string(index=False))

    plot_fixed_richness_depth_sensitivity()

    save_manifest(depth_audit, shannon_summary, zero_summary)

    print("\nCompleted teacher-feedback supplemental outputs.")
    print(f"Revision output directory: {REVISION_DIR}")


if __name__ == "__main__":
    main()
