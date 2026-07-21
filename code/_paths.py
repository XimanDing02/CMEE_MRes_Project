#!/usr/bin/env python3
# Author: Ximan Ding (x.ding25@imperial.ac.uk)
# Script: _paths.py
# Description: Shared paths and small console helpers for the analysis scripts.
#
# Arguments: None
# Date: July 2026

from pathlib import Path


# ============================================================
# 1. Project directories
# ============================================================

# Project root:
# CMEE_MRES_PROJECT/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Original input data.
DATA_DIR = PROJECT_ROOT / "data"

# Main results directory.
RESULTS_DIR = PROJECT_ROOT / "results"

# Result subdirectories.
BASELINE_DIR = RESULTS_DIR / "baseline"
FIGURES_DIR = RESULTS_DIR / "figures"
FINAL_DIR = RESULTS_DIR / "final"
INTERMEDIATE_DIR = RESULTS_DIR / "intermediate"
MODELS_DIR = RESULTS_DIR / "models"
PREPARED_DIR = RESULTS_DIR / "prepared"
SHAP_DIR = RESULTS_DIR / "shap"
STATISTICS_DIR = RESULTS_DIR / "statistics"


# ============================================================
# 2. Console helper
# ============================================================

def banner(message: str) -> None:
    """Print a formatted section title in the terminal."""

    line = "=" * 60
    print(f"\n{line}")
    print(message)
    print(line)


# ============================================================
# 3. Output-path helper
# ============================================================

def out(filename: str, directory: Path) -> Path:
    """Return an output path and create the directory if needed."""

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory / filename