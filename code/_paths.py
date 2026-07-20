#!/usr/bin/env python3
# Author: Ximan Ding (x.ding25@imperial.ac.uk)
# Script: _paths.py
# Description: Shared paths and small console helpers for the analysis scripts.
#
# Arguments: None
# Date: July 2026

from pathlib import Path


# Project root directory:
# CMEE_MRes_Project/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Original input data.
DATA_DIR = PROJECT_ROOT / "data"

# Analysis outputs.
RESULTS_DIR = PROJECT_ROOT / "results"
INTERMEDIATE_DIR = RESULTS_DIR / "intermediate"
PREPARED_DIR = RESULTS_DIR / "prepared"
FINAL_DIR = RESULTS_DIR / "final"


def banner(message: str) -> None:
    """Print a formatted section title in the terminal."""

    line = "=" * 60
    print(f"\n{line}")
    print(message)
    print(line)


def out(filename: str, directory: Path) -> Path:
    """Return an output path and create the parent directory if needed."""

    directory.mkdir(parents=True, exist_ok=True)
    return directory / filename