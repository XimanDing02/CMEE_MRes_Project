#!/usr/bin/env python3
# Author: Ximan Ding (x.ding25@imperial.ac.uk)
# Script: 00_rdata_to_csv.py
# Description: Convert crosssecdata.RData into Python-readable CSV files.
#
# Arguments: 1 -> Path to the .RData file
#                (optional; default: data/crosssecdata.RData)
# Date: July 2026

"""
Step 0: Convert R data into Python-readable files
=================================================

This script reads crosssecdata.RData using pyreadr, records the objects
contained in the file, validates the expected datatax columns, and exports
the datatax object as a CSV file.

Input:
    data/crosssecdata.RData

Outputs:
    results/intermediate/00_RData_object_info.csv
    results/intermediate/rdata__datatax.csv
    results/intermediate/crosssec_datatax.csv

Usage:
    python3 code/00_rdata_to_csv.py

or:

    python3 code/00_rdata_to_csv.py data/crosssecdata.RData
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from _paths import DATA_DIR, INTERMEDIATE_DIR, banner, out


EXPECTED_COLUMNS = [
    "otu_id",
    "count",
    "project_id",
    "sample_id",
    "run_id",
    "nreads",
    "classification",
]


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove hidden BOM characters and surrounding spaces from column names."""

    df = df.copy()

    df.columns = [
        str(column).strip().lstrip("\ufeff")
        for column in df.columns
    ]

    return df


def main(rdata_path: str | Path) -> None:
    """Read the RData file and export readable objects as CSV files."""

    banner("Step 0: Read RData and export CSV files")

    import pyreadr

    rdata_path = Path(rdata_path)

    if not rdata_path.exists():
        raise FileNotFoundError(
            f"RData file not found: {rdata_path}"
        )

    result = pyreadr.read_r(rdata_path)

    print(
        "Objects found in the RData file:",
        list(result.keys()),
    )

    object_rows = []
    exported_datatax = False

    for object_name, dataframe in result.items():

        if dataframe is None:
            continue

        dataframe = clean_columns(dataframe)

        object_rows.append(
            {
                "object_name": object_name,
                "object_class": type(dataframe).__name__,
                "rows": dataframe.shape[0],
                "columns": dataframe.shape[1],
            }
        )

        # Export a general copy of every readable object.
        general_output_path = out(
            f"rdata__{object_name}.csv",
            INTERMEDIATE_DIR,
        )

        dataframe.to_csv(
            general_output_path,
            index=False,
        )

        print(
            f"  {object_name}: {dataframe.shape} "
            f"-> {general_output_path}"
        )

        # Export a validated standard copy of datatax.
        if object_name == "datatax":

            missing_columns = set(
                EXPECTED_COLUMNS
            ).difference(
                dataframe.columns
            )

            if missing_columns:
                raise ValueError(
                    "datatax is missing expected columns: "
                    f"{sorted(missing_columns)}"
                )

            standard_output_path = out(
                "crosssec_datatax.csv",
                INTERMEDIATE_DIR,
            )

            dataframe[
                EXPECTED_COLUMNS
            ].to_csv(
                standard_output_path,
                index=False,
            )

            exported_datatax = True

            print(
                f"  datatax standard copy "
                f"-> {standard_output_path}"
            )

    # Save information about all readable RData objects.
    object_info_path = out(
        "00_RData_object_info.csv",
        INTERMEDIATE_DIR,
    )

    pd.DataFrame(
        object_rows
    ).to_csv(
        object_info_path,
        index=False,
    )

    print(
        f"  object information -> {object_info_path}"
    )

    if not exported_datatax:
        raise ValueError(
            "The expected object 'datatax' was not found. "
            "Check the object names printed above."
        )

    print(
        "\nStep 0 completed successfully.\n"
        "Next script:\n"
        "  python3 code/01_prepare_count_matrix.py\n"
    )


if __name__ == "__main__":

    rdata_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else DATA_DIR / "crosssecdata.RData"
    )

    main(rdata_path)