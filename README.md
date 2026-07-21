# 16S Relative Abundance Recovery Pipeline

Author: Ximan Ding (x.ding25@imperial.ac.uk)  
Date: July 2026

The original file is:
```text
Data/crosssecdata.RData
```

Inside that RData file there is one table called `datatax` with these columns:

```text
otu_id, count, project_id, sample_id, run_id, nreads, classification
```


The goal of the first stage is to convert that long count table into Python-readable CSV files, check that the reads are internally consistent, create a sample-by-OTU count matrix, split biological samples into train/valid/test groups, and select a fixed OTU vocabulary.

## Why There Is One R data

Python does not reliably read every `.RData` file by itself. The clean beginner workflow is:

1. Use a very small Python script once to export `datatax` from `.RData` to `.csv`.
2. Do the rest of the project in Python.

This keeps the project reproducible while still letting the main analysis be Python-based.

## Folder Structure

```text
code
data
results
```

## Requirements （very important）
numpy
pandas
pyreadr
scikit-learn (/usr/local/bin/python3 -m pip install scikit-learn)
python environment need: /usr/local/bin/python3 -m pip
/usr/local/bin/python3 -m pip install shap
/usr/local/bin/python3 -m pip install seaborn