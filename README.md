# Project Pipeline

## Table of Contents
- [16S Relative Abundance Recovery Pipeline](#16s-relative-abundance-recovery-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Brief Description](#brief-description)
  - [Input Data](#input-data)
  - [Languages](#languages)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Scripts List](#scripts-list)
  - [Scripts Basic Usage](#scripts-basic-usage)
  - [Expected Output](#expected-output)
  - [Author and Contact](#author-and-contact)

## Project Structure

This project structure is:

```plaintext
CMEE_MRes_Project/
│
├── code/
│   ├── _paths.py
│   ├── 00_rdata_to_csv.py
│   ├── 01_prepare_count_matrix.py
│   ├── 02_filter_reference_samples.py
│   ├── 03_split_samples.py
│   ├── 04_select_otu_vocabulary.py
│   ├── 05_build_reference_matrix.py
│   ├── 06_compute_train_statistics.py
│   ├── 07_generate_supervised_data.py
│   ├── 08_check_stage1_quality.py
│   ├── 09_evaluate_baselines.py
│   ├── 10_train_models.py
│   ├── 11_generate_predictions_and_metrics.py
│   ├── 12_prepare_statistical_tables.py
│   ├── 13_run_statistical_comparisons.py
│   ├── 14_compute_shap_results.py
│   ├── 15_prepare_shap_tables.py
│   ├── 16_plot_core_figures.py
│   ├── 17_plot_shap_figures.py
│   ├── 18_run_depth_sensitivity.py
│   ├── 19_plot_depth_sensitivity.py
│   └── 20_prepare_revision_outputs.py
│
├── data/
│   └── crosssecdata.RData
│
├── results/
│   ├── baseline/
│   ├── depth_sensitivity/
│   ├── figures/
│   ├── intermediate/
│   ├── models/
│   ├── prepared/
│   ├── revision_outputs/
│   ├── shap/
│   └── statistics/
│
├── requirements.txt
└── README.md
```

## Brief Description

1. This project studies whether high-depth 16S relative-abundance profiles can be recovered from shallow sequencing simulations.
2. The workflow starts from the processed `crosssecdata.RData` file and extracts the MGnify seawater project `MGYS00002437`.
3. Multiple runs belonging to the same biological sample are aggregated before relative abundances are recalculated.
4. Samples with at least 10,000 aggregated reads are retained as high-depth reference observations.
5. Biological samples are split into training, validation and test sets before shallow sequencing simulations are generated.
6. Shallow sequencing is simulated using multinomial sampling at 2,000 reads with five repeats per biological sample.
7. The project compares Raw shallow relative abundance, Training-mean relative abundance, Random Forest and XGBoost.
8. Model performance is assessed using OTU-level, whole-community, rank-abundance, diversity, richness, abundance-stratum and shallow-zero recovery metrics.
9. Statistical comparisons, SHAP interpretation and sequencing-depth sensitivity analyses are included.

## Input Data

The original input file is:

```text
data/crosssecdata.RData
```

Inside the RData file there is one table called `datatax` with these columns:

```text
otu_id, count, project_id, sample_id, run_id, nreads, classification
```

The analysis retains records where:

```text
project_id = MGYS00002437
classification = seawater
```

Here, `classification` refers to the broad environmental category in the processed dataset, not OTU taxonomy.

## Languages

```text
Python
# used for data preparation, shallow sequencing simulation, machine learning,
# statistical analysis, SHAP interpretation and plotting
```

## Installation

Clone the repository from GitHub:

```bash
git clone https://github.com/XimanDing02/CMEE_MRes_Project.git
cd CMEE_MRes_Project
```

Optional: create a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
```

Install required packages:

```bash
python3 -m pip install -r requirements.txt
```

## Dependencies

The required Python packages are listed in `requirements.txt`.

```text
numpy==2.4.4
pandas==3.0.2
scipy==1.17.1
scikit-learn==1.9.0
xgboost==3.3.0
joblib==1.5.3
matplotlib==3.10.9
seaborn==0.13.2
shap==0.52.0
pyreadr==0.5.6
```

## Scripts List

| Script Name | Description | Arguments |
|---|---|---|
| `_paths.py` | Stores shared project paths and output helpers | None |
| `00_rdata_to_csv.py` | Converts `crosssecdata.RData` into Python-readable CSV files | Optional RData path |
| `01_prepare_count_matrix.py` | Filters the target project, validates read counts and builds the sample-by-OTU count matrix | Optional input CSV path |
| `02_filter_reference_samples.py` | Keeps samples with at least 10,000 aggregated reads | Optional metadata and count-matrix paths |
| `03_split_samples.py` | Splits eligible biological samples into train, validation and test sets | Optional eligible-sample metadata path |
| `04_select_otu_vocabulary.py` | Selects the training-derived OTU vocabulary | Optional count-matrix and split paths |
| `05_build_reference_matrix.py` | Builds the reference relative-abundance matrix with selected OTUs and `OTHER` | Optional count-matrix, vocabulary and split paths |
| `06_compute_train_statistics.py` | Computes training-only OTU statistics used as model prior features | Optional reference-matrix and split paths |
| `07_generate_supervised_data.py` | Generates shallow-sequencing supervised-learning datasets | Optional reference matrix, split and statistics paths |
| `08_check_stage1_quality.py` | Checks Stage 1 data quality and saves configuration files | Optional Stage 1 file paths |
| `09_evaluate_baselines.py` | Evaluates Raw shallow RA and Training-mean RA baselines | None |
| `10_train_models.py` | Trains Random Forest and XGBoost models | None |
| `11_generate_predictions_and_metrics.py` | Generates predictions and evaluation metrics for all methods | None |
| `12_prepare_statistical_tables.py` | Prepares biological-sample-level statistical tables | None |
| `13_run_statistical_comparisons.py` | Runs Friedman tests, Wilcoxon tests, Holm correction and bootstrap CIs | None |
| `14_compute_shap_results.py` | Computes SHAP values for Random Forest and XGBoost | None |
| `15_prepare_shap_tables.py` | Summarises SHAP results into interpretation tables | None |
| `16_plot_core_figures.py` | Generates the main model-performance figures | None |
| `17_plot_shap_figures.py` | Generates SHAP interpretation figures | None |
| `18_run_depth_sensitivity.py` | Runs sequencing-depth sensitivity analyses | None |
| `19_plot_depth_sensitivity.py` | Plots sequencing-depth sensitivity results | None |
| `20_prepare_revision_outputs.py` | Prepares supplemental revision outputs | None |

## Scripts Basic Usage

All scripts should be run from the project root directory:

```bash
cd CMEE_MRes_Project
```

| Script Name | Basic Usage |
|---|---|
| `00_rdata_to_csv.py` | `python3 code/00_rdata_to_csv.py` |
| `01_prepare_count_matrix.py` | `python3 code/01_prepare_count_matrix.py` |
| `02_filter_reference_samples.py` | `python3 code/02_filter_reference_samples.py` |
| `03_split_samples.py` | `python3 code/03_split_samples.py` |
| `04_select_otu_vocabulary.py` | `python3 code/04_select_otu_vocabulary.py` |
| `05_build_reference_matrix.py` | `python3 code/05_build_reference_matrix.py` |
| `06_compute_train_statistics.py` | `python3 code/06_compute_train_statistics.py` |
| `07_generate_supervised_data.py` | `python3 code/07_generate_supervised_data.py` |
| `08_check_stage1_quality.py` | `python3 code/08_check_stage1_quality.py` |
| `09_evaluate_baselines.py` | `python3 code/09_evaluate_baselines.py` |
| `10_train_models.py` | `python3 code/10_train_models.py` |
| `11_generate_predictions_and_metrics.py` | `python3 code/11_generate_predictions_and_metrics.py` |
| `12_prepare_statistical_tables.py` | `python3 code/12_prepare_statistical_tables.py` |
| `13_run_statistical_comparisons.py` | `python3 code/13_run_statistical_comparisons.py` |
| `14_compute_shap_results.py` | `python3 code/14_compute_shap_results.py` |
| `15_prepare_shap_tables.py` | `python3 code/15_prepare_shap_tables.py` |
| `16_plot_core_figures.py` | `python3 code/16_plot_core_figures.py` |
| `17_plot_shap_figures.py` | `python3 code/17_plot_shap_figures.py` |
| `18_run_depth_sensitivity.py` | `python3 code/18_run_depth_sensitivity.py` |
| `19_plot_depth_sensitivity.py` | `python3 code/19_plot_depth_sensitivity.py` |
| `20_prepare_revision_outputs.py` | `python3 code/20_prepare_revision_outputs.py` |

## Expected Output

| Script Name | Expected Output |
|---|---|
| `00_rdata_to_csv.py` | CSV exports of the RData object and object summary files in `results/intermediate/` |
| `01_prepare_count_matrix.py` | Sample metadata, run-consistency checks and sample-by-OTU count matrix |
| `02_filter_reference_samples.py` | Eligible reference-sample table and filtered count matrix |
| `03_split_samples.py` | Fixed train/validation/test biological-sample split |
| `04_select_otu_vocabulary.py` | Training-derived selected OTU vocabulary |
| `05_build_reference_matrix.py` | Reference relative-abundance matrix with selected OTUs and `OTHER` |
| `06_compute_train_statistics.py` | Training-only OTU statistics table |
| `07_generate_supervised_data.py` | Train, validation and test supervised-learning datasets |
| `08_check_stage1_quality.py` | Stage 1 quality summary and analysis configuration |
| `09_evaluate_baselines.py` | Baseline predictions, row metrics and sample metrics |
| `10_train_models.py` | Trained Random Forest and XGBoost model files and validation metrics |
| `11_generate_predictions_and_metrics.py` | Predictions and full evaluation metric tables |
| `12_prepare_statistical_tables.py` | Test-set biological-sample-level tables and summary statistics |
| `13_run_statistical_comparisons.py` | Friedman tests, Wilcoxon tests, Holm-adjusted results and bootstrap intervals |
| `14_compute_shap_results.py` | SHAP sample rows, SHAP values and additivity checks |
| `15_prepare_shap_tables.py` | Global, subgroup, zero-recovery and local SHAP summary tables |
| `16_plot_core_figures.py` | Main model-performance figures in PNG, PDF and SVG formats |
| `17_plot_shap_figures.py` | SHAP interpretation figures in PNG, PDF and SVG formats |
| `18_run_depth_sensitivity.py` | Cross-depth performance, model-selection and improvement tables |
| `19_plot_depth_sensitivity.py` | Sequencing-depth sensitivity figures |
| `20_prepare_revision_outputs.py` | Supplemental revision tables, manifest and figures |

Main result folders include:

```text
results/prepared/
results/baseline/
results/models/
results/intermediate/
results/statistics/
results/shap/
results/figures/
results/depth_sensitivity/
results/revision_outputs/
```

## Author and Contact

Name: Ximan Ding

Email: x.ding25@imperial.ac.uk

Institution: Imperial College London

Programme: CMEE MRes