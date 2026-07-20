# MAC4112 Data Science and Software Engineering in Manufacturing 
![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)
## Overview
This project was developed as part of the **MAC4112 Data Science and Research Software** module at the University of Sheffield.
The software provides a complete data analysis pipeline for the diagnosis of machine tool and machine process health using multi-sensor data collected from CNC milling experiments. Raw MATLAB (.mat) files are processed through a reproducible workflow consisting of feature extraction, data cleaning, feature standardisation, anomaly detection, PCA, and machine learning classification.

---

## Objectives
The objectives of this software are to:
- Load and process raw machining sensor data stored in MATLAB files.
- Extract meaningful features from multiple sensors.
- Clean and preprocess these extracted features.
- Detect and remove anomalous runs.
- Reduce feature dimensionality using Principle Componenet Analysis (PCA).
- Train and compare multiple machine learning (ML) classifiers.

---

## Repository Structure

MAC4112_DATA_SCIENCE/
│
├── data/                     # Raw MATLAB datasets (not tracked by Git)
├── results/                  # Generated CSV files and figures (not tracked by Git)
│
├── src/
│   ├── features.py           # Feature extraction
│   ├── clean_data.py         # Missing value handling
│   ├── Standardise.py        # Feature scaling
│   ├── remove_outliers.py    # Isolation Forest outlier detection
│   ├── PCA.py                # Principal Component Analysis
│   └── classifier.py         # Machine learning models
│
├── Tests/
│   ├── test_pipeline.py      # Pytest unit tests
│   └── test_notebook.ipynb   # Snippets of code being developed
│
├── main.py                   # Main command line entry point
├── pytest.ini
├── .gitignore
├── LICENSE
└── README.md

---

## Software Pipeline
```
Raw MATLAB files (.mat)
            │
            ▼
Feature Extraction
            │
            ▼
Missing Value Imputation
            │
            ▼
Feature Standardisation
            │
            ▼
Isolation Forest Outlier Removal
            │
            ▼
Principal Component Analysis (PCA)
            │
            ▼
Machine Learning Classification
            │
            ▼
Performance Evaluation & Visualisation
```

---

## Extracted Features
Statistical Features: Mean, Root Mean Square (RMS), MAximum Peak, Skewness, Kurtosis
Dimensionless Features: Crest Factor, Shape Factor, Impulse Factor, Margin Factor.
Energy Feature: Signal Energy.
Time-Frequency Features: Spectral Kurtosis, Continuous Wavelet Transform Energy Bands.

---

## Machine Learning Models
The followign algorithms are evaluated:
- Random Forest
- Decision Tree
- Support Vector Machining
- k-Nearest Neighbours

Eaach is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## Installation
Clone the repository:
```bash
git clone https://github.com/gry2109/MAC4112_Data_Science.git
```
Move into the project folder :
```bash
cd MAC4112_DATA_SCIENCE
```
Create a virtual environment (recommended)

Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```
Install dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset
The dataset is **NOT INCLUDED** within this repository due to size and licensing.
Download the dataset from:
**Sensor signals for machine tool and process health assessment** Dominguez Caballero, J. A., Moore, J., & Stammers, J. (2023)
DOI: https://doi.org/10.15131/shef.data.24125715.v1

After downloading, place all `.mat` files inside the `data/` directory

---

## Usage
### Run specific datasets
```bash
python main.py --datasets Segmented_Linear_Baseline.mat Segmented_Machining_Baseline.mat
```
### Run using keyword matching
```bash
python main.py --datasets Baseline
```
### Run all available datasets
```bash
python main.py --datasets all
```
All code was ran through VS Code 
**At least two datasets must be selected for comparison.**

---

## Outputs
Running the pipeline automatically generates:
### CSV files
- master_features.csv
- master_features_cleaned.csv
- master_features_standardised.csv
- master_features_no_outliers.csv
- master_pca_features.csv
### Figures
- PCA Scatter Plot
- Confusion Matrix (one per classifier)
- Classifier Performance Comparison

All outputs are written to the `results/` directory.

---

## Testing
Automated testing is implemented using **pytest**.
Run all tests using:
```bash
pytest
```
The test suite validates:
- Feature extraction
- RMS calculations
- Data loading
- PCA outputs
- Missing value handling
- Machine learning pipeline
- Output figure generation

---

## Reproducibility
The software has been designed to support reproducible research through:
- Modular function design
- Fixed random seeds (`random_state = 42`)
- Version control using Git
- Public GitHub repository
- Automated testing with pytest
- Clear documentation
- Command-line interface
- Separation of raw data from source code

---

Key Python packages included in the requirments file:

- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- seaborn
- pywavelets
- mat73
- pytest

---

## License
This project is released under the MIT License.
See the LICENSE file for further details.

---

## Author
George Riley
PhD student
The University of Sheffield

---

##Acknowledgements
This work was completed as part of the **MAC4112 Data Science and Research Software** module.
Dataset provided by: Dominguez Caballero, J. A., Moore, J., & Stammers, J.
Teaching provided by Lindsay Lee and the RSE team: https://rse.shef.ac.uk/

The University of Sheffield.

---
