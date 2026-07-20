# MAC4112 Data Science and Software Engineering in Manufacturing 

## Overview
This project was developed as part of the **MAC4112 Data Science and Research Software** module at the University of Sheffield.
The software provides a complete data analysis pipeline for the diagnosis of machine tool and machine process health using multi-sensor data collected from CNC milling experiments. Raw MATLAB (.mat) files are processed through a reproducible workflow consisting of feature extraction, data cleaning, feature standardisation, anomaly detection, PCA, and machine learning classification.

## Objectives
The objectives of this software are to:
- Load and process raw machining sensor data stored in MATLAB files.
- Extract meaningful features from multiple sensors.
- Clean and preprocess these extracted features.
- Detect and remove anomalous runs.
- Reduce feature dimensionality using Principle Componenet Analysis (PCA).
- Train and compare multiple machine learning (ML) classifiers.

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

## Extracted Features
Statistical Features: Mean, Root Mean Square (RMS), MAximum Peak, Skewness, Kurtosis
Dimensionless Features: Crest Factor, Shape Factor, Impulse Factor, Margin Factor.
Energy Feature: Signal Energy.
Time-Frequency Features: Spectral Kurtosis, Continuous Wavelet Transform Energy Bands.







This repository contains the code used for the research software requirements section of the module.
The goal of the code is to predict when manufacturing and machine defects occur.

This README file needs editing further when more is aded to the project files.

Need to add instructions on how to use the software, what needs downloading to use it etc. 





pip install pandas numpy mat73 scikit-learn seaborn ensemble PyWavelets pytest 
Need to create a data and results fodler seperately 





This project is licensed under the MIT Liscense - see the LICENSE file for complete details.