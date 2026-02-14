# Lab 2: Advanced Data Drift Detection with Evidently AI

## Overview

This lab demonstrates **data drift detection and monitoring** in machine learning systems using Evidently AI. It explores how production data distributions change over time and why continuous monitoring is critical for maintaining model performance.

### Dataset: Wine Quality
- **Source**: UCI Wine Quality Dataset (Red Wine)
- **Features**: 11 physicochemical properties (pH, alcohol, acidity, sulfur dioxide, etc.)
- **Target**: Wine quality rating (0-10)
- **Size**: 1,599 samples

---

## What This Lab Does

### 1. **Feature Engineering**
Creates derived features to simulate real-world ML pipelines:
- `acidity_ratio`: Fixed/Volatile acidity ratio
- `sulfur_ratio`: Free/Total sulfur dioxide ratio
- `alcohol_sugar_interaction`: Product of alcohol × sugar
- `quality_indicator`: Binary high/low alcohol classifier

### 2. **Temporal Drift Simulation**
Simulates three production time periods with increasing drift:

| Scenario | Timeline | Drift Type | Severity |
|----------|----------|------------|----------|
| **Scenario 1** | Week 1 | Minimal measurement noise | Stable |
| **Scenario 2** | Week 4 | pH shift (+0.3), sulfur variability | Moderate |
| **Scenario 3** | Week 12 | pH (+0.5), alcohol (+1.0), 8% missing data |  Critical |

### 3. **Multi-Dimensional Monitoring**
Tracks both:
- **Data Drift**: Distribution changes in features (using `DataDriftPreset`)
- **Data Quality**: Missing values, statistics, anomalies (using `DataSummaryPreset`)

### 4. **Comparative Analysis**
- Monitors drift evolution over time
- Visualizes drift progression with matplotlib
- Generates actionable insights for each scenario

---

## Key Differences from Original Lab

| Aspect | Original Lab | This Modified Lab |
|--------|--------------|-------------------|
| **Dataset** | Adult Census | Wine Quality |
| **Drift Creation** | Education-based split | Synthetic noise + systematic shifts |
| **Time Dimension** | Single comparison | 3 temporal scenarios |
| **Features** | Raw features only | Raw + 4 engineered features |
| **Monitoring** | Drift only | Drift + Quality (DataSummaryPreset) |
| **Analysis** | Basic report | Comparative analysis + visualization |




## Prerequisites

### Required
- Python 3.11.x (recommended) or 3.10.x


### Python Packages
```bash
pip install evidently==0.7.0
pip install pandas numpy scikit-learn matplotlib
```

## Installation and setup

```bash
# Install Python 3.11.7
pyenv install 3.11.7
pyenv local 3.11.7

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  

# Verify Python version
python --version 
```
## Lab structure

``` bash
Lab2-Data_Monitoring-Evidently_AI/
├── Lab2_Modified.ipynb          
├── README.md                    
├── .python-version             
├── .venv/                       
└── outputs/                     
    ├── scenario1_early_production.html
    ├── scenario2_mid_production.html
    └── scenario3_late_production.html
```
