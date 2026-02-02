# Breast Cancer Classification with GitHub Actions MLOps Pipeline


## Project Overview

This Lab Assignment demonstrates an automated Machine Learning pipeline using **GitHub Actions** for continuous training and evaluation. The pipeline trains a **Gradient Boosting Classifier** on the **Breast Cancer Wisconsin (Diagnostic)** dataset to classify tumors as malignant or benign.

### Key Features

- **Automated Model Training**: Triggers on every push to main branch
- **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
- **Model Versioning**: Timestamped model files for tracking
- **Experiment Tracking**: MLflow integration for logging parameters and metrics
- **Scheduled Retraining**: Daily model calibration via cron jobs

## Dataset

**Breast Cancer Wisconsin (Diagnostic) Dataset**

| Attribute | Value |
|-----------|-------|
| Source | UCI Machine Learning Repository |
| Samples | 569 |
| Features | 30 numeric features |
| Classes | Malignant (212), Benign (357) |
| Task | Binary Classification |

Features are computed from digitized images of fine needle aspirates (FNA) of breast masses, describing characteristics of cell nuclei.

## Model Architecture

### Algorithm: Gradient Boosting Classifier

| Hyperparameter | Search Space | Best Value |
|----------------|--------------|------------|
| n_estimators | [50, 100, 150] | 50 |
| learning_rate | [0.05, 0.1, 0.2] | 0.2 |
| max_depth | [3, 4, 5] | 3 |
| min_samples_split | [2, 5] | 2 |

### Pipeline Components

```
Raw Data → StandardScaler → Train/Test Split → GridSearchCV → GradientBoostingClassifier
```

## Project Structure

```
MLOps_Labs/
├── .github/
│   └── workflows/
│       ├── model_calibration_on_push.yml   # Triggers on push to main
│       └── model_calibration.yml           # Scheduled daily training
│
└── Lab1-Github_actions_lab2/
    ├── src/
    │   ├── __init__.py
    │   ├── train_model.py                  # Training script with GridSearchCV
    │   └── evaluate_model.py               # Evaluation with comprehensive metrics
    ├── models/                             # Saved model files (.joblib)
    ├── metrics/                            # Evaluation metrics (.json)
    ├── data/                               # Pickled data files
    └── requirements.txt                    # Python dependencies
```

## GitHub Actions Workflow

### Workflow Triggers

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `model_calibration_on_push.yml` | Push to main | Train on code changes |
| `model_calibration.yml` | Daily at midnight UTC | Scheduled retraining |

### Pipeline Steps

1. **Checkout Code** - Clone repository
2. **Setup Python** - Install Python 3.9
3. **Install Dependencies** - Install required packages
4. **Generate Timestamp** - Create unique version identifier
5. **Train Model** - Run GridSearchCV and train best model
6. **Evaluate Model** - Generate comprehensive metrics
7. **Store Artifacts** - Move model and metrics to designated folders
8. **Commit & Push** - Version control trained artifacts

## How to Run Locally

### Prerequisites

```bash
Python 3.9+
pip
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Ackshay206/MLOps_Labs.git
cd MLOps_Labs/Lab1-Github_actions_lab2

# Install dependencies
pip install -r requirements.txt
```

### Run Training

```bash
# Generate timestamp
timestamp=$(date '+%Y%m%d%H%M%S')

# Train model
python src/train_model.py --timestamp "$timestamp"

# Evaluate model
python src/evaluate_model.py --timestamp "$timestamp"
```

## Technologies Used

- **Python 3.9** - Programming language
- **scikit-learn** - Machine learning library
- **MLflow** - Experiment tracking
- **GitHub Actions** - CI/CD automation
- **Joblib** - Model serialization

## Modifications from Original Lab

This project is modified from the original lab template with the following changes:

| Component | Original | Modified |
|-----------|----------|----------|
| Dataset | Synthetic (`make_classification`) | Real (Breast Cancer Wisconsin) |
| Model | RandomForestClassifier | GradientBoostingClassifier |
| Tuning | None | GridSearchCV (54 combinations) |
| Validation | None | 5-fold Cross-Validation |
| Preprocessing | None | StandardScaler |
| Metrics | F1 Score only | Accuracy, F1, Precision, Recall, ROC-AUC, Confusion Matrix |



## Author

**Ackshay206**

---

*Built with ❤️ using GitHub Actions for automated ML pipelines*