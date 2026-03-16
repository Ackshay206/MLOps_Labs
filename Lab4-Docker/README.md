# Lab 4 - Dockerized ML Pipeline: Iris Classification with Gradient Boosting

## Overview
This lab demonstrates how to containerize a machine learning pipeline using Docker. A Gradient Boosting classifier is trained on the classic Iris dataset, evaluated with standard metrics, and the trained model is saved as a `.pkl` file — all within a Docker container.

## What This Lab Does
1. Loads the Iris dataset (150 samples, 4 features, 3 classes)
2. Splits the data into 70% training and 30% testing sets
3. Trains a **Gradient Boosting Classifier** with 150 estimators
4. Evaluates the model and prints accuracy, a classification report, and feature importances
5. Saves the trained model to `gb_iris_model.pkl` using joblib

## Modifications from the Original Template
- Replaced **Random Forest** with **Gradient Boosting** for improved sequential learning
- Changed the train/test split from 80/20 to **70/30**
- Added **model evaluation** (accuracy score and full classification report)
- Added **feature importance logging** to understand which features drive predictions
- Added **numpy** as a dependency in `requirements.txt`

## Project Structure
```
Lab4-Docker/
├── dockerfile
├── ReadMe.md
└── src/
    ├── main.py
    └── requirements.txt
```

## How to Run

### Build the Docker image
```bash
docker build -t iris-gb-model:v1 .
```

### Run the container
```bash
docker run iris-gb-model:v1
```

### (Optional) Save the image as a tar file
```bash
docker save iris-gb-model:v1 > iris_gb_image.tar
```

## Expected Output
```
Test Accuracy: 0.9556
Classification Report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        19
  versicolor       0.90      0.95      0.92        19
   virginica       0.94      0.88      0.91         7

Feature Importances:
  sepal length (cm): 0.0262
  sepal width (cm): 0.0148
  petal length (cm): 0.4392
  petal width (cm): 0.5198

Model saved as gb_iris_model.pkl
```

## Technologies Used
- Python 3.10
- scikit-learn (Gradient Boosting Classifier)
- joblib (model serialization)
- Docker (containerization)# Lab 4 - Dockerized ML Pipeline: Iris Classification with Gradient Boosting

## Overview
This lab demonstrates how to containerize a machine learning pipeline using Docker. A Gradient Boosting classifier is trained on the classic Iris dataset, evaluated with standard metrics, and the trained model is saved as a `.pkl` file — all within a Docker container.

## What This Lab Does
1. Loads the Iris dataset (150 samples, 4 features, 3 classes)
2. Splits the data into 70% training and 30% testing sets
3. Trains a **Gradient Boosting Classifier** with 150 estimators
4. Evaluates the model and prints accuracy, a classification report, and feature importances
5. Saves the trained model to `gb_iris_model.pkl` using joblib

## Modifications from the Original Template
- Replaced **Random Forest** with **Gradient Boosting** for improved sequential learning
- Changed the train/test split from 80/20 to **70/30**
- Added **model evaluation** (accuracy score and full classification report)
- Added **feature importance logging** to understand which features drive predictions
- Added **numpy** as a dependency in `requirements.txt`

## Project Structure
```
Lab4-Docker/
├── dockerfile
├── ReadMe.md
└── src/
    ├── main.py
    └── requirements.txt
```

## How to Run

### Build the Docker image
```bash
docker build -t iris-gb-model:v1 .
```

### Run the container
```bash
docker run iris-gb-model:v1
```

### (Optional) Save the image as a tar file
```bash
docker save iris-gb-model:v1 > iris_gb_image.tar
```

## Expected Output
```
Test Accuracy: 0.9556
Classification Report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        19
  versicolor       0.90      0.95      0.92        19
   virginica       0.94      0.88      0.91         7

Feature Importances:
  sepal length (cm): 0.0262
  sepal width (cm): 0.0148
  petal length (cm): 0.4392
  petal width (cm): 0.5198

Model saved as gb_iris_model.pkl
```

## Technologies Used
- Python 3.10
- scikit-learn (Gradient Boosting Classifier)
- joblib (model serialization)
- Docker (containerization)