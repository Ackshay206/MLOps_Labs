"""
train_model.py - Breast Cancer Classification
Dataset: Breast Cancer Wisconsin (Diagnostic)
Model: GradientBoostingClassifier with GridSearchCV
"""
import mlflow
import datetime
import os
import pickle
import argparse
import sys
from joblib import dump
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, os.path.abspath('..'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    # Access the timestamp
    timestamp = args.timestamp

    # Use the timestamp in your script
    print(f"Timestamp received from GitHub Actions: {timestamp}")

    # Load Breast Cancer dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names.tolist()
    target_names = data.target_names.tolist()

    print(f"\nDataset: Breast Cancer Wisconsin (Diagnostic)")
    print(f"Total samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {target_names}")
    print(f"Class distribution: Malignant={sum(y==0)}, Benign={sum(y==1)}")

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    train_X, test_X, train_y, test_y = train_test_split(
        X_scaled, y,
        test_size=0.2,
        shuffle=True,
        random_state=42,
        stratify=y
    )

    # Save data (matching professor's style)
    if not os.path.exists('data'):
        os.makedirs('data/')

    with open('data/data.pickle', 'wb') as f:
        pickle.dump(X_scaled, f)

    with open('data/target.pickle', 'wb') as f:
        pickle.dump(y, f)

    # Save test split for evaluation
    with open('data/X_train.pickle', 'wb') as f:
        pickle.dump(train_X, f)

    with open('data/y_train.pickle', 'wb') as f:
        pickle.dump(train_y, f)

    # Save test split for evaluation
    with open('data/X_test.pickle', 'wb') as f:
        pickle.dump(test_X, f)

    with open('data/y_test.pickle', 'wb') as f:
        pickle.dump(test_y, f)

    # MLflow setup
    mlflow.set_tracking_uri("./mlruns")
    dataset_name = "Breast Cancer Wisconsin"
    current_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    experiment_name = f"{dataset_name}_{current_time}"
    experiment_id = mlflow.create_experiment(f"{experiment_name}")

    with mlflow.start_run(experiment_id=experiment_id,
                          run_name=f"{dataset_name}"):

        params = {
            "dataset_name": dataset_name,
            "number of datapoint": X.shape[0],
            "number of dimensions": X.shape[1],
            "test_size": 0.2,
            "model": "GradientBoostingClassifier"
        }

        mlflow.log_params(params)

        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5]
        }

        # GridSearchCV for hyperparameter tuning
        print("\nPerforming GridSearchCV for hyperparameter tuning...")
        gb = GradientBoostingClassifier(random_state=42)

        grid_search = GridSearchCV(
            gb,
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(train_X, train_y)

        # Get best model and params
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_cv_score = grid_search.best_score_

        print(f"\nBest parameters: {best_params}")
        print(f"Best CV F1 Score: {best_cv_score:.4f}")

        # Log best hyperparameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"best_{param_name}", param_value)

        # Cross-validation scores
        cv_scores = cross_val_score(best_model, train_X, train_y, cv=5, scoring='f1')
        print(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

        # Predictions
        y_train_predict = best_model.predict(train_X)
        y_test_predict = best_model.predict(test_X)
        y_test_proba = best_model.predict_proba(test_X)[:, 1]

        # Calculate metrics
        train_metrics = {
            'train_accuracy': accuracy_score(train_y, y_train_predict),
            'train_f1': f1_score(train_y, y_train_predict),
            'train_precision': precision_score(train_y, y_train_predict),
            'train_recall': recall_score(train_y, y_train_predict)
        }

        test_metrics = {
            'test_accuracy': accuracy_score(test_y, y_test_predict),
            'test_f1': f1_score(test_y, y_test_predict),
            'test_precision': precision_score(test_y, y_test_predict),
            'test_recall': recall_score(test_y, y_test_predict),
            'test_roc_auc': roc_auc_score(test_y, y_test_proba)
        }

        cv_metrics = {
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std()
        }

        # Log all metrics
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)
        mlflow.log_metrics(cv_metrics)

        # Print results
        print(f"\n{'='*50}")
        print("TRAINING RESULTS")
        print(f"{'='*50}")

        print(f"\nTrain Metrics:")
        print(f"  Accuracy:  {train_metrics['train_accuracy']:.4f}")
        print(f"  F1 Score:  {train_metrics['train_f1']:.4f}")
        print(f"  Precision: {train_metrics['train_precision']:.4f}")
        print(f"  Recall:    {train_metrics['train_recall']:.4f}")

        print(f"\nTest Metrics:")
        print(f"  Accuracy:  {test_metrics['test_accuracy']:.4f}")
        print(f"  F1 Score:  {test_metrics['test_f1']:.4f}")
        print(f"  Precision: {test_metrics['test_precision']:.4f}")
        print(f"  Recall:    {test_metrics['test_recall']:.4f}")
        print(f"  ROC AUC:   {test_metrics['test_roc_auc']:.4f}")

        # Feature importance
        feature_importance = dict(zip(feature_names, best_model.feature_importances_.tolist()))
        feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

        print(f"\nTop 5 Important Features:")
        for i, (feat, imp) in enumerate(list(feature_importance.items())[:5]):
            print(f"  {i+1}. {feat}: {imp:.4f}")

        # Save model
        if not os.path.exists('models/'):
            os.makedirs("models/")

        model_version = f'model_{timestamp}'
        model_filename = f'{model_version}_gb_model.joblib'
        dump(best_model, model_filename)

        print(f"\nModel saved: {model_filename}")
        print(f"{'='*50}")