"""
evaluate_model.py - Breast Cancer Model Evaluation
Evaluates GradientBoostingClassifier with comprehensive metrics
"""
import pickle
import os
import json
import joblib
import argparse
import sys
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)

sys.path.insert(0, os.path.abspath('..'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True,
                        help="Timestamp from GitHub Actions")
    args = parser.parse_args()

    # Access the timestamp
    timestamp = args.timestamp
    print(f"Timestamp received: {timestamp}")

    # Load the model
    try:
        model_version = f'model_{timestamp}_gb_model'
        model = joblib.load(f'{model_version}.joblib')
        print(f"Loaded model: {model_version}")
    except:
        raise ValueError('Failed to load the latest model')

    # Load test data
    try:
        with open('data/X_test.pickle', 'rb') as f:
            X_test = pickle.load(f)

        with open('data/y_test.pickle', 'rb') as f:
            y_test = pickle.load(f)

        print(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    except:
        raise ValueError('Failed to load the test data')

    # Make predictions
    y_predict = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_predict)
    tn, fp, fn, tp = cm.ravel()

    # Confidence analysis
    confidence = np.where(y_predict == 1, y_proba, 1 - y_proba)
    correct_mask = y_predict == y_test
    mean_confidence = float(np.mean(confidence))
    mean_confidence_correct = float(np.mean(confidence[correct_mask]))
    mean_confidence_incorrect = float(np.mean(confidence[~correct_mask])) if (~correct_mask).sum() > 0 else 0

    # Build metrics dictionary
    metrics = {
        "F1_Score": f1,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "ROC_AUC": roc_auc,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "confidence_analysis": {
            "mean_confidence": mean_confidence,
            "mean_confidence_correct": mean_confidence_correct,
            "mean_confidence_incorrect": mean_confidence_incorrect
        },
        "test_samples": int(len(y_test)),
        "correct_predictions": int(correct_mask.sum()),
        "incorrect_predictions": int((~correct_mask).sum())
    }

    # Print results
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")

    print(f"\nClassification Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")

    print(f"\nConfidence Analysis:")
    print(f"  Mean Confidence (all):       {mean_confidence:.4f}")
    print(f"  Mean Confidence (correct):   {mean_confidence_correct:.4f}")
    print(f"  Mean Confidence (incorrect): {mean_confidence_incorrect:.4f}")

    print(f"\nPredictions: {metrics['correct_predictions']}/{metrics['test_samples']} correct")
    print(f"{'='*50}")

    # Save metrics to JSON file
    if not os.path.exists('metrics/'):
        os.makedirs("metrics/")

    with open(f'{timestamp}_metrics.json', 'w') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    print(f"\nMetrics saved: {timestamp}_metrics.json")