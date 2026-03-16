# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

if __name__ == '__main__':
    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data into training and testing sets (70/30 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7
    )

    # Train a Gradient Boosting classifier
    model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=3, random_state=7
    )
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

    # Print feature importances
    print("Feature Importances:")
    for name, imp in zip(iris.feature_names, model.feature_importances_):
        print(f"  {name}: {imp:.4f}")

    # Save the model to a file
    joblib.dump(model, 'gb_iris_model.pkl')

    print("\nModel saved as gb_iris_model.pkl")
