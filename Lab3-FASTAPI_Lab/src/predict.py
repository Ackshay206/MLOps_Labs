import joblib

def predict_data(X):
    """
    Predict the class labels for the input data.
    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        y_pred (numpy.ndarray): Predicted class labels (0 = malignant, 1 = benign).
    """
    model = joblib.load("../model/breast_cancer_model.pkl")
    y_pred = model.predict(X)
    return y_pred
