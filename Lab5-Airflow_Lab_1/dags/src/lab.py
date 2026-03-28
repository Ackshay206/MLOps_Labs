import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64


def load_data():
    """
    Loads mall customer data from CSV, serializes it, and returns base64-encoded string.
    """
    print("Loading mall customer segmentation dataset...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    print(f"Loaded {len(df)} customers with columns: {list(df.columns)}")
    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Decodes data, selects AGE/ANNUAL_INCOME/SPENDING_SCORE features,
    applies StandardScaler, and returns base64-encoded payload with
    both the scaled data and the fitted scaler (so predictions can be
    made consistently in the final task).
    """
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["AGE", "ANNUAL_INCOME", "SPENDING_SCORE"]]

    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)
    print(f"Scaled {len(clustering_data_scaled)} rows using StandardScaler.")

    # Bundle scaled data + fitted scaler so build_save_model can persist the scaler
    payload = {"data": clustering_data_scaled, "scaler": scaler}
    serialized = pickle.dumps(payload)
    return base64.b64encode(serialized).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Trains KMeans for k=1..11, saves all models + the scaler to disk,
    and returns the SSE list for the elbow method.
    Uses k-means++ initialisation (better than random for convergence).
    """
    data_bytes = base64.b64decode(data_b64)
    payload = pickle.loads(data_bytes)
    df = payload["data"]
    scaler = payload["scaler"]

    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 10,
        "max_iter": 300,
        "random_state": 42,
    }

    sse = []
    models = {}
    for k in range(1, 12):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)
        models[k] = kmeans
        print(f"  k={k:2d}  SSE={kmeans.inertia_:.2f}")

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Save all trained models + the scaler so the final task can pick the best k
    save_payload = {"models": models, "scaler": scaler}
    with open(output_path, "wb") as f:
        pickle.dump(save_payload, f)
    print(f"Saved models and scaler to {output_path}")

    return sse


def load_model_elbow(filename: str, sse: list):
    """
    Loads saved models + scaler, finds the optimal k via the elbow method,
    predicts clusters for test.csv using that best model, and prints the result.
    """
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    save_payload = pickle.load(open(output_path, "rb"))
    models = save_payload["models"]
    scaler = save_payload["scaler"]

    # Elbow method to find optimal number of clusters
    kl = KneeLocator(range(1, 12), sse, curve="convex", direction="decreasing")
    optimal_k = kl.elbow
    print(f"Optimal number of clusters (elbow method): {optimal_k}")

    best_model = models[optimal_k]

    # Load test data, apply the same scaler, then predict
    df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    df_test_scaled = scaler.transform(df_test[["AGE", "ANNUAL_INCOME", "SPENDING_SCORE"]])
    predictions = best_model.predict(df_test_scaled)
    print(f"Test customer cluster predictions: {predictions}")

    try:
        return int(predictions[0])
    except Exception:
        return predictions[0].item() if hasattr(predictions[0], "item") else predictions[0]
