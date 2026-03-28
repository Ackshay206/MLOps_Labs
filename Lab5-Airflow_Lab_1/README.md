# Airflow Lab 1 — Mall Customer Segmentation


This lab builds a machine-learning pipeline using **Apache Airflow** and **Docker** to segment mall customers into groups using K-Means clustering and the elbow method.

---

## What I Changed from the Original Lab

| Item | Original | My Version |
|------|----------|------------|
| Dataset | Credit card transactions (8,950 rows) | Mall customer data (200 rows) |
| Features | BALANCE, PURCHASES, CREDIT_LIMIT | AGE, ANNUAL_INCOME, SPENDING_SCORE |
| Scaler | MinMaxScaler | StandardScaler |
| KMeans init | `random` | `k-means++` (faster, better convergence) |
| k range | 1 – 49 | 1 – 11 |
| Model saved | Only last model (k=49) | All k models + scaler saved together |
| Test predictions | Raw unscaled data fed to model (bug) | Test data properly scaled before prediction |
| DAG name | `Airflow_Lab1` | `Mall_Customer_Segmentation` |

---

## Project Structure

```
Lab5-Airflow_Lab_1/
├── dags/
│   ├── airflow.py          # Airflow DAG definition
│   ├── data/
│   │   ├── file.csv        # Mall customer training data (200 rows)
│   │   └── test.csv        # Two test customers for prediction
│   ├── model/              # Auto-created — stores model.sav (gitignored)
│   └── src/
│       ├── __init__.py
│       └── lab.py          # ML pipeline functions
├── logs/                   # Airflow task logs (gitignored)
├── plugins/                # Airflow plugins dir (gitignored)
├── config/                 # Airflow config (gitignored)
├── .env                    # Credentials and pip packages (gitignored)
├── .gitignore
├── docker-compose.yaml     # Fetched from Apache Airflow (gitignored)
├── setup.sh
└── README.md
```

---

## DAG Pipeline

The DAG `Mall_Customer_Segmentation` runs 4 tasks in sequence:

```
load_data_task  >>  data_preprocessing_task  >>  build_save_model_task  >>  load_model_task
```

| Task | What it does |
|------|-------------|
| `load_data_task` | Reads `file.csv`, serializes with pickle + base64 for XCom |
| `data_preprocessing_task` | Selects AGE/ANNUAL_INCOME/SPENDING_SCORE, applies StandardScaler |
| `build_save_model_task` | Trains KMeans for k=1–11 (k-means++ init), saves all models + scaler to `model.sav` |
| `load_model_task` | Finds optimal k via elbow method, scales test data, predicts customer clusters |

---

## ML Functions (`dags/src/lab.py`)

**`load_data()`** — Loads `file.csv` into a DataFrame, returns base64-encoded pickled data.

**`data_preprocessing(data_b64)`** — Drops nulls, selects the 3 features, fits a StandardScaler, returns scaled data + scaler bundled together.

**`build_save_model(data_b64, filename)`** — Trains KMeans for each k from 1 to 11, collects SSE for elbow method, saves all models and the scaler to `dags/model/filename`.

**`load_model_elbow(filename, sse)`** — Finds the elbow k using KneeLocator, loads the best model, applies the saved scaler to test data, and predicts which cluster each test customer belongs to.

---

## How to Run (Docker only — no local Python install needed)

### Prerequisites
- Docker Desktop installed and running (allocate at least 4 GB RAM)

### Step 1 — Go to the lab folder

```bash
cd /path/to/Lab5-Airflow_Lab_1
```

### Step 2 — Download docker-compose.yaml

```bash
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml'
```

### Step 3 — Create required directories

```bash
mkdir -p ./logs ./plugins
```

### Step 4 — Create `.env` file

```bash
cat > .env << 'EOF'
AIRFLOW_UID=50000
_AIRFLOW_WWW_USER_USERNAME=admin
_AIRFLOW_WWW_USER_PASSWORD=admin123
_PIP_ADDITIONAL_REQUIREMENTS=pandas scikit-learn kneed
EOF
```

### Step 5 — Edit `docker-compose.yaml`

Find the `environment:` block under `x-airflow-common` and add:

```yaml
AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
AIRFLOW__CORE__ENABLE_XCOM_PICKLING: 'true'
```

### Step 6 — Initialize the database (one-time, ~2 min)

```bash
docker compose up airflow-init
```

Wait for: `airflow-init exited with code 0`

### Step 7 — Start Airflow

```bash
docker compose up
```

Wait until you see:
```
airflow-webserver-1 | 127.0.0.1 - - [...] "GET /health HTTP/1.1" 200
```

### Step 8 — Open the UI

Go to **http://localhost:8080** and log in:
- Username: `admin`
- Password: `admin123`

### Step 9 — Trigger the DAG

1. Find `Mall_Customer_Segmentation` in the DAGs list
2. Toggle it **On**
3. Click the **Play (▶)** button → Trigger DAG

### Step 10 — View results

Click `Mall_Customer_Segmentation` → **Graph** tab → click `load_model_task` → **Logs** tab.

You will see output like:
```
Optimal number of clusters (elbow method): 5
Test customer cluster predictions: [1 2]
```

### Step 11 — Shut down

Open a new terminal (current one is occupied by Airflow logs):

```bash
docker compose down
```

---

## Dataset

The `file.csv` training data contains 200 synthetic mall customers with 5 natural segments:

| Cluster | Age | Income | Spending |
|---------|-----|--------|----------|
| 1 | Young (19–33) | Low (15–40k) | High (66–95) |
| 2 | Young/Mid (27–38) | High (73–137k) | High (72–92) |
| 3 | Mid/Older (42–65) | Low (15–40k) | Low (5–19) |
| 4 | Older (51–70) | High (75–137k) | Low (7–35) |
| 5 | Mid (35–50) | Medium (50–70k) | Medium (42–65) |

The `test.csv` contains 2 test customers the trained model predicts clusters for.
