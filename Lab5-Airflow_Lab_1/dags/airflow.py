# Import necessary libraries and modules
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

# NOTE:
# In Airflow 3.x, XCom pickling is enabled via environment variable in docker-compose.yaml:
# AIRFLOW__CORE__ENABLE_XCOM_PICKLING: 'true'

# Define default arguments for your DAG
default_args = {
    'owner': 'ackshay',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# Create a DAG for Mall Customer Segmentation using K-Means clustering
with DAG(
    'Mall_Customer_Segmentation',
    default_args=default_args,
    description='K-Means customer segmentation pipeline on Mall Customers dataset',
    catchup=False,
) as dag:

    # Task 1: Load customer data from CSV
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    # Task 2: Preprocess data — select features, apply StandardScaler
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    # Task 3: Train KMeans (k=1..11) with k-means++ init, save models + scaler
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.sav"],
    )

    # Task 4: Find optimal k via elbow method, predict clusters for test customers
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=["model.sav", build_save_model_task.output],
    )

    # Set task dependencies (linear pipeline)
    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

if __name__ == "__main__":
    dag.test()
