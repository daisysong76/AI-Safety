from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from data.scripts.preprocess_data import preprocess_dataset
from models.training_scripts.fine_tune import fine_tune_model

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1),
    "retries": 1,
}

with DAG("data_and_model_pipeline", default_args=default_args, schedule_interval=None) as dag:
    preprocess_task = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_dataset,
        op_kwargs={
            "input_path": "./data/raw/raw_data.csv",
            "output_path": "./data/processed/cleaned_data.csv",
        },
    )

    train_task = PythonOperator(
        task_id="fine_tune_model",
        python_callable=fine_tune_model,
    )

    preprocess_task >> train_task  # Define task dependency


# How to Use This DAG
# Place your raw data file in the data/raw/ directory (e.g., raw_data.csv).
# Define the preprocessing logic in the preprocess_dataset function in data/scripts/preprocess_data.py.
# Define the fine-tuning logic in the fine_tune_model function in models/training_scripts/fine_tune.py.
# Run the DAG using the Airflow CLI:
# airflow dags trigger data_and_model_pipeline
# The DAG will execute the preprocessing task followed by the training task.
# Advantages of This Approach
# Scalability: Easily add more tasks or scripts to the pipeline.
# Monitoring: View task logs and status in the Airflow UI.
# Scheduling: Set up a schedule for periodic execution of the pipeline.
# Dependency Management: Define task dependencies for a sequential workflow.
# Flexibility: Integrate with other Airflow features like sensors, triggers, and operators.
# Note: Make sure to install Apache Airflow and set up the necessary configurations before running the DAG.
# Summary