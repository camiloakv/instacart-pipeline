"""
DAG: dag_train_model
Trains the XGBoost reorder-prediction model from dbt's marts table.
Triggered automatically after dag_run_dbt completes successfully.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id="dag_train_model",
    description="Train the XGBoost reorder-prediction model",
    schedule=None,  # triggered by dag_run_dbt, or run manually
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "instacart"],
) as dag:

    train_model = BashOperator(
        task_id="train_model",
        bash_command="python /opt/airflow/ml/train.py",
    )
