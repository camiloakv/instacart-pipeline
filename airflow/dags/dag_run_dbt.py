"""
DAG: dag_run_dbt
Runs the full dbt project (staging -> intermediate -> marts) and its tests.
Triggered automatically after dag_ingest_raw completes successfully.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

DBT_DIR = "/opt/airflow/dbt"

with DAG(
    dag_id="dag_run_dbt",
    description="Run dbt models and tests on top of the raw Instacart tables",
    schedule=None,  # triggered by dag_ingest_raw, or run manually
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["dbt", "instacart"],
) as dag:

    dbt_run = BashOperator(
        task_id="dbt_run",
        bash_command=f"cd {DBT_DIR} && dbt run",
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        bash_command=f"cd {DBT_DIR} && dbt test",
    )

    trigger_training = TriggerDagRunOperator(
        task_id="trigger_train_model",
        trigger_dag_id="dag_train_model",
    )

    dbt_run >> dbt_test >> trigger_training
