"""
DAG: dag_ingest_raw
Loads the 6 Instacart CSV files into the Postgres 'raw' schema.
Each file gets its own task so failures are isolated and tasks run in parallel.
"""

from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine

from airflow import DAG
from airflow.operators.python import PythonOperator

# Maps table name -> CSV filename. All files live under /opt/airflow/data/raw
# (mounted from ./data/raw on the host, per docker-compose.yml)
TABLES = {
    "orders": "orders.csv",
    "order_products_prior": "order_products__prior.csv",
    "order_products_train": "order_products__train.csv",
    "products": "products.csv",
    "aisles": "aisles.csv",
    "departments": "departments.csv",
}

DATA_DIR = "/opt/airflow/data/raw"

# Connection string points at the 'postgres' service name (Docker network),
# credentials come from Airflow's own container env (same .env as docker-compose.yml)
PG_CONN_STRING = "postgresql+psycopg2://{user}:{password}@postgres:5432/{db}"


def load_table_to_postgres(table_name: str, csv_filename: str):
    import os

    conn_string = PG_CONN_STRING.format(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        db=os.environ["POSTGRES_DB"],
    )
    engine = create_engine(conn_string)

    csv_path = f"{DATA_DIR}/{csv_filename}"
    # chunksize keeps memory bounded for the larger files (order_products__prior is ~32M rows)
    chunk_iter = pd.read_csv(csv_path, chunksize=200_000)

    for i, chunk in enumerate(chunk_iter):
        chunk.to_sql(
            name=table_name,
            con=engine,
            schema="raw",
            if_exists="replace" if i == 0 else "append",
            index=False,
        )

    engine.dispose()


with DAG(
    dag_id="dag_ingest_raw",
    description="Load Instacart CSVs into Postgres raw schema",
    schedule=None,  # manual trigger for now; can add a cron schedule later
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ingest", "instacart"],
) as dag:

    for table_name, csv_filename in TABLES.items():
        PythonOperator(
            task_id=f"load_{table_name}",
            python_callable=load_table_to_postgres,
            op_kwargs={"table_name": table_name, "csv_filename": csv_filename},
        )
