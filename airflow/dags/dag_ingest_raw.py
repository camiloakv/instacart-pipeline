"""
DAG: dag_ingest_raw
Loads the 6 Instacart CSV files into the Postgres 'raw' schema.
Each file gets its own task so failures are isolated and tasks run in parallel.

Note: loading uses psycopg2's COPY directly rather than pandas.to_sql().
Pandas >=2.0's SQL I/O requires SQLAlchemy >=2.0, but Airflow 2.9.x pins
SQLAlchemy 1.4.x internally, so pandas.to_sql() is not usable in this
environment. COPY is also the faster, more standard way to bulk-load into
Postgres regardless.
"""

from datetime import datetime
from io import StringIO

import pandas as pd
import psycopg2

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

# Maps pandas dtype kind -> Postgres column type, for auto-generating CREATE TABLE
DTYPE_TO_PG = {
    "i": "BIGINT",
    "f": "DOUBLE PRECISION",
    "O": "TEXT",
    "b": "BOOLEAN",
}


def _pg_type_for(dtype) -> str:
    return DTYPE_TO_PG.get(dtype.kind, "TEXT")


def load_table_to_postgres(table_name: str, csv_filename: str):
    import os

    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        dbname=os.environ["POSTGRES_DB"],
    )
    conn.autocommit = False

    csv_path = f"{DATA_DIR}/{csv_filename}"
    # chunksize keeps memory bounded for the larger files (order_products__prior is ~32M rows)
    chunk_iter = pd.read_csv(csv_path, chunksize=200_000)

    with conn.cursor() as cur:
        for i, chunk in enumerate(chunk_iter):
            if i == 0:
                # Build CREATE TABLE from the first chunk's inferred dtypes
                cols_sql = ", ".join(
                    f'"{col}" {_pg_type_for(dtype)}'
                    for col, dtype in zip(chunk.columns, chunk.dtypes)
                )
                cur.execute(f'DROP TABLE IF EXISTS raw."{table_name}";')
                cur.execute(f'CREATE TABLE raw."{table_name}" ({cols_sql});')

            # Stream the chunk into a CSV buffer, then COPY it in
            buffer = StringIO()
            chunk.to_csv(buffer, index=False, header=False)
            buffer.seek(0)
            cur.copy_expert(
                f'COPY raw."{table_name}" FROM STDIN WITH CSV', buffer
            )
            conn.commit()

    conn.close()


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
