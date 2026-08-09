Simple yet complete ML pipeline ready for production.

<p>
<img src="https://img.shields.io/badge/PostgreSQL-316192?style=flat&logo=postgresql&logoColor=white"/>
<img src="https://img.shields.io/badge/dbt-FF694B?style=flat&logo=dbt&logoColor=white"/>
<img src="https://img.shields.io/badge/Airflow-017CEE?style=flat&logo=Apache%20Airflow&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-404D59?style=flat&logo=docker"/>
<img src="https://img.shields.io/badge/XGBoost-FFFFFF?style=flat&color=2887D7"/>
<img src="https://img.shields.io/badge/FastAPI-109989?style=flat&logo=FASTAPI&logoColor=white"/>
</p>


From the [Instacart market basket analysis dataset](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis), trains and deploys a XGBoost model to predict the reorder probability for a (user, product) pair. Uses standard best practices such as:
 - Orchestration via Airflow.
 - ELT via DBT.
 - Containerization via Docker.
 - Serving via FastAPI.

<!--

## Pipeline

1. Ingest raw data from local csv files into a postgres database.
2.

Simple ML model served via API
-->

## Fire it up!

- Download the [Kaggle dataset](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) (six csv files) into the `data/raw/` folder (yep, manually).

- Add these to your `.env`:
  ```bash
  POSTGRES_USER=some_name
  POSTGRES_PASSWORD=some_password
  POSTGRES_DB=some_dbname
  POSTGRES_HOST=postgres
  POSTGRES_PORT=5432
  AIRFLOW_ADMIN_USER=some_name
  AIRFLOW_ADMIN_PASSWORD=some_password
  ```

- Open Docker Desktop to start the engine.

- Set up the containers:
  ```bash
  # build all images from scratch, it could take a few minutes
  docker compose build --no-cache

  # start everything
  docker compose up -d

  # one-time dbt package install
  docker compose exec airflow-scheduler bash -c "cd /opt/airflow/dbt && dbt deps"

  # kick off the whole chain
  docker compose exec airflow-scheduler airflow dags trigger dag_ingest_raw

  # optionally, check the status of the project
  docker compose ps -a
  ```

- Check/trigger DAGs at `http://localhost:8080/home` > `dag_ingest_raw`. Login with `AIRFLOW_ADMIN_USER`, `AIRFLOW_ADMIN_PASSWORD` set above.
- (_Optionally_) Query from dbt view, e.g.:
  ```bash
  docker compose exec postgres psql -U airflow_user -d instacart -c "SELECT user_id, product_id FROM dbt_dev.fct_user_product_features LIMIT 1;"
  ```

- Make an API request:
  ```bash
  Invoke-WebRequest -UseBasicParsing -Uri http://localhost:8000/predict `
    -Method POST `
    -ContentType "application/json" `
    -Body '{"user_id": 175009, "product_id": 40706}'
  ```


## Features

- ✅ Postgres (raw data)
- ✅ Airflow (ingestion DAG, parallel tasks)
- ✅ dbt (staging → intermediate → marts, with tests + version-controllable models)
- ✅ XGBoost (trained on dbt's marts output)
- ✅ FastAPI (serving live predictions from the trained model)


## Backlog

- Download using Kaggle API token.
- Simple frontend to select user and product.
