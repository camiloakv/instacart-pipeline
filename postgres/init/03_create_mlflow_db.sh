#!/bin/bash
# Creates a separate 'mlflow' database (isolated from 'instacart') for
# MLflow's backend store, with its own dedicated role. Ownership is scoped
# to just this one throwaway database -- not superuser, not visibility
# into 'instacart' -- consistent with the least-privilege pattern used
# for api_reader/db_viewer elsewhere in this project.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE mlflow;
    CREATE ROLE mlflow_user WITH LOGIN PASSWORD '$MLFLOW_DB_PASSWORD';
    ALTER DATABASE mlflow OWNER TO mlflow_user;
EOSQL
