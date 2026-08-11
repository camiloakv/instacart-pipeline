#!/bin/bash
# Creates a read-only Postgres role for the FastAPI service, scoped to
# SELECT-only access on the dbt_dev schema. Runs automatically on Postgres's
# first-ever startup (docker-entrypoint-initdb.d convention), using the
# API_READER_PASSWORD env var passed into the postgres container.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE ROLE api_reader WITH LOGIN PASSWORD '$API_READER_PASSWORD';
    GRANT CONNECT ON DATABASE $POSTGRES_DB TO api_reader;
    GRANT USAGE ON SCHEMA dbt_dev TO api_reader;
    GRANT SELECT ON ALL TABLES IN SCHEMA dbt_dev TO api_reader;
    ALTER DEFAULT PRIVILEGES IN SCHEMA dbt_dev GRANT SELECT ON TABLES TO api_reader;
EOSQL
