#!/usr/bin/env bash

set -euo pipefail

set -a
source .env
set +a

ROOT_DIR="$(pwd)"
export HOSTNAME="$(hostname)"

# Setup TPC-H.
./setup/tpch_setup.sh

# Setup all versions of PostgreSQL.
./setup/postgres_setup.sh

# Read all versions of PostgreSQL.
declare -A PG_TAGS
declare -a PG_TAGS_ORDER
while IFS="=" read -r key value; do
  PG_TAGS[${key}]="${value}"
  PG_TAGS_ORDER+=("${key}")
done < <(jq --raw-output 'to_entries[] | "\(.key)=\(.value)"' ./setup/postgres_tags.json)

# Load TPC-H.
export TPCH_SCHEMA_ROOT="${ROOT_DIR}/setup/"
export TPCH_DATA_ROOT="${TPCH_ARTIFACT_DIR}/data/"
TPCH_SFS=("1" "10")

for PG_VER in "${PG_TAGS_ORDER[@]}"; do
  PG_DIR="${PG_ARTIFACT_DIR}/${PG_VER}"
  PG_BIN_DIR="${PG_ARTIFACT_DIR}/${PG_VER}/bin/"
  PGDATA="${PG_DATA_DIR}/${PG_VER}/"
  RESULT_DIR="${RESULT_ARTIFACT_ROOT}/${HOSTNAME}/${PG_VER}/"
  
  echo "Running: ${PG_VER}"

  # Start PG.
  ${PG_BIN_DIR}/pg_ctl -D "${PGDATA}" -l "${PGDATA}/pg.log" start -o "-p ${PG_PORT}"
  until ${PG_BIN_DIR}/pg_isready -p ${PG_PORT} &> /dev/null; do sleep 1; done
  # Load TPC-H if necessary.
  for tpch_sf in "${TPCH_SFS[@]}"; do
    export TPCH_SF="${tpch_sf}"
    export PG_DB="${PG_DB_PREFIX}_tpch_sf_${TPCH_SF}"
    # Create DB if not exists.
    ${PG_BIN_DIR}/psql -h "${PG_HOST}" -p "${PG_PORT}" -U "${PG_USER}" -tc "SELECT 1 FROM pg_database WHERE datname = '${PG_DB}'" -d "${PG_DB_PREFIX}" | grep -q 1 || ${PG_BIN_DIR}/psql -h "${PG_HOST}" -p "${PG_PORT}" -c "create database ${PG_DB} with owner = '${PG_USER}'" postgres
    python3 ./setup/tpch_load.py
    # Collect TPC-H plans.
    export TPCH_QUERY_START=15721
    export TPCH_QUERY_STOP=15721
    export TPCH_OUTPUT_DIR="${RESULT_DIR}/tpch/sf_${TPCH_SF}/"
    python3 ./setup/tpch_run.py
  done

  # Stop PG.
  ${PG_BIN_DIR}/pg_ctl -D "${PGDATA}" stop -m fast
done

python3 ./setup/dataset_result_tpch.py
