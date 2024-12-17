#!/usr/bin/env bash

set -euo pipefail

TPCH_DIR="$1"
TPCH_ARTIFACT_DIR="$2"
TPCH_QUERY_START="$3"
TPCH_QUERY_STOP="$4"
TPCH_QUERY_DIR="${TPCH_ARTIFACT_DIR}/queries"

ROOT_DIR="$(pwd)"

cd "${TPCH_DIR}/dbgen"
echo "Generating TPC-H queries in ${TPCH_QUERY_DIR}."
for seed in $(seq "${TPCH_QUERY_START}" "${TPCH_QUERY_STOP}"); do
  if [ ! -d "${TPCH_QUERY_DIR}/${seed}" ]; then
    mkdir -p "${TPCH_QUERY_DIR}/${seed}"
    for qnum in {1..22}; do
      DSS_QUERY="./queries" ./qgen "${qnum}" -r "${seed}" > "${TPCH_QUERY_DIR}/${seed}/${qnum}.sql"
    done
  fi
done
echo "Generated TPC-H queries in ${TPCH_QUERY_DIR}."
cd "${ROOT_DIR}"
