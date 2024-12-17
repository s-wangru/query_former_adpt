#!/usr/bin/env bash

set -euo pipefail

set -a
source .env
set +a

ROOT_DIR="$(pwd)"

if [ ! -d "${TPCH_DIR}" ]; then
  echo "Cloning: ${TPCH_DIR}."
  git clone "${TPCH_REPO}" "${TPCH_DIR}"
fi

cd "${TPCH_DIR}/dbgen"
make MACHINE=LINUX DATABASE=POSTGRESQL
cd "${ROOT_DIR}"

# Start seed, inclusive.
TPCH_QUERY_START='15721'
# End seed, inclusive.
TPCH_QUERY_STOP='15721'
./setup/tpch_queries.sh "${TPCH_DIR}" "${TPCH_ARTIFACT_DIR}" "${TPCH_QUERY_START}" "${TPCH_QUERY_STOP}"

# List of scale factors to pre-generate data for.
TPCH_SFS=("1" "10" "100")
for sf in "${TPCH_SFS[@]}"; do
  ./setup/tpch_data.sh "${TPCH_DIR}" "${TPCH_ARTIFACT_DIR}" "${sf}"
done
