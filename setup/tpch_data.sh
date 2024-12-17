#!/usr/bin/env bash

set -euo pipefail

TPCH_DIR="$1"
TPCH_ARTIFACT_DIR="$2"
TPCH_SF="$3"
TPCH_DATA_DIR="${TPCH_ARTIFACT_DIR}/data/sf_${TPCH_SF}/"

ROOT_DIR="$(pwd)"

cd "${TPCH_DIR}/dbgen"
echo "Generating TPC-H data in ${TPCH_DATA_DIR}."
if [ ! -d "${TPCH_DATA_DIR}" ]; then
    ./dbgen -vf -s "${TPCH_SF}"
    mkdir -p "${TPCH_DATA_DIR}/"
    mv ./*.tbl "${TPCH_DATA_DIR}/"
fi
echo "Generated TPC-H data in ${TPCH_DATA_DIR}."
cd "${ROOT_DIR}"
