#!/usr/bin/env bash

set -euo pipefail

set -a
source .env
set +a

TGT="$1"
CMD="$2"
ARG1="${3-}"
ARG2="${4-}"

ROOT_DIR="$(pwd)"

PG_VER="${TGT}"
PG_DIR="${PG_ARTIFACT_DIR}/${PG_VER}"
PG_BIN_DIR="${PG_ARTIFACT_DIR}/${PG_VER}/bin/"
PGDATA="${PG_DATA_DIR}/${PG_VER}/"

if [ "${CMD}" = "start" ]; then
  ${PG_BIN_DIR}/pg_ctl -D "${PGDATA}" -l "${PGDATA}/pg.log" start -o "-p ${PG_PORT}"
  until ${PG_BIN_DIR}/pg_isready -p ${PG_PORT} &> /dev/null; do sleep 1; done
elif [ "${CMD}" = "stop" ]; then
  ${PG_BIN_DIR}/pg_ctl -D "${PGDATA}" stop -m fast
elif [ "${CMD}" = "psql" ]; then
  ${PG_BIN_DIR}/psql -h "${PG_HOST}" -p "${PG_PORT}" -U "${PG_USER}" -d "${ARG1}"
fi
