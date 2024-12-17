#!/usr/bin/env bash

set -euo pipefail

set -a
source .env
set +a

ROOT_DIR="$(pwd)"
HOSTNAME="$(hostname)"

if [ ! -d "${PG_DIR}" ]; then
  echo "Cloning: ${PG_DIR}."
  git clone "${PG_REPO}" "${PG_DIR}"
fi

set +x
declare -A tags
while IFS="=" read -r key value; do
  tags[${key}]="${value}"
done < <(jq --raw-output 'to_entries[] | "\(.key)=\(.value)"' ./setup/postgres_tags.json)

for key in "${!tags[@]}"; do
  CHECKOUT_NAME="${key}"
  CHECKOUT_TAG="${tags[${key}]}"
  PG_BUILD_DEST="${PG_ARTIFACT_DIR}/${CHECKOUT_NAME}"
  PG_BIN_DIR="${PG_BUILD_DEST}/bin"
  PGDATA="${PG_DATA_DIR}/${CHECKOUT_NAME}"
  PG_LOG_DEST="${PGDATA}/${CHECKOUT_NAME}.log"
  PGTUNE_CONFIG="${ROOT_DIR}/setup/pgtune_${CHECKOUT_NAME}_${HOSTNAME}.sql"

  # Clone.
  if [ -d "${PG_BUILD_DEST}" ]; then
    echo "Skipping clone: ${CHECKOUT_NAME} ${CHECKOUT_TAG}"
  else
    echo "Cloning: ${CHECKOUT_NAME} ${CHECKOUT_TAG}"
    cd ${PG_DIR}
    git checkout "${CHECKOUT_TAG}"
    git reset --hard HEAD
    git clean -fd
    set +e
    make clean
    set -e
    mkdir -p "${PG_BUILD_DEST}"
    cd ${ROOT_DIR}
    ./setup/postgres_config.sh "release" ${PG_BUILD_DEST} "${PG_DIR}"
    cd ${PG_DIR}
    make install -j40
    # pg_prewarm
    make install-world-bin -j40
  fi

  # Init.
  if [ -d "${PGDATA}" ]; then
    echo "Skipping init: ${CHECKOUT_NAME} ${CHECKOUT_TAG}"
  else
    echo "Init: ${CHECKOUT_NAME} ${CHECKOUT_TAG}"
    cd ${PG_BIN_DIR}
    ./pg_ctl initdb -D "${PGDATA}"
    # Start PG.
    ./pg_ctl start -D "${PGDATA}" -l "${PG_LOG_DEST}" -o "-p ${PG_PORT}"
    until ./pg_isready -p ${PG_PORT} &> /dev/null; do sleep 1; done
    # Create user and default DB.
    ./psql -h "${PG_HOST}" -p "${PG_PORT}" -c "create user ${PG_USER} with login superuser password '${PG_PASS}'" postgres
    ./psql -h "${PG_HOST}" -p "${PG_PORT}" -c "create database ${PG_DB_PREFIX} with owner = '${PG_USER}'" postgres
    # PGTune.
    ./psql -h "${PG_HOST}" -p "${PG_PORT}" -f "${PGTUNE_CONFIG}" postgres
    # Stop PG.
    ./pg_ctl stop -D "${PGDATA}" -m fast
  fi
done

