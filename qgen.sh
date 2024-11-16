
set -euxo pipefail

cd "${DSS_CONFIG}"
set +x
for seed in $(seq "${TPCH_QUERY_START}" "${TPCH_QUERY_STOP}"); do
  if [ ! -d "${DSS_CONFIG}/${seed}" ]; then
    mkdir -p "${DSS_CONFIG}/${seed}"
    for qnum in {1..22}; do
      DSS_QUERY="./queries" ./qgen "${qnum}" -r "${seed}" > "${DSS_CONFIG}/${seed}/${qnum}.sql"
    done
  fi
done
set -x
cd "${ROOT_DIR}"