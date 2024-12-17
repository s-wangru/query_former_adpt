#!/bin/env bash

TAGS_DIR="$1"
if [ -z "${TAGS_DIR}" ]; then
  TAGS_DIR="$(pwd)/"
fi

mkdir -p ${TAGS_DIR}
cd ${TAGS_DIR}
for i in {1..999}; do
    dst="pg_tags_${i}.json"
    curl "https://api.github.com/repos/postgres/postgres/tags?per_page=100&page=${i}" > ${dst}
    if [[ $(jq length ${dst}) -eq 0 ]]; then
        rm ${dst}
        break
    fi
done
jq -s add pg_tags_*.json > pg_tags.json
rm pg_tags_*.json
cd -

# Then some hand editing to get the latest version.
# "pg8": "REL8_4_22"
# pg8 doesn't compile out of the box any more due to Perl changes.
