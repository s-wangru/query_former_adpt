#!/usr/bin/env bash

set -euo pipefail

set -a
source .env
set +a

TGT="${1-${PG_PORT}}"

sudo kill $(lsof -i tcp:15721 | awk 'NR!=1 {print $2}')