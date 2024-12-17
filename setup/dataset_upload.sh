#!/usr/bin/env bash

set -euo pipefail

set -a
source .env
set +a

python3 ./setup/dataset_upload_tpch.py
