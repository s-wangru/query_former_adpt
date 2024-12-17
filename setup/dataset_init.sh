#!/usr/bin/env bash

set -euo pipefail

set -a
source .env
set +a

HOSTNAME=$(hostname)
SPEC_FOLDER="${RESULT_ARTIFACT_ROOT}/${HOSTNAME}/spec"
mkdir -p "${SPEC_FOLDER}"

cat /proc/cpuinfo > ${SPEC_FOLDER}/cpuinfo.txt
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > ${SPEC_FOLDER}/scaling_governor.txt
cat /sys/devices/system/cpu/intel_pstate/no_turbo > ${SPEC_FOLDER}/no_turbo.txt
cat /sys/kernel/mm/transparent_hugepage/enabled > ${SPEC_FOLDER}/transparent_hugepage_enabled.txt
cat /sys/kernel/mm/transparent_hugepage/defrag > ${SPEC_FOLDER}/transparent_hugepage_defrag.txt
lsblk -o NAME,FSTYPE,ROTA,MOUNTPOINT,MODEL > ${SPEC_FOLDER}/lsblk.txt
sudo lshw -c memory > ${SPEC_FOLDER}/lshw_memory.txt
