#!/usr/bin/env bash

# Run one stable 32-paper segment of the confirmed 64-paper Ours cohort.
# The same command path and frozen workflow are used for both segment values.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "$#" -ne 1 || ( "$1" != "0" && "$1" != "1" ) ]]; then
  echo "Usage: bash scripts/run_ours_paired64_parallel.sh <0|1>" >&2
  exit 2
fi

SHARD_INDEX="$1"
if [[ "${SHARD_INDEX}" == "0" ]]; then
  SEGMENT_LABEL="a"
else
  SEGMENT_LABEL="b"
fi

INPUT_DIR="${OURS_PAIRED64_INPUT_DIR:-${REPO_ROOT}/data/nlp}"
OUTPUT_DIR="${OURS_PAIRED64_OUTPUT_DIR:-${REPO_ROOT}/outputs/ours_paired64_segment_${SEGMENT_LABEL}}"
BATCH_NAME="${OURS_PAIRED64_BATCH_NAME:-ours_paired64}"
WORKERS="${OURS_PAIRED64_WORKERS:-8}"

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input directory does not exist: ${INPUT_DIR}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" || -L "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output path: ${OUTPUT_DIR}" >&2
  exit 2
fi
if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OURS_PAIRED64_WORKERS must be a positive integer: ${WORKERS}" >&2
  exit 2
fi

# Keep the established real-run timeout unless explicitly overridden.
export DEEPSEEK_TIMEOUT_SECONDS="${DEEPSEEK_TIMEOUT_SECONDS:-180}"

echo "Starting Ours 64-paper segment ${SEGMENT_LABEL}: 32 papers"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Workers: ${WORKERS}"
echo "Order: confirmed cohort, filename ascending"

exec env PYTHONPATH=src \
  conda run -n myrag python scripts/run_ours_paired64_parallel.py \
  --shard-index "${SHARD_INDEX}" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --workers "${WORKERS}" \
  --batch-name "${BATCH_NAME}" \
  --semantic-scholar-config config/semantic_scholar.json
